from __future__ import annotations

from pathlib import Path

import numpy as np

from gtaiv_dubber.ai.demucs_adapter import load_vocal_stem
from gtaiv_dubber.audio.alignment import align_to_duration, fit_frame_count
from gtaiv_dubber.audio.io import mono_mix, read_audio, write_pcm16
from gtaiv_dubber.audio.loudness import limit_peak, match_rms, style_hints
from gtaiv_dubber.audio.metadata import read_wav_metadata
from gtaiv_dubber.audio.mixing import render_dialogue_replacements
from gtaiv_dubber.audio.reference import extract_reference
from gtaiv_dubber.audio.resampling import resample_audio
from gtaiv_dubber.audio.speakers import replace_overlapped_segments, simultaneous_source_segments
from gtaiv_dubber.audio.spatial import estimate_reference_weights
from gtaiv_dubber.audio.validation import validate_wav
from gtaiv_dubber.core.pipeline import BasePipeline, ProcessResult
from gtaiv_dubber.core.state import Status
from gtaiv_dubber.utils.atomic import atomic_copy_wav, atomic_publish
from gtaiv_dubber.utils.hashing import stable_key


class CutscenePipeline(BasePipeline):
    mode = "cutscene"
    exact_frames = True

    def _process(self, row, source: Path, destination: Path, digest: str) -> ProcessResult:
        file_id = int(row["id"])
        self.db.update_file(file_id, status=Status.ANALYZING, error_message=None)
        metadata = read_wav_metadata(source)
        original, sample_rate = read_audio(source)
        analysis_mono = mono_mix(original)
        key = stable_key(self.mode, row["relative_path"], digest)[:24]
        analysis_path = self.runtime["tmp"] / f"analysis_{key}.wav"
        write_pcm16(analysis_path, analysis_mono, sample_rate)
        self.logger.info("[SEPARATE] %s", row["relative_path"])
        vocals_path, _background_path = self.models.demucs.separate(
            analysis_mono, sample_rate, f"{row['relative_path']}:{digest}"
        )
        vocal_mono = load_vocal_stem(vocals_path, sample_rate, metadata.frames)
        self.models.unload_cosyvoice()
        self.logger.info("[ASR] %s", row["relative_path"])
        transcript_segments = self.models.whisper.transcribe(analysis_path)
        segments = [
            {
                "start": max(0.0, item.start),
                "end": min(metadata.duration, item.end),
                "text": item.text,
                "confidence": item.confidence,
                "status": Status.TRANSCRIBED,
            }
            for item in transcript_segments if item.end > item.start
        ]
        speaker_sources: dict[int, np.ndarray] = {}
        review_overlap = False
        separation_config = self.config.get("speaker_separation", {})
        if segments and bool(separation_config.get("enabled", True)):
            try:
                self.logger.info("[SPEAKERS] Buscando voces simultáneas en %s", row["relative_path"])
                separated = self.models.speaker_separator.separate(
                    vocal_mono, sample_rate, f"{row['relative_path']}:{digest}"
                )
                self.models.unload_speaker_separator()
                by_source: list[list[dict]] = []
                for source_index in range(min(2, separated.shape[1])):
                    separated_path = self.runtime["tmp"] / f"speakers_{key}_{source_index}.wav"
                    write_pcm16(separated_path, separated[:, source_index], sample_rate)
                    source_transcript = self.models.whisper.transcribe(separated_path)
                    separated_path.unlink(missing_ok=True)
                    by_source.append(
                        [
                            {
                                "start": max(0.0, item.start),
                                "end": min(metadata.duration, item.end),
                                "text": item.text,
                                "confidence": item.confidence,
                                "status": Status.TRANSCRIBED,
                            }
                            for item in source_transcript if item.end > item.start
                        ]
                    )
                simultaneous = simultaneous_source_segments(
                    by_source,
                    separated,
                    sample_rate,
                    min_overlap_seconds=float(separation_config.get("min_overlap_seconds", 0.12)),
                    min_source_rms_ratio=float(
                        separation_config.get("min_source_rms_ratio", 0.12)
                    ),
                    duplicate_similarity=float(
                        separation_config.get("duplicate_similarity", 0.72)
                    ),
                    activity_threshold=float(
                        separation_config.get("activity_threshold", 0.15)
                    ),
                    max_source_correlation=float(
                        separation_config.get("max_source_correlation", 0.25)
                    ),
                )
                if simultaneous:
                    segments = replace_overlapped_segments(segments, simultaneous)
                    speaker_sources = {
                        index: separated[:, index] for index in range(min(2, separated.shape[1]))
                    }
                    review_overlap = True
                    self.logger.warning(
                        "[SPEAKERS] %s: %d intervenciones simultáneas separadas",
                        row["relative_path"], len(simultaneous),
                    )
            except Exception as exc:
                self.logger.warning(
                    "[SPEAKERS] %s: separación no disponible; se conserva revisión manual: %s",
                    row["relative_path"], exc,
                )
            finally:
                self.models.unload_speaker_separator()
        self.db.replace_segments(file_id, segments)
        transcript = " ".join(item["text"] for item in segments)
        self.db.update_file(file_id, transcript_en=transcript, status=Status.TRANSCRIBED)
        self.models.unload_whisper()
        if not segments:
            original_meta = metadata

            def validator(path: Path):
                result = validate_wav(path, original_meta, exact_frames=True)
                if not result.valid:
                    raise ValueError(result.reason)

            atomic_copy_wav(source, destination, self.runtime["tmp"], validator)
            self.db.update_file(
                file_id, sha256=digest, config_hash=self.config_hash,
                status=Status.DONE, error_message="Sin diálogo detectable; original conservado",
            )
            return ProcessResult(Status.DONE, row["relative_path"], "Sin diálogo")
        failures = 0
        review_duration = False
        translations: list[str] = []
        dialogue_tracks: list[tuple[np.ndarray, int, int, np.ndarray]] = []
        for index, segment in enumerate(segments):
            if index and segment["start"] < segments[index - 1]["end"]:
                review_overlap = True
            try:
                previous = segments[index - 1]["text"] if index else ""
                following = segments[index + 1]["text"] if index + 1 < len(segments) else ""
                duration = segment["end"] - segment["start"]
                self.logger.info("[TRANSLATE] %s segmento %d", row["relative_path"], index)
                translation = self.models.qwen.adapt(
                    segment["text"], duration, previous_line=previous, next_line=following,
                    relative_path=row["relative_path"],
                )
                translations.append(translation)
                reference_path = self.runtime["references"] / "cutscenes" / f"{key}_{index:04d}.wav"
                speaker_source = segment.get("speaker_source")
                reference_audio = speaker_sources.get(speaker_source, vocal_mono)
                reference_padding = (
                    0.15 if speaker_source is not None
                    else float(self.config["cutscenes"].get("reference_padding_seconds", 0.75))
                )
                extract_reference(
                    reference_audio, sample_rate, segment["start"], segment["end"],
                    reference_padding,
                    reference_path,
                )
                first = max(0, round(segment["start"] * sample_rate))
                last = min(metadata.frames, round(segment["end"] * sample_rate))
                original_region = original[first:last]
                vocal_region = vocal_mono[first:last]
                level_reference = reference_audio[first:last]
                hints = style_hints(original_region, sample_rate)
                generated, generated_rate = self.synthesize_with_oom_retry(
                    reference_path, translation, duration, hints
                )
                generated = resample_audio(generated, generated_rate, sample_rate)
                aligned, error_pct = align_to_duration(
                    generated, sample_rate, duration, self.runtime["tmp"] / f"{key}_{index}"
                )
                if error_pct > self.config["alignment"]["warning_pct"]:
                    instruction = "Haz la frase más corta." if len(generated) > last - first else "Haz la frase más larga."
                    translation = self.models.qwen.adapt(
                        segment["text"], duration, previous_line=previous, next_line=following,
                        relative_path=row["relative_path"], length_instruction=instruction,
                    )
                    translations[-1] = translation
                    generated, generated_rate = self.synthesize_with_oom_retry(
                        reference_path, translation, duration, hints
                    )
                    generated = resample_audio(generated, generated_rate, sample_rate)
                    aligned, error_pct = align_to_duration(
                        generated, sample_rate, duration, self.runtime["tmp"] / f"{key}_{index}"
                    )
                aligned = match_rms(aligned, level_reference)
                weights = estimate_reference_weights(original_region, vocal_region)
                dialogue_tracks.append((aligned[:, 0], first, last, weights))
                segment_status = (
                    Status.REVIEW_DURATION
                    if error_pct > self.config["alignment"]["warning_pct"]
                    else Status.REVIEW_OVERLAP if review_overlap else Status.DONE
                )
                review_duration = review_duration or segment_status is Status.REVIEW_DURATION
                self.db.update_segment(
                    file_id, index, translation_es=translation,
                    reference_path=str(reference_path), generated_duration=len(generated) / sample_rate,
                    duration_error_pct=error_pct, status=segment_status, error_message=None,
                )
            except Exception as exc:
                failures += 1
                translations.append("")
                self.db.update_segment(
                    file_id, index, status=Status.REVIEW_AUDIO, error_message=str(exc)
                )
                self.logger.error("[REVIEW] %s segmento %d: %s", row["relative_path"], index, exc)
                continue
        output = render_dialogue_replacements(
            original,
            vocal_mono,
            dialogue_tracks,
            attenuation=float(self.config["cutscenes"].get("vocal_attenuation", 1.0)),
        )
        output = fit_frame_count(output, metadata.frames)
        output = limit_peak(output, self.config["audio"]["peak_limit_dbfs"])

        def producer(path: Path):
            write_pcm16(path, output, metadata.sample_rate)

        def validator(path: Path):
            result = validate_wav(path, metadata, exact_frames=True)
            if not result.valid:
                raise ValueError(result.reason)

        atomic_publish(destination, self.runtime["tmp"], producer, validator)
        if failures == len(segments):
            status = Status.FALLBACK
        elif failures:
            status = Status.REVIEW_AUDIO
        elif review_overlap:
            status = Status.REVIEW_OVERLAP
        elif review_duration:
            status = Status.REVIEW_DURATION
        else:
            status = Status.DONE
        self.db.update_file(
            file_id, sha256=digest, config_hash=self.config_hash,
            translation_es=" ".join(filter(None, translations)), status=status,
            error_message=f"{failures} segmentos conservados en inglés" if failures else None,
        )
        analysis_path.unlink(missing_ok=True)
        self.logger.info("[EXPORT] %s", row["relative_path"])
        return ProcessResult(status, row["relative_path"])
