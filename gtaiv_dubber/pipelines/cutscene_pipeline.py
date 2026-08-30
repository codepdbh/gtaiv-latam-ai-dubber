from __future__ import annotations

from pathlib import Path

import numpy as np

from gtaiv_dubber.ai.demucs_adapter import load_vocal_stem
from gtaiv_dubber.audio.alignment import align_to_duration, fit_frame_count
from gtaiv_dubber.audio.io import mono_mix, read_audio, write_pcm16
from gtaiv_dubber.audio.loudness import limit_peak, match_rms, style_hints
from gtaiv_dubber.audio.metadata import read_wav_metadata
from gtaiv_dubber.audio.mixing import replace_dialogue_region
from gtaiv_dubber.audio.reference import extract_reference
from gtaiv_dubber.audio.resampling import resample_audio
from gtaiv_dubber.audio.spatial import estimate_spatial_weights
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
        output = original.copy()
        failures = 0
        review_overlap = False
        review_duration = False
        translations: list[str] = []
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
                extract_reference(
                    vocal_mono, sample_rate, segment["start"], segment["end"],
                    float(self.config["cutscenes"].get("reference_padding_seconds", 0.75)),
                    reference_path,
                )
                first = max(0, round(segment["start"] * sample_rate))
                last = min(metadata.frames, round(segment["end"] * sample_rate))
                original_region = original[first:last]
                vocal_region = vocal_mono[first:last]
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
                aligned = match_rms(aligned, vocal_region)
                weights = estimate_spatial_weights(original_region)
                replace_dialogue_region(
                    output, aligned[:, 0], first, last, weights, vocal_region,
                    attenuation=float(self.config["cutscenes"].get("vocal_attenuation", 0.75)),
                )
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
