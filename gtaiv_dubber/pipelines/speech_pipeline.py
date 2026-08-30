from __future__ import annotations

from pathlib import Path

import numpy as np

from gtaiv_dubber.audio.alignment import align_to_duration
from gtaiv_dubber.audio.channels import preserve_channel_count
from gtaiv_dubber.audio.io import read_audio, write_pcm16
from gtaiv_dubber.audio.loudness import limit_peak, match_rms, style_hints
from gtaiv_dubber.audio.metadata import read_wav_metadata
from gtaiv_dubber.audio.reference import extract_reference
from gtaiv_dubber.audio.resampling import resample_audio
from gtaiv_dubber.audio.spatial import estimate_spatial_weights
from gtaiv_dubber.audio.speech_detection import SpeechKind, classify_speech
from gtaiv_dubber.audio.validation import validate_wav
from gtaiv_dubber.core.pipeline import BasePipeline, ProcessResult
from gtaiv_dubber.core.state import Status
from gtaiv_dubber.utils.atomic import atomic_copy_wav, atomic_publish
from gtaiv_dubber.utils.hashing import stable_key


class SpeechPipeline(BasePipeline):
    mode = "speech"
    exact_frames = True

    def _publish_original(self, row, source, destination, digest, transcript, reason) -> ProcessResult:
        original = read_wav_metadata(source)

        def validator(path: Path):
            result = validate_wav(path, original, exact_frames=True)
            if not result.valid:
                raise ValueError(result.reason)

        atomic_copy_wav(source, destination, self.runtime["tmp"], validator)
        self.db.update_file(
            int(row["id"]), sha256=digest, config_hash=self.config_hash,
            transcript_en=transcript, status=Status.DONE, error_message=reason,
        )
        return ProcessResult(Status.DONE, row["relative_path"], reason)

    def _process(self, row, source: Path, destination: Path, digest: str) -> ProcessResult:
        self.db.update_file(int(row["id"]), status=Status.ANALYZING, error_message=None)
        original_meta = read_wav_metadata(source)
        original_audio, sample_rate = read_audio(source)
        self.models.unload_cosyvoice()
        self.logger.info("[ASR] %s", row["relative_path"])
        segments = self.models.whisper.transcribe(source)
        transcript = " ".join(segment.text for segment in segments).strip()
        no_speech = min((segment.no_speech_probability for segment in segments), default=1.0)
        kind = classify_speech(source.name, transcript, no_speech)
        if kind is not SpeechKind.LINGUISTIC_SPEECH:
            self.models.unload_whisper()
            return self._publish_original(
                row, source, destination, digest, transcript, f"{kind}: original conservado"
            )
        self.db.update_file(int(row["id"]), transcript_en=transcript, status=Status.TRANSCRIBED)
        self.models.unload_whisper()
        self.logger.info("[TRANSLATE] %s", row["relative_path"])
        translation = self.models.qwen.adapt(
            transcript, original_meta.duration, relative_path=row["relative_path"]
        )
        self.db.update_file(int(row["id"]), translation_es=translation, status=Status.TRANSLATED)
        key = stable_key(self.mode, row["relative_path"], digest)[:24]
        reference_path = self.runtime["references"] / "speech" / f"{key}.wav"
        extract_reference(
            original_audio, sample_rate, 0.0, original_meta.duration, 0.0, reference_path
        )
        hints = style_hints(original_audio, sample_rate)
        self.logger.info("[TTS] %s", row["relative_path"])
        generated, generated_rate = self.synthesize_with_oom_retry(
            reference_path, translation, original_meta.duration, hints
        )
        generated = resample_audio(generated, generated_rate, sample_rate)
        self.db.update_file(int(row["id"]), status=Status.GENERATED)
        aligned, error_pct = align_to_duration(
            generated, sample_rate, original_meta.duration, self.runtime["tmp"] / key
        )
        if error_pct > self.config["alignment"]["warning_pct"]:
            instruction = "Haz la frase más corta." if len(generated) > original_meta.frames else "Haz la frase más larga."
            translation = self.models.qwen.adapt(
                transcript, original_meta.duration, relative_path=row["relative_path"],
                length_instruction=instruction,
            )
            generated, generated_rate = self.synthesize_with_oom_retry(
                reference_path, translation, original_meta.duration, hints
            )
            generated = resample_audio(generated, generated_rate, sample_rate)
            aligned, error_pct = align_to_duration(
                generated, sample_rate, original_meta.duration, self.runtime["tmp"] / key
            )
            self.db.update_file(int(row["id"]), translation_es=translation)
        weights = estimate_spatial_weights(original_audio)
        output = preserve_channel_count(aligned, original_meta.channels, weights)
        output = match_rms(output, original_audio)
        output = limit_peak(output, self.config["audio"]["peak_limit_dbfs"])
        self.db.update_file(int(row["id"]), status=Status.ALIGNED)

        def producer(path: Path):
            write_pcm16(path, output, original_meta.sample_rate)

        def validator(path: Path):
            result = validate_wav(path, original_meta, exact_frames=True)
            if not result.valid:
                raise ValueError(result.reason)

        atomic_publish(destination, self.runtime["tmp"], producer, validator)
        status = Status.REVIEW_DURATION if error_pct > self.config["alignment"]["warning_pct"] else Status.DONE
        self.db.update_file(
            int(row["id"]), sha256=digest, config_hash=self.config_hash,
            status=status, error_message=f"duration_error_pct={error_pct:.2f}" if status != Status.DONE else None,
        )
        self.logger.info("[EXPORT] %s", row["relative_path"])
        return ProcessResult(status, row["relative_path"])
