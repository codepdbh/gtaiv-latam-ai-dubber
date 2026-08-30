from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gtaiv_dubber.audio.metadata import WavMetadata, read_wav_metadata


@dataclass(slots=True)
class ValidationResult:
    valid: bool
    reason: str = ""
    metadata: WavMetadata | None = None


def validate_wav(
    path: Path,
    original: WavMetadata | None = None,
    *,
    exact_frames: bool = False,
    speech_duration_tolerance: float = 0.35,
) -> ValidationResult:
    if not path.is_file() or path.stat().st_size < 44:
        return ValidationResult(False, "Archivo ausente o demasiado pequeño")
    try:
        metadata = read_wav_metadata(path)
    except (OSError, ValueError) as exc:
        return ValidationResult(False, str(exc))
    if metadata.frames <= 0:
        return ValidationResult(False, "WAV vacío", metadata)
    if original:
        if metadata.sample_rate != original.sample_rate:
            return ValidationResult(False, "Sample rate distinto", metadata)
        if metadata.channels != original.channels:
            return ValidationResult(False, "Número de canales distinto", metadata)
        if exact_frames and metadata.frames != original.frames:
            return ValidationResult(False, "Frame count distinto", metadata)
        if not exact_frames and original.duration > 0:
            ratio = metadata.duration / original.duration
            if not (1 - speech_duration_tolerance <= ratio <= 1 + speech_duration_tolerance):
                return ValidationResult(False, "Duración fuera de tolerancia", metadata)
    return ValidationResult(True, metadata=metadata)

