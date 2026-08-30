from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import soundfile as sf


@dataclass(frozen=True, slots=True)
class WavMetadata:
    sample_rate: int
    channels: int
    sample_width: int
    frames: int
    duration: float
    subtype: str

    @property
    def bit_depth(self) -> int:
        return self.sample_width * 8

    def to_dict(self) -> dict:
        result = asdict(self)
        result["bit_depth"] = self.bit_depth
        return result


def read_wav_metadata(path: Path) -> WavMetadata:
    try:
        info = sf.info(str(path))
    except (RuntimeError, OSError) as exc:
        raise ValueError(f"WAV ilegible: {path}: {exc}") from exc
    if info.format not in ("WAV", "WAVEX"):
        raise ValueError(f"No es WAV/WAVE_FORMAT_EXTENSIBLE: {path}")
    widths = {"PCM_U8": 1, "PCM_16": 2, "PCM_24": 3, "PCM_32": 4}
    sample_width = widths.get(info.subtype)
    if sample_width is None:
        raise ValueError(f"Subtipo WAV no PCM o no soportado ({info.subtype}): {path}")
    channels = int(info.channels)
    sample_rate = int(info.samplerate)
    frames = int(info.frames)
    if channels < 1 or sample_rate < 1 or frames < 0:
        raise ValueError(f"WAV PCM no válido o no soportado: {path}")
    return WavMetadata(
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        frames=frames,
        duration=frames / sample_rate,
        subtype=info.subtype,
    )
