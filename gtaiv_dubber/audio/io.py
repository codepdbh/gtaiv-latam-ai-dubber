from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def read_audio(path: Path, *, always_2d: bool = True) -> tuple[np.ndarray, int]:
    data, sample_rate = sf.read(path, dtype="float32", always_2d=always_2d)
    if not np.isfinite(data).all():
        raise ValueError(f"Audio contiene NaN/Inf: {path}")
    return data, int(sample_rate)


def write_pcm16(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("No se puede escribir audio vacío o no finito")
    sf.write(path, np.clip(audio, -1.0, 1.0), sample_rate, subtype="PCM_16", format="WAV")


def mono_mix(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    return audio if audio.ndim == 1 else np.mean(audio, axis=1)

