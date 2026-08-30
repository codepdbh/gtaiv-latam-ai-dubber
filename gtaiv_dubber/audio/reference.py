from __future__ import annotations

from pathlib import Path

import numpy as np

from gtaiv_dubber.audio.io import mono_mix, write_pcm16


def extract_reference(
    audio: np.ndarray,
    sample_rate: int,
    start: float,
    end: float,
    padding: float,
    output_path: Path,
    minimum_seconds: float = 1.0,
) -> Path:
    duration = len(audio) / sample_rate
    left = max(0.0, start - padding)
    right = min(duration, end + padding)
    if right - left < minimum_seconds:
        extra = (minimum_seconds - (right - left)) / 2
        left = max(0.0, left - extra)
        right = min(duration, right + extra)
    first, last = round(left * sample_rate), round(right * sample_rate)
    clip = mono_mix(audio[first:last])
    if len(clip) < max(1, round(0.25 * sample_rate)):
        raise ValueError("Referencia vocal demasiado corta")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_pcm16(output_path, clip, sample_rate)
    return output_path

