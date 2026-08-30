from __future__ import annotations

import numpy as np


def rms(audio: np.ndarray) -> float:
    data = np.asarray(audio, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(data)) + 1e-12)) if data.size else 0.0


def peak(audio: np.ndarray) -> float:
    data = np.asarray(audio)
    return float(np.max(np.abs(data))) if data.size else 0.0


def match_rms(generated: np.ndarray, reference: np.ndarray, max_gain_db: float = 12.0) -> np.ndarray:
    source_rms = rms(generated)
    target_rms = rms(reference)
    if source_rms <= 1e-7 or target_rms <= 1e-7:
        return np.asarray(generated, dtype=np.float32)
    max_gain = 10 ** (max_gain_db / 20)
    gain = np.clip(target_rms / source_rms, 1 / max_gain, max_gain)
    return np.asarray(generated * gain, dtype=np.float32)


def limit_peak(audio: np.ndarray, limit_dbfs: float = -1.0) -> np.ndarray:
    data = np.asarray(audio, dtype=np.float32)
    ceiling = 10 ** (limit_dbfs / 20)
    current = peak(data)
    if current > ceiling:
        data = data * (ceiling / current)
    return np.asarray(data, dtype=np.float32)


def style_hints(audio: np.ndarray, sample_rate: int) -> dict[str, float | str]:
    data = np.asarray(audio, dtype=np.float32)
    level = rms(data)
    crest = peak(data) / max(level, 1e-8)
    intensity = "alta" if level > 0.16 or crest > 8 else "calma" if level < 0.045 else "media"
    return {"rms": level, "peak": peak(data), "crest_factor": crest, "intensity": intensity}

