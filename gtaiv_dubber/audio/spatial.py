from __future__ import annotations

import numpy as np


def estimate_spatial_weights(audio: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    data = np.asarray(audio, dtype=np.float32)
    if data.ndim == 1:
        return np.ones(1, dtype=np.float32)
    rms = np.sqrt(np.mean(np.square(data), axis=0) + epsilon)
    norm = np.linalg.norm(rms)
    if norm <= epsilon:
        return np.full(data.shape[1], 1.0 / np.sqrt(data.shape[1]), dtype=np.float32)
    return np.asarray(rms / norm, dtype=np.float32)


def apply_spatial_weights(mono: np.ndarray, weights: np.ndarray) -> np.ndarray:
    signal = np.asarray(mono, dtype=np.float32).reshape(-1)
    return signal[:, None] * np.asarray(weights, dtype=np.float32)[None, :]

