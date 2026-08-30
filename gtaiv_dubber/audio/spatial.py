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


def estimate_reference_weights(
    audio: np.ndarray,
    reference_mono: np.ndarray,
    epsilon: float = 1e-8,
    max_gain: float = 4.0,
) -> np.ndarray:
    """Proyecta una referencia mono sobre cada canal físico por mínimos cuadrados.

    A diferencia de los pesos RMS, estos coeficientes conservan la escala necesaria
    para retirar la voz estimada y evitan dejar una copia inglesa audible.
    """
    data = np.asarray(audio, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    reference = np.asarray(reference_mono, dtype=np.float32).reshape(-1)
    frames = min(len(data), len(reference))
    if frames == 0:
        return estimate_spatial_weights(data)
    data = data[:frames] - np.mean(data[:frames], axis=0, keepdims=True)
    reference = reference[:frames] - float(np.mean(reference[:frames]))
    denominator = float(np.dot(reference, reference))
    if denominator <= epsilon:
        return estimate_spatial_weights(data)
    weights = reference @ data / denominator
    if not np.all(np.isfinite(weights)) or np.max(np.abs(weights)) <= epsilon:
        return estimate_spatial_weights(data)
    return np.clip(weights, -max_gain, max_gain).astype(np.float32)
