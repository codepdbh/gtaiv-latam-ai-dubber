from __future__ import annotations

import numpy as np


def preserve_channel_count(audio: np.ndarray, channels: int, weights: np.ndarray | None = None) -> np.ndarray:
    data = np.asarray(audio, dtype=np.float32)
    if data.ndim == 2 and data.shape[1] == channels:
        return data
    mono = data if data.ndim == 1 else np.mean(data, axis=1)
    if channels == 1:
        return mono[:, None]
    if weights is None:
        weights = np.full(channels, 1.0 / np.sqrt(channels), dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    return mono[:, None] * weights[None, :]

