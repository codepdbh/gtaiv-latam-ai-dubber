from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(audio, dtype=np.float32)
    divisor = gcd(source_rate, target_rate)
    result = resample_poly(audio, target_rate // divisor, source_rate // divisor, axis=0)
    return np.asarray(result, dtype=np.float32)

