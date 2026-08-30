from __future__ import annotations

import numpy as np

from gtaiv_dubber.audio.alignment import fit_frame_count
from gtaiv_dubber.audio.spatial import apply_spatial_weights


def replace_dialogue_region(
    output: np.ndarray,
    generated_mono: np.ndarray,
    start_frame: int,
    end_frame: int,
    weights: np.ndarray,
    original_vocal_mono: np.ndarray | None = None,
    attenuation: float = 0.75,
) -> None:
    window = max(0, end_frame - start_frame)
    if window == 0:
        return
    generated = fit_frame_count(generated_mono, window)[:, 0]
    if original_vocal_mono is not None:
        vocal = fit_frame_count(original_vocal_mono, window)[:, 0]
        output[start_frame:end_frame] -= attenuation * apply_spatial_weights(vocal, weights)
    output[start_frame:end_frame] += apply_spatial_weights(generated, weights)

