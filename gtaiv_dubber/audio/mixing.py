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


def render_dialogue_replacements(
    original: np.ndarray,
    original_vocal_mono: np.ndarray,
    tracks: list[tuple[np.ndarray, int, int, np.ndarray]],
    attenuation: float = 1.0,
) -> np.ndarray:
    """Retira diálogo inglés una sola vez y suma todas las voces nuevas.

    Cuando dos pistas se solapan, sus voces traducidas se suman, pero la estimación
    vocal original se promedia antes de restarla para evitar una doble sustracción.
    """
    output = np.asarray(original, dtype=np.float32).copy()
    if output.ndim == 1:
        output = output[:, None]
    vocal = np.asarray(original_vocal_mono, dtype=np.float32).reshape(-1)
    removal = np.zeros_like(output)
    generated_mix = np.zeros_like(output)
    coverage = np.zeros(len(output), dtype=np.float32)
    for generated_mono, start_frame, end_frame, weights in tracks:
        first = max(0, int(start_frame))
        last = min(len(output), int(end_frame))
        window = last - first
        if window <= 0:
            continue
        generated = fit_frame_count(generated_mono, window)[:, 0]
        vocal_region = fit_frame_count(vocal[first:last], window)[:, 0]
        generated_mix[first:last] += apply_spatial_weights(generated, weights)
        removal[first:last] += apply_spatial_weights(vocal_region, weights)
        coverage[first:last] += 1.0
    active = coverage > 0
    if np.any(active):
        removal[active] /= coverage[active, None]
        output[active] -= float(attenuation) * removal[active]
    return output + generated_mix
