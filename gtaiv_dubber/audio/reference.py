from __future__ import annotations

from pathlib import Path

import numpy as np

from gtaiv_dubber.audio.io import mono_mix, write_pcm16


def reference_window(
    duration: float,
    start: float,
    end: float,
    padding: float,
    minimum_seconds: float = 1.0,
    *,
    limit_start: float | None = None,
    limit_end: float | None = None,
) -> tuple[float, float]:
    """Ventana de referencia con relleno, sin invadir a los personajes vecinos.

    ``limit_start`` y ``limit_end`` acotan el relleno al hueco disponible; sin
    ellos el comportamiento es el histórico de relleno simétrico.
    """
    floor = 0.0 if limit_start is None else max(0.0, min(float(limit_start), start))
    ceiling = duration if limit_end is None else min(duration, max(float(limit_end), end))
    left = max(floor, start - padding)
    right = min(ceiling, end + padding)
    if right - left < minimum_seconds:
        extra = (minimum_seconds - (right - left)) / 2
        left = max(floor, left - extra)
        right = min(ceiling, right + extra)
    return left, right


def extract_reference(
    audio: np.ndarray,
    sample_rate: int,
    start: float,
    end: float,
    padding: float,
    output_path: Path,
    minimum_seconds: float = 1.0,
    *,
    limit_start: float | None = None,
    limit_end: float | None = None,
    max_seconds: float | None = None,
) -> Path:
    duration = len(audio) / sample_rate
    left, right = reference_window(
        duration, start, end, padding, minimum_seconds,
        limit_start=limit_start, limit_end=limit_end,
    )
    if max_seconds is not None and right - left > max_seconds:
        # CosyVoice rechaza referencias de más de 30 s. Un monólogo largo daba
        # una ventana entera enorme, así que se recorta por el centro.
        middle = (left + right) / 2.0
        left, right = middle - max_seconds / 2.0, middle + max_seconds / 2.0
        left, right = max(0.0, left), min(duration, right)
    first, last = round(left * sample_rate), round(right * sample_rate)
    clip = mono_mix(audio[first:last])
    if len(clip) < max(1, round(0.25 * sample_rate)):
        raise ValueError("Referencia vocal demasiado corta")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_pcm16(output_path, clip, sample_rate)
    return output_path


def build_speaker_reference(
    clips: list[np.ndarray],
    sample_rate: int,
    output_path: Path,
    *,
    target_seconds: float = 6.0,
    max_seconds: float = 12.0,
    fade_seconds: float = 0.01,
    gap_seconds: float = 0.06,
    minimum_seconds: float = 2.0,
) -> tuple[Path, float]:
    """Concatena varias intervenciones del mismo personaje en una sola referencia.

    Una muestra larga y limpia da un clonado mucho más estable que un fragmento
    suelto de medio segundo, y mantiene la misma voz durante toda la escena. Un
    monólogo largo se recorta a ``max_seconds`` para no inflar el prompt del TTS.
    """
    fade = max(1, round(fade_seconds * sample_rate))
    gap = np.zeros(max(0, round(gap_seconds * sample_rate)), dtype=np.float32)
    ceiling = max(float(max_seconds), float(target_seconds))
    pieces: list[np.ndarray] = []
    pooled = 0.0
    for clip in clips:
        signal = mono_mix(np.asarray(clip, dtype=np.float32))
        remaining = ceiling - pooled
        if remaining <= 0:
            break
        signal = signal[: round(remaining * sample_rate)]
        if len(signal) < 2 * fade:
            continue
        shaped = signal.copy()
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        shaped[:fade] *= ramp
        shaped[-fade:] *= ramp[::-1]
        if pieces:
            pieces.append(gap)
        pieces.append(shaped)
        pooled += len(signal) / sample_rate
        if pooled >= target_seconds:
            break
    if pooled < minimum_seconds or not pieces:
        raise ValueError(f"Referencia de personaje insuficiente ({pooled:.2f}s)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_pcm16(output_path, np.concatenate(pieces), sample_rate)
    return output_path, pooled
