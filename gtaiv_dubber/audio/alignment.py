from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from gtaiv_dubber.audio.io import read_audio, write_pcm16


def atempo_chain(ratio: float) -> str:
    if ratio <= 0:
        raise ValueError("Ratio atempo inválido")
    factors: list[float] = []
    while ratio > 2.0:
        factors.append(2.0)
        ratio /= 2.0
    while ratio < 0.5:
        factors.append(0.5)
        ratio /= 0.5
    factors.append(ratio)
    return ",".join(f"atempo={factor:.8f}" for factor in factors)


def time_stretch_ffmpeg(
    audio: np.ndarray,
    sample_rate: int,
    target_frames: int,
    temp_dir: Path,
) -> np.ndarray:
    current = len(audio)
    if current == target_frames:
        # Pasa por fit_frame_count igual que las demás vías: esta devolvía el
        # array tal cual, y si venía en una dimensión el llamante fallaba al
        # pedirle el canal. Ocurre solo cuando la duración coincide al frame.
        return fit_frame_count(audio, target_frames)
    temp_dir.mkdir(parents=True, exist_ok=True)
    source = temp_dir / "stretch_source.wav"
    target = temp_dir / "stretch_target.wav"
    write_pcm16(source, audio, sample_rate)
    tempo = current / target_frames
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-af", atempo_chain(tempo), "-ar", str(sample_rate), str(target),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
        result, _ = read_audio(target, always_2d=True)
        return fit_frame_count(result, target_frames)
    finally:
        source.unlink(missing_ok=True)
        target.unlink(missing_ok=True)


def fit_frame_count(audio: np.ndarray, target_frames: int) -> np.ndarray:
    data = np.asarray(audio, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    if len(data) > target_frames:
        return data[:target_frames]
    if len(data) < target_frames:
        return np.pad(data, ((0, target_frames - len(data)), (0, 0)))
    return data


def align_to_duration(
    audio: np.ndarray,
    sample_rate: int,
    target_duration: float,
    temp_dir: Path,
) -> tuple[np.ndarray, float]:
    target_frames = max(1, round(target_duration * sample_rate))
    generated_duration = len(audio) / sample_rate
    error_pct = abs(generated_duration - target_duration) / max(target_duration, 1e-6) * 100
    aligned = time_stretch_ffmpeg(audio, sample_rate, target_frames, temp_dir)
    return aligned, error_pct



def fit_dialogue(
    audio: np.ndarray,
    sample_rate: int,
    slot_seconds: float,
    room_seconds: float,
    temp_dir: Path,
    *,
    hard_max_speedup: float = 1.35,
    fade_seconds: float = 0.04,
) -> tuple[np.ndarray, float, bool]:
    """Encaja una réplica doblada en su hueco sin deformar la voz.

    Devuelve el audio en 2D, el tempo aplicado y si hubo que recortar la cola.
    El audio puede ser más largo que el hueco: ocupa el silencio que le sigue.

    ``align_to_duration`` estiraba cada línea al frame exacto en ambos sentidos
    y sin límite. Medido el 27/09/2026 sobre 509 segmentos: el 42% se
    ralentizaba (voz arrastrada) y el 46% se comprimía, un 11% más del doble.
    Aquí la voz más corta que su hueco conserva su velocidad y se rellena con
    silencio, y la más larga ocupa primero el silencio siguiente (``room``);
    sólo si aun así no cabe se acelera, y nunca más de ``hard_max_speedup``.
    """
    data = np.asarray(audio, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    slot = max(1, round(slot_seconds * sample_rate))
    room = max(slot, round(room_seconds * sample_rate))
    current = len(data)
    if current <= slot:
        return fit_frame_count(data, slot), 1.0, False
    if current <= room:
        return data, 1.0, False
    tempo = min(current / room, float(hard_max_speedup))
    stretched = time_stretch_ffmpeg(data, sample_rate, max(1, round(current / tempo)), temp_dir)
    if len(stretched) <= room:
        return stretched, tempo, False
    # Último recurso: la traducción no cabe ni acelerada al máximo tolerable.
    # Se corta con un fundido corto para que no chasquee; el segmento queda
    # marcado para revisión.
    trimmed = stretched[:room].copy()
    fade = min(len(trimmed), max(1, round(fade_seconds * sample_rate)))
    trimmed[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)[:, None]
    return trimmed, tempo, True
