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
        return np.asarray(audio, dtype=np.float32)
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

