from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def wav_factory():
    def create(path: Path, *, rate: int = 24000, channels: int = 1, frames: int = 2400):
        path.parent.mkdir(parents=True, exist_ok=True)
        t = np.arange(frames) / rate
        mono = (np.sin(2 * np.pi * 220 * t) * 10000).astype("<i2")
        data = np.column_stack([mono] * channels).reshape(-1).astype("<i2").tobytes()
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(data)
        return path

    return create


@pytest.fixture
def sample_project(tmp_path: Path, wav_factory):
    wav_factory(tmp_path / "Speech" / "A" / "B" / "LINE.wav", rate=24000, channels=1, frames=2400)
    wav_factory(tmp_path / "Cutscenes" / "B_1" / "B_1.wav", rate=32000, channels=3, frames=3200)
    return tmp_path

