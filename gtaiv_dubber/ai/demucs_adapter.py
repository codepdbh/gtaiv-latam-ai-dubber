from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from gtaiv_dubber.audio.io import read_audio, write_pcm16
from gtaiv_dubber.audio.validation import validate_wav
from gtaiv_dubber.utils.hashing import stable_key


class DemucsAdapter:
    def __init__(self, config: dict, stems_root: Path):
        self.config = config
        self.stems_root = stems_root

    def separate(self, mono_audio: np.ndarray, sample_rate: int, cache_key: str) -> tuple[Path, Path]:
        key = stable_key(cache_key, self.config.get("model", "htdemucs"))[:24]
        model = self.config.get("model", "htdemucs")
        output_dir = self.stems_root / key
        vocals_cache = output_dir / "vocals.wav"
        background_cache = output_dir / "no_vocals.wav"
        if validate_wav(vocals_cache).valid and validate_wav(background_cache).valid:
            return vocals_cache, background_cache
        input_dir = self.stems_root / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        input_path = input_dir / f"{key}.wav"
        demucs_output = self.stems_root / "demucs_tmp" / key
        write_pcm16(input_path, mono_audio, sample_rate)
        command = [
            "python", "-m", "demucs.separate", "-n", model, "--two-stems=vocals",
            "-d", self.config.get("device", "cuda"), "-o", str(demucs_output), str(input_path),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            generated_dir = demucs_output / model / input_path.stem
            vocals = generated_dir / "vocals.wav"
            background = generated_dir / "no_vocals.wav"
            if not validate_wav(vocals).valid or not validate_wav(background).valid:
                raise RuntimeError("Demucs no produjo stems WAV válidos")
            output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(vocals, vocals_cache)
            shutil.copy2(background, background_cache)
            return vocals_cache, background_cache
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc))[-2000:]
            raise RuntimeError(f"Falló Demucs: {detail}") from exc
        finally:
            input_path.unlink(missing_ok=True)


def load_vocal_stem(path: Path, target_rate: int, target_frames: int) -> np.ndarray:
    from gtaiv_dubber.audio.alignment import fit_frame_count
    from gtaiv_dubber.audio.resampling import resample_audio

    audio, rate = read_audio(path)
    mono = np.mean(audio, axis=1)
    mono = resample_audio(mono, rate, target_rate)
    return fit_frame_count(mono, target_frames)[:, 0]

