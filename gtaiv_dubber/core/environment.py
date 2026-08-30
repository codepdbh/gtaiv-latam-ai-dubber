from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


def _command_version(command: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        output = (result.stdout or result.stderr).strip().splitlines()
        return result.returncode == 0, output[0] if output else ""
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)


def inspect_environment(config: dict | None = None) -> dict:
    is_wsl = "microsoft" in platform.release().lower() or bool(os.environ.get("WSL_DISTRO_NAME"))
    ffmpeg_ok, ffmpeg_version = _command_version(["ffmpeg", "-version"])
    ffprobe_ok, _ = _command_version(["ffprobe", "-version"])
    nvidia_ok, nvidia_output = _command_version(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    result = {
        "wsl2": is_wsl,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "ffmpeg": ffmpeg_ok and ffprobe_ok,
        "ffmpeg_version": ffmpeg_version,
        "nvidia_smi": nvidia_ok,
        "gpu": nvidia_output,
        "torch": False,
        "cuda": False,
        "vram_bytes": 0,
        "faster_whisper": False,
        "demucs": False,
        "cosyvoice": False,
        "ollama": False,
    }
    try:
        import torch

        result["torch"] = True
        result["cuda"] = bool(torch.cuda.is_available())
        if result["cuda"]:
            result["torch_gpu"] = torch.cuda.get_device_name(0)
            result["vram_bytes"] = int(torch.cuda.get_device_properties(0).total_memory)
    except Exception as exc:
        result["torch_error"] = str(exc)
    for package, key in (("faster_whisper", "faster_whisper"), ("demucs", "demucs")):
        try:
            __import__(package)
            result[key] = True
        except Exception:
            pass
    result["ollama"] = shutil.which("ollama") is not None
    if config:
        repo = Path(config["cosyvoice"]["repo_dir"]).expanduser()
        model = Path(config["cosyvoice"]["model_dir"]).expanduser()
        result["cosyvoice"] = repo.is_dir() and model.is_dir()
    return result


def format_environment(result: dict) -> str:
    ok = lambda value: "OK" if value else "ERROR"
    vram = result.get("vram_bytes", 0) / (1024**3)
    return "\n".join(
        [
            "========================================",
            "GTA IV LATAM AI DUBBER",
            "========================================",
            f"WSL2:            {ok(result['wsl2'])}",
            f"Python:          {result['python']}",
            f"FFmpeg:          {ok(result['ffmpeg'])}",
            f"NVIDIA GPU:      {result.get('gpu') or 'ERROR'}",
            f"CUDA:            {ok(result['cuda'])}",
            f"PyTorch CUDA:    {ok(result['torch'] and result['cuda'])}",
            f"VRAM:            {vram:.1f} GB" if vram else "VRAM:            ERROR",
            f"faster-whisper:  {ok(result['faster_whisper'])}",
            f"Qwen3/Ollama:    {ok(result['ollama'])}",
            f"CosyVoice 3:     {ok(result['cosyvoice'])}",
            f"Demucs:          {ok(result['demucs'])}",
            "========================================",
        ]
    )

