#!/usr/bin/env python3
"""Pruebas de inferencia mínimas para los tres modelos principales."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import soundfile as sf

from gtaiv_dubber.ai.cosyvoice_adapter import CosyVoiceAdapter
from gtaiv_dubber.ai.qwen_adapter import QwenAdapter
from gtaiv_dubber.ai.whisper_adapter import WhisperAdapter
from gtaiv_dubber.core.config import load_config, load_glossary


def release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("model", choices=("qwen", "whisper", "cosyvoice"))
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = load_config(root)

    if args.model == "qwen":
        adapter = QwenAdapter(config["ollama"], load_glossary(config))
        if not adapter.health():
            raise RuntimeError("Ollama/qwen3:8b no responde")
        result = adapter.adapt("Get in the car, now!", 1.8, relative_path="smoke-test")
        print(f"QWEN_OK: {result}")
        return 0

    reference = args.reference
    if reference is None:
        reference = next((root / config["project"]["speech_dir"]).rglob("*.wav"))
    reference = reference.resolve()

    if args.model == "whisper":
        adapter = WhisperAdapter(config["whisper"])
        segments = adapter.transcribe(reference)
        print("WHISPER_OK:", " | ".join(item.text for item in segments) or "<sin voz detectada>")
        adapter.unload()
        release_cuda()
        return 0

    adapter = CosyVoiceAdapter(config["cosyvoice"])
    audio, sample_rate = adapter.synthesize(reference, "¡Sube al auto, ahora!", 1.8)
    output = root / "work" / "tmp" / "cosyvoice-smoke.wav"
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, audio, sample_rate, subtype="PCM_16")
    print(f"COSYVOICE_OK: {output} ({len(audio)} frames @ {sample_rate} Hz)")
    adapter.unload()
    release_cuda()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
