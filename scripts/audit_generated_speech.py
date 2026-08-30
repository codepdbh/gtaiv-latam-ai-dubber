#!/usr/bin/env python3
"""Transcribe outputs en español para detectar repeticiones o TTS incorrecto."""

from __future__ import annotations

import argparse
from pathlib import Path

from faster_whisper import WhisperModel

from gtaiv_dubber.core.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--mode", choices=("speech", "cutscene"), default="speech")
    parser.add_argument("--glob", default="ALEX/*.wav")
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = load_config(root)["whisper"]
    model = WhisperModel(
        config["model"],
        device=config.get("device", "cuda"),
        compute_type=config.get("compute_type", "float16"),
        download_root=str(Path(config["download_root"]).expanduser()),
    )
    output_root = root / ("Speech_es" if args.mode == "speech" else "Cutscenes_es")
    for path in sorted(output_root.glob(args.glob))[: args.limit]:
        segments, _ = model.transcribe(
            str(path), language="es", task="transcribe", beam_size=5, vad_filter=False
        )
        spoken = " ".join(segment.text.strip() for segment in segments).strip()
        print(f"{path.relative_to(output_root).as_posix()}\t{spoken}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
