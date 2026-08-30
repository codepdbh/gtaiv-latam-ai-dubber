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
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--compute-type")
    parser.add_argument("--language", default="es")
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--show-times", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = load_config(root)["whisper"]
    model = WhisperModel(
        config["model"],
        device=args.device or config.get("device", "cuda"),
        compute_type=args.compute_type or config.get("compute_type", "float16"),
        download_root=str(Path(config["download_root"]).expanduser()),
    )
    output_root = args.input_root.resolve() if args.input_root else root / (
        "Speech_es" if args.mode == "speech" else "Cutscenes_es"
    )
    for path in sorted(output_root.glob(args.glob))[: args.limit]:
        segments, _ = model.transcribe(
            str(path),
            language=None if args.language == "auto" else args.language,
            task="transcribe",
            beam_size=5,
            vad_filter=False,
        )
        segment_list = list(segments)
        if args.show_times:
            spoken = " | ".join(
                f"{segment.start:.2f}-{segment.end:.2f} {segment.text.strip()}"
                for segment in segment_list
            )
        else:
            spoken = " ".join(segment.text.strip() for segment in segment_list).strip()
        print(f"{path.relative_to(output_root).as_posix()}\t{spoken}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
