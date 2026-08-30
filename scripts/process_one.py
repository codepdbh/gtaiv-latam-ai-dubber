#!/usr/bin/env python3
"""Procesa un único WAV por ruta relativa para pruebas y revisiones."""

from __future__ import annotations

import argparse
from pathlib import Path

from gtaiv_dubber.core.config import load_config, load_glossary, relevant_config_hash
from gtaiv_dubber.core.database import Database
from gtaiv_dubber.core.model_manager import ModelManager
from gtaiv_dubber.core.reports import write_manifests
from gtaiv_dubber.core.state import Status
from gtaiv_dubber.pipelines.cutscene_pipeline import CutscenePipeline
from gtaiv_dubber.pipelines.speech_pipeline import SpeechPipeline
from gtaiv_dubber.utils.logging import configure_logging
from gtaiv_dubber.utils.paths import ensure_runtime_dirs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("mode", choices=("speech", "cutscene"))
    parser.add_argument("relative_path")
    args = parser.parse_args()
    root = args.project_root.resolve()
    runtime = ensure_runtime_dirs(root)
    config = load_config(root)
    config["_config_hash"] = relevant_config_hash(config)
    config["_glossary"] = load_glossary(config)
    logger = configure_logging(runtime["logs"])
    with Database(runtime["work"] / "state.sqlite3") as db:
        row = db.get_file(args.mode, Path(args.relative_path).as_posix())
        if row is None:
            raise RuntimeError(f"Archivo no encontrado en SQLite: {args.relative_path}")
        models = ModelManager(config, config["_glossary"], runtime["stems"])
        pipeline_class = SpeechPipeline if args.mode == "speech" else CutscenePipeline
        result = pipeline_class(root, config, db, models, runtime, logger).process(row)
        write_manifests(db, runtime["reports"])
    print(f"{result.status.value}: {result.relative_path} ({result.elapsed:.1f}s)")
    return 1 if result.status in (Status.FALLBACK, Status.ERROR) else 0


if __name__ == "__main__":
    raise SystemExit(main())
