from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

from gtaiv_dubber.core.database import Database
from gtaiv_dubber.core.environment import inspect_environment
from gtaiv_dubber.core.model_manager import ModelManager
from gtaiv_dubber.core.reports import write_manifests
from gtaiv_dubber.core.scanner import scan_mode
from gtaiv_dubber.core.state import Status
from gtaiv_dubber.pipelines.cutscene_pipeline import CutscenePipeline
from gtaiv_dubber.pipelines.speech_pipeline import SpeechPipeline


def assert_models_ready(config: dict, models: ModelManager) -> None:
    environment = inspect_environment(config)
    missing = []
    for key, label in (
        ("ffmpeg", "FFmpeg/ffprobe"), ("cuda", "PyTorch CUDA"),
        ("faster_whisper", "faster-whisper"), ("demucs", "Demucs"),
        ("cosyvoice", "CosyVoice 3"),
    ):
        if not environment[key]:
            missing.append(label)
    if not models.qwen.health():
        missing.append("Ollama qwen3:8b")
    if missing:
        raise RuntimeError("Dependencias/modelos no listos: " + ", ".join(missing))


def select_canary(rows, mode: str):
    if not rows:
        raise RuntimeError(f"No hay archivos {mode} escaneados")
    if mode == "speech":
        candidates = [row for row in rows if 2.5 <= row["duration"] <= 5.0]
        linguistic_names = [
            row for row in candidates
            if not any(cue in Path(row["relative_path"]).stem.upper() for cue in
                       ("BUMP", "CRASH", "PAIN", "DODGE", "BREATH", "GRUNT", "MOAN"))
        ]
        pool = linguistic_names or candidates or rows
        return min(pool, key=lambda row: (abs(row["duration"] - 4.2), row["relative_path"]))
    candidates = [row for row in rows if 15.0 <= row["duration"] <= 30.0]
    return min(candidates or rows, key=lambda row: (abs(row["duration"] - 20.0), row["relative_path"]))


def process_project(
    project_root: Path,
    config: dict,
    db: Database,
    modes: list[str],
    runtime: dict,
    logger,
    *,
    canary_only: bool = False,
    allow_fallback: bool = False,
) -> dict:
    models = ModelManager(config, config["_glossary"], runtime["stems"])
    if not allow_fallback:
        assert_models_ready(config, models)
    pipelines = {
        "speech": SpeechPipeline(project_root, config, db, models, runtime, logger),
        "cutscene": CutscenePipeline(project_root, config, db, models, runtime, logger),
    }
    for mode in modes:
        if not db.list_files(mode):
            scan_mode(project_root, config, db, mode, logger)
    run_id = db.start_run("process " + ",".join(modes), config["_config_hash"])
    started = time.monotonic()
    counts = Counter()
    processed_paths: set[tuple[str, str]] = set()
    try:
        logger.info("[SCAN] Iniciando validación canary")
        for mode in modes:
            row = select_canary(db.list_files(mode), mode)
            result = pipelines[mode].process(row)
            counts[result.status.value] += 1
            processed_paths.add((mode, row["relative_path"]))
            if result.status in (Status.ERROR, Status.FALLBACK):
                raise RuntimeError(f"Canary {mode} no superó validación: {result.message}")
        if not canary_only:
            logger.info("[SCAN] Canarios aprobados; iniciando cola completa")
            for mode in modes:
                rows = db.list_files(mode)
                for index, row in enumerate(rows, 1):
                    if (mode, row["relative_path"]) in processed_paths:
                        continue
                    result = pipelines[mode].process(row)
                    counts[result.status.value] += 1
                    logger.info(
                        "[%s] %s %d/%d %s (%.1fs)", result.status.value, mode,
                        index, len(rows), result.relative_path, result.elapsed,
                    )
        errors = counts[Status.ERROR.value]
        db.finish_run(run_id, "DONE" if errors == 0 else "DONE_WITH_ERRORS", sum(counts.values()), errors)
    except Exception:
        db.finish_run(run_id, "ERROR", sum(counts.values()), counts[Status.ERROR.value] + 1)
        raise
    finally:
        write_manifests(db, runtime["reports"])
    return {"counts": dict(counts), "elapsed": time.monotonic() - started}
