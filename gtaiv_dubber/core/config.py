from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_CODE_ROOT = Path(__file__).resolve().parents[2]


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(project_root: Path, config_path: Path | None = None) -> dict[str, Any]:
    default_path = PROJECT_CODE_ROOT / "config" / "default.yaml"
    with default_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    local_path = config_path or project_root / "config" / "local.yaml"
    if local_path.exists() and local_path.resolve() != default_path.resolve():
        with local_path.open("r", encoding="utf-8") as handle:
            config = _merge(config, yaml.safe_load(handle) or {})
    config["_code_root"] = str(PROJECT_CODE_ROOT)
    config["_project_root"] = str(project_root.resolve())
    return config


# Ajustes de operación que no pueden cambiar ni una muestra del audio: reintentos,
# pausas y esperas. Quedan fuera del hash porque si entraran, subir un reintento
# marcaría como obsoletas las 62.895 voces de speech ya dobladas.
_HASH_IGNORED: frozenset[tuple[str, str]] = frozenset(
    {
        ("ollama", "request_attempts"),
        ("ollama", "retry_pause_seconds"),
    }
)


def relevant_config_hash(config: dict[str, Any], mode: str | None = None) -> str:
    clean = {k: v for k, v in config.items() if not k.startswith("_")}
    for section, key in _HASH_IGNORED:
        if isinstance(clean.get(section), dict) and key in clean[section]:
            if clean[section] is config.get(section):
                clean[section] = dict(clean[section])
            clean[section].pop(key, None)
    if mode is not None:
        common = {
            "project", "language", "glossary", "whisper", "ollama", "cosyvoice",
            "audio", "alignment", "processing",
        }
        mode_only = (
            {"cutscenes", "demucs", "speaker_separation", "diarization"}
            if mode == "cutscene"
            else set()
        )
        clean = {key: value for key, value in clean.items() if key in common | mode_only}
    payload = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_glossary(config: dict[str, Any]) -> dict[str, str]:
    path = Path(config["_code_root"]) / config.get("glossary", {}).get("path", "config/glossary.yaml")
    with path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    return {str(key): str(value) for key, value in values.items()}
