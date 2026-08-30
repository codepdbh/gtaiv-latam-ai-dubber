from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path, PureWindowsPath


WINDOWS_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")


def windows_to_wsl_path(value: str) -> str:
    win = PureWindowsPath(value)
    if not win.drive:
        raise ValueError(f"No es una ruta Windows absoluta: {value}")
    drive = win.drive.rstrip(":").lower()
    remainder = "/".join(win.parts[1:])
    return f"/mnt/{drive}/{remainder}" if remainder else f"/mnt/{drive}"


def normalize_project_path(value: str | os.PathLike[str]) -> Path:
    raw = str(value).strip().strip('"')
    if WINDOWS_PATH_RE.match(raw) and os.name != "nt":
        try:
            converted = subprocess.run(
                ["wslpath", "-u", raw], check=True, capture_output=True, text=True
            ).stdout.strip()
            return Path(converted).resolve()
        except (FileNotFoundError, subprocess.CalledProcessError):
            return Path(windows_to_wsl_path(raw))
    return Path(raw).expanduser().resolve()


def mode_paths(root: Path, config: dict, mode: str) -> tuple[Path, Path]:
    project = config["project"]
    if mode == "speech":
        return root / project["speech_dir"], root / project["speech_output"]
    if mode == "cutscene":
        return root / project["cutscenes_dir"], root / project["cutscenes_output"]
    raise ValueError(f"Modo desconocido: {mode}")


def safe_output_path(input_file: Path, input_root: Path, output_root: Path) -> Path:
    input_file = input_file.resolve()
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    relative = input_file.relative_to(input_root)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Ruta relativa insegura: {relative}")
    destination = (output_root / relative).resolve()
    destination.relative_to(output_root)
    if destination == input_file or input_root == output_root:
        raise RuntimeError("La salida jamás puede apuntar al original")
    return destination


def project_runtime_dirs(root: Path) -> dict[str, Path]:
    work = root / "work"
    return {
        "work": work,
        "tmp": work / "tmp",
        "stems": work / "stems",
        "references": work / "references",
        "generated": work / "generated",
        "cache": work / "cache",
        "reports": work / "reports",
        "logs": root / "logs",
    }


def ensure_runtime_dirs(root: Path) -> dict[str, Path]:
    paths = project_runtime_dirs(root)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
