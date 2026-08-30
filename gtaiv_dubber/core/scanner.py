from __future__ import annotations

import csv
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from gtaiv_dubber.audio.metadata import read_wav_metadata
from gtaiv_dubber.core.database import Database
from gtaiv_dubber.utils.paths import mode_paths, safe_output_path


@dataclass(slots=True)
class ScanStats:
    mode: str
    files: int = 0
    directories: int = 0
    unreadable: int = 0
    total_duration: float = 0.0
    total_bytes: int = 0
    sample_rates: Counter = field(default_factory=Counter)
    channels: Counter = field(default_factory=Counter)
    bit_depths: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "files": self.files,
            "directories": self.directories,
            "unreadable": self.unreadable,
            "total_duration_seconds": round(self.total_duration, 6),
            "total_bytes": self.total_bytes,
            "sample_rates": dict(sorted(self.sample_rates.items())),
            "channels": dict(sorted(self.channels.items())),
            "bit_depths": dict(sorted(self.bit_depths.items())),
        }


def iter_wavs(root: Path):
    if not root.is_dir():
        return
    for current, directories, files in os.walk(root):
        directories.sort(key=str.casefold)
        for name in sorted(files, key=str.casefold):
            if name.lower().endswith(".wav"):
                yield Path(current) / name


def scan_mode(project_root: Path, config: dict, db: Database, mode: str, logger=None) -> ScanStats:
    input_root, output_root = mode_paths(project_root, config, mode)
    if not input_root.is_dir():
        raise FileNotFoundError(f"No existe la carpeta requerida: {input_root}")
    if input_root.resolve() == output_root.resolve():
        raise RuntimeError("La salida no puede coincidir con la carpeta original")
    output_root.mkdir(parents=True, exist_ok=True)
    stats = ScanStats(mode=mode)
    index = 0
    for current, directories, files in os.walk(input_root):
        directories.sort(key=str.casefold)
        stats.directories += len(directories)
        current_relative = Path(current).relative_to(input_root)
        (output_root / current_relative).mkdir(parents=True, exist_ok=True)
        wav_names = sorted((name for name in files if name.lower().endswith(".wav")), key=str.casefold)
        for name in wav_names:
            index += 1
            input_file = Path(current) / name
            relative = input_file.relative_to(input_root)
            destination = safe_output_path(input_file, input_root, output_root)
            stats.files += 1
            stats.total_bytes += input_file.stat().st_size
            try:
                metadata = read_wav_metadata(input_file)
                stats.total_duration += metadata.duration
                stats.sample_rates[metadata.sample_rate] += 1
                stats.channels[metadata.channels] += 1
                stats.bit_depths[metadata.bit_depth] += 1
                db.upsert_file(
                    {
                        "mode": mode,
                        "input_path": str(input_file.resolve()),
                        "relative_path": relative.as_posix(),
                        "output_path": str(destination),
                        **metadata.to_dict(),
                    },
                    commit=False,
                )
            except (OSError, ValueError) as exc:
                stats.unreadable += 1
                if logger:
                    logger.error("[ERROR] %s", exc)
            if logger and index % 1000 == 0:
                db.connection.commit()
                logger.info("[SCAN] %s: %s WAV", mode, index)
    db.connection.commit()
    return stats


def write_scan_report(report_dir: Path, stats: list[ScanStats]) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "scan_report.json"
    csv_path = report_dir / "scan_report.csv"
    payload = {item.mode: item.to_dict() for item in stats}
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["mode", "files", "directories", "unreadable", "total_duration_seconds", "total_bytes"],
        )
        writer.writeheader()
        for item in stats:
            row = item.to_dict()
            writer.writerow({key: row[key] for key in writer.fieldnames})
    return json_path, csv_path
