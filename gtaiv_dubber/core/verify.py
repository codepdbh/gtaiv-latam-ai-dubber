from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

from gtaiv_dubber.audio.metadata import read_wav_metadata
from gtaiv_dubber.audio.validation import validate_wav
from gtaiv_dubber.core.database import Database
from gtaiv_dubber.core.scanner import iter_wavs
from gtaiv_dubber.utils.paths import mode_paths


def _relative_directories(root: Path) -> set[str]:
    result: set[str] = set()
    if not root.is_dir():
        return result
    for current, directories, _files in os.walk(root):
        current_path = Path(current)
        for name in directories:
            result.add((current_path / name).relative_to(root).as_posix())
    return result


def verify_project(project_root: Path, config: dict, db: Database) -> dict:
    report: dict = {}
    for mode in ("speech", "cutscene"):
        input_root, output_root = mode_paths(project_root, config, mode)
        results = []
        counts = Counter()
        originals = list(iter_wavs(input_root))
        output_files = [path for path in output_root.rglob("*") if path.is_file()] if output_root.exists() else []
        output_wavs = {
            path.relative_to(output_root).as_posix(): path
            for path in output_files if path.suffix.lower() == ".wav"
        }
        rows_by_relative = {row["relative_path"]: row for row in db.list_files(mode)}
        expected = set()
        for source in originals:
            relative = source.relative_to(input_root).as_posix()
            expected.add(relative)
            destination = output_wavs.get(relative)
            row = rows_by_relative.get(relative)
            if destination is None:
                validation = type("Result", (), {"valid": False, "reason": "Salida ausente"})()
            else:
                try:
                    metadata = read_wav_metadata(source)
                    validation = validate_wav(destination, metadata, exact_frames=True)
                except (OSError, ValueError) as exc:
                    validation = type("Result", (), {"valid": False, "reason": str(exc)})()
            db_status = row["status"] if row else "DISCOVERED"
            if validation.valid:
                if db_status.startswith("REVIEW"):
                    bucket = "REVIEW"
                elif db_status == "FALLBACK":
                    bucket = "FALLBACK"
                else:
                    bucket = "OK"
            else:
                bucket = "ERROR"
            counts[bucket] += 1
            if bucket != "OK":
                results.append(
                    {"relative_path": relative, "result": bucket, "status": db_status, "reason": validation.reason}
                )
        extras = [
            path.relative_to(output_root).as_posix() for path in output_files
            if path.suffix.lower() != ".wav" or path.relative_to(output_root).as_posix() not in expected
        ]
        input_directories = _relative_directories(input_root)
        output_directories = _relative_directories(output_root)
        missing_directories = sorted(input_directories - output_directories)
        report[mode] = {
            "total": len(originals),
            "ok": counts["OK"],
            "review": counts["REVIEW"],
            "fallback": counts["FALLBACK"],
            "error": counts["ERROR"],
            "unexpected_output_files": extras,
            "missing_directories": missing_directories,
            "details": results,
        }
    report_dir = project_root / "work" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "verify_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def format_verify(report: dict) -> str:
    lines = []
    for mode, title in (("speech", "SPEECH"), ("cutscene", "CUTSCENES")):
        item = report[mode]
        lines.extend(
            [
                title,
                f"Total: {item['total']}",
                f"OK: {item['ok']}",
                f"Review: {item['review']}",
                f"Fallback: {item['fallback']}",
                f"Error: {item['error']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip()
