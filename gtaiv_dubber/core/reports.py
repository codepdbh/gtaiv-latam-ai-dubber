from __future__ import annotations

import csv
import json
from pathlib import Path

from gtaiv_dubber.core.database import Database


def write_manifests(db: Database, report_dir: Path) -> list[Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for mode, basename in (("speech", "speech_manifest"), ("cutscene", "cutscene_files")):
        data = [dict(row) for row in db.list_files(mode)]
        json_path = report_dir / f"{basename}.json"
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        csv_path = report_dir / f"{basename}.csv"
        fieldnames = list(data[0]) if data else ["relative_path", "status", "error_message"]
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(data)
        outputs.extend((csv_path, json_path))
    segments = [
        dict(row) for row in db.connection.execute(
            """SELECT f.relative_path,s.segment_index,s.segment_start,s.segment_end,s.speaker_label,
                      s.transcript_en,s.translation_es,s.generated_duration,s.duration_error_pct,
                      s.status,s.error_message
               FROM segments s JOIN files f ON f.id=s.file_id ORDER BY f.relative_path,s.segment_index"""
        )
    ]
    for suffix in ("json", "csv"):
        path = report_dir / f"cutscene_manifest.{suffix}"
        if suffix == "json":
            path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            fields = list(segments[0]) if segments else ["relative_path", "status", "error_message"]
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(segments)
        outputs.append(path)
    return outputs
