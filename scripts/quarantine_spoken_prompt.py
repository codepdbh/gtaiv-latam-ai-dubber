#!/usr/bin/env python3
"""Cuarentena outputs creados antes de corregir el prompt TTS hablado."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from gtaiv_dubber.core.config import load_config
from gtaiv_dubber.core.database import Database
from gtaiv_dubber.utils.paths import mode_paths, safe_output_path


def process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    pid_file = root / "work" / "mass.pid"
    if pid_file.is_file():
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        if process_is_alive(pid):
            raise RuntimeError(f"La cola sigue activa (PID {pid}); no se puede reparar")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    quarantine = root / "work" / "quarantine" / f"spoken-style-prompt-{stamp}"
    quarantine.mkdir(parents=True, exist_ok=False)
    config = load_config(root)
    manifest = []
    with Database(root / "work" / "state.sqlite3") as db:
        rows = db.connection.execute(
            "SELECT * FROM files WHERE translation_es IS NOT NULL ORDER BY mode,relative_path"
        ).fetchall()
        with db.transaction() as connection:
            for row in rows:
                input_root, output_root = mode_paths(root, config, row["mode"])
                source = input_root / Path(row["relative_path"])
                output = safe_output_path(source, input_root, output_root)
                saved = None
                if output.is_file():
                    saved = quarantine / row["mode"] / Path(row["relative_path"])
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(output), str(saved))
                manifest.append(
                    {
                        "id": row["id"],
                        "mode": row["mode"],
                        "relative_path": row["relative_path"],
                        "previous_status": row["status"],
                        "transcript_en": row["transcript_en"],
                        "translation_es": row["translation_es"],
                        "quarantined_output": str(saved) if saved else None,
                    }
                )
                connection.execute("DELETE FROM segments WHERE file_id=?", (row["id"],))
                connection.execute(
                    """UPDATE files
                       SET sha256=NULL,config_hash=NULL,transcript_en=NULL,translation_es=NULL,
                           status='DISCOVERED',error_message=NULL,updated_at=?
                       WHERE id=?""",
                    (datetime.now(timezone.utc).isoformat(), row["id"]),
                )
            connection.execute(
                """UPDATE processing_runs
                   SET finished_at=?,status='ABORTED_BAD_TTS_PROMPT',errors=errors+1
                   WHERE status='RUNNING'""",
                (datetime.now(timezone.utc).isoformat(),),
            )
    (quarantine / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pid_file.unlink(missing_ok=True)
    print(f"Cuarentena: {quarantine}")
    print(f"Registros reiniciados: {len(manifest)}")
    print(f"Outputs movidos: {sum(item['quarantined_output'] is not None for item in manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
