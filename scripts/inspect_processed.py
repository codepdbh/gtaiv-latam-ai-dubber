#!/usr/bin/env python3
"""Resume metadatos y hashes de los outputs ya procesados."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gtaiv_dubber.audio.metadata import read_wav_metadata
from gtaiv_dubber.audio.validation import validate_wav
from gtaiv_dubber.core.config import load_config
from gtaiv_dubber.core.database import Database
from gtaiv_dubber.utils.hashing import sha256_file
from gtaiv_dubber.utils.paths import mode_paths, safe_output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = load_config(root)
    result = []
    with Database(root / "work" / "state.sqlite3") as db:
        for row in db.list_files():
            if row["status"] in ("DISCOVERED", "ANALYZING", "TRANSCRIBED", "TRANSLATED", "GENERATED", "ALIGNED"):
                continue
            input_root, output_root = mode_paths(root, config, row["mode"])
            source = input_root / Path(row["relative_path"])
            output = safe_output_path(source, input_root, output_root)
            original = read_wav_metadata(source)
            output_meta = read_wav_metadata(output) if output.is_file() else None
            validation = validate_wav(
                output, original, exact_frames=True
            )
            segment_statuses = []
            if row["mode"] == "cutscene":
                segment_statuses = [
                    item["status"] for item in db.connection.execute(
                        "SELECT status FROM segments WHERE file_id=? ORDER BY segment_index",
                        (row["id"],),
                    )
                ]
            result.append(
                {
                    "mode": row["mode"],
                    "relative_path": row["relative_path"],
                    "status": row["status"],
                    "valid": validation.valid,
                    "reason": validation.reason,
                    "source_hash_matches_db": sha256_file(source) == row["sha256"],
                    "original": original.to_dict(),
                    "output": output_meta.to_dict() if output_meta else None,
                    "segment_statuses": segment_statuses,
                }
            )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if all(item["valid"] and item["source_hash_matches_db"] for item in result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
