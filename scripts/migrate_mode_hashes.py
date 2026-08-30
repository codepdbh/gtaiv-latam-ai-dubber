#!/usr/bin/env python3
"""Migra outputs válidos del hash global antiguo a hashes por tipo de pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from gtaiv_dubber.audio.metadata import read_wav_metadata
from gtaiv_dubber.audio.validation import validate_wav
from gtaiv_dubber.core.config import load_config, relevant_config_hash
from gtaiv_dubber.core.database import Database
from gtaiv_dubber.core.state import REVIEW_STATUSES, Status
from gtaiv_dubber.utils.hashing import sha256_file
from gtaiv_dubber.utils.paths import mode_paths, safe_output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = load_config(root)
    allowed = {Status.DONE.value, *(status.value for status in REVIEW_STATUSES)}
    migrated = {"speech": 0, "cutscene": 0}
    skipped = 0
    with Database(root / "work" / "state.sqlite3") as db:
        for row in db.list_files():
            if row["status"] not in allowed:
                continue
            input_root, output_root = mode_paths(root, config, row["mode"])
            source = input_root / Path(row["relative_path"])
            destination = safe_output_path(source, input_root, output_root)
            if (
                not source.is_file()
                or sha256_file(source) != row["sha256"]
                or not validate_wav(
                    destination, read_wav_metadata(source), exact_frames=True
                ).valid
            ):
                skipped += 1
                continue
            db.update_file(
                int(row["id"]), config_hash=relevant_config_hash(config, row["mode"])
            )
            migrated[row["mode"]] += 1
    print(
        f"Migrados: Speech={migrated['speech']}, Cutscenes={migrated['cutscene']}; "
        f"omitidos por validación={skipped}"
    )
    return 0 if skipped == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
