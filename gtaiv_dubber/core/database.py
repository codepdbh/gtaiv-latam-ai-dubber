from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    mode TEXT NOT NULL CHECK(mode IN ('speech','cutscene')),
    input_path TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    output_path TEXT NOT NULL,
    sha256 TEXT,
    config_hash TEXT,
    sample_rate INTEGER NOT NULL,
    channels INTEGER NOT NULL,
    sample_width INTEGER NOT NULL,
    frames INTEGER NOT NULL,
    duration REAL NOT NULL,
    subtype TEXT NOT NULL,
    transcript_en TEXT,
    translation_es TEXT,
    status TEXT NOT NULL DEFAULT 'DISCOVERED',
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(mode, relative_path)
);
CREATE INDEX IF NOT EXISTS idx_files_mode_status ON files(mode, status);
CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    segment_index INTEGER NOT NULL,
    segment_start REAL NOT NULL,
    segment_end REAL NOT NULL,
    transcript_en TEXT,
    asr_confidence REAL,
    translation_es TEXT,
    speaker_label TEXT,
    reference_path TEXT,
    generated_duration REAL,
    duration_error_pct REAL,
    status TEXT NOT NULL DEFAULT 'DISCOVERED',
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(file_id, segment_index)
);
CREATE TABLE IF NOT EXISTS processing_runs (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    command TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    processed INTEGER NOT NULL DEFAULT 0,
    errors INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS config_hash (
    hash TEXT PRIMARY KEY,
    config_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)
        self._migrate()
        self.connection.commit()

    def _migrate(self) -> None:
        """Añade columnas nuevas a bases ya existentes sin tocar los datos."""
        for table, column, declaration in (("segments", "speaker_label", "TEXT"),):
            existing = {
                row["name"] for row in self.connection.execute(f"PRAGMA table_info({table})")
            }
            if column not in existing:
                self.connection.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {declaration}"
                )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.connection
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def upsert_file(self, item: dict, *, commit: bool = True) -> int:
        now = utc_now()
        values = {
            **item,
            "sha256": item.get("sha256"),
            "config_hash": item.get("config_hash"),
            "status": item.get("status", "DISCOVERED"),
            "error_message": item.get("error_message"),
            "created_at": now,
            "updated_at": now,
        }
        self.connection.execute(
            """
            INSERT INTO files (
                mode,input_path,relative_path,output_path,sha256,config_hash,
                sample_rate,channels,sample_width,frames,duration,subtype,status,
                error_message,created_at,updated_at
            ) VALUES (
                :mode,:input_path,:relative_path,:output_path,:sha256,:config_hash,
                :sample_rate,:channels,:sample_width,:frames,:duration,:subtype,:status,
                :error_message,:created_at,:updated_at
            )
            ON CONFLICT(mode,relative_path) DO UPDATE SET
                input_path=excluded.input_path,
                output_path=excluded.output_path,
                sample_rate=excluded.sample_rate,
                channels=excluded.channels,
                sample_width=excluded.sample_width,
                frames=excluded.frames,
                duration=excluded.duration,
                subtype=excluded.subtype,
                updated_at=excluded.updated_at
            """,
            values,
        )
        if commit:
            self.connection.commit()
        row = self.connection.execute(
            "SELECT id FROM files WHERE mode=? AND relative_path=?",
            (item["mode"], item["relative_path"]),
        ).fetchone()
        return int(row["id"])

    def update_file(self, file_id: int, **changes) -> None:
        if not changes:
            return
        allowed = {
            "sha256", "config_hash", "transcript_en", "translation_es", "status", "error_message"
        }
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"Campos de archivo no permitidos: {sorted(unknown)}")
        changes["updated_at"] = utc_now()
        assignments = ",".join(f"{key}=?" for key in changes)
        self.connection.execute(
            f"UPDATE files SET {assignments} WHERE id=?", (*changes.values(), file_id)
        )
        self.connection.commit()

    def replace_segments(self, file_id: int, segments: list[dict]) -> None:
        now = utc_now()
        with self.transaction() as connection:
            connection.execute("DELETE FROM segments WHERE file_id=?", (file_id,))
            for index, segment in enumerate(segments):
                connection.execute(
                    """
                    INSERT INTO segments(
                        file_id,segment_index,segment_start,segment_end,transcript_en,
                        asr_confidence,translation_es,speaker_label,reference_path,
                        generated_duration,duration_error_pct,status,error_message,
                        created_at,updated_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        file_id, index, segment["start"], segment["end"], segment.get("text"),
                        segment.get("confidence"), segment.get("translation_es"),
                        segment.get("speaker_label"), segment.get("reference_path"),
                        segment.get("generated_duration"),
                        segment.get("duration_error_pct"), segment.get("status", "TRANSCRIBED"),
                        segment.get("error_message"), now, now,
                    ),
                )

    def update_segment(self, file_id: int, index: int, **changes) -> None:
        allowed = {
            "translation_es", "speaker_label", "reference_path", "generated_duration",
            "duration_error_pct", "status", "error_message",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"Campos de segmento no permitidos: {sorted(unknown)}")
        changes["updated_at"] = utc_now()
        assignments = ",".join(f"{key}=?" for key in changes)
        self.connection.execute(
            f"UPDATE segments SET {assignments} WHERE file_id=? AND segment_index=?",
            (*changes.values(), file_id, index),
        )
        self.connection.commit()

    def get_file(self, mode: str, relative_path: str):
        return self.connection.execute(
            "SELECT * FROM files WHERE mode=? AND relative_path=?", (mode, relative_path)
        ).fetchone()

    def list_files(self, mode: str | None = None):
        if mode:
            return self.connection.execute(
                "SELECT * FROM files WHERE mode=? ORDER BY relative_path", (mode,)
            ).fetchall()
        return self.connection.execute("SELECT * FROM files ORDER BY mode,relative_path").fetchall()

    def status_counts(self) -> list[sqlite3.Row]:
        return self.connection.execute(
            "SELECT mode,status,COUNT(*) count FROM files GROUP BY mode,status ORDER BY mode,status"
        ).fetchall()

    def start_run(self, command: str, config_hash: str) -> int:
        cursor = self.connection.execute(
            "INSERT INTO processing_runs(started_at,command,config_hash,status) VALUES(?,?,?,'RUNNING')",
            (utc_now(), command, config_hash),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def finish_run(self, run_id: int, status: str, processed: int, errors: int) -> None:
        self.connection.execute(
            "UPDATE processing_runs SET finished_at=?,status=?,processed=?,errors=? WHERE id=?",
            (utc_now(), status, processed, errors, run_id),
        )
        self.connection.commit()
