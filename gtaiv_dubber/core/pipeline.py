from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from gtaiv_dubber.audio.metadata import read_wav_metadata
from gtaiv_dubber.audio.validation import validate_wav
from gtaiv_dubber.core.database import Database
from gtaiv_dubber.core.state import Status
from gtaiv_dubber.utils.atomic import atomic_copy_wav
from gtaiv_dubber.utils.hashing import sha256_file
from gtaiv_dubber.utils.paths import mode_paths, safe_output_path


@dataclass(slots=True)
class ProcessResult:
    status: Status
    relative_path: str
    message: str = ""
    elapsed: float = 0.0


class BasePipeline:
    mode = ""
    exact_frames = False

    def __init__(self, root: Path, config: dict, db: Database, models, runtime: dict, logger=None):
        self.root = root
        self.config = config
        self.db = db
        self.models = models
        self.runtime = runtime
        self.logger = logger or logging.getLogger("gtaiv_dubber")
        self.config_hash = config["_config_hash"]

    def _prepare(self, row) -> tuple[Path, Path, str]:
        input_root, output_root = mode_paths(self.root, self.config, self.mode)
        source = input_root / Path(row["relative_path"])
        destination = safe_output_path(source, input_root, output_root)
        if not source.is_file():
            raise FileNotFoundError(source)
        digest = sha256_file(source)
        return source, destination, digest

    def should_skip(self, row, digest: str) -> bool:
        if row["status"] != Status.DONE:
            return False
        if row["sha256"] != digest or row["config_hash"] != self.config_hash:
            return False
        input_root, output_root = mode_paths(self.root, self.config, self.mode)
        source = input_root / Path(row["relative_path"])
        destination = safe_output_path(source, input_root, output_root)
        original = read_wav_metadata(source)
        validation = validate_wav(
            destination, original, exact_frames=self.exact_frames
        )
        return validation.valid

    def fallback(self, row, source: Path, destination: Path, digest: str, error: Exception) -> ProcessResult:
        original = read_wav_metadata(source)

        def validator(path: Path) -> None:
            result = validate_wav(path, original, exact_frames=self.exact_frames)
            if not result.valid:
                raise ValueError(result.reason)

        atomic_copy_wav(source, destination, self.runtime["tmp"], validator)
        self.db.update_file(
            int(row["id"]), sha256=digest, config_hash=self.config_hash,
            status=Status.FALLBACK, error_message=str(error),
        )
        self.logger.error("[FALLBACK] %s: %s", row["relative_path"], error)
        return ProcessResult(Status.FALLBACK, row["relative_path"], str(error))

    def process(self, row) -> ProcessResult:
        started = time.monotonic()
        input_root, output_root = mode_paths(self.root, self.config, self.mode)
        source = input_root / Path(row["relative_path"])
        destination = safe_output_path(source, input_root, output_root)
        digest = ""
        try:
            source, destination, digest = self._prepare(row)
            if self.should_skip(row, digest):
                self.logger.info("[SKIP] %s", row["relative_path"])
                return ProcessResult(Status.DONE, row["relative_path"], "SKIP", time.monotonic() - started)
            result = self._process(row, source, destination, digest)
            result.elapsed = time.monotonic() - started
            return result
        except Exception as exc:
            try:
                result = self.fallback(row, source, destination, digest, exc)
            except Exception as fallback_exc:
                self.db.update_file(int(row["id"]), status=Status.ERROR, error_message=str(fallback_exc))
                self.logger.exception("[ERROR] %s", row["relative_path"])
                result = ProcessResult(Status.ERROR, row["relative_path"], str(fallback_exc))
            result.elapsed = time.monotonic() - started
            return result

    def _process(self, row, source: Path, destination: Path, digest: str) -> ProcessResult:
        raise NotImplementedError

    def synthesize_with_oom_retry(self, *args, **kwargs):
        try:
            return self.models.cosyvoice.synthesize(*args, **kwargs)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            self.logger.warning("[TTS] CUDA OOM; limpiando VRAM y reintentando una vez")
            self.models.clear_cuda()
            return self.models.cosyvoice.synthesize(*args, **kwargs)
