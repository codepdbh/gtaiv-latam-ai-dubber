from pathlib import Path

import pytest
import yaml

from gtaiv_dubber.audio.metadata import read_wav_metadata
from gtaiv_dubber.core.config import load_config, relevant_config_hash
from gtaiv_dubber.core.database import Database
from gtaiv_dubber.core.pipeline import BasePipeline
from gtaiv_dubber.core.reports import write_manifests


def test_config_loading_and_hash(sample_project):
    config_dir = sample_project / "config"
    config_dir.mkdir()
    (config_dir / "local.yaml").write_text("audio:\n  peak_limit_dbfs: -2.0\n", encoding="utf-8")
    config = load_config(sample_project)
    assert config["audio"]["peak_limit_dbfs"] == -2.0
    assert relevant_config_hash(config) == relevant_config_hash(config)


def test_corrupt_input_metadata(tmp_path):
    path = tmp_path / "bad.wav"
    path.write_bytes(b"not a wav")
    with pytest.raises(ValueError):
        read_wav_metadata(path)


def test_report_generation(sample_project):
    with Database(sample_project / "work" / "state.sqlite3") as db:
        paths = write_manifests(db, sample_project / "work" / "reports")
    assert len(paths) == 6
    assert all(path.is_file() for path in paths)


class AlwaysFailsPipeline(BasePipeline):
    mode = "speech"
    exact_frames = True

    def _process(self, row, source, destination, digest):
        raise RuntimeError("forced")


def test_safe_fallback_copies_original(sample_project):
    source = sample_project / "Speech" / "A" / "B" / "LINE.wav"
    destination = sample_project / "Speech_es" / "A" / "B" / "LINE.wav"
    metadata = read_wav_metadata(source)
    config = {
        "_config_hash": "cfg",
        "project": {
            "speech_dir": "Speech", "speech_output": "Speech_es",
            "cutscenes_dir": "Cutscenes", "cutscenes_output": "Cutscenes_es",
        },
    }
    runtime = {"tmp": sample_project / "work" / "tmp"}
    with Database(sample_project / "work" / "state.sqlite3") as db:
        file_id = db.upsert_file({
            "mode": "speech", "input_path": str(source), "relative_path": "A/B/LINE.wav",
            "output_path": str(destination), **metadata.to_dict(),
        })
        row = db.get_file("speech", "A/B/LINE.wav")
        result = AlwaysFailsPipeline(sample_project, config, db, None, runtime).process(row)
        saved = db.get_file("speech", "A/B/LINE.wav")
    assert result.status.value == "FALLBACK"
    assert saved["status"] == "FALLBACK"
    assert destination.read_bytes() == source.read_bytes()


def test_resume_ignores_foreign_absolute_output_path(sample_project):
    from gtaiv_dubber.core.state import Status
    from gtaiv_dubber.utils.hashing import sha256_file

    source = sample_project / "Speech" / "A" / "B" / "LINE.wav"
    destination = sample_project / "Speech_es" / "A" / "B" / "LINE.wav"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())
    digest = sha256_file(source)
    config = {
        "_config_hash": "same-config",
        "project": {
            "speech_dir": "Speech", "speech_output": "Speech_es",
            "cutscenes_dir": "Cutscenes", "cutscenes_output": "Cutscenes_es",
        },
    }
    row = {
        "status": Status.DONE,
        "sha256": digest,
        "config_hash": "same-config",
        "relative_path": "A/B/LINE.wav",
        "output_path": r"C:\foreign\Speech_es\A\B\LINE.wav",
    }
    pipeline = BasePipeline(sample_project, config, None, None, {})
    pipeline.mode = "speech"
    assert pipeline.should_skip(row, digest)
