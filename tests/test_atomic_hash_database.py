from pathlib import Path

import pytest

from gtaiv_dubber.audio.metadata import read_wav_metadata
from gtaiv_dubber.audio.validation import validate_wav
from gtaiv_dubber.core.database import Database
from gtaiv_dubber.utils.atomic import atomic_copy_wav, atomic_publish
from gtaiv_dubber.utils.hashing import sha256_file


def test_sha256_is_stable(wav_factory, tmp_path):
    path = wav_factory(tmp_path / "x.wav")
    assert sha256_file(path) == sha256_file(path)


def test_atomic_copy_and_no_temporary_left(wav_factory, tmp_path):
    source = wav_factory(tmp_path / "original" / "x.wav")
    destination = tmp_path / "output" / "x.wav"
    temporary = tmp_path / "work" / "tmp"
    metadata = read_wav_metadata(source)

    def validator(path):
        assert validate_wav(path, metadata, exact_frames=True).valid

    atomic_copy_wav(source, destination, temporary, validator)
    assert destination.is_file()
    assert not list(temporary.glob("*.tmp.wav"))
    assert sha256_file(source) == sha256_file(destination)


def test_atomic_failure_does_not_replace_existing(wav_factory, tmp_path):
    destination = wav_factory(tmp_path / "output.wav", frames=50)
    before = sha256_file(destination)

    def producer(path):
        path.write_bytes(b"broken")

    with pytest.raises(ValueError):
        atomic_publish(destination, tmp_path / "tmp", producer, lambda _p: (_ for _ in ()).throw(ValueError("bad")))
    assert sha256_file(destination) == before


def test_sqlite_upsert_and_status(sample_project):
    db_path = sample_project / "work" / "state.sqlite3"
    item = {
        "mode": "speech", "input_path": "in.wav", "relative_path": "A/in.wav",
        "output_path": "out.wav", "sample_rate": 24000, "channels": 1,
        "sample_width": 2, "frames": 10, "duration": 10 / 24000, "subtype": "PCM_16",
    }
    with Database(db_path) as db:
        file_id = db.upsert_file(item)
        db.update_file(file_id, status="DONE", sha256="abc", config_hash="cfg")
    with Database(db_path) as db:
        row = db.get_file("speech", "A/in.wav")
        assert row["status"] == "DONE"
        assert row["sha256"] == "abc"

