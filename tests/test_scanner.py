from gtaiv_dubber.core.database import Database
from gtaiv_dubber.core.scanner import iter_wavs, scan_mode


def test_recursive_scanner_and_exact_relative_path(sample_project):
    config = {
        "project": {
            "speech_dir": "Speech", "speech_output": "Speech_es",
            "cutscenes_dir": "Cutscenes", "cutscenes_output": "Cutscenes_es",
        }
    }
    with Database(sample_project / "work" / "state.sqlite3") as db:
        stats = scan_mode(sample_project, config, db, "speech")
        row = db.list_files("speech")[0]
    assert stats.files == 1
    assert row["relative_path"] == "A/B/LINE.wav"
    assert row["output_path"].endswith("Speech_es/A/B/LINE.wav")
    assert [p.name for p in iter_wavs(sample_project / "Speech")] == ["LINE.wav"]


def test_scanner_preserves_cutscene_metadata(sample_project):
    config = {
        "project": {
            "speech_dir": "Speech", "speech_output": "Speech_es",
            "cutscenes_dir": "Cutscenes", "cutscenes_output": "Cutscenes_es",
        }
    }
    with Database(sample_project / "work" / "state.sqlite3") as db:
        stats = scan_mode(sample_project, config, db, "cutscene")
        row = db.list_files("cutscene")[0]
    assert stats.sample_rates[32000] == 1
    assert row["channels"] == 3
    assert row["frames"] == 3200

