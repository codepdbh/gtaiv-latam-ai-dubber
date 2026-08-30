from pathlib import Path

import pytest

from gtaiv_dubber.utils.paths import safe_output_path, windows_to_wsl_path


def test_output_mapping_keeps_name_and_tree(tmp_path):
    source_root = tmp_path / "Speech"
    source = source_root / "deep" / "bank" / "VOICE.wav"
    source.parent.mkdir(parents=True)
    source.touch()
    output = safe_output_path(source, source_root, tmp_path / "Speech_es")
    assert output.relative_to(tmp_path).as_posix() == "Speech_es/deep/bank/VOICE.wav"


def test_refuses_output_into_original(tmp_path):
    root = tmp_path / "Speech"
    source = root / "x.wav"
    root.mkdir()
    source.touch()
    with pytest.raises(RuntimeError):
        safe_output_path(source, root, root)


def test_windows_path_to_wsl():
    assert windows_to_wsl_path(r"D:\Games\GTA IV\audios") == "/mnt/d/Games/GTA IV/audios"

