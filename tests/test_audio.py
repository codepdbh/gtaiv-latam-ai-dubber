from pathlib import Path

import numpy as np

from gtaiv_dubber.audio.alignment import atempo_chain, fit_frame_count, time_stretch_ffmpeg
from gtaiv_dubber.audio.metadata import read_wav_metadata
from gtaiv_dubber.audio.spatial import apply_spatial_weights, estimate_spatial_weights
from gtaiv_dubber.audio.validation import validate_wav


def test_wav_metadata(wav_factory, tmp_path):
    path = wav_factory(tmp_path / "x.wav", rate=32000, channels=3, frames=1234)
    metadata = read_wav_metadata(path)
    assert (metadata.sample_rate, metadata.channels, metadata.frames, metadata.bit_depth) == (32000, 3, 1234, 16)


def test_wav_extensible_metadata(tmp_path):
    import numpy as np
    import soundfile as sf

    path = tmp_path / "multichannel-wavex.wav"
    sf.write(path, np.zeros((321, 5), dtype=np.float32), 32000, format="WAVEX", subtype="PCM_16")
    metadata = read_wav_metadata(path)
    assert (metadata.sample_rate, metadata.channels, metadata.frames, metadata.bit_depth) == (32000, 5, 321, 16)


def test_exact_frame_validation(wav_factory, tmp_path):
    original = wav_factory(tmp_path / "a.wav", frames=1000)
    same = wav_factory(tmp_path / "b.wav", frames=1000)
    different = wav_factory(tmp_path / "c.wav", frames=999)
    metadata = read_wav_metadata(original)
    assert validate_wav(same, metadata, exact_frames=True).valid
    assert not validate_wav(different, metadata, exact_frames=True).valid


def test_fit_frames_and_channel_spatial_order():
    data = np.ones((50, 3), dtype=np.float32)
    assert fit_frame_count(data, 40).shape == (40, 3)
    assert fit_frame_count(data, 60).shape == (60, 3)
    source = np.column_stack([np.ones(100), np.ones(100) * 0.5, np.ones(100) * 0.1])
    weights = estimate_spatial_weights(source)
    projected = apply_spatial_weights(np.ones(100), weights)
    assert projected.shape == (100, 3)
    assert weights[0] > weights[1] > weights[2]


def test_atempo_extreme_ratios_are_chained():
    assert atempo_chain(4.0).count("atempo=") == 2
    assert atempo_chain(0.25).count("atempo=") == 2


def test_ffmpeg_time_stretch_has_exact_target_frames(tmp_path):
    signal = np.sin(np.linspace(0, 20, 2400, dtype=np.float32))
    stretched = time_stretch_ffmpeg(signal, 24000, 3000, tmp_path)
    assert stretched.shape == (3000, 1)


def test_speech_pipeline_requires_exact_output_frames():
    from gtaiv_dubber.pipelines.speech_pipeline import SpeechPipeline

    assert SpeechPipeline.exact_frames is True
