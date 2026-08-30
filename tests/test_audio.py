from pathlib import Path

import numpy as np

from gtaiv_dubber.audio.alignment import atempo_chain, fit_frame_count, time_stretch_ffmpeg
from gtaiv_dubber.audio.metadata import read_wav_metadata
from gtaiv_dubber.audio.mixing import render_dialogue_replacements
from gtaiv_dubber.audio.speakers import (
    replace_overlapped_segments,
    simultaneous_source_segments,
    text_similarity,
)
from gtaiv_dubber.audio.spatial import (
    apply_spatial_weights,
    estimate_reference_weights,
    estimate_spatial_weights,
)
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


def test_reference_projection_recovers_channel_scale():
    reference = np.sin(np.linspace(0, 30, 4000, dtype=np.float32))
    expected = np.array([3.0, 0.4, -0.2], dtype=np.float32)
    multichannel = reference[:, None] * expected[None, :]
    measured = estimate_reference_weights(multichannel, reference)
    assert np.allclose(measured, expected, atol=1e-3)


def test_overlapping_replacements_remove_english_only_once():
    vocal = np.ones(100, dtype=np.float32) * 0.2
    original = np.column_stack((vocal, vocal))
    tracks = [
        (np.ones(60, dtype=np.float32) * 0.1, 10, 70, np.ones(2, dtype=np.float32)),
        (np.ones(50, dtype=np.float32) * 0.05, 40, 90, np.ones(2, dtype=np.float32)),
    ]
    result = render_dialogue_replacements(original, vocal, tracks, attenuation=1.0)
    assert np.allclose(result[20], [0.1, 0.1])
    assert np.allclose(result[50], [0.15, 0.15])
    assert np.allclose(result[80], [0.05, 0.05])
    assert np.allclose(result[0], [0.2, 0.2])


def test_simultaneous_sources_reject_duplicate_leakage_and_keep_two_speakers():
    timeline = np.arange(4000, dtype=np.float32) / 8000
    sources = np.column_stack(
        (np.sin(2 * np.pi * 220 * timeline), np.sin(2 * np.pi * 337 * timeline))
    )
    distinct = simultaneous_source_segments(
        [[{"start": 0.0, "end": 0.4, "text": "go away"}],
         [{"start": 0.1, "end": 0.5, "text": "leave him alone"}]],
        sources,
        8000,
    )
    duplicate = simultaneous_source_segments(
        [[{"start": 0.0, "end": 0.4, "text": "go away now"}],
         [{"start": 0.1, "end": 0.5, "text": "go away now"}]],
        sources,
        8000,
    )
    assert len(distinct) == 2
    assert duplicate == []
    assert text_similarity("Go away!", "go away") == 1.0

    correlated = np.column_stack((sources[:, 0], -sources[:, 0]))
    assert simultaneous_source_segments(
        [[{"start": 0.0, "end": 0.4, "text": "go away"}],
         [{"start": 0.1, "end": 0.5, "text": "leave him alone"}]],
        correlated,
        8000,
    ) == []


def test_original_segment_is_replaced_by_two_simultaneous_sources():
    original = [{"start": 0.0, "end": 1.0, "text": "combined"}]
    simultaneous = [
        {"start": 0.1, "end": 0.9, "text": "one", "speaker_source": 0},
        {"start": 0.2, "end": 0.8, "text": "two", "speaker_source": 1},
    ]
    result = replace_overlapped_segments(original, simultaneous)
    assert [item["text"] for item in result] == ["one", "two"]
