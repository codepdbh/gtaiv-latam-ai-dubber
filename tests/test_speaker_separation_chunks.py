from pathlib import Path

import numpy as np
import pytest

from gtaiv_dubber.ai.speaker_separator_adapter import (
    SpeakerSeparatorAdapter,
    _crossfade,
    _match_sources,
)


RATE = 8000


def two_tones(seconds: float, rate: int = RATE) -> np.ndarray:
    """Dos fuentes bien distintas, para seguirles la pista entre trozos."""
    t = np.arange(int(seconds * rate)) / rate
    low = np.sin(2 * np.pi * 180 * t).astype(np.float32)
    high = np.sin(2 * np.pi * 900 * t).astype(np.float32)
    return np.column_stack([low, high])


class FakeSepformer:
    """Imita a SepFormer: separa bien, pero alterna el orden de sus dos salidas."""

    def __init__(self, sources: np.ndarray, swap_every: int = 2):
        self.hparams = type("H", (), {"sample_rate": RATE})()
        self.sources = sources
        self.swap_every = swap_every
        self.calls = 0
        self.longest = 0

    def separate_batch(self, mix):
        import torch

        offset = self.calls * 0  # el trozo se localiza por longitud, ver abajo
        length = mix.shape[-1]
        self.longest = max(self.longest, length)
        start = self._find_start(mix.numpy()[0], length)
        piece = self.sources[start : start + length].copy()
        if self.calls % self.swap_every == 1:
            piece = piece[:, ::-1].copy()
        self.calls += 1
        return torch.from_numpy(piece[None, :, :] + offset)

    def _find_start(self, chunk: np.ndarray, length: int) -> int:
        mixed = self.sources.sum(axis=1)
        for start in range(0, len(mixed) - length + 1):
            if np.allclose(mixed[start : start + length], chunk, atol=1e-5):
                return start
        return 0


def make_adapter(tmp_path: Path, window: float, overlap: float) -> SpeakerSeparatorAdapter:
    return SpeakerSeparatorAdapter(
        {"enabled": True, "window_seconds": window, "overlap_seconds": overlap},
        tmp_path,
    )


def test_adapter_keeps_the_interface_the_pipeline_uses(tmp_path):
    """El pipeline llama a estos metodos; perder uno tumba el canario entero."""
    adapter = make_adapter(tmp_path, window=15.0, overlap=1.5)
    for name in ("separate", "unload", "load", "enabled"):
        assert hasattr(adapter, name), f"falta {name}"
    adapter._model = object()
    adapter.unload()
    assert adapter._model is None


def test_match_sources_keeps_order_when_aligned():
    tail = two_tones(1.0)
    piece = two_tones(1.0)
    assert np.allclose(_match_sources(tail, piece, len(piece)), piece)


def test_match_sources_swaps_when_reversed():
    tail = two_tones(1.0)
    piece = two_tones(1.0)[:, ::-1].copy()
    fixed = _match_sources(tail, piece, len(piece))
    assert np.allclose(fixed[:, 0], tail[:, 0], atol=1e-5)


def test_match_sources_keeps_order_when_the_overlap_is_silence():
    """Un volteo a ciegas en un silencio cambiaria de personaje a mitad de escena."""
    tail = np.zeros((800, 2), dtype=np.float32)
    piece = two_tones(1.0)[:, ::-1].copy()
    piece[:800] = 0.0
    assert np.allclose(_match_sources(tail, piece, 800), piece)


def test_match_sources_keeps_order_when_the_evidence_is_weak():
    rng = np.random.default_rng(3)
    tail = rng.standard_normal((800, 2)).astype(np.float32)
    piece = rng.standard_normal((800, 2)).astype(np.float32)
    assert np.allclose(_match_sources(tail, piece, 800), piece)


def test_crossfade_length_and_continuity():
    left = np.ones((100, 2), dtype=np.float32)
    right = np.full((100, 2), 3.0, dtype=np.float32)
    merged = _crossfade(left, right, 20)
    assert len(merged) == 180
    assert merged[0, 0] == pytest.approx(1.0)
    assert merged[-1, 0] == pytest.approx(3.0)
    assert 1.0 < merged[85, 0] < 3.0


def test_chunked_output_covers_the_whole_signal(tmp_path):
    sources = two_tones(40.0)
    adapter = make_adapter(tmp_path, window=15.0, overlap=1.5)
    adapter._model = FakeSepformer(sources)
    result = adapter._separate_chunked(sources.sum(axis=1), RATE)
    assert result.shape == (len(sources), 2)
    assert adapter._model.calls >= 3


def test_chunked_never_feeds_more_than_one_window(tmp_path):
    """La razón de ser del troceo: el pico de memoria no puede crecer con el audio."""
    sources = two_tones(120.0)
    adapter = make_adapter(tmp_path, window=15.0, overlap=1.5)
    adapter._model = FakeSepformer(sources, swap_every=99)
    adapter._separate_chunked(sources.sum(axis=1), RATE)
    assert adapter._model.longest <= int(15.0 * RATE) + 1


def test_chunked_realigns_swapped_chunks(tmp_path):
    """SepFormer no garantiza el orden de sus salidas entre trozos."""
    sources = two_tones(40.0)
    adapter = make_adapter(tmp_path, window=15.0, overlap=1.5)
    adapter._model = FakeSepformer(sources, swap_every=2)
    result = adapter._separate_chunked(sources.sum(axis=1), RATE)
    half = len(sources) // 2
    corr_straight = float(np.corrcoef(result[half:, 0], sources[half:, 0])[0, 1])
    corr_swapped = float(np.corrcoef(result[half:, 0], sources[half:, 1])[0, 1])
    assert corr_straight > corr_swapped


def test_chunked_handles_audio_shorter_than_one_window(tmp_path):
    sources = two_tones(4.0)
    adapter = make_adapter(tmp_path, window=15.0, overlap=1.5)
    adapter._model = FakeSepformer(sources, swap_every=99)
    result = adapter._separate_chunked(sources.sum(axis=1), RATE)
    assert result.shape == (len(sources), 2)
    assert adapter._model.calls == 1
