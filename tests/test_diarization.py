import json
from pathlib import Path

import numpy as np
import pytest

from gtaiv_dubber.ai.qwen_adapter import QwenAdapter, format_scene_context
from gtaiv_dubber.audio.diarization import (
    change_points,
    cluster_cohesion,
    cluster_separation,
    cluster_embeddings,
    diarize_scene,
    neighbour_limits,
    normalize_embeddings,
    overlap_constraints,
    smooth_labels,
    spectral_embedding,
    speaker_profile_segments,
    speech_windows,
    split_segment,
)
from gtaiv_dubber.audio.reference import (
    build_speaker_reference,
    extract_reference,
    reference_window,
)


RATE = 24000


def voice(seed: int, base_hz: float, tilt: float, seconds: float, rate: int = RATE) -> np.ndarray:
    """Voz sintética: dos timbres claramente distintos para probar el agrupamiento."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    signal = np.zeros_like(t)
    for harmonic in range(1, 12):
        signal += (harmonic ** -tilt) * np.sin(
            2 * np.pi * base_hz * harmonic * t + rng.uniform(0.0, 2 * np.pi)
        )
    signal *= 0.5 * (1.0 + np.sin(2 * np.pi * rng.uniform(3.0, 5.0) * t))
    signal += 0.01 * rng.standard_normal(len(t))
    return (signal / np.max(np.abs(signal)) * 0.7).astype(np.float32)


def dark(seed: int, seconds: float = 1.2) -> np.ndarray:
    return voice(seed, 110.0, 1.6, seconds)


def bright(seed: int, seconds: float = 1.2) -> np.ndarray:
    return voice(seed, 215.0, 0.7, seconds)


def test_spectral_embedding_separates_two_timbres():
    clips = [dark(1), bright(2), dark(3), bright(4)]
    embeddings = normalize_embeddings(
        np.stack([spectral_embedding(clip, RATE) for clip in clips])
    )
    labels = cluster_embeddings(embeddings, threshold=0.32, max_speakers=6)
    assert labels[0] == labels[2]
    assert labels[1] == labels[3]
    assert labels[0] != labels[1]


def test_cluster_respects_cannot_link():
    embeddings = normalize_embeddings(np.stack([dark(5)[:64], dark(6)[:64], bright(7)[:64]]))
    labels = cluster_embeddings(
        embeddings, threshold=2.0, max_speakers=1, cannot_link={(0, 1)}
    )
    assert labels[0] != labels[1]


def test_overlap_constraints_marks_simultaneous_lines():
    segments = [
        {"start": 0.0, "end": 2.0},
        {"start": 1.5, "end": 3.0},
        {"start": 4.0, "end": 5.0},
    ]
    assert overlap_constraints(segments) == {(0, 1)}


def test_normalize_embeddings_center_flag():
    data = np.array([[1.0, 0.0], [1.0, 0.1]], dtype=np.float32)
    uncentered = normalize_embeddings(data, center=False)
    assert float(uncentered[0] @ uncentered[1]) > 0.99
    centered = normalize_embeddings(data, center=True)
    assert float(centered[0] @ centered[1]) < -0.99


def test_speech_windows_covers_a_long_turn():
    windows = speech_windows(0.0, 6.0, window_seconds=1.5, hop_seconds=0.75)
    assert windows[0] == (0.0, 1.5)
    assert windows[1] == (0.75, 2.25)
    assert windows[-1][1] == 6.0
    assert len(windows) == 8


def test_speech_windows_short_turn_is_a_single_window():
    assert speech_windows(1.0, 2.0, window_seconds=1.5, hop_seconds=0.75) == [(1.0, 2.0)]
    assert speech_windows(1.0, 1.0) == []


def test_smooth_labels_removes_a_single_flicker():
    assert smooth_labels([0, 0, 1, 0, 0]) == [0, 0, 0, 0, 0]
    assert smooth_labels([0, 0, 1, 1, 1]) == [0, 0, 1, 1, 1]


def test_change_points_lands_between_the_two_windows():
    windows = [(0.0, 1.5), (0.75, 2.25), (1.5, 3.0)]
    assert change_points(windows, [0, 0, 1]) == [pytest.approx(1.875)]
    assert change_points(windows, [0, 0, 0]) == []


def test_split_segment_uses_word_timestamps():
    segment = {
        "start": 0.0, "end": 6.0, "text": "one two three four five six",
        "words": (
            (0.1, 0.5, "one"), (0.8, 1.2, "two"), (2.0, 2.5, "three"),
            (3.4, 3.8, "four"), (4.2, 4.6, "five"), (5.0, 5.5, "six"),
        ),
    }
    pieces = split_segment(segment, [3.0])
    assert [piece["text"] for piece in pieces] == ["one two three", "four five six"]
    assert pieces[0]["start"] == 0.0 and pieces[0]["end"] == pytest.approx(3.0)
    assert pieces[1]["end"] == 6.0
    assert "words" not in pieces[0]


def test_split_segment_without_words_splits_proportionally():
    segment = {"start": 0.0, "end": 6.0, "text": "uno dos tres cuatro"}
    pieces = split_segment(segment, [3.0])
    assert len(pieces) == 2
    assert " ".join(piece["text"] for piece in pieces) == "uno dos tres cuatro"


def test_split_segment_ignores_boundaries_that_are_too_short():
    segment = {"start": 0.0, "end": 2.0, "text": "hola"}
    assert split_segment(segment, [0.2], min_turn_seconds=0.6) == [segment]


def test_split_segment_keeps_full_coverage():
    segment = {
        "start": 1.0, "end": 7.0, "text": "a b c d",
        "words": ((1.1, 1.3, "a"), (1.8, 2.0, "b"), (4.5, 4.7, "c"), (5.5, 5.7, "d")),
    }
    pieces = split_segment(segment, [3.5])
    assert pieces[0]["start"] == 1.0
    assert pieces[-1]["end"] == 7.0
    assert all(
        pieces[index]["start"] == pieces[index - 1]["end"] for index in range(1, len(pieces))
    )


def test_has_multiple_voices_rejects_a_single_speaker():
    from gtaiv_dubber.audio.diarization import has_multiple_voices

    one = normalize_embeddings(
        np.stack([spectral_embedding(dark(101 + i, seconds=1.5), RATE) for i in range(6)])
    )
    assert not has_multiple_voices(one)
    two = normalize_embeddings(
        np.stack(
            [spectral_embedding(clip, RATE) for clip in
             [dark(111), dark(112), dark(113), bright(114), bright(115), bright(116)]]
        )
    )
    assert has_multiple_voices(two)


def test_cluster_cohesion_reports_spread_per_group():
    embeddings = normalize_embeddings(
        np.stack([spectral_embedding(clip, RATE) for clip in [dark(81), dark(82), bright(83)]])
    )
    cohesion = cluster_cohesion(embeddings, [0, 0, 1])
    assert cohesion[1] == 0.0
    assert 0.0 <= cohesion[0] <= 2.0


def test_diarize_scene_splits_a_turn_that_holds_two_voices():
    """El caso real: Whisper devuelve un bloque con dos personajes dentro."""
    scene = np.concatenate([dark(91, seconds=3.0), bright(92, seconds=3.0)])
    segment = {
        "start": 0.0, "end": 6.0, "text": "one two three four five six",
        "words": (
            (0.2, 0.6, "one"), (1.2, 1.6, "two"), (2.2, 2.6, "three"),
            (3.4, 3.8, "four"), (4.4, 4.8, "five"), (5.4, 5.8, "six"),
        ),
    }
    split, labels, cohesion = diarize_scene(
        [segment], lambda _segment: scene, RATE, threshold=0.32,
        min_separation_ratio=8.0,
    )
    assert len(split) == 2
    assert labels[0] != labels[1]
    assert split[0]["end"] == pytest.approx(3.0, abs=1.2)
    assert split[0]["start"] == 0.0 and split[1]["end"] == 6.0
    assert split[0]["text"].startswith("one")
    assert split[1]["text"].endswith("six")
    assert set(cohesion) == set(labels)


def test_diarize_scene_keeps_a_single_voice_intact():
    scene = dark(93, seconds=6.0)
    segment = {"start": 0.0, "end": 6.0, "text": "uno dos tres", "words": ()}
    split, labels, _cohesion = diarize_scene(
        [segment], lambda _segment: scene, RATE, threshold=0.32,
        min_separation_ratio=8.0,
    )
    assert len(split) == 1
    assert labels == [0]
    assert split[0]["text"] == "uno dos tres"


def test_diarize_scene_reuses_a_label_for_the_same_voice():
    scene = np.concatenate(
        [dark(94, seconds=2.5), bright(95, seconds=2.5), dark(96, seconds=2.5)]
    )
    segments = [
        {"start": 0.0, "end": 2.5, "text": "a"},
        {"start": 2.5, "end": 5.0, "text": "b"},
        {"start": 5.0, "end": 7.5, "text": "c"},
    ]
    split, labels, _cohesion = diarize_scene(
        segments, lambda _segment: scene, RATE, threshold=0.32,
        min_separation_ratio=8.0,
    )
    assert len(split) == 3
    assert labels[0] == labels[2] != labels[1]


def test_diarize_scene_reads_simultaneous_lines_from_their_own_source():
    """Dos personajes que se pisan: cada uno debe medirse en su fuente separada."""
    oscura = np.concatenate([dark(97, seconds=4.0), np.zeros(int(2.0 * RATE), np.float32),
                             dark(98, seconds=2.5)])
    clara = bright(99, seconds=len(oscura) / RATE)
    sources = {0: oscura, 1: clara}
    segments = [
        {"start": 0.0, "end": 3.0, "text": "a", "speaker_source": 0},
        {"start": 1.0, "end": 4.0, "text": "b", "speaker_source": 1},
        {"start": 6.0, "end": 8.5, "text": "c", "speaker_source": 0},
    ]
    _split, labels, _cohesion = diarize_scene(
        segments, lambda segment: sources[segment["speaker_source"]], RATE, threshold=0.32
    )
    assert labels[0] != labels[1]
    assert labels[0] == labels[2]


def test_diarize_scene_survives_an_empty_scene():
    assert diarize_scene([], lambda _segment: np.zeros(10), RATE, threshold=0.5) == ([], [], {})


def test_neighbour_limits_keeps_padding_off_the_next_speaker():
    segments = [
        {"start": 0.0, "end": 1.0},
        {"start": 1.2, "end": 2.0},
        {"start": 5.0, "end": 6.0},
    ]
    limits = neighbour_limits(segments, guard_seconds=0.05)
    assert limits[0] == (0.0, pytest.approx(1.15))
    assert limits[1] == (pytest.approx(1.05), pytest.approx(4.95))
    assert limits[2][0] == pytest.approx(2.05)


def test_neighbour_limits_blocks_padding_on_overlap():
    segments = [{"start": 0.0, "end": 2.0}, {"start": 1.5, "end": 3.0}]
    limits = neighbour_limits(segments)
    assert limits[0] == (0.0, 2.0)
    assert limits[1] == (1.5, 3.0)


def test_reference_window_without_limits_is_unchanged():
    assert reference_window(10.0, 4.0, 5.0, 0.75) == (pytest.approx(3.25), pytest.approx(5.75))
    assert reference_window(10.0, 4.0, 4.2, 0.0) == (pytest.approx(3.6), pytest.approx(4.6))


def test_reference_window_respects_limits():
    left, right = reference_window(10.0, 4.0, 4.2, 0.75, limit_start=3.9, limit_end=4.4)
    assert (left, right) == (pytest.approx(3.9), pytest.approx(4.4))


def test_extract_reference_does_not_cross_into_the_neighbour(tmp_path: Path):
    audio = np.concatenate([dark(41, seconds=1.0), bright(42, seconds=1.0)])
    path = extract_reference(
        audio, RATE, 0.0, 1.0, 0.75, tmp_path / "ref.wav", limit_start=0.0, limit_end=1.0
    )
    import soundfile as sf

    data, rate = sf.read(path)
    assert rate == RATE
    assert len(data) == pytest.approx(RATE, rel=0.02)


def test_speaker_profile_segments_prefers_the_longest_lines():
    segments = [
        {"start": 0.0, "end": 0.5},
        {"start": 1.0, "end": 4.0},
        {"start": 5.0, "end": 7.0},
        {"start": 8.0, "end": 8.2},
    ]
    labels = [0, 0, 1, 1]
    chosen = speaker_profile_segments(segments, labels, target_seconds=6.0, max_windows=2)
    assert chosen[0] == [0, 1]
    assert chosen[1] == [2]


def test_build_speaker_reference_pools_clips(tmp_path: Path):
    clips = [dark(51, seconds=1.5), dark(52, seconds=1.5)]
    path, pooled = build_speaker_reference(
        clips, RATE, tmp_path / "speaker.wav", target_seconds=6.0, minimum_seconds=2.0
    )
    assert pooled == pytest.approx(3.0, abs=0.05)
    import soundfile as sf

    data, _rate = sf.read(path)
    assert len(data) > 3.0 * RATE


def test_build_speaker_reference_caps_a_long_monologue(tmp_path: Path):
    path, pooled = build_speaker_reference(
        [dark(53, seconds=30.0)], RATE, tmp_path / "long.wav",
        target_seconds=6.0, max_seconds=12.0,
    )
    assert pooled == pytest.approx(12.0, abs=0.05)
    import soundfile as sf

    data, _rate = sf.read(path)
    assert len(data) <= 12.1 * RATE


def test_build_speaker_reference_rejects_short_material(tmp_path: Path):
    with pytest.raises(ValueError):
        build_speaker_reference(
            [dark(61, seconds=0.4)], RATE, tmp_path / "short.wav", minimum_seconds=2.0
        )


def test_embedding_adapter_falls_back_when_ecapa_fails():
    from gtaiv_dubber.ai.speaker_embedding_adapter import SpeakerEmbeddingAdapter

    adapter = SpeakerEmbeddingAdapter({"enabled": True, "backend": "ecapa"})

    def broken():
        raise RuntimeError("sin speechbrain")

    adapter.load = broken
    assert adapter.prepare() == "spectral"
    vectors = adapter.embed([dark(71), bright(72)], RATE)
    assert vectors.shape[0] == 2


def test_embedding_adapter_disabled_uses_spectral():
    from gtaiv_dubber.ai.speaker_embedding_adapter import SpeakerEmbeddingAdapter

    assert SpeakerEmbeddingAdapter({"enabled": False}).backend == "spectral"


def test_format_scene_context_marks_the_current_line():
    entries = [("PERSONAJE 1", "hello"), ("PERSONAJE 2", "hi"), ("PERSONAJE 1", "bye")]
    context = format_scene_context(entries, 1, window=1)
    assert context.splitlines() == [
        "   PERSONAJE 1: hello",
        ">> PERSONAJE 2: hi",
        "   PERSONAJE 1: bye",
    ]


def _capture(adapter: QwenAdapter) -> list[dict]:
    captured: list[dict] = []

    def fake_request(messages):
        captured.extend(messages)
        return "traducción"

    adapter._request = fake_request
    return captured


def test_speech_prompt_stays_byte_identical():
    """Speech no debe cambiar de prompt: 62.000 clips ya se doblaron con este."""
    adapter = QwenAdapter({}, {"Niko": "Niko"})
    captured = _capture(adapter)
    adapter.adapt("Get out", 1.5, relative_path="A/B.wav")
    system, user = captured
    assert "ESCENA" not in system["content"] and "PERSONAJE" not in system["content"]
    assert user["content"] == (
        "TARGET_DURATION: 1.500 segundos\n"
        "ARCHIVO: A/B.wav\n"
        "LÍNEA ANTERIOR: (ninguna)\n"
        "LÍNEA ACTUAL: Get out\n"
        "LÍNEA SIGUIENTE: (ninguna)\n"
        "GLOSARIO OBLIGATORIO:\n"
        "- Niko: Niko\n"
        "\n"
        "Traduce solamente LÍNEA ACTUAL."
    )


def test_cutscene_prompt_carries_speaker_and_scene():
    adapter = QwenAdapter({}, {})
    captured = _capture(adapter)
    adapter.adapt(
        "Get out", 1.5, relative_path="CM1/CM1.wav",
        speaker_label="PERSONAJE 2",
        scene_context=">> PERSONAJE 2: Get out",
    )
    system, user = captured
    assert "PERSONAJE ACTUAL: PERSONAJE 2" in user["content"]
    assert "ESCENA:\n>> PERSONAJE 2: Get out" in user["content"]
    assert "mismo género" in system["content"]


def test_extract_reference_caps_a_long_monologue(tmp_path: Path):
    """CosyVoice rechaza referencias de mas de 30 s."""
    import soundfile as sf

    audio = dark(201, seconds=45.0)
    path = extract_reference(
        audio, RATE, 0.0, 45.0, 0.0, tmp_path / "larga.wav", max_seconds=20.0
    )
    data, rate = sf.read(path)
    assert rate == RATE
    assert len(data) / RATE == pytest.approx(20.0, abs=0.1)


def test_extract_reference_without_cap_is_unchanged(tmp_path: Path):
    import soundfile as sf

    audio = dark(202, seconds=8.0)
    path = extract_reference(audio, RATE, 0.0, 8.0, 0.0, tmp_path / "corta.wav")
    data, _rate = sf.read(path)
    assert len(data) / RATE == pytest.approx(8.0, abs=0.1)


def test_neighbour_limits_can_starve_a_reference(tmp_path: Path):
    """El caso de BC_3: vecinos pegados dejan la ventana por debajo del minimo.

    Es la condicion que obliga al tercer escalon del respaldo en la tuberia, asi
    que conviene que quede fijada: sin limites la misma linea si da referencia.
    """
    audio = dark(203, seconds=10.0)
    start, end = 5.0, 5.05
    with pytest.raises(ValueError):
        extract_reference(
            audio, RATE, start, end, 0.75, tmp_path / "ahogada.wav",
            limit_start=start, limit_end=start + 0.1,
        )
    libre = extract_reference(
        audio, RATE, start, end, 0.75, tmp_path / "libre.wav",
    )
    assert libre.exists()


def test_build_speaker_reference_accepts_the_lowered_minimum(tmp_path: Path):
    """PERSONAJE 1 de BC_3 sumaba 1.88 s y se quedaba sin voz agrupada."""
    clips = [dark(204, seconds=0.95), dark(205, seconds=0.93)]
    path, pooled = build_speaker_reference(
        clips, RATE, tmp_path / "agrupada.wav", minimum_seconds=1.2
    )
    assert path.exists()
    assert pooled >= 1.2
    with pytest.raises(ValueError):
        build_speaker_reference(
            clips, RATE, tmp_path / "rechazada.wav", minimum_seconds=2.0
        )


def test_cluster_separation_reports_the_nearest_other_group():
    """Dos voces bien distintas: el contraste debe ser claramente positivo."""
    embeddings = normalize_embeddings(
        np.array(
            [[1.0, 0.0], [0.98, 0.20], [0.0, 1.0], [0.20, 0.98]], dtype=np.float32
        )
    )
    labels = [0, 0, 1, 1]
    dentro = cluster_cohesion(embeddings, labels)
    vecino = cluster_separation(embeddings, labels)
    for label in (0, 1):
        assert vecino[label] > dentro[label]


def test_cluster_separation_of_a_lone_group_is_infinite():
    embeddings = normalize_embeddings(np.array([[1.0, 0.0], [0.99, 0.1]], dtype=np.float32))
    assert cluster_separation(embeddings, [0, 0])[0] == float("inf")


def test_qwen_retries_a_timeout_and_succeeds(monkeypatch):
    """Un plazo agotado de Ollama no debe costarnos la linea."""
    import urllib.request

    from gtaiv_dubber.ai import qwen_adapter as modulo

    adapter = QwenAdapter(
        {"request_attempts": 3, "retry_pause_seconds": 0.0}, {}
    )
    intentos = {"n": 0}

    class Respuesta:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"message": {"content": "Vamos, primo."}}'

    def falso_urlopen(_request, timeout=None):
        intentos["n"] += 1
        if intentos["n"] < 3:
            raise TimeoutError("timed out")
        return Respuesta()

    monkeypatch.setattr(urllib.request, "urlopen", falso_urlopen)
    monkeypatch.setattr(modulo.time, "sleep", lambda _s: None)
    assert adapter._request([{"role": "user", "content": "hola"}]) == "Vamos, primo."
    assert intentos["n"] == 3


def test_qwen_gives_up_after_the_configured_attempts(monkeypatch):
    import urllib.request

    from gtaiv_dubber.ai import qwen_adapter as modulo

    adapter = QwenAdapter({"request_attempts": 2, "retry_pause_seconds": 0.0}, {})
    intentos = {"n": 0}

    def siempre_falla(_request, timeout=None):
        intentos["n"] += 1
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", siempre_falla)
    monkeypatch.setattr(modulo.time, "sleep", lambda _s: None)
    with pytest.raises(RuntimeError, match="tras 2 intentos"):
        adapter._request([{"role": "user", "content": "hola"}])
    assert intentos["n"] == 2


def test_qwen_bounds_the_generation_length(monkeypatch):
    """BR_3 segmento 23 se desbocaba y agotaba el plazo en los tres intentos."""
    import urllib.request

    adapter = QwenAdapter({"max_output_tokens": 256}, {})
    enviado = {}

    class Respuesta:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"message": {"content": "Corto."}, "eval_count": 5}'

    def espia(request, timeout=None):
        enviado.update(json.loads(request.data.decode("utf-8")))
        return Respuesta()

    monkeypatch.setattr(urllib.request, "urlopen", espia)
    assert adapter._request([{"role": "user", "content": "hola"}]) == "Corto."
    assert enviado["options"]["num_predict"] == 256
    assert enviado["keep_alive"] == 0


def test_qwen_warns_when_the_cap_is_reached(monkeypatch, caplog):
    import logging
    import urllib.request

    adapter = QwenAdapter({"max_output_tokens": 8}, {})

    class Respuesta:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"message": {"content": "Se corto"}, "eval_count": 8}'

    monkeypatch.setattr(urllib.request, "urlopen", lambda _r, timeout=None: Respuesta())
    with caplog.at_level(logging.WARNING):
        adapter._request([{"role": "user", "content": "hola"}])
    assert "agotó el tope" in caplog.text


def test_diarize_scene_reports_separation_when_asked():
    """La tubería decide la voz común por contraste: necesita la distancia al vecino."""
    scene = np.concatenate(
        [dark(97, seconds=2.5), bright(98, seconds=2.5), dark(99, seconds=2.5)]
    )
    segments = [
        {"start": 0.0, "end": 2.5, "text": "a"},
        {"start": 2.5, "end": 5.0, "text": "b"},
        {"start": 5.0, "end": 7.5, "text": "c"},
    ]
    separation: dict[int, float] = {}
    _split, labels, cohesion = diarize_scene(
        segments, lambda _segment: scene, RATE, threshold=0.32,
        min_separation_ratio=8.0, separation_out=separation,
    )
    assert set(separation) == set(labels) == set(cohesion)
    for label in separation:
        assert separation[label] > cohesion[label]


def test_diarize_scene_single_voice_has_no_neighbour():
    separation: dict[int, float] = {}
    segment = {"start": 0.0, "end": 6.0, "text": "uno dos tres", "words": ()}
    diarize_scene(
        [segment], lambda _segment: dark(100, seconds=6.0), RATE, threshold=0.32,
        min_separation_ratio=8.0, separation_out=separation,
    )
    assert separation == {0: float("inf")}
