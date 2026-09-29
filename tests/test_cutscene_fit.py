"""Encaje de réplicas en cutscenes: la voz no se deforma para cuadrar el hueco."""
from pathlib import Path

import numpy as np
import pytest

from gtaiv_dubber.audio.alignment import fit_dialogue
from gtaiv_dubber.pipelines.cutscene_pipeline import (
    fit_badness,
    is_vocalization,
    length_instruction,
    same_words,
)

RATE = 16000


def tone(seconds: float) -> np.ndarray:
    t = np.arange(round(seconds * RATE)) / RATE
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_a_short_voice_keeps_its_speed_and_gets_silence(tmp_path: Path):
    """El 42% de las voces salían más cortas y se ralentizaban."""
    voice = tone(1.0)
    fitted, tempo, trimmed = fit_dialogue(voice, RATE, 2.0, 2.0, tmp_path)
    assert tempo == 1.0 and not trimmed
    assert len(fitted) == 2 * RATE
    assert np.allclose(fitted[: RATE, 0], voice)
    assert not np.any(fitted[RATE:, 0])


def test_a_long_voice_uses_the_following_silence_first(tmp_path: Path):
    voice = tone(2.4)
    fitted, tempo, trimmed = fit_dialogue(voice, RATE, 2.0, 2.6, tmp_path)
    assert tempo == 1.0 and not trimmed
    assert len(fitted) == len(voice)


def test_speedup_is_capped(tmp_path: Path):
    voice = tone(2.4)
    fitted, tempo, trimmed = fit_dialogue(voice, RATE, 2.0, 2.0, tmp_path, hard_max_speedup=1.35)
    assert tempo == pytest.approx(1.2, abs=0.01)
    assert not trimmed
    assert len(fitted) <= 2 * RATE


def test_a_hopeless_voice_is_trimmed_not_crushed(tmp_path: Path):
    """El peor caso medido se comprimía 32 veces: eso ya no puede pasar."""
    voice = tone(4.0)
    fitted, tempo, trimmed = fit_dialogue(voice, RATE, 1.0, 1.0, tmp_path, hard_max_speedup=1.35)
    assert tempo == pytest.approx(1.35)
    assert trimmed
    assert len(fitted) == RATE
    assert abs(fitted[-1, 0]) < 1e-3


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Ha-ha-ha-ha-ha!", True),
        ("Oh!", True),
        ("Argh... ugh.", True),
        ("Oh no!", False),
        ("Hey!", False),
        ("You", False),
    ],
)
def test_vocalizations_are_not_dubbed(text, expected):
    assert is_vocalization(text) is expected


def test_same_words_ignores_punctuation():
    assert same_words("No!", "¡No!")
    assert not same_words("You", "Tú")


def test_badness_prefers_a_fitting_voice():
    fits = fit_badness(90, 100, 110, 0.55)
    too_long = fit_badness(160, 100, 110, 0.55)
    too_short = fit_badness(30, 100, 110, 0.55)
    assert fits == 0.0
    assert too_long > too_short > fits


def test_length_instruction_gives_numbers():
    shorter = length_instruction("uno dos tres cuatro cinco seis", 3.0, 1.5, 1.6)
    assert "6 palabras" in shorter and "1.6 s" in shorter and "como máximo" in shorter
    longer = length_instruction("vale", 0.4, 2.0, 2.0)
    assert "más completa" in longer


def test_a_looping_translation_is_detected():
    from gtaiv_dubber.pipelines.cutscene_pipeline import runaway_translation

    source = "It's not what you have to play."
    loop = ("No es lo que tienes que jugar, es lo que tienes que ser, es lo que tienes "
            "que hacer, es lo que tienes que decir")
    assert runaway_translation(source, loop)
    assert not runaway_translation(source, "No es lo que tienes que interpretar.")
    assert not runaway_translation("Yeah.", "Sí, claro, por supuesto.")
