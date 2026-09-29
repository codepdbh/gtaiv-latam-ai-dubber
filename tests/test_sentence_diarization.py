"""Diarización por frases: se corta donde cambia quien habla, no a mitad de frase."""
import numpy as np

from gtaiv_dubber.audio.diarization import (
    cluster_labels,
    join_unfinished,
    merge_same_speaker,
    sentence_boundaries,
    split_segment,
)
from gtaiv_dubber.pipelines.cutscene_pipeline import strip_asr_hallucinations


def piece(start, end, text, **extra):
    return {"start": start, "end": end, "text": text, **extra}


def test_boundaries_fall_between_a_closing_word_and_the_next():
    """BC_3: Whisper juntó a Niko y a Bernie en un solo bloque."""
    segment = piece(3.66, 11.05, "Yeah, Bernie, what's wrong? Nothing, sweetie. Oh my god", words=(
        (3.7, 4.0, "Yeah,"), (4.1, 4.5, "Bernie,"), (4.6, 5.0, "what's"), (5.1, 5.6, "wrong?"),
        (7.5, 8.0, "Nothing,"), (8.1, 8.6, "sweetie."), (9.0, 9.4, "Oh"), (9.5, 9.8, "my"),
        (9.9, 10.3, "god"),
    ))
    cuts = sentence_boundaries(segment)
    assert cuts == [(5.6 + 7.5) / 2.0, (8.6 + 9.0) / 2.0]
    pieces = split_segment(segment, cuts, min_turn_seconds=0.6)
    assert [p["text"] for p in pieces] == [
        "Yeah, Bernie, what's wrong?", "Nothing, sweetie.", "Oh my god",
    ]


def test_consecutive_sentences_of_one_voice_are_rejoined():
    pieces = [piece(0, 2, "Hola."), piece(2, 4, "Qué tal."), piece(4, 6, "Bien."), piece(7, 9, "Y tú.")]
    merged, labels = merge_same_speaker(pieces, [0, 0, 1, 1], parents=[0, 0, 0, 1])
    assert [p["text"] for p in merged] == ["Hola. Qué tal.", "Bien.", "Y tú."]
    assert labels == [0, 1, 1]
    assert merged[0]["end"] == 4


def test_an_unfinished_line_is_completed_by_the_same_voice():
    """El monólogo de BC_3 venía en cuatro trozos con pausas de ~0,55 s."""
    pieces = [piece(45.79, 47.93, "I'm working for the mafia"),
              piece(48.47, 52.83, "the police are on my case.")]
    joined, labels = join_unfinished(pieces, [0, 0])
    assert len(joined) == 1
    assert joined[0]["text"] == "I'm working for the mafia the police are on my case."


def test_a_finished_sentence_is_never_joined():
    pieces = [piece(0, 2, "Goodbye, Florian."), piece(2.1, 5, "No, no, no")]
    joined, _labels = join_unfinished(pieces, [0, 0])
    assert len(joined) == 2


def test_two_long_pieces_of_different_voices_stay_apart():
    """Whisper omite el punto: "Goodbye, Florian. No" + la réplica de Bernie."""
    pieces = [piece(77.14, 78.48, "Goodbye, Florian. No"), piece(78.48, 84.8, "No, no, I need you")]
    joined, labels = join_unfinished(pieces, [0, 1])
    assert len(joined) == 2 and labels == [0, 1]


def test_a_tiny_fragment_joins_its_continuation_despite_the_label():
    pieces = [piece(73.30, 74.22, "You are"), piece(74.22, 75.76, "stressed.")]
    joined, labels = join_unfinished(pieces, [1, 2])
    assert [p["text"] for p in joined] == ["You are stressed."]
    assert labels == [1]


def test_joining_respects_the_length_cap_and_simultaneous_speech():
    long = [piece(0, 6, "uno dos"), piece(6.2, 12, "tres cuatro.")]
    assert len(join_unfinished(long, [0, 0])[0]) == 2
    overlap = [piece(0, 2, "uno", speaker_source=0), piece(2.1, 3, "dos.")]
    assert len(join_unfinished(overlap, [0, 0])[0]) == 2


def test_short_lines_join_an_existing_voice_instead_of_founding_one():
    """"Yeah." de 0,28 s fundaba PERSONAJE 4 en B_1."""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    odd = np.array([0.6, 0.1, 0.79])
    voices = np.stack([a, a, b, b, odd])
    voices = voices / np.linalg.norm(voices, axis=1, keepdims=True)
    pieces = [piece(0, 3, "a"), piece(3, 6, "a"), piece(6, 9, "b"), piece(9, 12, "b"),
              piece(12, 12.3, "Yeah.")]
    founded = cluster_labels(voices, pieces, set(), threshold=0.3, max_speakers=8)
    assert len(set(founded)) == 3
    guarded = cluster_labels(
        voices, pieces, set(), threshold=0.3, max_speakers=8, min_cluster_seconds=1.5
    )
    assert len(set(guarded)) == 2
    assert guarded[4] == guarded[0]


def test_short_lines_respect_simultaneous_speech():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    voices = np.stack([a, a, b, a])
    pieces = [piece(0, 3, "a"), piece(3, 6, "a"), piece(6, 9, "b"), piece(1, 1.4, "x")]
    labels = cluster_labels(
        voices, pieces, {(0, 3)}, threshold=0.3, max_speakers=8, min_cluster_seconds=1.5
    )
    assert labels[3] != labels[0]


def test_whisper_you_hallucination_is_removed():
    assert strip_asr_hallucinations("All right, let's have fun Oh You You You You") == (
        "All right, let's have fun Oh"
    )
    assert strip_asr_hallucinations("You") == ""
    assert strip_asr_hallucinations("You.") == ""


def test_real_repetition_is_kept():
    assert strip_asr_hallucinations("No, no, no, no, you don't") == "No, no, no, no, you don't"
    assert strip_asr_hallucinations("You know what?") == "You know what?"
