from __future__ import annotations

import re
from enum import Enum


class SpeechKind(str, Enum):
    LINGUISTIC_SPEECH = "LINGUISTIC_SPEECH"
    NONVERBAL_VOCALIZATION = "NONVERBAL_VOCALIZATION"
    NO_SPEECH = "NO_SPEECH"


NONVERBAL_CUES = re.compile(
    r"(?:^|_)(BUMP|CRASH|PAIN|DODGE|BREATH|GRUNT|EFFORT|MOAN|SCREAM|GASP|COUGH|LAUGH)(?:_|$)",
    re.IGNORECASE,
)
NO_SPEECH_CUES = re.compile(r"(?:^|_)(SILENCE|EMPTY|SFX|IMPACT)(?:_|$)", re.IGNORECASE)


def classify_speech(filename: str, transcript: str, no_speech_probability: float) -> SpeechKind:
    text = transcript.strip()
    if text and no_speech_probability < 0.78:
        return SpeechKind.LINGUISTIC_SPEECH
    stem = filename.rsplit(".", 1)[0]
    if NONVERBAL_CUES.search(stem):
        return SpeechKind.NONVERBAL_VOCALIZATION
    if NO_SPEECH_CUES.search(stem) or no_speech_probability >= 0.92:
        return SpeechKind.NO_SPEECH
    if text:
        return SpeechKind.LINGUISTIC_SPEECH
    return SpeechKind.NONVERBAL_VOCALIZATION
