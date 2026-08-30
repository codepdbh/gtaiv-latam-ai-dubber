from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

import numpy as np


def text_similarity(left: str, right: str) -> float:
    def normalize(value: str) -> str:
        value = unicodedata.normalize("NFKD", value.casefold())
        value = "".join(char for char in value if not unicodedata.combining(char))
        return " ".join(re.findall(r"[a-z0-9]+", value))

    a, b = normalize(left), normalize(right)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def interval_overlap(left: dict, right: dict) -> tuple[float, float]:
    start = max(float(left["start"]), float(right["start"]))
    end = min(float(left["end"]), float(right["end"]))
    return start, max(start, end)


def simultaneous_source_segments(
    by_source: list[list[dict]],
    sources: np.ndarray,
    sample_rate: int,
    *,
    min_overlap_seconds: float = 0.12,
    min_source_rms_ratio: float = 0.12,
    duplicate_similarity: float = 0.72,
    activity_threshold: float = 0.15,
    max_source_correlation: float = 0.25,
) -> list[dict]:
    """Encuentra habla simultánea real entre dos fuentes ya separadas.

    Rechaza solapamientos demasiado cortos, una segunda fuente casi silenciosa y
    texto prácticamente duplicado (filtración del mismo hablante).
    """
    if len(by_source) < 2 or sources.ndim != 2 or sources.shape[1] < 2:
        return []
    selected: dict[tuple[int, int], dict] = {}
    for left_index, left in enumerate(by_source[0]):
        for right_index, right in enumerate(by_source[1]):
            start, end = interval_overlap(left, right)
            if end - start < min_overlap_seconds:
                continue
            first = max(0, round(start * sample_rate))
            last = min(len(sources), round(end * sample_rate))
            if last <= first:
                continue
            region = sources[first:last, :2]
            rms = np.sqrt(np.mean(np.square(region), axis=0) + 1e-10)
            ratio = float(np.min(rms[:2]) / max(float(np.max(rms[:2])), 1e-8))
            if ratio < min_source_rms_ratio:
                continue
            centered = region - np.mean(region, axis=0, keepdims=True)
            norms = np.sqrt(np.sum(np.square(centered), axis=0) + 1e-10)
            correlation = float(np.dot(centered[:, 0], centered[:, 1]) / np.prod(norms))
            if abs(correlation) > max_source_correlation:
                continue
            frame_size = max(1, round(sample_rate * 0.02))
            usable = len(region) // frame_size * frame_size
            if usable < frame_size:
                continue
            framed = region[:usable].reshape(-1, frame_size, 2)
            frame_rms = np.sqrt(np.mean(np.square(framed), axis=1) + 1e-10)
            thresholds = np.max(frame_rms, axis=0) * activity_threshold
            both_active_seconds = float(
                np.count_nonzero(np.all(frame_rms >= thresholds[None, :], axis=1))
                * frame_size / sample_rate
            )
            if both_active_seconds < min_overlap_seconds:
                continue
            if text_similarity(left.get("text", ""), right.get("text", "")) >= duplicate_similarity:
                continue
            selected[(0, left_index)] = {**left, "speaker_source": 0}
            selected[(1, right_index)] = {**right, "speaker_source": 1}
    return sorted(selected.values(), key=lambda item: (item["start"], item["speaker_source"]))


def replace_overlapped_segments(original: list[dict], simultaneous: list[dict]) -> list[dict]:
    if not simultaneous:
        return original
    windows = [(float(item["start"]), float(item["end"])) for item in simultaneous]

    def touches_window(item: dict) -> bool:
        return any(
            min(float(item["end"]), end) - max(float(item["start"]), start) > 0.05
            for start, end in windows
        )

    kept = [item for item in original if not touches_window(item)]
    return sorted(kept + simultaneous, key=lambda item: (item["start"], item.get("speaker_source", -1)))
