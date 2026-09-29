from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    confidence: float | None
    no_speech_probability: float
    words: tuple[tuple[float, float, str], ...] = ()


class WhisperAdapter:
    def __init__(self, config: dict):
        self.config = config
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        download_root = self.config.get("download_root")
        if download_root:
            download_root = str(Path(download_root).expanduser())
        self._model = WhisperModel(
            self.config["model"],
            device=self.config.get("device", "cuda"),
            compute_type=self.config.get("compute_type", "float16"),
            download_root=download_root,
        )

    def transcribe(
        self, audio_path: Path, *, word_timestamps: bool = False
    ) -> list[TranscriptSegment]:
        """``word_timestamps`` sólo lo piden las cutscenes, para poder repartir el
        texto cuando dos personajes caen dentro de la misma intervención."""
        self.load()
        raw_segments, _info = self._model.transcribe(
            str(audio_path),
            language="en",
            task="transcribe",
            beam_size=int(self.config.get("beam_size", 5)),
            vad_filter=bool(self.config.get("vad_filter", True)),
            condition_on_previous_text=True,
            word_timestamps=word_timestamps,
        )
        result = []
        for segment in raw_segments:
            confidence = None
            if segment.avg_logprob is not None:
                confidence = max(0.0, min(1.0, 1.0 + float(segment.avg_logprob)))
            text = segment.text.strip()
            if text:
                result.append(
                    TranscriptSegment(
                        start=float(segment.start),
                        end=float(segment.end),
                        text=text,
                        confidence=confidence,
                        no_speech_probability=float(segment.no_speech_prob or 0.0),
                        words=tuple(
                            (float(word.start), float(word.end), str(word.word))
                            for word in (getattr(segment, "words", None) or ())
                        ),
                    )
                )
        return result

    def unload(self) -> None:
        self._model = None
