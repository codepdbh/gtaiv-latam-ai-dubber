from __future__ import annotations

from pathlib import Path

import numpy as np

from gtaiv_dubber.audio.alignment import fit_frame_count
from gtaiv_dubber.audio.io import read_audio, write_pcm16
from gtaiv_dubber.audio.resampling import resample_audio
from gtaiv_dubber.audio.validation import validate_wav
from gtaiv_dubber.utils.hashing import stable_key


class SpeakerSeparatorAdapter:
    """Separa una mezcla vocal en dos fuentes con SepFormer, de forma local."""

    def __init__(self, config: dict, stems_root: Path):
        self.config = config
        self.stems_root = stems_root
        self._model = None

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enabled", True))

    def load(self) -> None:
        if self._model is not None or not self.enabled:
            return
        try:
            from speechbrain.inference.separation import SepformerSeparation
        except ImportError as exc:
            raise RuntimeError(
                "La separación de hablantes requiere speechbrain. "
                "Reinstale el proyecto con el extra [ai]."
            ) from exc

        model_dir = Path(self.config["model_dir"]).expanduser()
        model_dir.mkdir(parents=True, exist_ok=True)
        self._model = SepformerSeparation.from_hparams(
            source=self.config.get("model_id", "speechbrain/sepformer-wsj02mix"),
            savedir=str(model_dir),
            run_opts={"device": self.config.get("device", "cpu")},
        )

    def separate(
        self,
        mono_audio: np.ndarray,
        sample_rate: int,
        cache_key: str,
    ) -> np.ndarray:
        """Devuelve ``(frames, 2)`` al sample rate original."""
        if not self.enabled:
            raise RuntimeError("La separación de hablantes está desactivada")
        key = stable_key(
            cache_key, self.config.get("model_id", "speechbrain/sepformer-wsj02mix")
        )[:24]
        output_dir = self.stems_root / "speakers" / key
        cached = [output_dir / "speaker_0.wav", output_dir / "speaker_1.wav"]
        if all(validate_wav(path).valid for path in cached):
            sources = [read_audio(path)[0][:, 0] for path in cached]
            return np.column_stack(
                [fit_frame_count(source, len(mono_audio))[:, 0] for source in sources]
            ).astype(np.float32)

        self.load()
        model_rate = int(getattr(self._model.hparams, "sample_rate", 8000))
        signal = resample_audio(
            np.asarray(mono_audio, dtype=np.float32).reshape(-1), sample_rate, model_rate
        ).reshape(-1)
        data = self._separate_chunked(signal, model_rate)
        sources = []
        output_dir.mkdir(parents=True, exist_ok=True)
        for index in range(2):
            source = resample_audio(data[:, index], model_rate, sample_rate)
            source = fit_frame_count(source, len(mono_audio))[:, 0]
            write_pcm16(cached[index], source, sample_rate)
            sources.append(source)
        return np.column_stack(sources).astype(np.float32)

    def _separate_chunked(self, signal: np.ndarray, model_rate: int) -> np.ndarray:
        """Separa por ventanas solapadas: la memoria de SepFormer crece con la duración.

        Medido el 25/09/2026: el pico de RAM sube casi en línea recta con el audio
        de entrada (1,1 GB a 5 s, 2,7 GB a 30 s), así que una cutscene entera de
        dos minutos pedía ~7 GB y mataba el proceso. Troceando, el pico queda fijo.
        Cada ventana se separa por su cuenta y SepFormer no garantiza el orden de
        sus dos salidas, por eso cada trozo se alinea con el anterior comparando la
        zona solapada antes de encadenarlos con un fundido.
        """
        import torch

        window = max(1, round(float(self.config.get("window_seconds", 15.0)) * model_rate))
        overlap = max(1, round(float(self.config.get("overlap_seconds", 1.5)) * model_rate))
        overlap = min(overlap, window // 2)
        hop = window - overlap
        starts = [0]
        while starts[-1] + window < len(signal):
            starts.append(starts[-1] + hop)
        output = np.zeros((0, 2), dtype=np.float32)
        for index, start in enumerate(starts):
            last = index == len(starts) - 1
            chunk = signal[start:] if last else signal[start : start + window]
            if len(chunk) < 2:
                break
            with torch.no_grad():
                estimated = self._model.separate_batch(torch.from_numpy(chunk.copy())[None, :])
            piece = estimated.squeeze(0).detach().cpu().numpy().astype(np.float32)
            del estimated
            if piece.ndim != 2 or piece.shape[1] < 2:
                raise RuntimeError(f"SepFormer devolvió una forma inesperada: {piece.shape}")
            piece = piece[:, :2]
            if not len(output):
                output = piece
            else:
                piece = _match_sources(output, piece, overlap)
                output = _crossfade(output, piece, overlap)
        return output

    def unload(self) -> None:
        self._model = None


def _match_sources(
    tail: np.ndarray,
    piece: np.ndarray,
    overlap: int,
    min_rms: float = 1e-3,
    min_margin: float = 0.05,
) -> np.ndarray:
    """Ordena las dos salidas del trozo nuevo como venían en el anterior.

    Solo invierte cuando la prueba es concluyente. Si la zona solapada cae en un
    silencio, o las dos opciones puntúan casi igual, mantiene el orden que traía:
    un volteo a ciegas cambiaría de personaje a mitad de escena, que es mucho
    peor que arrastrar una duda.
    """
    head = piece[: min(overlap, len(piece))]
    span = min(len(tail), len(head))
    if span < 2:
        return piece
    left, right = tail[-span:], head[:span]
    if min(_rms(left), _rms(right)) < min_rms:
        return piece

    def score(a: np.ndarray, b: np.ndarray) -> float:
        a = a - a.mean()
        b = b - b.mean()
        norm = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / norm) if norm > 1e-8 else 0.0

    straight = score(left[:, 0], right[:, 0]) + score(left[:, 1], right[:, 1])
    swapped = score(left[:, 0], right[:, 1]) + score(left[:, 1], right[:, 0])
    if swapped - straight <= min_margin:
        return piece
    return piece[:, ::-1].copy()


def _rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(block)) + 1e-12))


def _crossfade(output: np.ndarray, piece: np.ndarray, overlap: int) -> np.ndarray:
    """Encadena el trozo nuevo fundiendo la zona solapada."""
    span = min(overlap, len(output), len(piece))
    if span <= 0:
        return np.concatenate([output, piece])
    ramp = np.linspace(0.0, 1.0, span, dtype=np.float32)[:, None]
    merged = output.copy()
    merged[-span:] = merged[-span:] * (1.0 - ramp) + piece[:span] * ramp
    return np.concatenate([merged, piece[span:]])
