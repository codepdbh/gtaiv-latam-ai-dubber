from __future__ import annotations

from pathlib import Path

import numpy as np

from gtaiv_dubber.audio.diarization import spectral_embeddings
from gtaiv_dubber.audio.resampling import resample_audio


MODEL_RATE = 16000


class SpeakerEmbeddingAdapter:
    """Huellas vocales ECAPA-TDNN para identificar personajes dentro de una escena.

    Si SpeechBrain o el modelo no están disponibles cae al respaldo espectral en
    numpy: el doblaje nunca se detiene por no poder cargar el identificador.
    """

    def __init__(self, config: dict):
        self.config = config or {}
        self._model = None
        self._unavailable = ""

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enabled", True))

    @property
    def backend(self) -> str:
        if not self.enabled or self._unavailable:
            return "spectral"
        return str(self.config.get("backend", "ecapa")).strip().lower()

    def load(self) -> None:
        if self._model is not None or self.backend != "ecapa":
            return
        from speechbrain.inference.speaker import EncoderClassifier

        model_dir = Path(self.config["model_dir"]).expanduser()
        model_dir.mkdir(parents=True, exist_ok=True)
        self._model = EncoderClassifier.from_hparams(
            source=self.config.get("model_id", "speechbrain/spkrec-ecapa-voxceleb"),
            savedir=str(model_dir),
            run_opts={"device": self.config.get("device", "cpu")},
        )

    def prepare(self) -> str:
        """Intenta cargar ECAPA y devuelve el backend que se usará realmente."""
        if self.backend == "ecapa":
            try:
                self.load()
            except Exception as exc:
                self._unavailable = str(exc)
                self._model = None
        return self.backend

    def embed(self, clips: list[np.ndarray], sample_rate: int) -> np.ndarray:
        """Devuelve ``(len(clips), dim)``; una fila por intervención."""
        if not clips:
            return np.zeros((0, 1), dtype=np.float32)
        if self.backend != "ecapa":
            return spectral_embeddings(clips, sample_rate)
        try:
            return self._embed_ecapa(clips, sample_rate)
        except Exception as exc:
            self._unavailable = str(exc)
            self._model = None
            return spectral_embeddings(clips, sample_rate)

    def _embed_ecapa(self, clips: list[np.ndarray], sample_rate: int) -> np.ndarray:
        import torch

        self.load()
        minimum = max(1, round(0.20 * MODEL_RATE))
        prepared = []
        for clip in clips:
            signal = resample_audio(
                np.asarray(clip, dtype=np.float32).reshape(-1), sample_rate, MODEL_RATE
            ).reshape(-1)
            if len(signal) < minimum:
                signal = np.pad(signal, (0, minimum - len(signal)))
            prepared.append(signal)
        # En lotes: una cutscene de cuatro minutos genera casi 300 ventanas y
        # pasarlas juntas reservaba memoria suficiente para matar el proceso.
        # Además cada lote se rellena a su propio largo, no al mayor de todos.
        size = max(1, int(self.config.get("embed_batch_size", 32)))
        outputs = []
        for start in range(0, len(prepared), size):
            group = prepared[start : start + size]
            width = max(len(signal) for signal in group)
            batch = np.zeros((len(group), width), dtype=np.float32)
            lengths = np.zeros(len(group), dtype=np.float32)
            for index, signal in enumerate(group):
                batch[index, : len(signal)] = signal
                lengths[index] = len(signal) / width
            with torch.no_grad():
                embeddings = self._model.encode_batch(
                    torch.from_numpy(batch), torch.from_numpy(lengths)
                )
            outputs.append(embeddings.squeeze(1).detach().cpu().numpy().astype(np.float32))
            del embeddings
        return np.concatenate(outputs, axis=0)

    def unload(self) -> None:
        self._model = None
