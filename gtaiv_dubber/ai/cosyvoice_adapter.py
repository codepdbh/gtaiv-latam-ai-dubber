from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


class CosyVoiceAdapter:
    def __init__(self, config: dict):
        self.config = config
        self._model = None
        self.sample_rate = 24000

    def load(self) -> None:
        if self._model is not None:
            return
        # Torchaudio 2.11 delega la lectura a TorchCodec, que requiere una
        # instalación compartida completa de FFmpeg aunque aquí sólo abrimos
        # PCM WAV. CosyVoice llama a torchaudio.load desde file_utils; este
        # lector equivalente mantiene el tensor [canales, muestras] y evita
        # introducir dependencias de vídeo ajenas al pipeline.
        import soundfile as sf
        import torch
        import torchaudio

        def load_pcm_wav(path, *args, **kwargs):
            del args, kwargs
            samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
            return torch.from_numpy(samples.T.copy()), sample_rate

        torchaudio.load = load_pcm_wav
        repo_dir = Path(self.config["repo_dir"]).expanduser()
        model_dir = Path(self.config["model_dir"]).expanduser()
        matcha = repo_dir / "third_party" / "Matcha-TTS"
        for path in (repo_dir, matcha):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        if not model_dir.is_dir():
            raise RuntimeError(f"Falta modelo CosyVoice: {model_dir}")
        try:
            from cosyvoice.cli.cosyvoice import AutoModel
        except ImportError as exc:
            raise RuntimeError(f"No se pudo importar CosyVoice desde {repo_dir}: {exc}") from exc
        self._model = AutoModel(model_dir=str(model_dir))
        self.sample_rate = int(self._model.sample_rate)

    def synthesize(
        self,
        reference_audio: Path,
        text_es: str,
        target_duration: float,
        style_hints: dict | None = None,
    ) -> tuple[np.ndarray, int]:
        self.load()
        if not reference_audio.is_file():
            raise FileNotFoundError(reference_audio)
        # CosyVoice 3 exige el prefijo neutro oficial y <|endofprompt|>. No se
        # añaden instrucciones de estilo en español: el modelo acababa
        # pronunciándolas en vez del diálogo. La referencia aporta la prosodia.
        del target_duration, style_hints
        spoken_text = text_es.strip()
        if not spoken_text:
            raise ValueError("Texto TTS vacío")
        tts_text = f"You are a helpful assistant.<|endofprompt|>{spoken_text}"
        chunks = []
        for result in self._model.inference_cross_lingual(
            tts_text, str(reference_audio), stream=False, text_frontend=False
        ):
            tensor = result["tts_speech"].detach().float().cpu().numpy().reshape(-1)
            chunks.append(tensor)
        if not chunks:
            raise RuntimeError("CosyVoice no produjo audio")
        return np.concatenate(chunks).astype(np.float32), self.sample_rate

    def unload(self) -> None:
        self._model = None
