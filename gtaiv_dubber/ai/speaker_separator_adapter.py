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
        input_dir = self.stems_root / "speaker_inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        input_path = input_dir / f"{key}.wav"
        write_pcm16(input_path, mono_audio, sample_rate)
        try:
            separated = self._model.separate_file(str(input_path))
            data = separated.squeeze(0).detach().cpu().numpy()
            if data.ndim != 2 or data.shape[1] < 2:
                raise RuntimeError(f"SepFormer devolvió una forma inesperada: {data.shape}")
            model_rate = int(getattr(self._model.hparams, "sample_rate", 8000))
            sources = []
            output_dir.mkdir(parents=True, exist_ok=True)
            for index in range(2):
                source = resample_audio(data[:, index], model_rate, sample_rate)
                source = fit_frame_count(source, len(mono_audio))[:, 0]
                write_pcm16(cached[index], source, sample_rate)
                sources.append(source)
            return np.column_stack(sources).astype(np.float32)
        finally:
            input_path.unlink(missing_ok=True)

    def unload(self) -> None:
        self._model = None
