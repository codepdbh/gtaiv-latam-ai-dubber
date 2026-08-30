from __future__ import annotations

import gc

from gtaiv_dubber.ai.cosyvoice_adapter import CosyVoiceAdapter
from gtaiv_dubber.ai.demucs_adapter import DemucsAdapter
from gtaiv_dubber.ai.qwen_adapter import QwenAdapter
from gtaiv_dubber.ai.speaker_separator_adapter import SpeakerSeparatorAdapter
from gtaiv_dubber.ai.whisper_adapter import WhisperAdapter


class ModelManager:
    def __init__(self, config: dict, glossary: dict[str, str], stems_root):
        self.whisper = WhisperAdapter(config["whisper"])
        self.qwen = QwenAdapter(config["ollama"], glossary)
        self.cosyvoice = CosyVoiceAdapter(config["cosyvoice"])
        self.demucs = DemucsAdapter(config["demucs"], stems_root)
        self.speaker_separator = SpeakerSeparatorAdapter(
            config.get("speaker_separation", {}), stems_root
        )

    @staticmethod
    def clear_cuda() -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def unload_whisper(self) -> None:
        self.whisper.unload()
        self.clear_cuda()

    def unload_cosyvoice(self) -> None:
        self.cosyvoice.unload()
        self.clear_cuda()

    def unload_speaker_separator(self) -> None:
        self.speaker_separator.unload()
        self.clear_cuda()
