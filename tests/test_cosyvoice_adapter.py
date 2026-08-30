from pathlib import Path

import numpy as np

from gtaiv_dubber.ai.cosyvoice_adapter import CosyVoiceAdapter


class FakeTensor:
    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.zeros((1, 16), dtype=np.float32)


class FakeCosyVoice:
    sample_rate = 24000

    def __init__(self):
        self.received_text = None

    def inference_cross_lingual(self, text, _reference, **_kwargs):
        self.received_text = text
        yield {"tts_speech": FakeTensor()}


def test_cross_lingual_never_speaks_style_instruction(tmp_path: Path):
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference-placeholder")
    fake = FakeCosyVoice()
    adapter = CosyVoiceAdapter({})
    adapter._model = fake

    audio, rate = adapter.synthesize(
        reference,
        "¿Quieres entrar?",
        1.0,
        {"intensity": "media"},
    )

    assert fake.received_text == "You are a helpful assistant.<|endofprompt|>¿Quieres entrar?"
    assert "intensidad" not in fake.received_text
    assert fake.received_text.count("<|endofprompt|>") == 1
    assert audio.shape == (16,)
    assert rate == 24000
