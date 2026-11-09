from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

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
        self.received_reference = None
        self.reference_info = None

    def inference_cross_lingual(self, text, _reference, **_kwargs):
        self.received_text = text
        self.received_reference = Path(_reference)
        self.reference_info = sf.info(_reference)
        yield {"tts_speech": FakeTensor()}


def test_cross_lingual_never_speaks_style_instruction(tmp_path: Path):
    reference = tmp_path / "reference.wav"
    sf.write(reference, np.zeros(24000, dtype=np.float32), 24000, subtype="PCM_16")
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


@pytest.mark.parametrize("source_rate", [8000, 11025, 16000, 22050, 24000, 32000, 48000])
def test_reference_rate_is_adapted_without_changing_original(tmp_path, source_rate):
    reference = tmp_path / "reference.wav"
    samples = np.sin(np.arange(source_rate) * 2 * np.pi * 300 / source_rate).astype(np.float32)
    sf.write(reference, samples, source_rate, subtype="PCM_16")
    original_bytes = reference.read_bytes()
    fake = FakeCosyVoice()
    adapter = CosyVoiceAdapter({})
    adapter._model = fake

    adapter.synthesize(reference, "Hola.", 1.0)

    assert fake.reference_info.samplerate == max(24000, source_rate)
    assert fake.reference_info.duration == pytest.approx(1.0)
    assert reference.read_bytes() == original_bytes
    if source_rate < 24000:
        assert fake.received_reference != reference
        assert not fake.received_reference.exists()
    else:
        assert fake.received_reference == reference


def test_temporary_reference_is_cleaned_on_model_error(tmp_path):
    reference = tmp_path / "reference.wav"
    sf.write(reference, np.zeros(8000, dtype=np.float32), 8000, subtype="PCM_16")
    adapter = CosyVoiceAdapter({})
    prepared_paths = []

    class BrokenModel:
        def inference_cross_lingual(self, text, prepared, **kwargs):
            prepared_paths.append(Path(prepared))
            assert Path(prepared).is_file()
            raise RuntimeError("model failure")

    adapter._model = BrokenModel()
    with pytest.raises(RuntimeError, match="model failure"):
        adapter.synthesize(reference, "Hola.", 1.0)
    assert not prepared_paths[0].exists()
    assert reference.is_file()
