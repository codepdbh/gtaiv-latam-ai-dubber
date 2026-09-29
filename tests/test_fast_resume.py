from gtaiv_dubber.core.runner import resume_prefix_length


def _rows(*statuses):
    return [
        {"relative_path": f"{index:03}.wav", "status": status}
        for index, status in enumerate(statuses, 1)
    ]


def test_resume_prefix_uses_saved_checkpoint_without_reading_wavs():
    assert resume_prefix_length(_rows("DONE", "REVIEW_DURATION", "DISCOVERED")) == 2


def test_resume_prefix_counts_completed_silent_and_fallback_rows():
    assert resume_prefix_length(_rows("DONE", "FALLBACK", "REVIEW_ASR")) == 3


def test_resume_prefix_retries_an_interrupted_incomplete_row():
    assert resume_prefix_length(_rows("DONE", "TRANSLATED", "DONE")) == 1
