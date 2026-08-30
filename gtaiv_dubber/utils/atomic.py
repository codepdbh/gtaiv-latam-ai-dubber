from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable


Validator = Callable[[Path], None]


def atomic_publish(
    destination: Path,
    temp_root: Path,
    producer: Callable[[Path], None],
    validator: Validator,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f"{destination.stem}.", suffix=".tmp.wav", dir=temp_root
    )
    os.close(fd)
    temporary = Path(name)
    try:
        producer(temporary)
        validator(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_copy_wav(source: Path, destination: Path, temp_root: Path, validator: Validator) -> None:
    def copy(temp: Path) -> None:
        with source.open("rb") as input_handle, temp.open("wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, 1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())

    atomic_publish(destination, temp_root, copy, validator)

