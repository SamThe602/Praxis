"""Timer utilities scaffold."""

from __future__ import annotations

import contextlib
import time
from typing import Iterator


@contextlib.contextmanager
def deadline(seconds: float) -> Iterator[None]:
    """Simple deadline context placeholder."""
    start = time.time()
    yield
    if time.time() - start > seconds:
        raise TimeoutError("Deadline exceeded in scaffold timer.")
