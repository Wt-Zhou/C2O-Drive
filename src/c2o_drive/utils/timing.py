from __future__ import annotations
import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def time_block(name: str) -> Iterator[None]:
    start = time.time()
    try:
        yield
    finally:
        dur = (time.time() - start) * 1000.0
        print(f"[timing] {name}: {dur:.2f} ms")
