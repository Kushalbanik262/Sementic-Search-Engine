"""A stand-in engine so queue tests exercise the transport, not torch."""

from __future__ import annotations

import threading
import time
from typing import Sequence

from main import EncodeResult


class FakeEngine:
    model_name = "fake-model"
    dimensions = 4

    def __init__(self, *, delay_s: float = 0.0, poison: str = "boom") -> None:
        self.delay_s = delay_s
        self.poison = poison
        self.calls: list[list[str]] = []
        self._lock = threading.Lock()

    def encode(self, texts: Sequence[str], *, input_type: str, normalize: bool) -> EncodeResult:
        with self._lock:
            self.calls.append(list(texts))
        if any(self.poison in text for text in texts):
            raise RuntimeError("poison text")
        time.sleep(self.delay_s)
        return EncodeResult(
            embeddings=[[float(len(text)), 0.0, 0.0, 1.0 if input_type == "query" else 0.0] for text in texts],
            truncated=[False] * len(texts),
            took_ms=self.delay_s * 1000,
        )
