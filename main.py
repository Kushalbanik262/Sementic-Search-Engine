# Embedding Service - sentence embeddings over HTTP.
# Copyright (C) 2026 Kushal Banik
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Sentence embedding engine.

Wraps a SentenceTransformer with the parts a long-running service needs:
load-once semantics, warmup, query/passage handling and truncation reporting.
The FastAPI layer lives in `apis.py` and holds no model logic of its own.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config import Settings, get_settings

logger = logging.getLogger(__name__)

InputType = Literal["passage", "query"]


class ModelNotLoadedError(RuntimeError):
    """Raised when the engine is used before `load()` has completed."""


@dataclass(slots=True)
class EncodeResult:
    embeddings: list[list[float]]
    truncated: list[bool]
    took_ms: float

class EmbeddingEngine:
    """Holds the model for the lifetime of the process.

    Loading is guarded by a lock so concurrent callers cannot start two
    downloads; encoding itself is left unsynchronised because the caller
    (`apis.py`) bounds concurrency with a capacity limiter.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model: SentenceTransformer | None = None
        self._load_lock = threading.Lock()

    # --- lifecycle -------------------------------------------------------

    def load(self) -> None:
        """Load the model. Idempotent, so it is safe to call from anywhere."""
        if self._model is not None:
            return

        with self._load_lock:
            # trying to acquire the lock
            if self._model is not None:  # another thread won the race
                return

            s = self._settings
            if s.torch_num_threads:
                torch.set_num_threads(s.torch_num_threads)

            started = time.perf_counter()
            model = SentenceTransformer(s.model_name, device=s.device)
            model.max_seq_length = s.max_seq_length
            self._model = model

            logger.info(
                "model loaded name=%s device=%s dim=%d max_seq_length=%d took=%.1fs",
                s.model_name,
                model.device,
                model.get_embedding_dimension(),
                model.max_seq_length,
                time.perf_counter() - started,
            )

    def warmup(self) -> None:
        """Run one throwaway encode so the first real request is not the slow one."""
        started = time.perf_counter()
        self.encode(["warmup"], input_type="passage")
        logger.info("warmup complete took=%.2fs", time.perf_counter() - started)

    # --- introspection ---------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def model_name(self) -> str:
        return self._settings.model_name

    @property
    def dimensions(self) -> int:
        return self._require_model().get_embedding_dimension()

    @property
    def max_seq_length(self) -> int:
        return self._require_model().max_seq_length

    # --- encoding --------------------------------------------------------

    def encode(
        self,
        texts: Sequence[str],
        *,
        input_type: InputType = "passage",
        normalize: bool = True,
    ) -> EncodeResult:
        """Embed `texts`. Blocking and CPU-bound -- call it from a worker thread.

        `input_type="query"` prepends the BGE retrieval instruction. Passages are
        always embedded bare, which is what the model was trained for.
        """
        model = self._require_model()
        s = self._settings

        # SentenceTransformer.encode raises IndexError on an empty list, so
        # short-circuit rather than leaking that out of a script call.
        if not texts:
            return EncodeResult(embeddings=[], truncated=[], took_ms=0.0)

        prepared = list(texts)
        if input_type == "query" and s.query_instruction:
            prepared = [s.query_instruction + text for text in prepared]

        started = time.perf_counter()
        vectors = model.encode(
            prepared,
            batch_size=s.encode_batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        took_ms = (time.perf_counter() - started) * 1000

        return EncodeResult(
            # float32 -> plain lists, because numpy arrays are not JSON serialisable
            embeddings=np.asarray(vectors, dtype=np.float32).tolist(),
            truncated=self._detect_truncation(prepared),
            took_ms=round(took_ms, 2),
        )

    # --- internals -------------------------------------------------------

    def _require_model(self) -> SentenceTransformer:
        if self._model is None:
            raise ModelNotLoadedError("call EmbeddingEngine.load() first")
        return self._model

    def _detect_truncation(self, texts: Sequence[str]) -> list[bool]:
        """Flag texts the model silently cut short at `max_seq_length`.

        Worth the extra tokenizer pass: quietly dropping the tail of a document
        produces an embedding that looks fine and retrieves badly.
        """
        model = self._require_model()

        # transformers warns on every over-length sequence; here we are measuring
        # over-length on purpose, so keep it out of the logs.
        hf_logger = logging.getLogger("transformers.tokenization_utils_base")
        previous_level = hf_logger.level
        hf_logger.setLevel(logging.ERROR)
        try:
            lengths = model.tokenizer(
                list(texts),
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=False,
                return_length=True,
            )["length"]
        finally:
            hf_logger.setLevel(previous_level)

        return [length > model.max_seq_length for length in lengths]


# Process-wide singleton. `apis.py` loads it during startup.
engine = EmbeddingEngine()


def generate_embedding(
    sentences: str | Sequence[str],
    *,
    input_type: InputType = "passage",
) -> list[list[float]]:
    """Convenience wrapper for scripts and notebooks.

    Loads the model on first use, so prefer the `engine` object inside the API
    where loading is done up front during startup.
    """
    texts = [sentences] if isinstance(sentences, str) else list(sentences)
    engine.load()
    return engine.encode(texts, input_type=input_type).embeddings