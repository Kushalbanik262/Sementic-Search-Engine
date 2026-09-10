"""Unit tests for EmbeddingEngine, exercised without going through HTTP."""

from __future__ import annotations

import pytest

from config import get_settings
from main import EmbeddingEngine, ModelNotLoadedError

from .conftest import l2_norm


def test_unloaded_engine_refuses_to_encode() -> None:
    """A fresh engine must fail loudly rather than lazily loading mid-request."""
    fresh = EmbeddingEngine()
    assert fresh.is_ready is False
    with pytest.raises(ModelNotLoadedError):
        fresh.encode(["anything"])


def test_load_is_idempotent(engine) -> None:
    """Startup calls load(); a stray second call must not reload the model."""
    model_before = engine._model
    engine.load()
    assert engine._model is model_before


def test_reports_model_geometry(engine) -> None:
    assert engine.model_name == "BAAI/bge-small-en-v1.5"
    assert engine.dimensions == 384
    assert engine.max_seq_length == 512


def test_encode_shape_matches_input(engine) -> None:
    result = engine.encode(["one", "two", "three"])
    assert len(result.embeddings) == 3
    assert all(len(vector) == 384 for vector in result.embeddings)
    assert len(result.truncated) == 3
    assert result.took_ms > 0


def test_vectors_are_unit_length(engine) -> None:
    result = engine.encode(["how do I reset my password"])
    assert l2_norm(result.embeddings[0]) == pytest.approx(1.0, abs=1e-4)


def test_normalize_false_still_returns_unit_vectors(engine) -> None:
    """Documents a model-specific quirk, not a bug in our code.

    bge-small-en-v1.5 ships a Normalize module inside its SentenceTransformer
    pipeline, so the `normalize` flag cannot produce raw unnormalised vectors.
    Anyone who needs true raw output has to bypass that module. If this test
    ever starts failing, the model was swapped for one without that module.
    """
    result = engine.encode(["hello"], normalize=False)
    assert l2_norm(result.embeddings[0]) == pytest.approx(1.0, abs=1e-4)


def test_encoding_is_deterministic(engine) -> None:
    """Embeddings are persisted, so the same text must not drift between runs."""
    first = engine.encode(["stable text"]).embeddings[0]
    second = engine.encode(["stable text"]).embeddings[0]
    assert first == second


def test_query_type_applies_the_bge_instruction(engine) -> None:
    """A query must embed exactly as its instruction-prefixed passage would.

    This is the check that catches the prefix silently going missing, which
    degrades retrieval without producing any error.
    """
    text = "how do I reset my password"
    instruction = get_settings().query_instruction

    as_query = engine.encode([text], input_type="query").embeddings[0]
    as_prefixed_passage = engine.encode(
        [instruction + text], input_type="passage"
    ).embeddings[0]

    assert as_query == as_prefixed_passage


def test_passage_type_does_not_apply_the_instruction(engine) -> None:
    text = "how do I reset my password"
    as_query = engine.encode([text], input_type="query").embeddings[0]
    as_passage = engine.encode([text], input_type="passage").embeddings[0]
    assert as_query != as_passage


def test_truncation_is_detected(engine) -> None:
    short = "a short sentence"
    long = "word " * 700  # ~700 tokens, over the 512 limit

    result = engine.encode([short, long])
    assert result.truncated == [False, True]


def test_truncation_accounts_for_the_query_instruction(engine) -> None:
    """The instruction eats into the token budget, so it must be measured
    on the prefixed text, not the raw input."""
    # Sized to sit just under the limit bare, and just over it once prefixed.
    text = "word " * 505
    assert engine.encode([text], input_type="passage").truncated == [False]
    assert engine.encode([text], input_type="query").truncated == [True]


def test_empty_batch_returns_nothing(engine) -> None:
    """The API rejects empty input, but the engine should not blow up on it."""
    result = engine.encode([])
    assert result.embeddings == []
    assert result.truncated == []
