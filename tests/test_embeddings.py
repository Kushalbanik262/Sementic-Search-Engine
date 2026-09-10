"""The /v1/embeddings happy paths and response contract."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from .conftest import embed, l2_norm


def test_accepts_a_bare_string(client: TestClient) -> None:
    body = embed(client, "How do I reset my password?")
    assert body["count"] == 1
    assert len(body["embeddings"]) == 1


def test_accepts_a_list(client: TestClient) -> None:
    body = embed(client, ["first text", "second text", "third text"])
    assert body["count"] == 3
    assert [len(v) for v in body["embeddings"]] == [384, 384, 384]


def test_response_contract(client: TestClient) -> None:
    """Clients persist these vectors, so the shape is a contract worth pinning."""
    body = embed(client, "hello")

    assert set(body) == {
        "model_name",
        "dimensions",
        "input_type",
        "normalized",
        "count",
        "truncated",
        "embeddings",
        "took_ms",
    }
    assert body["model_name"] == "BAAI/bge-small-en-v1.5"
    assert body["dimensions"] == 384
    assert body["input_type"] == "passage"
    assert body["normalized"] is True
    assert body["truncated"] == [False]
    assert body["took_ms"] > 0
    assert all(isinstance(value, float) for value in body["embeddings"][0])


def test_vectors_are_unit_length(client: TestClient) -> None:
    body = embed(client, "cosine similarity should be a plain dot product")
    assert l2_norm(body["embeddings"][0]) == pytest.approx(1.0, abs=1e-4)


def test_input_type_defaults_to_passage(client: TestClient) -> None:
    """Indexing is the common case, and passages must stay instruction-free."""
    assert embed(client, "text")["input_type"] == "passage"


def test_query_and_passage_differ(client: TestClient) -> None:
    as_query = embed(client, "password reset", input_type="query")["embeddings"][0]
    as_passage = embed(client, "password reset", input_type="passage")["embeddings"][0]
    assert as_query != as_passage


def test_truncated_flag_is_per_item(client: TestClient) -> None:
    body = embed(client, ["short", "word " * 700, "also short"])
    assert body["truncated"] == [False, True, False]


def test_full_batch_is_accepted(client: TestClient) -> None:
    """The documented limit must actually work, not just not be exceeded."""
    body = embed(client, [f"document number {i}" for i in range(128)])
    assert body["count"] == 128


def test_response_carries_a_request_id(client: TestClient) -> None:
    response = client.post("/v1/embeddings", json={"input": "hi"})
    assert response.headers["x-request-id"]


def test_incoming_request_id_is_preserved(client: TestClient) -> None:
    """Lets a caller's trace id survive into our logs."""
    response = client.post(
        "/v1/embeddings",
        json={"input": "hi"},
        headers={"X-Request-ID": "trace-abc-123"},
    )
    assert response.headers["x-request-id"] == "trace-abc-123"
