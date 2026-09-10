"""Input validation. Every case here must be a 422, never a 500."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from config import get_settings

SETTINGS = get_settings()


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        ("empty list", {"input": []}),
        ("empty string", {"input": ""}),
        ("whitespace only", {"input": "   \n\t "}),
        ("empty item in list", {"input": ["fine", ""]}),
        ("non-string item", {"input": [123]}),
        ("nested list", {"input": [["nope"]]}),
        ("null input", {"input": None}),
        ("missing input", {}),
        ("wrong input_type", {"input": "hi", "input_type": "document"}),
        ("wrong normalize type", {"input": "hi", "normalize": "yes please"}),
        (
            "batch too large",
            {"input": ["x"] * (SETTINGS.max_batch_items + 1)},
        ),
        (
            "text too long",
            {"input": ["x" * (SETTINGS.max_chars_per_text + 1)]},
        ),
    ],
)
def test_bad_input_is_rejected(client: TestClient, name: str, payload: dict) -> None:
    response = client.post("/v1/embeddings", json=payload)
    assert response.status_code == 422, f"{name}: got {response.status_code}"


def test_rejection_explains_the_limit(client: TestClient) -> None:
    """A 422 should tell the caller what the ceiling is, not just say 'invalid'."""
    response = client.post(
        "/v1/embeddings", json={"input": ["x"] * (SETTINGS.max_batch_items + 1)}
    )
    assert "max_batch_items" in response.text


def test_boundary_values_are_accepted(client: TestClient) -> None:
    """The limits are inclusive; off-by-one here would reject valid traffic."""
    at_char_limit = "x" * SETTINGS.max_chars_per_text
    assert client.post("/v1/embeddings", json={"input": at_char_limit}).status_code == 200

    at_batch_limit = ["x"] * SETTINGS.max_batch_items
    assert client.post("/v1/embeddings", json={"input": at_batch_limit}).status_code == 200


def test_malformed_json_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/embeddings",
        content=b"{not json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
