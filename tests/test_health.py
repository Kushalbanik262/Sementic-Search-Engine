"""Health and metadata endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_root(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "AI server Up"}


def test_liveness_does_not_require_the_model(client: TestClient) -> None:
    """Liveness must answer without touching the model.

    A required response field with no default breaks this even though the
    handler code looks fine, so assert the status explicitly.
    """
    response = client.get("/health/live")
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["status"] == "alive"
    assert body["model_name"] == "BAAI/bge-small-en-v1.5"
    assert body["dimensions"] is None
    assert body["pending_jobs"] is None


def test_readiness_reports_model_geometry(client: TestClient) -> None:
    response = client.get("/health/ready")
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["status"] == "ready"
    assert body["dimensions"] == 384
    assert body["max_seq_length"] == 512
    assert body["pending_jobs"] == 0


def test_openapi_schema_builds(client: TestClient) -> None:
    """Catches response models that cannot be turned into a schema."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/v1/embeddings" in response.json()["paths"]
