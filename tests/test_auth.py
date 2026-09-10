"""API key enforcement.

Auth is off unless API_KEY is set, so these tests switch it on through the
settings dependency rather than mutating the environment.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

KEY = "test-secret-key"


def test_auth_is_off_by_default(client: TestClient) -> None:
    """Local development must not need a header."""
    response = client.post("/v1/embeddings", json={"input": "hi"})
    assert response.status_code == 200


def test_missing_key_is_rejected(client: TestClient, override_settings) -> None:
    override_settings(api_key=KEY)
    response = client.post("/v1/embeddings", json={"input": "hi"})
    assert response.status_code == 401


def test_wrong_key_is_rejected(client: TestClient, override_settings) -> None:
    override_settings(api_key=KEY)
    response = client.post(
        "/v1/embeddings", json={"input": "hi"}, headers={"X-API-Key": "wrong"}
    )
    assert response.status_code == 401


def test_correct_key_is_accepted(client: TestClient, override_settings) -> None:
    override_settings(api_key=KEY)
    response = client.post(
        "/v1/embeddings", json={"input": "hi"}, headers={"X-API-Key": KEY}
    )
    assert response.status_code == 200


def test_health_stays_open_when_auth_is_on(
    client: TestClient, override_settings
) -> None:
    """Probes are unauthenticated, so locking these would fail every deploy."""
    override_settings(api_key=KEY)
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 200


def test_auth_runs_before_validation(client: TestClient, override_settings) -> None:
    """An unauthenticated caller should not learn our limits from 422 bodies."""
    override_settings(api_key=KEY)
    response = client.post("/v1/embeddings", json={"input": []})
    assert response.status_code == 401
