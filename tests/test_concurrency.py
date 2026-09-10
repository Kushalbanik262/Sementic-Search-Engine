"""Load shedding.

When every encode slot is busy, requests must queue and then fail fast with
a 503 rather than piling up until the client times out.
"""

from __future__ import annotations

import anyio
import httpx
import pytest
from fastapi.testclient import TestClient

from apis import _encode_limiter, app
from config import Settings, get_settings


@pytest.fixture
def asgi_client(client: TestClient):
    """Async client over the already-loaded app.

    Depends on `client` so the model is loaded; this fixture only adds a way
    to drive the app from inside our own event loop.

    `raise_app_exceptions=False` because Starlette re-raises after running the
    500 handler, and one test needs to inspect that 500 response.
    """
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.anyio
async def test_returns_503_when_all_slots_are_busy(asgi_client, monkeypatch) -> None:
    merged = get_settings().model_dump()
    merged["encode_queue_timeout_s"] = 0.05
    app.dependency_overrides[get_settings] = lambda: Settings(**merged)

    # Hold the only slot so the request cannot get one.
    await _encode_limiter.acquire()
    try:
        async with asgi_client as http:
            response = await http.post("/v1/embeddings", json={"input": "hi"})
        assert response.status_code == 503
        assert "busy" in response.json()["detail"]
    finally:
        _encode_limiter.release()
        app.dependency_overrides.pop(get_settings, None)


@pytest.mark.anyio
async def test_slot_is_released_after_a_request(asgi_client) -> None:
    """A leaked slot would wedge the whole worker after one request."""
    async with asgi_client as http:
        first = await http.post("/v1/embeddings", json={"input": "one"})
        second = await http.post("/v1/embeddings", json={"input": "two"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert _encode_limiter.statistics().borrowed_tokens == 0


@pytest.mark.anyio
async def test_slot_is_released_after_a_failure(asgi_client, monkeypatch) -> None:
    """The `finally` around the encode must survive an exception too."""
    from main import engine

    def boom(*args, **kwargs):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(engine, "encode", boom)

    async with asgi_client as http:
        response = await http.post("/v1/embeddings", json={"input": "hi"})

    assert response.status_code == 500
    assert _encode_limiter.statistics().borrowed_tokens == 0


@pytest.mark.anyio
async def test_concurrent_requests_all_succeed(asgi_client) -> None:
    """Serialised by the limiter, but nothing should be dropped."""
    results: list[int] = []

    async with asgi_client as http:
        async with anyio.create_task_group() as group:

            async def one(text: str) -> None:
                response = await http.post("/v1/embeddings", json={"input": text})
                results.append(response.status_code)

            for index in range(8):
                group.start_soon(one, f"concurrent request {index}")

    assert results == [200] * 8
