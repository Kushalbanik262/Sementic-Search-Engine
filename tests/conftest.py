"""Shared fixtures.

Loading the model takes several seconds, so the app is started once per test
session and every test shares that one instance.
"""

from __future__ import annotations

import math
from typing import Callable, Iterator, Sequence

import pytest
from fastapi.testclient import TestClient

from apis import app
from config import Settings, get_settings
from main import engine as _engine


@pytest.fixture
def anyio_backend() -> str:
    """Async tests run on asyncio only; there is no trio in this project."""
    return "asyncio"


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    """App with lifespan run, so the model is loaded and warm."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def engine(client: TestClient):
    """The loaded engine singleton.

    Depends on `client` purely so the lifespan has already loaded the model.
    """
    return _engine


@pytest.fixture
def override_settings(client: TestClient) -> Iterator[Callable[..., None]]:
    """Temporarily swap Settings for one test.

    Only reaches code that reads settings through `Depends(get_settings)`.
    The request validators call `get_settings()` directly and keep the
    real defaults, so validation-limit tests use the actual configured values.
    """

    def apply(**changes: object) -> None:
        merged = get_settings().model_dump()
        merged.update(changes)
        app.dependency_overrides[get_settings] = lambda: Settings(**merged)

    yield apply
    app.dependency_overrides.pop(get_settings, None)


# --- helpers ---------------------------------------------------------------


def l2_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Both models' outputs are unit-length, but divide anyway so the helper
    stays correct if that ever changes."""
    return sum(x * y for x, y in zip(a, b)) / (l2_norm(a) * l2_norm(b))


def embed(client: TestClient, text: str | list[str], **kwargs: object) -> dict:
    response = client.post("/v1/embeddings", json={"input": text, **kwargs})
    assert response.status_code == 200, response.text
    return response.json()
