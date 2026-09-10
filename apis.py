"""HTTP layer for the embedding service.

Run it with:
    uvicorn apis:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import functools
import logging
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator

from anyio import CapacityLimiter, fail_after, to_thread
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from config import Settings, get_settings
from main import EncodeResult, InputType, engine

logger = logging.getLogger("embeddings.api")

# Bounds how many encodes run at once. Requests beyond this queue up rather
# than fighting each other for cores.
_encode_limiter = CapacityLimiter(get_settings().max_concurrent_encodes)

_tasks_run = 0
_total_time_to_process_till_now = 0.0


# --- schemas ---------------------------------------------------------------


class EmbeddingRequest(BaseModel):
    input: list[str] = Field(
        ...,
        description="One or more texts to embed. A bare string is also accepted.",
        examples=[["How do I reset my password?"]],
    )
    input_type: InputType = Field(
        default="passage",
        description=(
            "'passage' for documents you are indexing, 'query' for search terms. "
            "Queries get the BGE retrieval instruction prepended."
        ),
    )
    normalize: bool = Field(
        default=True,
        description="Return unit-length vectors so cosine similarity is a dot product.",
    )

    @field_validator("input", mode="before")
    @classmethod
    def _coerce_and_validate(cls, value: object) -> list[str]:
        settings = get_settings()

        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise ValueError("input must be a string or a list of strings")
        if not value:
            raise ValueError("input must contain at least one text")
        if len(value) > settings.max_batch_items:
            raise ValueError(
                f"batch of {len(value)} exceeds max_batch_items={settings.max_batch_items}"
            )

        for index, text in enumerate(value):
            if not isinstance(text, str):
                raise ValueError(f"input[{index}] must be a string")
            if not text.strip():
                raise ValueError(f"input[{index}] is empty")
            if len(text) > settings.max_chars_per_text:
                raise ValueError(
                    f"input[{index}] has {len(text)} chars, "
                    f"over max_chars_per_text={settings.max_chars_per_text}"
                )
        return value


class EmbeddingResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    dimensions: int
    input_type: InputType
    normalized: bool
    count: int
    truncated: list[bool] = Field(
        description="Per-text flag: True means the text was cut at the model token limit."
    )
    embeddings: list[list[float]]
    took_ms: float


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_name: str
    dimensions: int | None = None
    max_seq_length: int | None = None
    # Needs a default: /health/live reports no queue depth, and in pydantic v2
    # an `int | None` field with no default is still required.
    pending_jobs: int | None = None



class EngineResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    engine_status : str
    models_supported : list[str]
    total_processed_tasks : int
    average_time_to_process : float
    device: str | None

# --- dependencies ----------------------------------------------------------


async def require_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    settings: Settings = Depends(get_settings),
) -> None:
    """No-op when API_KEY is unset, so local development needs no header."""
    if settings.api_key is None:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing API key",
        )


# --- app -------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    )

    # Load and warm up before the port starts accepting traffic, so readiness
    # flips only once the first request can actually be served fast.
    await to_thread.run_sync(engine.load)
    await to_thread.run_sync(engine.warmup)
    logger.info("service ready")

    yield

    logger.info("service shutting down")


settings = get_settings()

app = FastAPI(
    title="Embedding Service",
    version="1.0.0",
    summary=f"Sentence embeddings from {settings.model_name}",
    lifespan=lifespan,
)

if settings.cors_allow_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Tag every request with an id and log how long it took."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id

    started = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - started) * 1000

    response.headers["X-Request-ID"] = request_id
    logger.info(
        "%s %s -> %d %.1fms request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        request_id,
    )
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception("unhandled error request_id=%s", request_id)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "internal server error", "request_id": request_id},
    )


# --- routes ----------------------------------------------------------------


@app.get("/", tags=["meta"])
def read_root() -> dict[str, str]:
    return {"message": "AI server Up"}


@app.get("/health/live", response_model=HealthResponse, tags=["meta"])
def health_live() -> HealthResponse:
    """Liveness: the process is up. Never touches the model."""
    return HealthResponse(status="alive", model_name=engine.model_name)


@app.get("/health/ready", response_model=HealthResponse, tags=["meta"])
def health_ready() -> HealthResponse:
    """Readiness: the model is loaded and warm. Point your load balancer here."""
    if not engine.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model is still loading",
        )
    return HealthResponse(
        status="ready",
        model_name=engine.model_name,
        dimensions=engine.dimensions,
        max_seq_length=engine.max_seq_length,
        pending_jobs=_encode_limiter.statistics().tasks_waiting
    )


@app.get("/health/engine",response_model=EngineResponse,tags=["meta"])
def engine_health() -> EngineResponse:
    """Engine Health Check"""
    device = get_settings().device
    return EngineResponse(
        engine_status= "RUNNING" if engine.is_ready  else "NOT_RUNNING",
        total_processed_tasks=_tasks_run,
        models_supported=["BAAI/bge-small-en-v1.5"],
        average_time_to_process= (_total_time_to_process_till_now / max(_tasks_run,1)),
        device=device
    )


@app.post(
    "/v1/embeddings",
    response_model=EmbeddingResponse,
    dependencies=[Depends(require_api_key)],
    tags=["embeddings"],
)
async def create_embeddings(
    payload: EmbeddingRequest,
    settings: Settings = Depends(get_settings),
) -> EmbeddingResponse:
    if not engine.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model is still loading",
        )

    work = functools.partial(
        engine.encode,
        payload.input,
        input_type=payload.input_type,
        normalize=payload.normalize,
    )

    # Wait for a slot, but shed load instead of queueing forever.
    try:
        with fail_after(settings.encode_queue_timeout_s):
            await _encode_limiter.acquire()
    except TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="server busy, retry shortly",
        ) from None

    try:
        global _tasks_run
        global _total_time_to_process_till_now
        _tasks_run += 1
        result: EncodeResult = await to_thread.run_sync(work)
        _total_time_to_process_till_now += result.took_ms
    finally:
        _encode_limiter.release()

    return EmbeddingResponse(
        model_name=engine.model_name,
        dimensions=engine.dimensions,
        input_type=payload.input_type,
        normalized=payload.normalize,
        count=len(result.embeddings),
        truncated=result.truncated,
        embeddings=result.embeddings,
        took_ms=result.took_ms,
    )
