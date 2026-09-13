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

"""Wire contracts shared by the HTTP API and the message queue.

Keeping both transports on the same models means a text that is rejected over
HTTP is rejected over the queue too, with the same message.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from config import get_settings
from main import InputType

SCHEMA_VERSION = 1
REQUEST_TYPE = "embedding.request.v1"
RESULT_TYPE = "embedding.result.v1"

Lane = Literal["query", "index"]
ErrorCode = Literal["validation_error", "expired", "internal_error", "dead_lettered"]


# --- HTTP --------------------------------------------------------------------


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


# --- message queue -----------------------------------------------------------


def new_job_id() -> str:
    return uuid.uuid4().hex


class EmbeddingJob(EmbeddingRequest):
    """Body of a message on `embedding.requests`."""

    schema_version: int = SCHEMA_VERSION
    job_id: str = Field(default_factory=new_job_id, min_length=1, max_length=128)
    # Past this instant nobody is waiting for the answer, so workers skip it.
    deadline: datetime | None = None
    # Opaque to the service and echoed back verbatim, e.g. document ids so the
    # indexer can map vectors to documents without its own lookup table.
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def _cap_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        limit = get_settings().mq_metadata_max_bytes
        size = len(json.dumps(value, separators=(",", ":")).encode())
        if size > limit:
            raise ValueError(f"metadata is {size} bytes, over mq_metadata_max_bytes={limit}")
        return value


class JobError(BaseModel):
    code: ErrorCode
    message: str


class EmbeddingResult(BaseModel):
    """Body of a reply. `status="error"` replies carry `error` and no vectors."""

    model_config = ConfigDict(protected_namespaces=())

    schema_version: int = SCHEMA_VERSION
    job_id: str
    status: Literal["ok", "error"]
    error: JobError | None = None
    model_name: str | None = None
    dimensions: int | None = None
    input_type: InputType | None = None
    normalized: bool | None = None
    count: int = 0
    truncated: list[bool] = Field(default_factory=list)
    embeddings: list[list[float]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Encode time of the whole micro-batch this job rode in, and its size.
    took_ms: float = 0.0
    batch_texts: int = 0
    worker_id: str | None = None
