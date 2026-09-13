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

"""What a worker does with a message, minus the broker.

Everything here is plain functions over bytes and pydantic models, so the
decisions (reject, drop as expired, how a batch is split) are testable without
RabbitMQ.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any, Protocol, Sequence

from pydantic import ValidationError

from main import EncodeResult, InputType
from schemas import EmbeddingJob, EmbeddingResult, ErrorCode, JobError

logger = logging.getLogger("embeddings.worker")


class Engine(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def encode(
        self, texts: Sequence[str], *, input_type: InputType, normalize: bool
    ) -> EncodeResult: ...


def error_result(
    job_id: str,
    code: ErrorCode,
    message: str,
    *,
    metadata: dict[str, Any] | None = None,
    worker_id: str | None = None,
) -> EmbeddingResult:
    return EmbeddingResult(
        job_id=job_id,
        status="error",
        error=JobError(code=code, message=message),
        metadata=metadata or {},
        worker_id=worker_id,
    )


def parse_job(
    body: bytes,
    *,
    fallback_id: str | None,
    worker_id: str | None = None,
) -> tuple[EmbeddingJob | None, EmbeddingResult | None]:
    """Return `(job, None)` for a valid message or `(None, error_reply)`.

    The job id comes from the body, else the AMQP message/correlation id, so a
    producer that only set the property still gets a correlatable error.
    """
    try:
        raw = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, error_result(
            fallback_id or "unknown", "validation_error", f"body is not JSON: {exc}",
            worker_id=worker_id,
        )
    if not isinstance(raw, dict):
        return None, error_result(
            fallback_id or "unknown", "validation_error", "body must be a JSON object",
            worker_id=worker_id,
        )

    if not raw.get("job_id") and fallback_id:
        raw["job_id"] = fallback_id
    try:
        return EmbeddingJob.model_validate(raw), None
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'body'}: {error['msg']}"
            for error in exc.errors()
        )
        metadata = raw.get("metadata")
        return None, error_result(
            str(raw.get("job_id") or "unknown"),
            "validation_error",
            details,
            metadata=metadata if isinstance(metadata, dict) else None,
            worker_id=worker_id,
        )


def is_expired(job: EmbeddingJob, now: datetime | None = None) -> bool:
    if job.deadline is None:
        return False
    deadline = job.deadline
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=UTC)
    return deadline <= (now or datetime.now(UTC))


def encode_jobs(
    engine: Engine, jobs: Sequence[EmbeddingJob], *, worker_id: str | None = None
) -> list[EmbeddingResult]:
    """Embed several jobs with ONE model call and split the vectors back out.

    All jobs must share `input_type` and `normalize`; the batcher groups by
    exactly that key. Blocking and CPU-bound: call it from a worker thread.
    """
    if not jobs:
        return []
    input_type, normalize = jobs[0].input_type, jobs[0].normalize
    if any(job.input_type != input_type or job.normalize != normalize for job in jobs):
        raise ValueError("jobs in one batch must share input_type and normalize")

    texts = [text for job in jobs for text in job.input]
    encoded = engine.encode(texts, input_type=input_type, normalize=normalize)

    results = []
    offset = 0
    for job in jobs:
        end = offset + len(job.input)
        results.append(
            EmbeddingResult(
                job_id=job.job_id,
                status="ok",
                model_name=engine.model_name,
                dimensions=engine.dimensions,
                input_type=input_type,
                normalized=normalize,
                count=len(job.input),
                truncated=encoded.truncated[offset:end],
                embeddings=encoded.embeddings[offset:end],
                metadata=job.metadata,
                took_ms=encoded.took_ms,
                batch_texts=len(texts),
                worker_id=worker_id,
            )
        )
        offset = end
    return results


def encode_jobs_isolating_failures(
    engine: Engine, jobs: Sequence[EmbeddingJob], *, worker_id: str | None = None
) -> list[EmbeddingResult | Exception]:
    """Like `encode_jobs`, but one bad job cannot fail its batch-mates.

    If the combined encode raises, every job is retried on its own, so only
    the job that actually breaks the model is reported as failed.
    """
    try:
        return list(encode_jobs(engine, jobs, worker_id=worker_id))
    except Exception as exc:
        if len(jobs) == 1:
            return [exc]
        logger.warning("batch of %d jobs failed, retrying one by one: %r", len(jobs), exc)

    outcomes: list[EmbeddingResult | Exception] = []
    for job in jobs:
        try:
            outcomes.extend(encode_jobs(engine, [job], worker_id=worker_id))
        except Exception as exc:
            outcomes.append(exc)
    return outcomes
