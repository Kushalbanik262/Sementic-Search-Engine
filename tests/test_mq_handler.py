"""Message handling decisions, with no broker."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from mq.handler import encode_jobs, encode_jobs_isolating_failures, is_expired, parse_job
from mq.worker import delivery_attempt, parse_lanes
from schemas import EmbeddingJob
from tests.mq_fakes import FakeEngine


def body(**fields: object) -> bytes:
    return json.dumps({"input": ["hello"], **fields}).encode()


def test_valid_message_parses() -> None:
    job, error = parse_job(body(job_id="j1", metadata={"doc_ids": ["d1"]}), fallback_id=None)
    assert error is None
    assert job.job_id == "j1" and job.metadata == {"doc_ids": ["d1"]}


def test_job_id_falls_back_to_the_amqp_property() -> None:
    job, _ = parse_job(body(), fallback_id="from-props")
    assert job.job_id == "from-props"


@pytest.mark.parametrize("raw", [b"not json", b"[1, 2]", b"\xff\xfe"])
def test_non_object_bodies_become_validation_errors(raw: bytes) -> None:
    job, error = parse_job(raw, fallback_id="j9")
    assert job is None
    assert error.status == "error" and error.error.code == "validation_error" and error.job_id == "j9"


def test_validation_uses_the_http_rules() -> None:
    _, error = parse_job(body(input=["   "], job_id="j2", metadata={"k": "v"}), fallback_id=None)
    assert error.error.code == "validation_error"
    assert "empty" in error.error.message
    assert error.metadata == {"k": "v"}  # echoed so the producer can still correlate


def test_oversized_metadata_is_rejected() -> None:
    _, error = parse_job(body(metadata={"blob": "x" * 20_000}), fallback_id="j3")
    assert error.error.code == "validation_error"
    assert "metadata" in error.error.message


def test_expiry() -> None:
    now = datetime.now(UTC)
    assert not is_expired(EmbeddingJob(input=["a"]))
    assert is_expired(EmbeddingJob(input=["a"], deadline=now - timedelta(seconds=1)))
    assert not is_expired(EmbeddingJob(input=["a"], deadline=now + timedelta(seconds=5)))
    # naive timestamps are read as UTC
    assert is_expired(EmbeddingJob(input=["a"], deadline=(now - timedelta(seconds=1)).replace(tzinfo=None)))


def test_one_encode_call_is_split_back_per_job() -> None:
    engine = FakeEngine()
    jobs = [
        EmbeddingJob(input=["a", "bb"], metadata={"n": 1}),
        EmbeddingJob(input=["ccc"], metadata={"n": 2}),
    ]
    results = encode_jobs(engine, jobs, worker_id="w1")

    assert engine.calls == [["a", "bb", "ccc"]]
    assert [r.count for r in results] == [2, 1]
    assert [e[0] for e in results[0].embeddings] == [1.0, 2.0]
    assert results[1].embeddings[0][0] == 3.0
    assert [r.metadata for r in results] == [{"n": 1}, {"n": 2}]
    assert all(r.batch_texts == 3 and r.worker_id == "w1" for r in results)


def test_mixed_keys_are_refused() -> None:
    with pytest.raises(ValueError):
        encode_jobs(FakeEngine(), [EmbeddingJob(input=["a"]), EmbeddingJob(input=["b"], input_type="query")])


def test_a_poison_job_does_not_fail_its_batch_mates() -> None:
    engine = FakeEngine()
    outcomes = encode_jobs_isolating_failures(
        engine, [EmbeddingJob(input=["fine"]), EmbeddingJob(input=["boom"]), EmbeddingJob(input=["ok"])]
    )
    assert outcomes[0].status == "ok"
    assert isinstance(outcomes[1], RuntimeError)
    assert outcomes[2].status == "ok"


def test_the_real_engine_splits_identically(engine) -> None:
    """Batched vectors must equal one-job-at-a-time vectors."""
    jobs = [EmbeddingJob(input=["reset my password"]), EmbeddingJob(input=["billing", "refunds"])]
    batched = encode_jobs(engine, jobs)
    alone = [encode_jobs(engine, [job])[0] for job in jobs]
    for together, single in zip(batched, alone):
        for a, b in zip(together.embeddings, single.embeddings):
            assert max(abs(x - y) for x, y in zip(a, b)) < 1e-4


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        (None, 1),
        ({}, 1),
        ({"x-acquired-count": 2}, 3),  # RabbitMQ 4.x
        ({"x-delivery-count": 1}, 2),  # RabbitMQ 3.13
        ({"x-acquired-count": 1, "x-delivery-count": 4}, 5),
    ],
)
def test_delivery_attempt_reads_either_broker_header(headers, expected) -> None:
    class Message:
        pass

    message = Message()
    message.headers = headers
    assert delivery_attempt(message) == expected


def test_parse_lanes() -> None:
    assert parse_lanes("query, index,query") == ["query", "index"]
    with pytest.raises(ValueError):
        parse_lanes("bulk")
    with pytest.raises(ValueError):
        parse_lanes(" , ")
