"""End to end over a real RabbitMQ.

Skipped when RABBITMQ_URL is unreachable. Start one with:
    docker run -d -p 5672:5672 rabbitmq:4-management

Every test gets its own exchange and queue names, so tests cannot see each
other's messages and nothing leaks into a real deployment on the same broker.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

import aio_pika
import pytest

from config import Settings, get_settings
from mq.client import EmbeddingClient, EmbeddingJobFailed, EmbeddingOverloaded
from mq.worker import EmbeddingWorker
from schemas import EmbeddingResult
from tests.mq_fakes import FakeEngine

pytestmark = [pytest.mark.anyio, pytest.mark.integration]


@pytest.fixture(scope="module")
def broker_url() -> str:
    url = get_settings().rabbitmq_url

    async def probe() -> None:
        connection = await asyncio.wait_for(aio_pika.connect(url), 3)
        await connection.close()

    try:
        asyncio.run(probe())
    except Exception as exc:
        pytest.skip(f"RabbitMQ not reachable at {url}: {exc!r}")
    return url


@pytest.fixture
async def settings(broker_url: str) -> AsyncIterator[Settings]:
    prefix = f"test.{uuid.uuid4().hex[:8]}"
    merged = get_settings().model_dump()
    merged.update(
        rabbitmq_url=broker_url,
        mq_exchange_requests=f"{prefix}.requests",
        mq_exchange_events=f"{prefix}.events",
        mq_exchange_dlx=f"{prefix}.dlx",
        mq_dead_letter_queue=f"{prefix}.dead",
        query_queue=f"{prefix}.query",
        index_queue=f"{prefix}.index",
        index_results_queue=f"{prefix}.results",
        query_batch_window_ms=2,
        index_batch_window_ms=100,
        index_delivery_limit=2,
        worker_shutdown_grace_s=10,
    )
    s = Settings(**merged)
    yield s

    connection = await aio_pika.connect(broker_url)
    channel = await connection.channel()
    for queue in (s.query_queue, s.index_queue, s.index_results_queue, s.mq_dead_letter_queue):
        await channel.queue_delete(queue)
    for exchange in (s.mq_exchange_requests, s.mq_exchange_events, s.mq_exchange_dlx):
        await channel.exchange_delete(exchange)
    await connection.close()


@asynccontextmanager
async def running(worker: EmbeddingWorker) -> AsyncIterator[EmbeddingWorker]:
    task = asyncio.create_task(worker.run())
    ready = asyncio.create_task(worker.wait_ready())
    done, _ = await asyncio.wait({task, ready}, timeout=15, return_when=asyncio.FIRST_COMPLETED)
    if task in done:
        task.result()  # surface the startup error
    assert ready in done, "worker did not become ready"
    try:
        yield worker
    finally:
        worker.stop()
        await asyncio.wait_for(task, 20)


def make_worker(settings: Settings, lanes: list[str], engine: FakeEngine | None = None, name: str = "w") -> EmbeddingWorker:
    return EmbeddingWorker(
        engine or FakeEngine(), lanes, settings=settings, worker_id=f"{name}-{uuid.uuid4().hex[:4]}", load_model=False
    )


async def collect_results(client: EmbeddingClient, settings: Settings) -> asyncio.Queue[EmbeddingResult]:
    inbox: asyncio.Queue[EmbeddingResult] = asyncio.Queue()

    async def handle(result: EmbeddingResult) -> None:
        await inbox.put(result)

    await client.consume_results(handle, queue=settings.index_results_queue)
    return inbox


async def test_query_round_trip(settings: Settings) -> None:
    async with running(make_worker(settings, ["query"])), EmbeddingClient(settings, app_id="search") as client:
        result = await client.embed_query(["red shoes", "blue hat"], timeout=5)

    assert result.status == "ok"
    assert result.count == 2 and result.dimensions == 4
    assert result.input_type == "query"
    assert [vector[0] for vector in result.embeddings] == [9.0, 8.0]


async def test_query_round_trip_with_the_real_model(settings: Settings, engine) -> None:
    worker = EmbeddingWorker(engine, ["query"], settings=settings, load_model=False)
    async with running(worker), EmbeddingClient(settings) as client:
        result = await client.embed_query("how do I reset my password", timeout=10)
    assert result.dimensions == engine.dimensions
    assert len(result.embeddings[0]) == engine.dimensions


async def test_index_round_trip_echoes_metadata_and_micro_batches(settings: Settings) -> None:
    engine = FakeEngine()
    async with running(make_worker(settings, ["index"], engine)), EmbeddingClient(settings, app_id="indexer") as client:
        inbox = await collect_results(client, settings)
        job_ids = [
            await client.submit_index([f"doc {i}", f"doc {i} part 2"], metadata={"doc_id": i})
            for i in range(10)
        ]
        results = [await asyncio.wait_for(inbox.get(), 10) for _ in job_ids]

    assert {r.job_id for r in results} == set(job_ids)
    assert all(r.status == "ok" and r.count == 2 for r in results)
    assert {r.metadata["doc_id"] for r in results} == set(range(10))
    # 10 messages arrived inside one 100ms window: far fewer than 10 model calls
    assert len(engine.calls) < 10
    assert max(r.batch_texts for r in results) > 2


async def test_invalid_message_gets_an_error_reply_not_a_retry(settings: Settings) -> None:
    async with running(make_worker(settings, ["index"])), EmbeddingClient(settings) as client:
        inbox = await collect_results(client, settings)
        connection = await aio_pika.connect(settings.rabbitmq_url)
        channel = await connection.channel()
        exchange = await channel.get_exchange(settings.mq_exchange_requests)
        await exchange.publish(
            aio_pika.Message(
                body=json.dumps({"input": []}).encode(),
                message_id="bad-1",
                reply_to=settings.index_results_queue,
            ),
            routing_key="index",
        )
        await connection.close()
        result = await asyncio.wait_for(inbox.get(), 10)

    assert result.job_id == "bad-1"
    assert result.status == "error" and result.error.code == "validation_error"


async def test_poison_job_is_dead_lettered_and_batch_mates_succeed(settings: Settings) -> None:
    async with running(make_worker(settings, ["index"])), EmbeddingClient(settings) as client:
        inbox = await collect_results(client, settings)
        good = await client.submit_index(["fine text"])
        bad = await client.submit_index(["boom"])
        result = await asyncio.wait_for(inbox.get(), 10)
        assert result.job_id == good and result.status == "ok"

        connection = await aio_pika.connect(settings.rabbitmq_url)
        channel = await connection.channel()
        dead_queue = await channel.get_queue(settings.mq_dead_letter_queue)
        dead = None
        deadline = time.monotonic() + 15
        while dead is None and time.monotonic() < deadline:
            dead = await dead_queue.get(fail=False, no_ack=True)
            if dead is None:
                await asyncio.sleep(0.2)
        await connection.close()

    assert dead is not None, "poison message never reached the dead-letter queue"
    assert dead.message_id == bad


async def test_full_query_lane_fails_fast(settings: Settings) -> None:
    settings = Settings(**{**settings.model_dump(), "query_max_length": 1})
    # No worker: the first query fills the lane, the second must be refused.
    async with EmbeddingClient(settings) as client:
        first = asyncio.create_task(client.embed_query("one", timeout=3))
        await asyncio.sleep(0.3)
        started = time.monotonic()
        with pytest.raises(EmbeddingOverloaded):
            await client.embed_query("two", timeout=3)
        assert time.monotonic() - started < 1
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)


async def test_failed_query_is_answered_with_an_error_not_retried(settings: Settings) -> None:
    async with running(make_worker(settings, ["query"])), EmbeddingClient(settings) as client:
        with pytest.raises(EmbeddingJobFailed) as caught:
            await client.embed_query("boom", timeout=5)
    assert caught.value.result.error.code == "internal_error"


async def test_graceful_shutdown_loses_nothing(settings: Settings) -> None:
    slow = FakeEngine(delay_s=0.3)
    settings = Settings(**{**settings.model_dump(), "index_batch_max_texts": 2, "index_batch_window_ms": 0})
    async with EmbeddingClient(settings) as client:
        inbox = await collect_results(client, settings)
        job_ids = {await client.submit_index([f"doc {i}"]) for i in range(8)}

        first = make_worker(settings, ["index"], slow, name="first")
        async with running(first):
            await asyncio.sleep(0.4)  # stop while batches are in flight and buffered
        async with running(make_worker(settings, ["index"], name="second")):
            seen: set[str] = set()
            while seen != job_ids:
                seen.add((await asyncio.wait_for(inbox.get(), 15)).job_id)

    assert seen == job_ids


async def test_index_backlog_does_not_delay_queries(settings: Settings) -> None:
    slow_index = FakeEngine(delay_s=0.25)
    async with (
        EmbeddingClient(settings) as client,
        running(make_worker(settings, ["index"], slow_index, name="index")),
        running(make_worker(settings, ["query"], name="query")),
    ):
        await client.declare_results_queue(settings.index_results_queue)
        for i in range(100):
            await client.submit_index([f"bulk document {i}"] * 4)

        latencies = []
        for _ in range(10):
            started = time.monotonic()
            await client.embed_query("red shoes", timeout=3)
            latencies.append(time.monotonic() - started)

    assert max(latencies) < 0.5, latencies
