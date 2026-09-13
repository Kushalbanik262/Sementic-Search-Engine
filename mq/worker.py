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

"""Queue-driven embedding worker.

Run one process per replica:
    python -m mq.worker --lanes query          # search queries, latency first
    python -m mq.worker --lanes index          # bulk indexing, throughput first
    python -m mq.worker --lanes query,index    # small deployments

Scale by starting more processes: every replica is a competing consumer on
the same queue. Within a process there is exactly one encode at a time, run
off the event loop, so heartbeats, deliveries and publisher confirms keep
flowing while torch works.

Delivery is at-least-once: a request is acked only after its reply has been
confirmed by the broker.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import socket
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable, Sequence

import aio_pika
from aio_pika.abc import (
    AbstractChannel,
    AbstractIncomingMessage,
    AbstractQueue,
    AbstractRobustConnection,
)

from config import Settings, get_settings
from mq.batcher import Batch, MicroBatcher, PrioritySlot
from mq.handler import (
    Engine,
    encode_jobs_isolating_failures,
    error_result,
    is_expired,
    parse_job,
)
from mq.topology import declare_topology
from schemas import RESULT_TYPE, EmbeddingJob, EmbeddingResult, Lane

logger = logging.getLogger("embeddings.worker")

LANES: tuple[Lane, ...] = ("query", "index")


@dataclass(frozen=True, slots=True)
class LaneSpec:
    name: Lane
    queue: str
    prefetch: int
    max_texts: int
    window_s: float
    # Lower runs first when both lanes share a process.
    priority: int
    # Index replies must survive a broker restart and must reach a real queue;
    # a query reply is worthless once its caller has gone.
    persistent_reply: bool
    mandatory_reply: bool
    # Index requests are retried through the quorum queue's delivery limit;
    # a query that failed once is answered with an error instead.
    retry_on_failure: bool


def lane_specs(settings: Settings) -> dict[Lane, LaneSpec]:
    return {
        "query": LaneSpec(
            name="query",
            queue=settings.query_queue,
            prefetch=settings.query_prefetch,
            max_texts=settings.query_batch_max_texts,
            window_s=settings.query_batch_window_ms / 1000,
            priority=0,
            persistent_reply=False,
            mandatory_reply=False,
            retry_on_failure=False,
        ),
        "index": LaneSpec(
            name="index",
            queue=settings.index_queue,
            prefetch=settings.index_prefetch,
            max_texts=settings.index_batch_max_texts,
            window_s=settings.index_batch_window_ms / 1000,
            priority=1,
            persistent_reply=True,
            mandatory_reply=True,
            retry_on_failure=True,
        ),
    }


def parse_lanes(value: str) -> list[Lane]:
    lanes: list[Lane] = []
    for part in value.split(","):
        name = part.strip().lower()
        if not name:
            continue
        if name not in LANES:
            raise ValueError(f"unknown lane {name!r}; expected one of {', '.join(LANES)}")
        if name not in lanes:
            lanes.append(name)  # type: ignore[arg-type]
    if not lanes:
        raise ValueError("at least one lane is required")
    return lanes


def delivery_attempt(message: AbstractIncomingMessage) -> int:
    """1-based attempt number of this delivery.

    Counted here rather than left to `x-delivery-limit`: since RabbitMQ 4.0 a
    `nack(requeue=True)` over AMQP 0.9.1 bumps `x-acquired-count` but not the
    delivery count, so the broker's limit would never trip. Both headers are
    read so this also works on 3.13. A requeue caused by a worker shutting
    down counts too, so the limit is slightly conservative.
    """
    headers = message.headers or {}
    previous = max(
        int(headers.get("x-acquired-count") or 0),
        int(headers.get("x-delivery-count") or 0),
    )
    return previous + 1


def default_worker_id() -> str:
    return f"{socket.gethostname()}-{os.getpid()}"


@dataclass(slots=True)
class Delivery:
    message: AbstractIncomingMessage
    job: EmbeddingJob


@dataclass(slots=True)
class _Lane:
    spec: LaneSpec
    batcher: MicroBatcher[Delivery]
    consume_channel: AbstractChannel | None = None
    publish_channel: AbstractChannel | None = None
    queue: AbstractQueue | None = None
    consumer_tag: str | None = None


class EmbeddingWorker:
    def __init__(
        self,
        engine: Engine,
        lanes: Sequence[Lane],
        *,
        settings: Settings | None = None,
        worker_id: str | None = None,
        connect: Callable[..., object] = aio_pika.connect_robust,
        load_model: bool = True,
    ) -> None:
        self._settings = settings or get_settings()
        self._engine = engine
        self._connect = connect
        self._load_model = load_model
        self.worker_id = worker_id or self._settings.worker_id or default_worker_id()

        specs = lane_specs(self._settings)
        self._lanes: dict[Lane, _Lane] = {
            name: _Lane(
                spec=specs[name],
                batcher=MicroBatcher(
                    max_texts=specs[name].max_texts,
                    window_s=specs[name].window_s,
                    size_of=lambda d: len(d.job.input),
                    key_of=lambda d: (d.job.input_type, d.job.normalize),
                ),
            )
            for name in parse_lanes(",".join(lanes))
        }

        self._slot = PrioritySlot()
        self._stop = asyncio.Event()
        self._draining = False
        self._ready = asyncio.Event()
        self._connection: AbstractRobustConnection | None = None
        self._loops: list[asyncio.Task[None]] = []
        self._inflight: set[asyncio.Task[None]] = set()
        self.stats: Counter[str] = Counter()

    # --- lifecycle -------------------------------------------------------

    @property
    def lanes(self) -> list[Lane]:
        return list(self._lanes)

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set() and not self._draining

    async def wait_ready(self) -> None:
        await self._ready.wait()

    def stop(self) -> None:
        """Ask `run()` to shut down gracefully. Safe to call more than once."""
        self._stop.set()

    async def run(self) -> None:
        """Consume until `stop()` is called or the task is cancelled."""
        s = self._settings
        if self._load_model:
            load = getattr(self._engine, "load", None)
            warmup = getattr(self._engine, "warmup", None)
            if load:
                await asyncio.to_thread(load)
            if warmup:
                await asyncio.to_thread(warmup)

        self._connection = await self._connect(
            s.rabbitmq_url, client_properties={"connection_name": self.worker_id}
        )
        try:
            setup_channel = await self._connection.channel()
            await declare_topology(setup_channel, s)
            await setup_channel.close()

            for lane in self._lanes.values():
                await self._start_lane(lane)

            self._ready.set()
            logger.info("worker ready id=%s lanes=%s", self.worker_id, ",".join(self._lanes))
            await self._stop.wait()
        finally:
            await self._shutdown()

    async def _start_lane(self, lane: _Lane) -> None:
        assert self._connection is not None
        spec = lane.spec
        # Each lane publishes on its own confirm channel, so a failure answering
        # a vanished query caller can never disturb index replies.
        lane.publish_channel = await self._connection.channel(
            publisher_confirms=True, on_return_raises=True
        )
        lane.consume_channel = await self._connection.channel()
        await lane.consume_channel.set_qos(prefetch_count=spec.prefetch)
        lane.queue = await lane.consume_channel.get_queue(spec.queue)
        lane.consumer_tag = await lane.queue.consume(
            lambda message, lane=lane: self._on_message(lane, message)
        )
        self._loops.append(asyncio.create_task(self._lane_loop(lane), name=f"lane-{spec.name}"))

    async def _shutdown(self) -> None:
        """Stop taking work, finish what we hold, then disconnect.

        Anything still unacked when the grace period ends is returned to the
        queue by the broker when the connection closes, so nothing is lost.
        """
        self._draining = True
        grace = self._settings.worker_shutdown_grace_s
        logger.info("worker draining id=%s grace=%.0fs", self.worker_id, grace)

        for lane in self._lanes.values():
            if lane.queue is not None and lane.consumer_tag is not None:
                try:
                    await lane.queue.cancel(lane.consumer_tag)
                except Exception:
                    logger.warning("could not cancel consumer lane=%s", lane.spec.name, exc_info=True)

        for task in self._loops:
            task.cancel()
        await asyncio.gather(*self._loops, return_exceptions=True)

        try:
            async with asyncio.timeout(grace):
                if self._inflight:
                    await asyncio.gather(*self._inflight, return_exceptions=True)
                for lane in self._lanes.values():
                    for batch in lane.batcher.drain():
                        await self._process(lane, batch)
        except TimeoutError:
            logger.warning("grace period over; unacked messages return to the queue")

        if self._connection is not None:
            await self._connection.close()
        logger.info("worker stopped id=%s stats=%s", self.worker_id, dict(self.stats))

    # --- consuming -------------------------------------------------------

    async def _on_message(self, lane: _Lane, message: AbstractIncomingMessage) -> None:
        spec = lane.spec
        self.stats[f"{spec.name}.received"] += 1

        if self._draining:
            await message.nack(requeue=True)
            return

        job, rejection = parse_job(
            message.body,
            fallback_id=message.message_id or message.correlation_id,
            worker_id=self.worker_id,
        )
        if rejection is not None:
            self.stats[f"{spec.name}.rejected"] += 1
            logger.warning(
                "rejected lane=%s job_id=%s reason=%s",
                spec.name, rejection.job_id, rejection.error.message if rejection.error else "",
            )
            # Retrying a malformed message cannot help: answer and drop it.
            await self._settle(lane, message, rejection, retry=False)
            return

        assert job is not None
        if is_expired(job):
            await self._expire(lane, Delivery(message, job))
            return

        lane.batcher.add(Delivery(message, job))

    async def _lane_loop(self, lane: _Lane) -> None:
        while True:
            batch = await lane.batcher.next_batch()
            task = asyncio.create_task(self._process(lane, batch))
            self._inflight.add(task)
            task.add_done_callback(self._inflight.discard)
            # Shielded: cancelling the loop on shutdown must not abandon a
            # batch mid-encode; `_shutdown` waits for it instead.
            await asyncio.shield(task)

    async def _process(self, lane: _Lane, batch: Batch[Delivery]) -> None:
        spec = lane.spec
        live: list[Delivery] = []
        for delivery in batch.items:
            # It may have expired while buffered behind other batches.
            if is_expired(delivery.job):
                await self._expire(lane, delivery)
            else:
                live.append(delivery)
        if not live:
            return

        jobs = [delivery.job for delivery in live]
        async with self._slot.acquire(spec.priority):
            outcomes = await asyncio.to_thread(
                encode_jobs_isolating_failures, self._engine, jobs, worker_id=self.worker_id
            )
        logger.debug(
            "batch lane=%s jobs=%d texts=%d", spec.name, len(jobs), sum(len(j.input) for j in jobs)
        )

        await asyncio.gather(
            *(self._finish(lane, delivery, outcome) for delivery, outcome in zip(live, outcomes))
        )

    async def _finish(
        self, lane: _Lane, delivery: Delivery, outcome: EmbeddingResult | Exception
    ) -> None:
        spec = lane.spec
        if isinstance(outcome, Exception):
            self.stats[f"{spec.name}.failed"] += 1
            logger.error(
                "encode failed lane=%s job_id=%s error=%r", spec.name, delivery.job.job_id, outcome
            )
            if spec.retry_on_failure:
                attempt = delivery_attempt(delivery.message)
                if attempt >= self._settings.index_delivery_limit:
                    # requeue=False on a queue with a DLX dead-letters it.
                    self.stats[f"{spec.name}.dead_lettered"] += 1
                    logger.error(
                        "giving up lane=%s job_id=%s attempts=%d; dead-lettering",
                        spec.name, delivery.job.job_id, attempt,
                    )
                    await delivery.message.nack(requeue=False)
                else:
                    await delivery.message.nack(requeue=True)
                return
            outcome = error_result(
                delivery.job.job_id, "internal_error", "embedding failed",
                metadata=delivery.job.metadata, worker_id=self.worker_id,
            )
        else:
            self.stats[f"{spec.name}.completed"] += 1
        await self._settle(lane, delivery.message, outcome, retry=spec.retry_on_failure)

    async def _expire(self, lane: _Lane, delivery: Delivery) -> None:
        spec = lane.spec
        self.stats[f"{spec.name}.expired"] += 1
        logger.info("expired lane=%s job_id=%s", spec.name, delivery.job.job_id)
        if spec.name == "query":
            # Its caller has already given up; a reply would go nowhere.
            await delivery.message.ack()
            return
        result = error_result(
            delivery.job.job_id, "expired", "deadline passed before the job was processed",
            metadata=delivery.job.metadata, worker_id=self.worker_id,
        )
        await self._settle(lane, delivery.message, result, retry=False)

    # --- replying --------------------------------------------------------

    async def _settle(
        self,
        lane: _Lane,
        message: AbstractIncomingMessage,
        result: EmbeddingResult,
        *,
        retry: bool,
    ) -> None:
        """Publish the reply, wait for the broker's confirm, then ack."""
        try:
            await self._reply(lane, message, result)
        except Exception:
            self.stats[f"{lane.spec.name}.reply_failed"] += 1
            logger.exception(
                "reply failed lane=%s job_id=%s reply_to=%s",
                lane.spec.name, result.job_id, message.reply_to,
            )
            await message.nack(requeue=retry)
            return
        await message.ack()

    async def _reply(
        self, lane: _Lane, message: AbstractIncomingMessage, result: EmbeddingResult
    ) -> None:
        if not message.reply_to:
            logger.warning("no reply_to lane=%s job_id=%s; result dropped", lane.spec.name, result.job_id)
            return
        assert lane.publish_channel is not None
        spec = lane.spec
        await lane.publish_channel.default_exchange.publish(
            aio_pika.Message(
                body=result.model_dump_json().encode(),
                content_type="application/json",
                correlation_id=message.correlation_id or result.job_id,
                message_id=result.job_id,
                type=RESULT_TYPE,
                app_id=self.worker_id,
                timestamp=datetime.now(UTC),
                delivery_mode=(
                    aio_pika.DeliveryMode.PERSISTENT
                    if spec.persistent_reply
                    else aio_pika.DeliveryMode.NOT_PERSISTENT
                ),
            ),
            routing_key=message.reply_to,
            mandatory=spec.mandatory_reply,
        )


# --- entry point -------------------------------------------------------------


async def _serve(worker: EmbeddingWorker) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, worker.stop)
        except (NotImplementedError, RuntimeError):
            # Windows' event loop has no add_signal_handler.
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(worker.stop))
    await worker.run()


def main(argv: Sequence[str] | None = None) -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Embedding worker (RabbitMQ consumer)")
    parser.add_argument("--lanes", default=settings.worker_lanes, help="query, index or query,index")
    parser.add_argument("--worker-id", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    )
    from main import engine

    worker = EmbeddingWorker(engine, parse_lanes(args.lanes), settings=settings, worker_id=args.worker_id)
    asyncio.run(_serve(worker))


if __name__ == "__main__":
    main()
