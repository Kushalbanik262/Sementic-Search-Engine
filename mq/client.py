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

"""Async producer SDK for the embedding queue.

    async with EmbeddingClient(app_id="search-api") as client:
        result = await client.embed_query(["red running shoes"], timeout=2)

    async with EmbeddingClient(app_id="indexer") as client:
        await client.consume_results(handle_result)          # upsert into the index
        job_id = await client.submit_index(texts, metadata={"doc_ids": ids})
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Awaitable, Callable, Sequence

import aio_pika
from aio_pika.abc import (
    AbstractChannel,
    AbstractIncomingMessage,
    AbstractQueue,
    AbstractRobustConnection,
)
from aio_pika.exceptions import DeliveryError, PublishError
from pydantic import ValidationError

from config import Settings, get_settings
from mq.topology import INDEX_KEY, QUERY_KEY, Topology, declare_topology
from schemas import REQUEST_TYPE, EmbeddingJob, EmbeddingResult

logger = logging.getLogger("embeddings.client")

DIRECT_REPLY_TO = "amq.rabbitmq.reply-to"
RESULTS_QUEUE_ARGUMENTS = {"x-queue-type": "quorum"}


class EmbeddingClientError(RuntimeError):
    pass


class EmbeddingOverloaded(EmbeddingClientError):
    """The broker refused the request because the lane's queue is full."""


class EmbeddingTimeout(EmbeddingClientError, TimeoutError):
    pass


class EmbeddingJobFailed(EmbeddingClientError):
    def __init__(self, result: EmbeddingResult) -> None:
        error = result.error
        super().__init__(f"job {result.job_id} failed: {error.code}: {error.message}" if error else "")
        self.result = result


@dataclass(slots=True)
class ResultsSubscription:
    queue: AbstractQueue
    consumer_tag: str
    channel: AbstractChannel

    async def cancel(self) -> None:
        await self.queue.cancel(self.consumer_tag)
        await self.channel.close()


class EmbeddingClient:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        url: str | None = None,
        app_id: str = "embedding-client",
    ) -> None:
        self._settings = settings or get_settings()
        self._url = url or self._settings.rabbitmq_url
        self.app_id = app_id
        self._connection: AbstractRobustConnection | None = None
        self._channel: AbstractChannel | None = None
        self._topology: Topology | None = None
        self._pending: dict[str, asyncio.Future[EmbeddingResult]] = {}
        self._declared_results: set[str] = set()

    async def __aenter__(self) -> EmbeddingClient:
        return await self.connect()

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def connect(self) -> EmbeddingClient:
        self._connection = await aio_pika.connect_robust(
            self._url, client_properties={"connection_name": self.app_id}
        )
        # Confirms turn "queue full" (reject-publish) and "unroutable" into
        # exceptions at publish time instead of silent loss.
        self._channel = await self._connection.channel(publisher_confirms=True, on_return_raises=True)
        self._topology = await declare_topology(self._channel, self._settings)

        # Direct reply-to: no reply queue to create or clean up. RabbitMQ
        # requires consuming it, with no_ack, on the channel we publish from.
        reply_queue = await self._channel.get_queue(DIRECT_REPLY_TO, ensure=False)
        await reply_queue.consume(self._on_reply, no_ack=True)
        return self

    async def close(self) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(EmbeddingClientError("client closed"))
        self._pending.clear()
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    # --- query lane --------------------------------------------------------

    async def embed_query(
        self,
        texts: str | Sequence[str],
        *,
        normalize: bool = True,
        timeout: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EmbeddingResult:
        """Embed search queries and wait for the vectors.

        Raises `EmbeddingOverloaded` straight away when the query lane is full,
        `EmbeddingTimeout` when no reply arrives in time, and
        `EmbeddingJobFailed` for an error reply.
        """
        timeout = timeout if timeout is not None else self._settings.query_ttl_ms / 1000
        job = EmbeddingJob(
            input=texts,
            input_type="query",
            normalize=normalize,
            deadline=datetime.now(UTC) + timedelta(seconds=timeout),
            metadata=metadata or {},
        )
        future: asyncio.Future[EmbeddingResult] = asyncio.get_running_loop().create_future()
        self._pending[job.job_id] = future
        try:
            await self._publish(job, QUERY_KEY, reply_to=DIRECT_REPLY_TO, expiration=timeout, persistent=False)
            result = await asyncio.wait_for(future, timeout)
        except TimeoutError:
            raise EmbeddingTimeout(f"no reply for query job {job.job_id} within {timeout}s") from None
        finally:
            self._pending.pop(job.job_id, None)

        if result.status == "error":
            raise EmbeddingJobFailed(result)
        return result

    async def _on_reply(self, message: AbstractIncomingMessage) -> None:
        try:
            result = EmbeddingResult.model_validate_json(message.body)
        except ValidationError:
            logger.warning("malformed reply correlation_id=%s", message.correlation_id)
            return
        future = self._pending.get(message.correlation_id or result.job_id)
        if future is not None and not future.done():
            future.set_result(result)

    # --- index lane --------------------------------------------------------

    async def submit_index(
        self,
        texts: str | Sequence[str],
        *,
        metadata: dict[str, Any] | None = None,
        job_id: str | None = None,
        normalize: bool = True,
        reply_to: str | None = None,
    ) -> str:
        """Queue passages for embedding and return the job id immediately.

        The result is delivered to `reply_to` (default `index_results_queue`);
        consume it with `consume_results`.
        """
        reply_to = reply_to or self._settings.index_results_queue
        await self.declare_results_queue(reply_to)
        fields: dict[str, Any] = {"job_id": job_id} if job_id else {}
        job = EmbeddingJob(
            input=texts, input_type="passage", normalize=normalize, metadata=metadata or {}, **fields
        )
        await self._publish(job, INDEX_KEY, reply_to=reply_to, persistent=True)
        return job.job_id

    async def declare_results_queue(self, name: str | None = None) -> None:
        """Make sure the durable results queue exists before anything is sent to it.

        Workers publish index results as mandatory, so a missing queue would
        bounce every result back into retries.
        """
        name = name or self._settings.index_results_queue
        if name in self._declared_results:
            return
        assert self._channel is not None, "call connect() first"
        await self._channel.declare_queue(name, durable=True, arguments=RESULTS_QUEUE_ARGUMENTS)
        self._declared_results.add(name)

    async def consume_results(
        self,
        handler: Callable[[EmbeddingResult], Awaitable[None]],
        *,
        queue: str | None = None,
        prefetch: int | None = None,
    ) -> ResultsSubscription:
        """Feed index results to `handler`, acking each only after it returns.

        A handler exception requeues the result, so the handler must be
        idempotent (upsert by document id), which delivery is at-least-once anyway.
        """
        assert self._connection is not None, "call connect() first"
        name = queue or self._settings.index_results_queue
        channel = await self._connection.channel()
        await channel.set_qos(prefetch_count=prefetch or self._settings.index_prefetch)
        results_queue = await channel.declare_queue(name, durable=True, arguments=RESULTS_QUEUE_ARGUMENTS)
        self._declared_results.add(name)

        async def on_message(message: AbstractIncomingMessage) -> None:
            try:
                result = EmbeddingResult.model_validate_json(message.body)
            except ValidationError:
                logger.error("dropping malformed result message_id=%s", message.message_id)
                await message.reject(requeue=False)
                return
            try:
                await handler(result)
            except Exception:
                logger.exception("result handler failed job_id=%s; requeueing", result.job_id)
                await message.nack(requeue=True)
                return
            await message.ack()

        tag = await results_queue.consume(on_message)
        return ResultsSubscription(queue=results_queue, consumer_tag=tag, channel=channel)

    # --- internals ---------------------------------------------------------

    async def _publish(
        self,
        job: EmbeddingJob,
        routing_key: str,
        *,
        reply_to: str,
        persistent: bool,
        expiration: float | None = None,
    ) -> None:
        assert self._topology is not None, "call connect() first"
        message = aio_pika.Message(
            body=job.model_dump_json().encode(),
            content_type="application/json",
            message_id=job.job_id,
            correlation_id=job.job_id,
            reply_to=reply_to,
            type=REQUEST_TYPE,
            app_id=self.app_id,
            timestamp=datetime.now(UTC),
            expiration=expiration,
            delivery_mode=(
                aio_pika.DeliveryMode.PERSISTENT if persistent else aio_pika.DeliveryMode.NOT_PERSISTENT
            ),
        )
        try:
            await self._topology.requests.publish(message, routing_key=routing_key, mandatory=True)
        except PublishError as exc:  # returned: no queue bound for this key
            raise EmbeddingClientError(f"request for lane {routing_key!r} was unroutable") from exc
        except DeliveryError as exc:  # nacked: reject-publish on a full queue
            raise EmbeddingOverloaded(f"lane {routing_key!r} is at capacity, retry shortly") from exc
