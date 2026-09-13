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

"""RabbitMQ exchanges, queues and bindings.

Every process that touches the broker calls `declare_topology`. Declarations
are idempotent, so start order does not matter. They must also be *identical*
everywhere: RabbitMQ closes the channel if a queue is redeclared with
different arguments, so change these only together with a queue migration.

    embedding.requests (direct)
      ├─ rk=query ─▶ embedding.query   classic, TTL, max-length + reject-publish
      └─ rk=index ─▶ embedding.index   quorum, delivery-limit ─▶ embedding.dlx ─▶ embedding.dead
    embedding.events (topic)  ◀─ copy of rk=index requests, plus job.* lifecycle events
"""

from __future__ import annotations

from dataclasses import dataclass

import aio_pika
from aio_pika.abc import AbstractChannel, AbstractExchange, AbstractQueue

from config import Settings

QUERY_KEY = "query"
INDEX_KEY = "index"


@dataclass(slots=True)
class Topology:
    requests: AbstractExchange
    events: AbstractExchange
    dlx: AbstractExchange
    query_queue: AbstractQueue
    index_queue: AbstractQueue
    dead_queue: AbstractQueue


def query_queue_arguments(settings: Settings) -> dict[str, object]:
    # No dead-lettering: an expired or failed query has nobody waiting for it.
    return {
        "x-message-ttl": settings.query_ttl_ms,
        "x-max-length": settings.query_max_length,
        # Refuse new publishes when full, so the producer fails fast instead
        # of the oldest queued query being silently dropped.
        "x-overflow": "reject-publish",
    }


def index_queue_arguments(settings: Settings) -> dict[str, object]:
    return {
        # Replicated and crash safe, and it counts redeliveries for us.
        "x-queue-type": "quorum",
        "x-delivery-limit": settings.index_delivery_limit,
        "x-dead-letter-exchange": settings.mq_exchange_dlx,
    }


async def declare_topology(channel: AbstractChannel, settings: Settings) -> Topology:
    dlx = await channel.declare_exchange(
        settings.mq_exchange_dlx, aio_pika.ExchangeType.DIRECT, durable=True
    )
    dead_queue = await channel.declare_queue(
        settings.mq_dead_letter_queue,
        durable=True,
        arguments={"x-queue-type": "quorum"},
    )
    # Dead-lettered messages keep their original routing key.
    await dead_queue.bind(dlx, routing_key=INDEX_KEY)

    requests = await channel.declare_exchange(
        settings.mq_exchange_requests, aio_pika.ExchangeType.DIRECT, durable=True
    )
    query_queue = await channel.declare_queue(
        settings.query_queue, durable=True, arguments=query_queue_arguments(settings)
    )
    await query_queue.bind(requests, routing_key=QUERY_KEY)

    index_queue = await channel.declare_queue(
        settings.index_queue, durable=True, arguments=index_queue_arguments(settings)
    )
    await index_queue.bind(requests, routing_key=INDEX_KEY)

    # Lifecycle events. Nothing is bound here until the status tracker runs,
    # and RabbitMQ drops unroutable messages, so nothing piles up meanwhile.
    events = await channel.declare_exchange(
        settings.mq_exchange_events, aio_pika.ExchangeType.TOPIC, durable=True
    )
    # Exchange-to-exchange binding: every index request is also copied into
    # the events exchange, which is how a job becomes QUEUED.
    await events.bind(requests, routing_key=INDEX_KEY)

    return Topology(
        requests=requests,
        events=events,
        dlx=dlx,
        query_queue=query_queue,
        index_queue=index_queue,
        dead_queue=dead_queue,
    )
