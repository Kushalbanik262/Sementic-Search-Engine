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

"""Application configuration.

Every value can be overridden with an environment variable of the same name
(case-insensitive) or a line in a local `.env` file. See `.env.example`.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # our own fields start with "model_", which pydantic reserves by default
        protected_namespaces=(),
    )

    # --- model -----------------------------------------------------------
    model_name: str = "BAAI/bge-small-en-v1.5"
    # None -> pick cuda when available, otherwise cpu
    device: str | None = None
    max_seq_length: int = 512
    encode_batch_size: int = 32

    # BGE v1.5 retrieval instruction. Prepended to QUERIES ONLY -- passages
    # must never carry it, or query/passage vectors stop being comparable.
    query_instruction: str = "Represent this sentence for searching relevant passages: "

    # --- request limits --------------------------------------------------
    max_batch_items: int = 128
    max_chars_per_text: int = 8_000

    # --- concurrency -----------------------------------------------------
    # torch already spreads a single encode across cores, so running several
    # encodes at once on CPU mostly thrashes cache. Scale out with uvicorn
    # workers rather than raising this.
    max_concurrent_encodes: int = 1
    encode_queue_timeout_s: float = 30.0
    torch_num_threads: int | None = None  # None -> leave torch's default

    # --- service ---------------------------------------------------------
    api_key: str | None = None  # unset -> auth disabled
    cors_allow_origins: list[str] = Field(default_factory=list)
    log_level: str = "INFO"

    # --- message queue ---------------------------------------------------
    # Requests arrive on `mq_exchange_requests` with routing key "query" or
    # "index"; each key has its own queue and its own pool of workers, so a
    # bulk indexing backlog can never delay a search query.
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/"
    mq_exchange_requests: str = "embedding.requests"
    mq_exchange_events: str = "embedding.events"
    mq_exchange_dlx: str = "embedding.dlx"
    mq_dead_letter_queue: str = "embedding.dead"
    # opaque producer metadata is echoed back on the reply, so cap its size
    mq_metadata_max_bytes: int = 16_384

    # query lane: latency first. Low prefetch so a busy worker does not sit
    # on queries another replica could serve right now.
    query_queue: str = "embedding.query"
    query_prefetch: int = 16
    query_batch_max_texts: int = 32
    query_batch_window_ms: float = 5.0
    query_ttl_ms: int = 5_000
    query_max_length: int = 1_000

    # index lane: throughput first. Micro-batches across messages.
    index_queue: str = "embedding.index"
    index_results_queue: str = "indexer.results"
    index_prefetch: int = 32
    index_batch_max_texts: int = 256
    index_batch_window_ms: float = 50.0
    index_delivery_limit: int = 5

    # --- worker ----------------------------------------------------------
    # Comma separated: "query", "index" or "query,index".
    worker_lanes: str = "index"
    worker_id: str | None = None  # None -> hostname-pid
    worker_shutdown_grace_s: float = 30.0


@lru_cache
def get_settings() -> Settings:
    """Cached so the whole process shares one Settings instance."""
    return Settings()
