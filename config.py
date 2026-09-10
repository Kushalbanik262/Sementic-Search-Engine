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


@lru_cache
def get_settings() -> Settings:
    """Cached so the whole process shares one Settings instance."""
    return Settings()
