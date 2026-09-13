# CLAUDE.md

Embedding service for semantic search: `BAAI/bge-small-en-v1.5` served over
HTTP (FastAPI) and asynchronously over RabbitMQ. GPL-3.0-or-later; every
source file carries the license header.

README.md has the full user-facing docs, including the **Worker blueprint**
section. Architecture diagram artifact:
https://claude.ai/code/artifact/f1223607-f661-4d6f-9c72-ae09b6d3c7ab

## Commands (Windows dev box)

```bash
.venv/Scripts/python.exe -m pytest                       # full suite (95 tests)
HF_HUB_OFFLINE=1 .venv/Scripts/python.exe -m pytest      # skip HF revision checks
.venv/Scripts/python.exe -m pytest tests/test_mq_integration.py   # needs RabbitMQ
docker run -d --name embedding-rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:4-management
.venv/Scripts/python.exe -m mq.worker --lanes query,index
uvicorn apis:app --port 8000
```

Integration tests skip when RabbitMQ is unreachable, so a "pass" without a
broker proves nothing about the queue path. pytest runs with
`--strict-markers`; register new markers in `pytest.ini`.

## Layout

- `config.py` — pydantic-settings; every field overridable by env var. Fields
  **must be type-annotated** (un-annotated fields crash pydantic v2).
- `main.py` — `EmbeddingEngine` singleton `engine`: load, warmup, encode.
- `schemas.py` — HTTP models plus queue contracts (`EmbeddingJob`,
  `EmbeddingResult`). Shared so both transports validate identically.
- `apis.py` — FastAPI routes, auth, health.
- `mq/topology.py` — exchanges/queues; `mq/batcher.py` — `MicroBatcher`,
  `PrioritySlot`; `mq/handler.py` — broker-free logic; `mq/worker.py` —
  worker entry point; `mq/client.py` — `EmbeddingClient` producer SDK.
- `tests/mq_fakes.py` — `FakeEngine` for queue tests (text "boom" raises).

## Invariants — do not break

- **Query instruction only on queries.** `input_type="query"` prepends the BGE
  instruction; passages never get it.
- **One encode at a time per process** (`MAX_CONCURRENT_ENCODES=1` in HTTP,
  `PrioritySlot` in the worker). Scale with processes, not concurrency.
- **Encode runs off the event loop** (`to_thread`) so AMQP heartbeats and
  confirms keep flowing. Never call `engine.encode` directly in async code.
- **At-least-once:** ack a request only after its reply publish is confirmed.
- **Topology arguments must be identical in every process.** Changing
  `QUERY_TTL_MS`, `QUERY_MAX_LENGTH`, `INDEX_DELIVERY_LIMIT` means migrating
  the queue, or declarations fail with PRECONDITION_FAILED.
- `embedding.query` has **no DLX** (expired/failed queries are worthless);
  only `embedding.index` (quorum) dead-letters to `embedding.dlx` → `embedding.dead`.
- Query replies: Direct Reply-to, consumed `no_ack` on the same channel that
  publishes. Index replies: durable `indexer.results`, published `mandatory`.
- Each worker lane has its own publish channel, so a vanished query caller
  cannot disturb index replies.

## Gotchas learned

- **RabbitMQ 4.x does not count `nack(requeue=True)` toward
  `x-delivery-limit`**; it only bumps `x-acquired-count`. The worker counts
  attempts itself (`mq.worker.delivery_attempt`) and dead-letters with
  `requeue=False`. Verified on RabbitMQ 4.3.5.
- Windows asyncio has no `loop.add_signal_handler`; the worker falls back to
  `signal.signal`.
- Validation in `EmbeddingRequest` calls `get_settings()` directly, so
  `dependency_overrides` do not change validation limits in tests.

## Roadmap (agreed plan, status as of 2026-09-13)

- [x] Phase 0 — config fix, schemas extracted to `schemas.py`.
- [x] Phase 1 — topology, batcher, handler, worker, client, unit + integration
      tests, RabbitMQ service in CI. Uncommitted at time of writing.
- [ ] Phase 2 — job status: workers emit `job.started/completed/failed` to
      `embedding.events`; `status/tracker.py` (only Redis writer, forward-only
      state updates via Lua) consumes `embedding.status` + `embedding.dead`;
      `GET /v1/jobs/{job_id}` (read-only, no embeddings stored), `GET /health/queues`.
      States: QUEUED → PROCESSING → COMPLETED | FAILED | DEAD_LETTERED | EXPIRED.
      Query jobs untracked by default (`TRACK_QUERY_JOBS=false`).
      **Pending user confirmation:** Redis as the store (24h TTL), tracker as
      its own process (vs `STATUS_TRACKER_IN_API`).
- [ ] Phase 3 — worker `/health/live|ready` port, structured logs, reconnect hardening.
- [ ] Phase 4 — Dockerfile, docker-compose (api, worker-query, worker-index,
      tracker, rabbitmq, redis), `scripts/benchmark_encode.py`,
      `scripts/loadtest.py`, publish numbers.
- [ ] Phase 5 (optional) — Prometheus metrics, OpenTelemetry via AMQP headers,
      KEDA autoscaling, binary float32 payloads.

Hard requirement from the user: job **submission and result delivery happen
only via the message queue**; HTTP job endpoints are read-only status. The
sync `POST /v1/embeddings` stays as is.
