# Embedding Service

[![Tests](https://github.com/Kushalbanik262/Sementic-Search-Engine/actions/workflows/tests.yml/badge.svg)](https://github.com/Kushalbanik262/Sementic-Search-Engine/actions/workflows/tests.yml)

Sentence embeddings over HTTP and over RabbitMQ, backed by `BAAI/bge-small-en-v1.5` (384 dims, 512 token limit).

## Layout

| File | Role |
| --- | --- |
| `config.py` | Settings, read from env vars / `.env` |
| `main.py` | `EmbeddingEngine` — model loading, warmup, encoding |
| `schemas.py` | Wire contracts shared by HTTP and the queue |
| `apis.py` | FastAPI app — validation, auth, concurrency, health |
| `mq/topology.py` | RabbitMQ exchanges, queues and bindings |
| `mq/batcher.py` | Micro-batcher and the per-process priority encode slot |
| `mq/handler.py` | Broker-free message handling: validate, expire, batch-encode, split |
| `mq/worker.py` | Queue worker entry point (`python -m mq.worker`) |
| `mq/client.py` | Async producer SDK (`EmbeddingClient`) |

## Run

```bash
pip install -r requirements.txt
cp .env.example .env          # optional, defaults are sensible
uvicorn apis:app --host 0.0.0.0 --port 8000
```

Interactive docs at `http://localhost:8000/docs`.

For production, run behind a process manager with more than one worker:

```bash
uvicorn apis:app --host 0.0.0.0 --port 8000 --workers 4
```

Each worker loads its own copy of the model (~130 MB), so size workers against RAM.

## API

### `POST /v1/embeddings`

```json
{
  "input": ["How do I reset my password?"],
  "input_type": "passage",
  "normalize": true
}
```

`input` also accepts a bare string. Response:

```json
{
  "model_name": "BAAI/bge-small-en-v1.5",
  "dimensions": 384,
  "input_type": "passage",
  "normalized": true,
  "count": 1,
  "truncated": [false],
  "embeddings": [[-0.024, -0.030, "..."]],
  "took_ms": 12.4
}
```

**`input_type` matters.** Use `passage` for documents you are indexing and `query`
for search terms. Queries get BGE's retrieval instruction
(`Represent this sentence for searching relevant passages: `) prepended; passages
must never have it. Mixing this up quietly degrades retrieval quality.

**Check `truncated`.** Anything past 512 tokens is cut off. A truncated document
still produces a perfectly valid-looking vector that retrieves badly, so chunk
long documents before indexing rather than ignoring the flag.

### Health

- `GET /health/live` — process is up. Never touches the model.
- `GET /health/ready` — model loaded and warm. Point your load balancer here.

### Auth

Optional. Set `API_KEY` and every `/v1/embeddings` call must send a matching
`X-API-Key` header. Unset means auth is disabled, which is the local default.
Health endpoints are always open.

## Configuration

All keys in `.env.example`. The ones that matter under load:

| Key | Default | Notes |
| --- | --- | --- |
| `MAX_CONCURRENT_ENCODES` | `1` | torch already uses every core for one encode; raising this thrashes cache. Add workers instead. |
| `TORCH_NUM_THREADS` | unset | Set it to the cores you actually gave the container, or torch will assume the whole host. |
| `ENCODE_BATCH_SIZE` | `32` | Internal batching inside one request. |
| `MAX_BATCH_ITEMS` | `128` | Rejects oversized requests with a 422. |
| `ENCODE_QUEUE_TIMEOUT_S` | `30` | Waiting requests get a 503 rather than queueing forever. |

## Worker blueprint (RabbitMQ)

The same engine also serves embeddings asynchronously. Producers never call
the worker directly: requests go in through RabbitMQ and results come back
through RabbitMQ.

Two **lanes** keep search fast while bulk indexing runs:

| Lane | Routing key | Used for | Optimised for |
| --- | --- | --- | --- |
| `query` | `query` | Embedding a search term, caller waits for the vector | Latency |
| `index` | `index` | Embedding documents before they go into the search index | Throughput |

### Topology

```
                                 ┌──────────────────────────── RabbitMQ ────────────────────────────┐
 ┌────────────────┐  rk=query    │  embedding.requests (direct)                                     │
 │ Search service │─────────────▶│   ├─ rk=query ─▶ embedding.query   classic, TTL,                 │──▶ query workers (N)
 │  (awaits reply)│◀─────────────│   │                                max-length + reject-publish   │
 └────────────────┘ direct       │   └─ rk=index ─▶ embedding.index   quorum, retry limit           │──▶ index workers (M)
                    reply-to     │                        │ after INDEX_DELIVERY_LIMIT attempts     │
 ┌────────────────┐  rk=index    │                        ▼                                         │
 │ Indexer        │─────────────▶│  embedding.dlx ─▶ embedding.dead                                 │
 │                │◀─────────────│                                                                  │
 └───────┬────────┘ indexer.     │  embedding.events (topic) ◀─ copy of rk=index requests,          │
         ▼          results      │                              later job.* lifecycle events        │
    search index                 └──────────────────────────────────────────────────────────────────┘
```

- **Query replies** use RabbitMQ Direct Reply-to (`amq.rabbitmq.reply-to`), so
  callers create no reply queues. `embedding.query` has no dead-letter
  exchange: an expired or failed query has nobody waiting for it.
- **Index results** go to the durable queue named in the request's `reply_to`
  (default `indexer.results`), so they survive restarts of the indexer.
- Declarations are idempotent and done by every worker and client. They must
  be **identical** everywhere, because RabbitMQ refuses to redeclare a queue
  with different arguments.

### Inside one worker process

```
 RabbitMQ ──deliver──▶ consumer (channel qos = prefetch)          ┐
                          │ parse + validate (schemas.py)         │
                          │  ├─ invalid  → error reply, ack       │  asyncio event loop:
                          │  └─ expired  → drop / error reply     │  heartbeats, deliveries,
                          ▼                                       │  confirms never stall
                   MicroBatcher per lane                          │
             (flush at max_texts OR window_ms,                    │
              grouped by input_type + normalize)                  │
                          ▼                                       │
             PrioritySlot — ONE encode per process                │
             (query batches jump ahead of index batches)          ┘
                          ▼
         asyncio.to_thread ─▶ EmbeddingEngine.encode (torch, TORCH_NUM_THREADS)
                          ▼
         split vectors back per job ─▶ publish reply (publisher confirm)
                          ▼
                    ack the request
```

**Delivery is at-least-once.** A request is acked only after the broker has
confirmed its reply. A crash in between means a redelivery, so consumers of
results must be idempotent (upsert by document id).

**Failures.**

| What went wrong | Query lane | Index lane |
| --- | --- | --- |
| Malformed / invalid message | Error reply, ack | Error reply, ack |
| Deadline passed | Dropped silently | `expired` error reply, ack |
| Encode raised | `internal_error` reply, ack | Requeued; dead-lettered after `INDEX_DELIVERY_LIMIT` attempts |
| Reply could not be delivered | Nack, no requeue | Requeued (results are published `mandatory`) |

If a combined batch fails, each job is retried on its own, so one poison text
cannot fail its batch-mates.

> **RabbitMQ 4.x note.** A `nack(requeue=True)` bumps `x-acquired-count` but
> not the delivery count, so the queue's `x-delivery-limit` never trips on its
> own. The worker counts attempts from those headers and dead-letters with
> `requeue=False` itself; the queue argument is only a backstop.

### Message contract (v1)

Request, published to `embedding.requests` with routing key `query` or `index`.
AMQP properties: `message_id` and `correlation_id` = `job_id`, `reply_to`,
`type=embedding.request.v1`, plus `expiration` on the query lane.

```jsonc
{
  "schema_version": 1,
  "job_id": "3f1c…",                     // generated by the client if omitted
  "input": ["text", "..."],              // same limits as POST /v1/embeddings
  "input_type": "passage",               // or "query"
  "normalize": true,
  "deadline": "2026-09-13T10:00:02Z",    // optional
  "metadata": { "doc_ids": ["a1", "a2"] } // opaque, echoed back, ≤ MQ_METADATA_MAX_BYTES
}
```

Reply (`type=embedding.result.v1`, `correlation_id=job_id`):

```jsonc
{
  "schema_version": 1, "job_id": "3f1c…", "status": "ok",   // or "error"
  "error": null,        // {"code": "validation_error|expired|internal_error|dead_lettered", "message": "…"}
  "model_name": "BAAI/bge-small-en-v1.5", "dimensions": 384,
  "input_type": "passage", "normalized": true, "count": 2,
  "truncated": [false, false], "embeddings": [[…], […]],
  "metadata": { "doc_ids": ["a1", "a2"] },
  "took_ms": 41.2, "batch_texts": 64, "worker_id": "index-7f9c"
}
```

### Running

```bash
docker run -d --name embedding-rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:4-management

python -m mq.worker --lanes query          # search queries
python -m mq.worker --lanes index          # bulk indexing
python -m mq.worker --lanes query,index    # small deployments only
```

Producers use the client:

```python
from mq.client import EmbeddingClient

async with EmbeddingClient(app_id="search-api") as client:
    result = await client.embed_query(["red running shoes"], timeout=2)

async with EmbeddingClient(app_id="indexer") as client:
    async def handle(result):            # must be idempotent
        await search_index.upsert(result.metadata["doc_ids"], result.embeddings)

    await client.consume_results(handle)   # acks only after handle() returns
    job_id = await client.submit_index(texts, metadata={"doc_ids": ids})
```

`embed_query` raises `EmbeddingOverloaded` immediately when the query lane is
full, `EmbeddingTimeout` when no reply arrives in time, and
`EmbeddingJobFailed` on an error reply.

### Scaling horizontally

Every replica is a competing consumer on its lane's queue. There is no
coordination between workers.

```
                    ┌─▶ query-worker-1 ─┐
 embedding.query ───┼─▶ query-worker-2 ─┼──▶ replies
                    └─▶ query-worker-N ─┘
                    ┌─▶ index-worker-1 ─┐
 embedding.index ───┼─▶ index-worker-2 ─┼──▶ indexer.results
                    └─▶ index-worker-M ─┘
```

- **Scale out:** start another process with the same `WORKER_LANES`. It
  receives work immediately.
- **Scale in / deploy:** send SIGTERM. The worker cancels its consumers,
  finishes the batch it is encoding and any it holds, publishes and acks, then
  disconnects within `WORKER_SHUTDOWN_GRACE_S`. Whatever is still unacked goes
  back to the queue.
- **Crash:** unacked messages return to the queue automatically.
- **Autoscaling signal:** queue depth and consumer count per lane.
- **Production default:** separate query and index replicas. `query,index` in
  one process is for small setups; there a query can still wait for the index
  batch already encoding, bounded by `INDEX_BATCH_MAX_TEXTS`.

Example for one 8-core node:

```yaml
services:
  worker-query:
    command: python -m mq.worker --lanes query
    environment: { TORCH_NUM_THREADS: "2", RABBITMQ_URL: "amqp://guest:guest@rabbitmq:5672/" }
    deploy: { replicas: 1, resources: { limits: { cpus: "2" } } }
  worker-index:
    command: python -m mq.worker --lanes index
    environment: { TORCH_NUM_THREADS: "3", RABBITMQ_URL: "amqp://guest:guest@rabbitmq:5672/" }
    deploy: { replicas: 2, resources: { limits: { cpus: "3" } } }
# 1×2 + 2×3 = 8 threads = 8 cores
```

### Why torch does not become a bottleneck

Encoding is the real compute cost, so it sets the ceiling on throughput.
The design keeps it from also becoming a *contention* or *blocking* problem:

1. **One encode per process.** torch already spreads one encode across every
   core; concurrent encodes only thrash cache.
2. **Encoding runs off the event loop.** `asyncio.to_thread` keeps AMQP
   heartbeats, deliveries and publisher confirms flowing during a long batch,
   so the broker never drops the connection mid-encode.
3. **No oversubscription.** Keep `TORCH_NUM_THREADS × replicas ≤ cores` per
   node. Without `TORCH_NUM_THREADS`, every replica assumes the whole host.
4. **Micro-batching** amortises per-call overhead across many small messages.
5. **Separate lanes**, so an index backlog never sits in front of a query.
6. **Linear scale-out** until a node's cores are used up; then add nodes, or
   scale vertically with `DEVICE=cuda`.

Sizing: `replicas ≈ peak texts/s ÷ texts/s one worker sustains × 1.3`.
Measure the per-worker number on your hardware with your real text lengths.

### Worker configuration

| Key | Default | Notes |
| --- | --- | --- |
| `RABBITMQ_URL` | `amqp://guest:guest@localhost:5672/` | |
| `WORKER_LANES` | `index` | `query`, `index` or `query,index`; `--lanes` overrides |
| `WORKER_ID` | hostname-pid | Shown in logs and on every reply |
| `WORKER_SHUTDOWN_GRACE_S` | `30` | Time to finish in-flight work on SIGTERM |
| `QUERY_PREFETCH` | `16` | Keep low so an idle replica is not starved |
| `QUERY_BATCH_MAX_TEXTS` | `32` | |
| `QUERY_BATCH_WINDOW_MS` | `5` | Latency added at most to batch queries |
| `QUERY_TTL_MS` | `5000` | Queue TTL; clients also set per-message expiration |
| `QUERY_MAX_LENGTH` | `1000` | Beyond this, publishes are refused (fail fast) |
| `INDEX_PREFETCH` | `32` | |
| `INDEX_BATCH_MAX_TEXTS` | `256` | Also bounds how long a query can wait in a mixed-lane worker |
| `INDEX_BATCH_WINDOW_MS` | `50` | |
| `INDEX_DELIVERY_LIMIT` | `5` | Total attempts before dead-lettering |
| `INDEX_RESULTS_QUEUE` | `indexer.results` | Default `reply_to` for index jobs |
| `MQ_METADATA_MAX_BYTES` | `16384` | Cap on echoed metadata |
| `TORCH_NUM_THREADS` | unset | Cores per replica |

Changing `QUERY_TTL_MS`, `QUERY_MAX_LENGTH` or `INDEX_DELIVERY_LIMIT` changes
queue arguments: delete or migrate the existing queue first, or every process
will fail to declare it.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

95 tests, ~18s. The model loads once per session and every test shares it.
The integration tests need a RabbitMQ at `RABBITMQ_URL` and skip when there
is none; CI runs one as a service container.

| File | Covers |
| --- | --- |
| `test_engine.py` | Engine in isolation: geometry, determinism, query instruction, truncation detection |
| `test_health.py` | Root, liveness, readiness, OpenAPI schema |
| `test_embeddings.py` | Happy paths and the response contract |
| `test_validation.py` | Every bad input is a 422, never a 500 — plus the inclusive boundaries |
| `test_auth.py` | API key on/off, and that health probes stay open |
| `test_retrieval.py` | Semantic quality — the vectors are actually good for search |
| `test_concurrency.py` | Load shedding, and that the encode slot is never leaked |
| `test_mq_batcher.py` | Batch size/window rules, key grouping, priority encode slot |
| `test_mq_handler.py` | Message validation, expiry, batch splitting, poison isolation, attempt counting |
| `test_mq_integration.py` | Real broker: round trips, micro-batching, dead-lettering, back-pressure, graceful shutdown, queries unaffected by an index backlog |

`test_retrieval.py` is the unusual one and the most valuable. Everything else
checks that the service returns well-shaped vectors; those tests check the
vectors are *useful*. A model swap or a dropped query instruction keeps every
other test green while quietly destroying search quality.

## CI

`.github/workflows/tests.yml` runs the suite on every push to `main`, every
pull request, and on demand via **Actions → Tests → Run workflow**. Two choices
in there are worth knowing about, because both are easy to get wrong:

**Torch is installed from the CPU index before anything else.** A plain
`pip install torch` on a Linux runner pulls ~2.5 GB of CUDA libraries that a
runner with no GPU will never load. Installing from
`https://download.pytorch.org/whl/cpu` first satisfies the `torch>=2.14`
requirement, so the later `pip install -r requirements-dev.txt` leaves it alone.
Keep that step first if you reorder anything.

**The model is downloaded in its own step, then tests run offline.** The
download is cached across runs by `actions/cache`, and the test step sets
`HF_HUB_OFFLINE=1`. This buys three things: a Hugging Face outage shows up as a
failed download rather than 59 confusing test errors, unauthenticated rate
limits stop mattering, and the suite drops from ~11s to ~5s because each model
load skips its revision check.

To force a fresh model download, bump the trailing version in the cache key
(`hf-bge-small-en-v1.5-v1` → `-v2`).

## Deployment notes

- **Install the CPU torch build** unless you have a GPU — the default wheel pulls
  ~2.5 GB of CUDA libraries you will not use:
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- **Bake the model into the image.** Otherwise every cold start downloads from
  Hugging Face, and an outage there becomes your outage. Run
  `huggingface-cli download BAAI/bge-small-en-v1.5` at build time and set
  `HF_HUB_OFFLINE=1` at runtime.
- **Never change the model without reindexing.** Vectors from two different
  models are not comparable, so a model swap invalidates every stored embedding.

## License

Copyright (C) 2026 Kushal Banik.

Released under the [GNU General Public License v3.0 or later](LICENSE).

GPL-3.0 is a copyleft license: anyone who distributes this service, or a
derivative of it, must release their source under the GPL as well. Two
consequences worth knowing before you build on this:

- **Running it as a hosted service is not "distribution."** Someone can host a
  modified version and offer it over HTTP without publishing their changes.
  [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html) is the variant that
  closes that gap, if a network service is what you are protecting.
- **It constrains who can adopt it.** Most companies will not link GPL code into
  a closed-source product, so this rules out that kind of reuse by design.

Switching later is possible while you are the sole copyright holder, but gets
much harder once other people have contributed.
"# Sementic-Search-Engine" 
