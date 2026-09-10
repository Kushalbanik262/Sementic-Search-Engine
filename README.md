# Embedding Service

[![Tests](https://github.com/Kushalbanik262/Sementic-Search-Engine/actions/workflows/tests.yml/badge.svg)](https://github.com/Kushalbanik262/Sementic-Search-Engine/actions/workflows/tests.yml)

Sentence embeddings over HTTP, backed by `BAAI/bge-small-en-v1.5` (384 dims, 512 token limit).

## Layout

| File | Role |
| --- | --- |
| `config.py` | Settings, read from env vars / `.env` |
| `main.py` | `EmbeddingEngine` — model loading, warmup, encoding |
| `apis.py` | FastAPI app — validation, auth, concurrency, health |

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

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

59 tests, ~11s. The model loads once per session (~6s of that) and every test
shares it.

| File | Covers |
| --- | --- |
| `test_engine.py` | Engine in isolation: geometry, determinism, query instruction, truncation detection |
| `test_health.py` | Root, liveness, readiness, OpenAPI schema |
| `test_embeddings.py` | Happy paths and the response contract |
| `test_validation.py` | Every bad input is a 422, never a 500 — plus the inclusive boundaries |
| `test_auth.py` | API key on/off, and that health probes stay open |
| `test_retrieval.py` | Semantic quality — the vectors are actually good for search |
| `test_concurrency.py` | Load shedding, and that the encode slot is never leaked |

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
"# Sementic-Search-Engine" 
