# Embedding Service

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
