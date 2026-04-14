# pedro server

HTTP server that exposes the `ask` and `research` pipelines as streaming endpoints.

**Architecture decision:** [ADR-0005 — Client-server architecture](../../adr/0005-client-server-architecture.md)

## Security Notice

**⚠️ Default configuration is for local development only.**

The server enables CORS with `allow_origins=["*"]` and `allow_credentials=True` to simplify local testing. This combination is **insecure for production** and will be rejected by browsers.

**For production deployment:**
- Set explicit `allow_origins` to your trusted domains
- Remove or restrict `allow_credentials=True` unless explicitly needed
- Add authentication/authorization middleware
- Filter sensitive headers from request logging (see `app.py` logging middleware)

See `pdf_rag/server/app.py` CORS middleware configuration.

## Starting the server

```bash
pedro serve                         # 127.0.0.1:8000
pedro serve --host 0.0.0.0 --port 9000
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/v1/models` | List available models (OpenAI-compatible) |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat endpoint (streaming/non-streaming) |
| `POST` | `/v1/ask` | Pedro-native ask, streamed as SSE |
| `POST` | `/v1/research` | Pedro-native research, streamed as SSE |

**Interactive docs (Swagger UI):** http://localhost:8000/docs  
**OpenAPI schema:** http://localhost:8000/openapi.json

## Pedro-native SSE format

`/v1/ask` and `/v1/research` stream Server-Sent Events:

```
event: log
data: {"text": "🪅 Planning sub-questions..."}

event: token
data: {"text": "The answer is"}

event: done
data: {}
```

- `log` — pipeline step messages
- `token` — individual answer tokens
- `done` — end of stream

```bash
curl -N -X POST http://localhost:8000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is entropy?"}'
```

All request fields are optional — the server falls back to its config defaults.

## OpenAI-compatible usage

`/v1/chat/completions` works with any OpenAI-compatible client. Model names: `pedro-ask`, `pedro-research`.

```bash
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "pedro-ask", "messages": [{"role": "user", "content": "What is entropy?"}], "stream": true}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="pedro")

# Streaming ask
stream = client.chat.completions.create(
    model="pedro-ask",
    messages=[{"role": "user", "content": "What is backpropagation?"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)

# Research with extra params
response = client.chat.completions.create(
    model="pedro-research",
    messages=[{"role": "user", "content": "Compare LSTM and Transformer"}],
    stream=False,
    extra_body={"depth": 2, "n_subquestions": 3},
)
print(response.choices[0].message.content)
```

## Connecting the TUI to the server

```bash
# Terminal 1 — server
pedro serve

# Terminal 2 — TUI as a client
PEDRO_SERVER_URL=http://localhost:8000 pedro
```

Without `PEDRO_SERVER_URL` the TUI works standalone (no server needed).

## Implementation

`make_app(db_path, base_url, ...)` factory creates the FastAPI app with injected config — used by `pedro serve` and by tests (which inject an in-memory ChromaDB collection without touching the filesystem).

Sync pipeline functions (`run_ask`, `research`) run in a `ThreadPoolExecutor`. Callbacks (`on_token`, `log_fn`) push events into an `asyncio.Queue` that the SSE generator drains.

## Server Configuration

The server uses the following environment variables for model configuration:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `mistral:7b` | Quality model for research synthesis |
| `ASK_LLM_MODEL` | `LLM_MODEL` | Model for `/v1/ask` endpoint (falls back to `LLM_MODEL`) |
| `FAST_MODEL` | `LLM_MODEL` | Model for sub-questions, planning, reflection |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model for retrieval |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |

Set these before starting the server:

```bash
LLM_MODEL=command-r:35b ASK_LLM_MODEL=command-r:35b FAST_MODEL=mistral:7b pedro serve
```
