# ADR-0005: Client-server architecture

## Context

`researcher.py` owns all pipeline logic cleanly after ADR-0001's callback refactor.
The TUI is currently the only consumer of that logic. The goal is to expose the same
pipelines over HTTP so the TUI becomes one of several possible clients — a web
interface or a messenger bot can be added later without touching core logic.

## Decision

Add an HTTP server (`pedro serve`) that wraps `run_ask()` and `research()` and
streams results as SSE. The TUI becomes a thin client that connects to the server.

## Architecture overview

```
┌─────────────────────────────────────────────────────────┐
│                     pedro server                        │
│                                                         │
│  POST /v1/ask       →  run_ask()   → SSE stream         │
│  POST /v1/research  →  research()  → SSE stream         │
│                           │                             │
│                     researcher.py                       │
│                     ChromaDB + Ollama                   │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP + SSE
         ┌─────────────────┼─────────────────┐
         │                 │                 │
      TUI client     web client        bot client
   (Textual app)    (future)           (future)
```

## Server

**Framework: FastAPI + uvicorn**

FastAPI was chosen because:
- `StreamingResponse` with `text/event-stream` handles SSE natively
- `run_in_executor` bridges the sync Ollama SDK into the async request context
- Pydantic request models give free validation and documentation
- `TestClient` from Starlette allows synchronous integration tests without spinning up a real process

**Endpoints**

```
POST /v1/ask       — runs run_ask(), streams SSE
POST /v1/research  — runs research(), streams SSE
```

**Request bodies**

```json
// POST /v1/ask
{
  "question": "What is entropy?",
  "llm_model": "mistral:7b",
  "embed_model": "nomic-embed-text",
  "top_k": 5,
  "show_sources": true
}

// POST /v1/research
{
  "question": "What is entropy?",
  "llm_model": "mistral:7b",
  "fast_model": "mistral:7b",
  "tiny_model": "mistral:7b",
  "embed_model": "nomic-embed-text",
  "depth": 2,
  "n_subquestions": 3,
  "top_k": 5,
  "languages": [],
  "translate_model": "mistral:7b"
}
```

All fields are optional and fall back to server-side config defaults.

**SSE event format**

```
event: log
data: {"text": "🪅 Planning sub-questions..."}

event: token
data: {"text": "The answer is"}

event: done
data: {}
```

Two types: `log` (pipeline step messages) and `token` (answer tokens streamed one by
one). This is intentionally not the OpenAI `chat.completion.chunk` format — log events
have no clean mapping there. An `/v1/chat/completions` shim can be layered on later for
Open-WebUI / LiteLLM compatibility.

**Sync → async bridge**

`run_ask()` and `research()` are synchronous. The server runs them in a thread pool and
forwards callbacks into an `asyncio.Queue` that the SSE generator drains:

```
thread:  on_token("The") → q.put(("token", "The"))
async:   q.get() → yield SSE chunk
```

**`make_app()` factory**

The FastAPI app is created via `make_app(db_path, base_url, ...)` so tests can inject
config directly. `pedro serve` calls `make_app(**config_defaults)`.

## TUI client

**Dual-mode**: if `PEDRO_SERVER_URL` env var is set, TUI connects to the server;
otherwise it falls back to direct `run_ask()` / `research()` calls.

A new helper `pdf_rag/tui/stream_client.py` runs in worker threads (sync):
- Opens a streaming `requests.post()` call (no new dep — `requests` is already required)
- Parses SSE lines, dispatches `token` events to `on_token` and `log` events to `log_fn`
- Calls `check()` between lines to support Escape cancellation

## New `pedro serve` command

```
pedro serve [--host 127.0.0.1] [--port 8000]
```

## Files

| File | Change |
|------|--------|
| `pdf_rag/server/__init__.py` | new (empty) |
| `pdf_rag/server/app.py` | new — FastAPI app, `make_app()`, SSE streaming |
| `pdf_rag/tui/stream_client.py` | new — SSE client, `stream_ask()`, `stream_research()` |
| `pdf_rag/cli.py` | add `serve` command |
| `pdf_rag/config.py` | add `SERVER_URL = os.getenv("PEDRO_SERVER_URL", "")` |
| `pdf_rag/tui/app.py` | dual-mode dispatch in `_do_ask` / `_do_research` |
| `pyproject.toml` | add `fastapi>=0.110`, `uvicorn[standard]>=0.29` |
| `tests/test_server.py` | new — SSE stream tests via `TestClient` |

## Testing

`make_app()` factory allows injecting `db_path` and other config so tests never touch
the filesystem or real Ollama. `starlette.testclient.TestClient` drives the server
synchronously — no real port needed.

```python
# tests/test_server.py

def test_ask_streams_tokens(seeded_collection, mock_embed, mock_ollama_chat):
    app = make_app(db_path="unused", base_url=OLLAMA_URL, ...)
    with patch("pdf_rag.researcher._open_collection", return_value=seeded_collection):
        with TestClient(app) as client:
            with client.stream("POST", "/v1/ask",
                               json={"question": "What is entropy?"}) as r:
                events = parse_sse(r)
    assert any(e["kind"] == "token" for e in events)
    assert events[-1]["kind"] == "done"

def test_ask_streams_log_events(seeded_collection, mock_embed, mock_ollama_chat):
    ...
    assert any(e["kind"] == "log" for e in events)

def test_research_streams(seeded_collection, mock_embed, mock_ollama_chat):
    ...
    assert any("Planning" in e["text"] for e in events if e["kind"] == "log")
```

Helper `parse_sse(response)` reads lines from the streaming response and returns
`[{"kind": "token"|"log"|"done", "text": "..."}]`.

All 17 existing tests must continue to pass unchanged.

## Verification

```bash
# all tests including new server tests
.venv/bin/python -m pytest tests/ -v

# server in one terminal
pedro serve --port 8000

# TUI as a client in another
PEDRO_SERVER_URL=http://localhost:8000 pedro

# raw curl smoke test
curl -N -X POST http://localhost:8000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is entropy?"}'
```
