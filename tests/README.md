# Tests

Regression tests for the `pedro ask` and `pedro research` CLI pipelines.

## Stack

- **pytest** — test runner
- **responses** — mock for `requests.post` (Ollama embed endpoint)
- `unittest.mock.patch` — mock for `ollama.Client.chat`
- `chromadb.EphemeralClient` — in-memory vector DB, pre-seeded with fake chunks

No real Ollama instance or PDF files are needed.

## Setup

```bash
uv pip install -e ".[dev]" --system-certs
```

## Linting

```bash
uv run ruff check pdf_rag/ tests/ eval/
```

## Run

```bash
.venv/bin/python -m pytest tests/ -v
```

## Watch mode

Install `pytest-watch`:

```bash
uv pip install pytest-watch --system-certs
```

Then run:

```bash
.venv/bin/ptw tests/ -- -v
```

Reruns tests automatically on any file change.
