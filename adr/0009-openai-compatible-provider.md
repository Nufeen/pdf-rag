# ADR-0009: OpenAI-compatible provider support

## Context

The project is tightly coupled to Ollama for both LLM inference and embeddings. All chat calls use `ollama.Client` and all embedding calls hit `/api/embed` directly via `requests`. There is no abstraction layer, making it impossible to swap providers.

Users may want to run inference against OpenAI, local OpenAI-compatible servers (vLLM, LM Studio, Ollama's own `/v1` path), or other hosted endpoints without changing code.

## Decision

Introduce a `PROVIDER_TYPE` environment variable (`"ollama"` | `"openai"`, default `"ollama"`) and a `pdf_rag/provider.py` module that presents a unified `make_client()` / `embed()` interface. All existing call sites are updated to use this interface; Ollama remains the default and existing behaviour is unchanged when `PROVIDER_TYPE` is not set.

No new package dependencies — the OpenAI provider is implemented with `requests` using the standard OpenAI HTTP API (SSE streaming, `/chat/completions`, `/embeddings`).

## New environment variables

| Variable | Default | Purpose |
|---|---|---|
| `PROVIDER_TYPE` | `ollama` | Select provider: `ollama` or `openai` |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint |
| `OPENAI_API_KEY` | _(empty)_ | API key sent as `Bearer` token |

## Provider module (`pdf_rag/provider.py`)

### `make_client(base_url) → client`

Returns an object with a `.chat(model, messages, stream)` method whose return value matches Ollama's dict shape (`{"message": {"content": ...}}`), so all existing call sites in `llm.py`, `researcher.py`, and `context_manager.py` need only a one-line swap.

- **Ollama branch** — thin wrapper around `ollama.Client`
- **OpenAI branch** — `_OpenAIClient` posts to `{OPENAI_BASE_URL}/chat/completions` via `requests`; streaming parses SSE lines and yields `{"message": {"content": token}}` dicts; non-streaming returns a single dict

### `embed(texts, model, base_url, batch_size) → list[list[float]]`

- **Ollama branch** — existing `POST {base_url}/api/embed` logic (moved from `indexer.py`)
- **OpenAI branch** — `POST {OPENAI_BASE_URL}/embeddings` with `Authorization: Bearer {OPENAI_API_KEY}`; response sorted by `index`

When provider is `openai`, both functions ignore the passed-in `base_url` and use `OPENAI_BASE_URL` / `OPENAI_API_KEY` from config.

## Affected files

| File | Change |
|---|---|
| `pdf_rag/config.py` | Add `PROVIDER_TYPE`, `OPENAI_BASE_URL`, `OPENAI_API_KEY` |
| `pdf_rag/provider.py` | New file — `make_client()` and `embed()` |
| `pdf_rag/llm.py` | `Client(host=…)` → `make_client(…)`; provider-agnostic error message |
| `pdf_rag/researcher.py` | Same swap; remove `from ollama import Client`; drop `Client` type hints |
| `pdf_rag/context_manager.py` | Same swap |
| `pdf_rag/indexer.py` | `batch_embed()` delegates to `provider.embed()`; inline loop replaced by `batch_embed()` call |
| `pdf_rag/retriever.py` | Embedding call replaced by `provider.embed()`; `import requests` removed |
| `.env.example` | Document the three new variables |

## Alternatives considered

**Add `openai` SDK as a dependency** — cleaner API but adds a dependency for something achievable with `requests`. Rejected in favour of zero new deps.

**Abstract via a Protocol / ABC** — adds indirection without benefit at this scale. The duck-typed wrapper is sufficient.

**Rename `base_url` parameters to be provider-agnostic** — would require changing all function signatures and CLI flags. Deferred; for OpenAI the URL comes from config, not the existing `base_url` param.
