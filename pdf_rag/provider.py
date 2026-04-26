"""
Provider abstraction for LLM chat and embeddings.

Dispatches to Ollama or an OpenAI-compatible endpoint based on PROVIDER_TYPE.
Both return/yield dicts matching Ollama's shape so call sites need no changes.
"""

import json

import requests

from .config import OLLAMA_BASE_URL, OPENAI_API_KEY, OPENAI_BASE_URL, PROVIDER_TYPE


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

class _OllamaClient:
    def __init__(self, host: str) -> None:
        from ollama import Client
        self._client = Client(host=host)

    def chat(self, model: str, messages: list[dict], stream: bool = False):
        return self._client.chat(model=model, messages=messages, stream=stream)


# ---------------------------------------------------------------------------
# OpenAI-compatible
# ---------------------------------------------------------------------------

class _OpenAIClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def chat(self, model: str, messages: list[dict], stream: bool = False):
        payload = {"model": model, "messages": messages, "stream": stream}
        if stream:
            resp = requests.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=self._headers,
                stream=True,
                timeout=120,
            )
            resp.raise_for_status()
            return self._iter_stream(resp)
        else:
            resp = requests.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=self._headers,
                timeout=120,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return {"message": {"content": content}}

    @staticmethod
    def _iter_stream(resp):
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode() if isinstance(raw, bytes) else raw
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            content = chunk["choices"][0]["delta"].get("content") or ""
            if content:
                yield {"message": {"content": content}}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_client(base_url: str = OLLAMA_BASE_URL):
    """Return a chat client for the configured provider."""
    if PROVIDER_TYPE == "openai":
        return _OpenAIClient(OPENAI_BASE_URL, OPENAI_API_KEY)
    return _OllamaClient(base_url)


def embed(
    texts: list[str],
    model: str,
    base_url: str = OLLAMA_BASE_URL,
    batch_size: int = 32,
) -> list[list[float]]:
    """Embed texts using the configured provider. Returns list of float vectors."""
    if PROVIDER_TYPE == "openai":
        return _openai_embed(texts, model, batch_size)
    return _ollama_embed(texts, model, base_url, batch_size)


def _ollama_embed(texts: list[str], model: str, base_url: str, batch_size: int) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = requests.post(
            f"{base_url}/api/embed",
            json={"model": model, "input": batch},
            timeout=120,
        )
        resp.raise_for_status()
        embeddings.extend(resp.json()["embeddings"])
    return embeddings


def _openai_embed(texts: list[str], model: str, batch_size: int) -> list[list[float]]:
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    base_url = OPENAI_BASE_URL.rstrip("/")
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = requests.post(
            f"{base_url}/embeddings",
            json={"model": model, "input": batch},
            headers=headers,
            timeout=120,
        )
        resp.raise_for_status()
        body = resp.json()
        # Standard OpenAI: {"data": [{"embedding": [...], "index": N}, ...]}
        # Some servers return a bare list of vectors or {"embeddings": [...]}
        if isinstance(body, list):
            embeddings.extend(body)
        elif "data" in body:
            data = sorted(body["data"], key=lambda x: x.get("index", 0))
            embeddings.extend(item["embedding"] for item in data)
        elif "embeddings" in body:
            embeddings.extend(body["embeddings"])
        else:
            raise ValueError(f"Unrecognised embeddings response format: {list(body.keys())}")
    return embeddings
