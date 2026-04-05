"""Synchronous SSE client for the pedro server. Runs inside worker threads."""
from __future__ import annotations

import json
from collections.abc import Callable

import requests


def _iter_events(response: requests.Response):
    """Yield (kind, text) pairs from an SSE response."""
    kind = "token"
    for raw in response.iter_lines():
        if not raw:
            continue
        line = raw.decode() if isinstance(raw, bytes) else raw
        if line.startswith("event:"):
            kind = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            payload = json.loads(line.removeprefix("data:").strip())
            yield kind, payload.get("text", "")


def stream_ask(
    server_url: str,
    question: str,
    params: dict,
    on_token: Callable[[str], None],
    log_fn: Callable[[str], None],
    check: Callable[[], None],
) -> None:
    body = {"question": question, **params}
    with requests.post(f"{server_url}/v1/ask", json=body, stream=True, timeout=300) as r:
        r.raise_for_status()
        for kind, text in _iter_events(r):
            check()
            if kind == "token":
                on_token(text)
            elif kind == "log":
                log_fn(text)
            elif kind == "done":
                break


def stream_research(
    server_url: str,
    question: str,
    params: dict,
    on_token: Callable[[str], None],
    log_fn: Callable[[str], None],
    check: Callable[[], None],
) -> None:
    body = {"question": question, **params}
    with requests.post(f"{server_url}/v1/research", json=body, stream=True, timeout=600) as r:
        r.raise_for_status()
        for kind, text in _iter_events(r):
            check()
            if kind == "token":
                on_token(text)
            elif kind == "log":
                log_fn(text)
            elif kind == "done":
                break
