import json
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from pdf_rag.server import make_app
from tests.conftest import OLLAMA_URL


def make_test_app(seeded_col):
    app = make_app(db_path="unused", base_url=OLLAMA_URL)
    with patch("pdf_rag.researcher._open_collection", return_value=seeded_col):
        yield app


def parse_sse(response) -> list[dict]:
    """Parse SSE lines from a streaming TestClient response into event dicts."""
    events = []
    kind = "token"
    for line in response.iter_lines():
        if not line:
            continue
        if line.startswith("event:"):
            kind = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            payload = json.loads(line.removeprefix("data:").strip())
            events.append({"kind": kind, "text": payload.get("text", "")})
    return events


# ── /v1/ask ───────────────────────────────────────────────────────────────────

def test_ask_streams_tokens(seeded_collection, mock_embed, mock_ollama_chat):
    app = make_app(db_path="unused", base_url=OLLAMA_URL)
    with patch("pdf_rag.researcher._open_collection", return_value=seeded_collection):
        with TestClient(app) as client:
            with client.stream("POST", "/v1/ask", json={"question": "What is entropy?"}) as r:
                events = parse_sse(r)
    assert any(e["kind"] == "token" for e in events)


def test_ask_ends_with_done(seeded_collection, mock_embed, mock_ollama_chat):
    app = make_app(db_path="unused", base_url=OLLAMA_URL)
    with patch("pdf_rag.researcher._open_collection", return_value=seeded_collection):
        with TestClient(app) as client:
            with client.stream("POST", "/v1/ask", json={"question": "What is entropy?"}) as r:
                events = parse_sse(r)
    assert events[-1]["kind"] == "done"


def test_ask_streams_log_events(seeded_collection, mock_embed, mock_ollama_chat):
    app = make_app(db_path="unused", base_url=OLLAMA_URL)
    with patch("pdf_rag.researcher._open_collection", return_value=seeded_collection):
        with TestClient(app) as client:
            with client.stream("POST", "/v1/ask", json={"question": "What is entropy?"}) as r:
                events = parse_sse(r)
    assert any(e["kind"] == "log" for e in events)


def test_ask_token_contains_response(seeded_collection, mock_embed, mock_ollama_chat):
    app = make_app(db_path="unused", base_url=OLLAMA_URL)
    with patch("pdf_rag.researcher._open_collection", return_value=seeded_collection):
        with TestClient(app) as client:
            with client.stream("POST", "/v1/ask", json={"question": "What is entropy?"}) as r:
                events = parse_sse(r)
    tokens = "".join(e["text"] for e in events if e["kind"] == "token")
    assert "Response from" in tokens


# ── /v1/research ──────────────────────────────────────────────────────────────

def test_research_streams_tokens(seeded_collection, mock_embed, mock_ollama_chat):
    app = make_app(db_path="unused", base_url=OLLAMA_URL)
    with patch("pdf_rag.researcher._open_collection", return_value=seeded_collection):
        with TestClient(app) as client:
            with client.stream("POST", "/v1/research",
                               json={"question": "What is entropy?", "depth": 1, "n_subquestions": 1}) as r:
                events = parse_sse(r)
    assert any(e["kind"] == "token" for e in events)


def test_research_ends_with_done(seeded_collection, mock_embed, mock_ollama_chat):
    app = make_app(db_path="unused", base_url=OLLAMA_URL)
    with patch("pdf_rag.researcher._open_collection", return_value=seeded_collection):
        with TestClient(app) as client:
            with client.stream("POST", "/v1/research",
                               json={"question": "What is entropy?", "depth": 1, "n_subquestions": 1}) as r:
                events = parse_sse(r)
    assert events[-1]["kind"] == "done"


def test_research_streams_planning_log(seeded_collection, mock_embed, mock_ollama_chat):
    app = make_app(db_path="unused", base_url=OLLAMA_URL)
    with patch("pdf_rag.researcher._open_collection", return_value=seeded_collection):
        with TestClient(app) as client:
            with client.stream("POST", "/v1/research",
                               json={"question": "What is entropy?", "depth": 1, "n_subquestions": 1}) as r:
                events = parse_sse(r)
    log_texts = " ".join(e["text"] for e in events if e["kind"] == "log")
    assert "Planning" in log_texts
