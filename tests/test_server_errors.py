from starlette.testclient import TestClient

from pdf_rag.server import make_app
from tests.conftest import OLLAMA_URL


def _client():
    app = make_app(db_path="unused", base_url=OLLAMA_URL)
    return TestClient(app)


def test_ask_missing_question_returns_422():
    with _client() as client:
        r = client.post("/v1/ask", json={})
    assert r.status_code == 422


def test_research_missing_question_returns_422():
    with _client() as client:
        r = client.post("/v1/research", json={})
    assert r.status_code == 422


def test_chat_completions_missing_messages_returns_422():
    with _client() as client:
        r = client.post("/v1/chat/completions", json={"model": "pedro-ask"})
    assert r.status_code == 422


def test_chat_completions_missing_model_returns_422():
    with _client() as client:
        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 422


def test_ask_invalid_json_returns_422():
    with _client() as client:
        r = client.post(
            "/v1/ask",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
    assert r.status_code == 422
