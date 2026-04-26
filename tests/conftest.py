import pytest
import responses as resp_lib
import chromadb
from unittest.mock import patch
import pdf_rag.provider as _provider

FAKE_VECTOR = [0.1, 0.2, 0.3, 0.4]
OLLAMA_URL = "http://test-ollama:11434"

FAKE_CHUNKS = [
    {
        "source_file": "book.pdf",
        "page_num": 1,
        "source_hash": "abc",
        "text": "Entropy is a measure of disorder in thermodynamics.",
    },
    {
        "source_file": "book.pdf",
        "page_num": 2,
        "source_hash": "abc",
        "text": "Shannon entropy quantifies information uncertainty.",
    },
]


def _chat_response(model: str, stream: bool):
    """Return predictable chat content that encodes the model name."""
    content = f"Response from {model}."
    if not stream:
        return {"message": {"content": content}}
    return iter([{"message": {"content": content}}])


@pytest.fixture()
def seeded_collection():
    """In-memory ChromaDB collection pre-seeded with two fake chunks."""
    client = chromadb.EphemeralClient()
    col = client.get_or_create_collection("pdf_books", metadata={"hnsw:space": "cosine"})
    col.add(
        ids=["id1", "id2"],
        embeddings=[FAKE_VECTOR, FAKE_VECTOR],
        documents=[c["text"] for c in FAKE_CHUNKS],
        metadatas=[{k: v for k, v in c.items() if k != "text"} for c in FAKE_CHUNKS],
    )
    return col


@pytest.fixture(autouse=True)
def force_ollama_provider(monkeypatch):
    """Force PROVIDER_TYPE=ollama so tests are not affected by local .env."""
    monkeypatch.setattr(_provider, "PROVIDER_TYPE", "ollama")


@pytest.fixture()
def mock_embed():
    """Mock POST /api/embed to return FAKE_VECTOR for any input."""
    with resp_lib.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        rsps.add(
            resp_lib.POST,
            f"{OLLAMA_URL}/api/embed",
            json={"embeddings": [FAKE_VECTOR]},
        )
        yield rsps


@pytest.fixture()
def mock_ollama_chat():
    """Patch ollama.Client.chat to return content keyed by model name."""
    def _chat(self, model, messages, stream=False, **kwargs):
        return _chat_response(model, stream)

    with patch("ollama.Client.chat", _chat):
        yield
