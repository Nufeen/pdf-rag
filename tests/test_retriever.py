import chromadb

from pdf_rag.retriever import query
from tests.conftest import FAKE_VECTOR, OLLAMA_URL


def test_query_returns_list_of_dicts(seeded_collection, mock_embed):
    result = query("What is entropy?", seeded_collection, base_url=OLLAMA_URL, top_k=2)
    assert isinstance(result, list)
    assert all(isinstance(c, dict) for c in result)


def test_query_chunk_has_required_keys(seeded_collection, mock_embed):
    result = query("What is entropy?", seeded_collection, base_url=OLLAMA_URL, top_k=2)
    assert result
    for chunk in result:
        assert "text" in chunk
        assert "source_file" in chunk
        assert "page_num" in chunk
        assert "score" in chunk


def test_query_top_k_limits_results(seeded_collection, mock_embed):
    result = query("What is entropy?", seeded_collection, base_url=OLLAMA_URL, top_k=1)
    assert len(result) == 1


def test_query_top_k_returns_all_when_fewer_than_k(seeded_collection, mock_embed):
    # seeded_collection has 2 chunks; top_k=10 should return at most 2
    result = query("What is entropy?", seeded_collection, base_url=OLLAMA_URL, top_k=10)
    assert len(result) <= 2


def test_query_empty_collection_returns_empty_list(mock_embed):
    col = chromadb.EphemeralClient().get_or_create_collection(
        "empty", metadata={"hnsw:space": "cosine"}
    )
    result = query("What is entropy?", col, base_url=OLLAMA_URL, top_k=5)
    assert result == []


def test_query_score_is_between_0_and_1(seeded_collection, mock_embed):
    result = query("What is entropy?", seeded_collection, base_url=OLLAMA_URL, top_k=2)
    for chunk in result:
        assert 0.0 <= chunk["score"] <= 1.0
