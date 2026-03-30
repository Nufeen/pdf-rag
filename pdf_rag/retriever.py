import requests

from .config import EMBED_MODEL, OLLAMA_BASE_URL, TOP_K


def query(
    question: str,
    collection,
    embed_model: str = EMBED_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    top_k: int = TOP_K,
) -> list[dict]:
    resp = requests.post(
        f"{base_url}/api/embed",
        json={"model": embed_model, "input": [question]},
        timeout=60,
    )
    resp.raise_for_status()
    query_embedding = resp.json()["embeddings"][0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "source_file": meta["source_file"],
            "page_num": meta["page_num"],
            "score": 1 - dist,
        })

    return chunks
