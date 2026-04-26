from chromadb import Collection

from .config import EMBED_MODEL, OLLAMA_BASE_URL, TOP_K
from .provider import embed


def query(
    question: str,
    collection: Collection,
    embed_model: str = EMBED_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    top_k: int = TOP_K,
) -> list[dict]:
    try:
        query_embedding = embed([question], embed_model, base_url)[0]
    except Exception as e:
        if "connection" in str(e).lower() or "refused" in str(e).lower():
            from .config import OPENAI_BASE_URL, PROVIDER_TYPE
            url = OPENAI_BASE_URL if PROVIDER_TYPE == "openai" else base_url
            raise SystemExit(f"Cannot reach embed provider at {url}. Check your provider settings.\nError: {e}")
        raise

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
