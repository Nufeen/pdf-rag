import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    val = os.getenv(name, str(default))
    try:
        return int(val)
    except ValueError:
        raise SystemExit(f"Invalid value for {name}: {val!r} (must be an integer)")


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("RAG_DEEP_MODEL", "mistral:7b")
FAST_MODEL = os.getenv("RAG_FAST_MODEL", LLM_MODEL)
TINY_MODEL = os.getenv("RAG_TINY_MODEL", FAST_MODEL)
CHROMA_DB_PATH = os.getenv("RAG_DB_PATH", str(Path.home() / ".pdf-rag" / "chroma_db"))
SESSIONS_PATH = Path(os.getenv("RAG_SESSIONS_PATH", str(Path.home() / ".pdf-rag" / "sessions")))
CHUNK_SIZE = _int("RAG_CHUNK_SIZE", 800)
CHUNK_OVERLAP = _int("RAG_CHUNK_OVERLAP", 150)
TOP_K = _int("RAG_TOP_K", 5)
COLLECTION_NAME = "pdf_books"
EMBED_BATCH_SIZE = 32
CHROMA_BATCH_SIZE = 5000  # ChromaDB max is 5461 (SQLite variable limit)
RESEARCH_DEPTH = _int("RESEARCH_DEPTH", 2)
RESEARCH_N_SUBQUESTIONS = _int("RESEARCH_N_SUBQUESTIONS", 3)
SEARCH_LANGUAGES: list[str] = [
    l.strip() for l in os.environ.get("RAG_SEARCH_LANGUAGES", "").split(",") if l.strip()
]
TRANSLATE_MODEL: str = os.environ.get("RAG_TRANSLATE_MODEL", TINY_MODEL)
