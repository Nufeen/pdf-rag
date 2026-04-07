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
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL  = os.getenv("LLM_MODEL",  "mistral:7b")
FAST_MODEL = os.getenv("FAST_MODEL", LLM_MODEL)
DB_PATH = os.getenv("DB_PATH", str(Path.home() / ".pdf-rag" / "chroma_db"))
SESSIONS_PATH = Path(os.getenv("SESSIONS_PATH", str(Path.home() / ".pdf-rag" / "sessions")))
CHUNK_SIZE = _int("CHUNK_SIZE", 800)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 150)
TOP_K = _int("TOP_K", 5)
COLLECTION_NAME = "pdf_books"
EMBED_BATCH_SIZE = 32
CHROMA_BATCH_SIZE = 5000  # ChromaDB max is 5461 (SQLite variable limit)
RESEARCH_DEPTH = _int("RESEARCH_DEPTH", 2)
RESEARCH_N_SUBQUESTIONS = _int("RESEARCH_N_SUBQUESTIONS", 3)
SEARCH_LANGUAGES: list[str] = [
    l.strip() for l in os.environ.get("SEARCH_LANGUAGES", "").split(",") if l.strip()
]
TRANSLATE_MODEL: str = os.environ.get("TRANSLATE_MODEL", FAST_MODEL)
SERVER_URL: str = os.environ.get("PEDRO_SERVER_URL", "")
PDF_EXPORT_PATH: str = os.path.expanduser(
    os.environ.get("PEDRO_PDF_PATH", "~/.pedro/exports")
)
