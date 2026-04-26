import hashlib
from pathlib import Path

import chromadb
import click
import fitz  # PyMuPDF
from tqdm import tqdm

from .chunker import split_text
from .config import (
    CHROMA_BATCH_SIZE,
    DB_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBED_BATCH_SIZE,
    EMBED_MODEL,
    OLLAMA_BASE_URL,
)
from . import provider

def compute_file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def extract_pages(pdf_path: str) -> list[dict]:
    pages = []
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if len(text.strip()) > 100:
                pages.append({"page_num": page_num, "text": text})
    return pages


def batch_embed(texts: list[str], model: str, base_url: str, batch_size: int = EMBED_BATCH_SIZE) -> list[list[float]]:
    return provider.embed(texts, model, base_url, batch_size)


def chunk_id(source_file: str, page_num: int, chunk_idx: int) -> str:
    return hashlib.md5(f"{source_file}:{page_num}:{chunk_idx}".encode()).hexdigest()


def index_folder(
    folder_path: str,
    db_path: str = DB_PATH,
    embed_model: str = EMBED_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    force: bool = False,
) -> None:
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    pdf_files = sorted(Path(folder_path).glob("**/*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return

    for pdf_path in pdf_files:
        file_hash = compute_file_hash(str(pdf_path))

        if not force:
            existing = collection.get(
                where={"source_hash": file_hash},
                limit=1,
                include=[],
            )
            if existing["ids"]:
                print(f"Skipping (already indexed): {pdf_path.name}")
                continue

        # Remove stale chunks for this filename (handles modified files)
        old = collection.get(
            where={"source_file": pdf_path.name},
            include=[],
        )
        if old["ids"]:
            # delete() is not subject to the SQLite batch size limit
            collection.delete(ids=old["ids"])

        pages = extract_pages(str(pdf_path))
        if not pages:
            print(f"No extractable text in: {pdf_path.name}")
            continue

        all_chunks = []
        for page_data in pages:
            chunks = split_text(page_data["text"], CHUNK_SIZE, CHUNK_OVERLAP)
            for chunk in chunks:
                all_chunks.append({
                    "text": chunk,
                    "page_num": page_data["page_num"],
                    "source_file": pdf_path.name,
                    "source_hash": file_hash,
                })

        texts = [c["text"] for c in all_chunks]

        print(f"Embedding: {pdf_path.name} ({len(all_chunks)} chunks)...")
        try:
            embeddings = batch_embed(texts, embed_model, base_url)
        except Exception as e:
            msg = str(e)
            if "connection" in msg.lower() or "refused" in msg.lower():
                from .config import OPENAI_BASE_URL, PROVIDER_TYPE
                url = OPENAI_BASE_URL if PROVIDER_TYPE == "openai" else base_url
                raise SystemExit(f"Cannot reach embed provider at {url}. Check your provider settings.\nError: {e}")
            click.echo(click.style(f"Warning: skipping {pdf_path.name} — embed error: {e}", fg="yellow"))
            continue

        ids = [
            chunk_id(c["source_file"], c["page_num"], i)
            for i, c in enumerate(all_chunks)
        ]
        metadatas = [
            {k: v for k, v in c.items() if k != "text"}
            for c in all_chunks
        ]

        for i in range(0, len(all_chunks), CHROMA_BATCH_SIZE):
            collection.add(
                ids=ids[i : i + CHROMA_BATCH_SIZE],
                embeddings=embeddings[i : i + CHROMA_BATCH_SIZE],
                documents=texts[i : i + CHROMA_BATCH_SIZE],
                metadatas=metadatas[i : i + CHROMA_BATCH_SIZE],
            )
        print(f"Indexed: {pdf_path.name} ({len(all_chunks)} chunks)")
