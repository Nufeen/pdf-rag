from pathlib import Path

from ollama import Client

from .config import LLM_MODEL, OLLAMA_BASE_URL


_PROMPT_FILE = Path(__file__).parent.parent / "prompt.txt"


def _load_system_prompt() -> str:
    if _PROMPT_FILE.exists():
        return _PROMPT_FILE.read_text().strip()
    return (
        "You are a research assistant. Answer the user's question using ONLY the provided context "
        "excerpts from PDF books. For each statement you make, cite the source as "
        "[Book: <filename>, Page: <page_num>]. If the context does not contain enough information "
        "to answer the question, say so explicitly. Do not use any knowledge outside the provided excerpts."
    )


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Excerpt {i} — {chunk['source_file']}, Page {chunk['page_num']}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def generate_answer(
    question: str,
    chunks: list[dict],
    base_url: str = OLLAMA_BASE_URL,
    llm_model: str = LLM_MODEL,
) -> None:
    context = build_context(chunks)
    user_message = (
        f"Context excerpts:\n\n{context}\n\n---\n\n"
        f"Question: {question}\n\n"
        "Answer (cite sources as [Book: filename, Page: N]):"
    )

    client = Client(host=base_url)
    try:
        stream = client.chat(
            model=llm_model,
            messages=[
                {"role": "system", "content": _load_system_prompt()},
                {"role": "user", "content": user_message},
            ],
            stream=True,
        )
        for part in stream:
            print(part["message"]["content"], end="", flush=True)
        print()
    except Exception as e:
        if "connection" in str(e).lower() or "refused" in str(e).lower():
            raise SystemExit(f"Cannot reach Ollama at {base_url}. Is it running and is OLLAMA_BASE_URL correct?")
        raise
