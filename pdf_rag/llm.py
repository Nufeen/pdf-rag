from pathlib import Path

from ollama import Client

from .config import LLM_MODEL, OLLAMA_BASE_URL


_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

_ANSWER_FALLBACK = (
    "You are a research assistant. Answer the user's question using ONLY the provided context "
    "excerpts from PDF books. For each statement you make, cite the source as "
    "[Book: <filename>, Page: <page_num>]. If the context does not contain enough information "
    "to answer the question, say so explicitly. Do not use any knowledge outside the provided excerpts."
)


def load_prompt(name: str, fallback: str = "", **kwargs: str) -> str:
    path = _PROMPTS_DIR / f"{name}.txt"
    template = path.read_text().strip() if path.exists() else fallback
    return template.format(**kwargs) if kwargs else template


def _load_system_prompt() -> str:
    return load_prompt("answer", fallback=_ANSWER_FALLBACK)


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
    stream: bool = True,
) -> str:
    context = build_context(chunks)
    user_message = (
        f"Context excerpts:\n\n{context}\n\n---\n\n"
        f"Question: {question}\n\n"
        "Answer (cite sources as [Book: filename, Page: N]):"
    )

    client = Client(host=base_url)
    try:
        response = client.chat(
            model=llm_model,
            messages=[
                {"role": "system", "content": _load_system_prompt()},
                {"role": "user", "content": user_message},
            ],
            stream=stream,
        )
        if stream:
            full = ""
            for part in response:
                token = part["message"]["content"]
                print(token, end="", flush=True)
                full += token
            print()
            return full
        else:
            return response["message"]["content"]
    except Exception as e:
        if "connection" in str(e).lower() or "refused" in str(e).lower():
            raise SystemExit(f"Cannot reach Ollama at {base_url}. Is it running and is OLLAMA_BASE_URL correct?")
        raise
