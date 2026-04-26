from collections.abc import Callable
from pathlib import Path

from .config import LLM_MODEL, OLLAMA_BASE_URL
from .provider import make_client


_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def load_prompt(name: str, **kwargs: str) -> str:
    user_path = _PROMPTS_DIR / "user_prompts" / f"{name}.txt"
    if user_path.exists():
        template = user_path.read_text().strip()
    else:
        default_path = _PROMPTS_DIR / "default" / f"{name}.txt"
        template = default_path.read_text().strip()
    return template.format(**kwargs) if kwargs else template


def _load_system_prompt() -> str:
    return load_prompt("answer")


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
    on_token: Callable[[str], None] | None = None,
    session_context: str = "",
    system_prompt: str | None = None,
) -> str:
    context = build_context(chunks)
    if session_context:
        context = f"Session context:\n{session_context}\n\n---\n\n{context}"
    user_message = (
        f"Context excerpts:\n\n{context}\n\n---\n\n"
        f"Question: {question}\n\n"
        "Answer (cite sources as [Book: filename, Page: N]):"
    )

    client = make_client(base_url)
    try:
        response = client.chat(
            model=llm_model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                    if system_prompt is not None
                    else _load_system_prompt(),
                },
                {"role": "user", "content": user_message},
            ],
            stream=stream,
        )
        if stream:
            _emit = on_token if on_token is not None else lambda t: print(t, end="", flush=True)
            full = ""
            for part in response:
                token = part["message"]["content"]
                _emit(token)
                full += token
            if on_token is None:
                print("\n🌵")
            return full
        else:
            return response["message"]["content"]
    except Exception as e:
        if "connection" in str(e).lower() or "refused" in str(e).lower():
            raise SystemExit(
                f"Cannot reach LLM provider at {base_url}. Check your provider settings."
            )
        raise
