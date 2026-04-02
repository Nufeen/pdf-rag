from collections.abc import Callable

import click
from ollama import Client
from chromadb import Collection

from .config import (
    EMBED_MODEL,
    FAST_MODEL,
    LLM_MODEL,
    OLLAMA_BASE_URL,
    RESEARCH_DEPTH,
    RESEARCH_N_SUBQUESTIONS,
    TINY_MODEL,
    TOP_K,
)
from .llm import generate_answer, load_prompt
from .retriever import query


def plan_subquestions(question: str, n: int, client: Client, model: str) -> list[str]:
    prompt = load_prompt("plan_subquestions", n=str(n), question=question)
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    lines = [l.strip() for l in response["message"]["content"].strip().splitlines() if l.strip()]
    return lines[:n]


def reflect(question: str, answer: str, client: Client, model: str) -> list[str] | None:
    prompt = load_prompt("reflect", question=question, answer=answer)
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    content = response["message"]["content"].strip()
    if content.upper().startswith("SUFFICIENT"):
        return None
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    return lines if lines else None


def synthesize(
    question: str,
    findings: list[dict],
    client: Client,
    model: str,
    base_url: str,
    stream: bool = True,
    on_token: Callable[[str], None] | None = None,
) -> str:
    parts = []
    for i, f in enumerate(findings, 1):
        parts.append(f"[Finding {i} — sub-question: {f['subquestion']}]\n{f['answer']}")
    context = "\n\n---\n\n".join(parts)
    prompt = load_prompt("synthesize", question=question, context=context)

    try:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
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
                print()
            return full
        else:
            return response["message"]["content"]
    except Exception as e:
        if "connection" in str(e).lower() or "refused" in str(e).lower():
            raise SystemExit(f"Cannot reach Ollama at {base_url}. Is it running and is OLLAMA_BASE_URL correct?")
        raise


def _step(msg: str) -> None:
    click.echo(click.style(f"\n🪅 {msg}", fg="yellow", bold=True))


def _subq(i: int, total: int, text: str) -> None:
    prefix = click.style(f"  [{i}/{total}]", fg="cyan", bold=True)
    click.echo(f"{prefix} {click.style(text, fg='cyan')}")


def _info(msg: str) -> None:
    click.echo(click.style(f"  {msg}", fg="bright_black"))


def _ok(msg: str) -> None:
    click.echo(click.style(f"  ✓ {msg}", fg="green"))


def research(
    question: str,
    collection: Collection,
    base_url: str = OLLAMA_BASE_URL,
    llm_model: str = LLM_MODEL,
    fast_model: str = FAST_MODEL,
    tiny_model: str = TINY_MODEL,
    embed_model: str = EMBED_MODEL,
    depth: int = RESEARCH_DEPTH,
    n_subquestions: int = RESEARCH_N_SUBQUESTIONS,
    top_k: int = TOP_K,
    log_fn: Callable[[str], None] | None = None,
    on_token: Callable[[str], None] | None = None,
) -> None:
    step = (lambda msg: log_fn(f"\n🪅 {msg}")) if log_fn else _step
    info = (lambda msg: log_fn(f"  {msg}")) if log_fn else _info
    ok = (lambda msg: log_fn(f"  ✓ {msg}")) if log_fn else _ok
    subq = (lambda i, total, text: log_fn(f"  [{i}/{total}] {text}")) if log_fn else _subq

    client = Client(host=base_url)
    all_findings: list[dict] = []

    for iteration in range(depth):
        if iteration == 0:
            step(f"Planning {n_subquestions} sub-questions...")
            subquestions = plan_subquestions(question, n_subquestions, client, tiny_model)
            for i, sq in enumerate(subquestions, 1):
                info(f"{i}. {sq}")
        else:
            step(f"Reflecting (pass {iteration}/{depth - 1})...")
            current_answer = synthesize(
                question, all_findings, client, fast_model, base_url, stream=False
            )
            followups = reflect(question, current_answer, client, tiny_model)
            if followups is None:
                ok("Answer is sufficient.")
                step("Final answer:\n")
                synthesize(question, all_findings, client, llm_model, base_url, stream=True, on_token=on_token)
                return
            info(f"→ {len(followups)} follow-up sub-question(s) identified:")
            for i, sq in enumerate(followups, 1):
                info(f"  {i}. {sq}")
            subquestions = followups

        step(f"Executing {len(subquestions)} sub-question(s)...")
        for i, sq in enumerate(subquestions, 1):
            subq(i, len(subquestions), sq)
            chunks = query(
                question=sq,
                collection=collection,
                embed_model=embed_model,
                base_url=base_url,
                top_k=top_k,
            )
            if not chunks:
                info("(no relevant content found)")
                continue
            info(f"retrieved {len(chunks)} chunks — generating partial answer...")
            answer = generate_answer(
                question=sq,
                chunks=chunks,
                base_url=base_url,
                llm_model=fast_model,
                stream=False,
            )
            all_findings.append({"subquestion": sq, "answer": answer, "chunks": chunks})

    step("Synthesizing final answer...\n")
    synthesize(question, all_findings, client, llm_model, base_url, stream=True, on_token=on_token)
