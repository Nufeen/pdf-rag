# ADR-0000: Deep Research Mode

## Context

The basic `pedro ask` command performs a single-pass RAG: one query → retrieve chunks → generate answer. This works for focused questions but falls short for complex, multi-faceted topics that require reasoning across several angles.

## Decision

Add a `pedro research` command implementing **Query Decomposition + Iterative Refinement**:

1. **Plan** — LLM decomposes the question into N sub-questions
2. **Execute** — each sub-question is answered independently via standard RAG
3. **Synthesize** — LLM combines partial answers into a coherent response
4. **Reflect** — LLM evaluates completeness; if gaps found, generates follow-up sub-questions and repeats
5. **Output** — final answer with all sources cited

Iteration is bounded by a `DEPTH` parameter. At `DEPTH=1` there is no reflection — single decompose-execute-synthesize pass.

https://arxiv.org/html/2510.18633v1 on Query Decomposition

https://arxiv.org/pdf/2303.17651 on Self Refine

## Approach

Maps to three established ideas:

**Sub-Question Querying** (LlamaIndex) — break a complex question into smaller, focused sub-questions, answer each independently via retrieval, then merge the results. The insight is that a single embedding of a complex question is a poor retrieval signal; N targeted embeddings are much better.

**DSP — Demonstrate-Search-Predict** (Khattab et al. 2022) — a framework where the LLM alternates between _reasoning_ (what do I need to know?) and _retrieval_ (go find it), rather than retrieving once upfront. This ADR uses the same alternation pattern: plan → retrieve → synthesize → plan again if needed.

**Self-RAG** (Asai et al. 2023) — the model learns to reflect on its own outputs using special critique tokens: "Is this retrieval relevant? Is my answer supported? Is it complete?" The reflection step in this ADR approximates the same idea via a plain prompt ("Is this answer sufficient?") rather than a fine-tuned model, trading some precision for zero additional infrastructure.

Considered and deferred:

- **HyDE** (Hypothetical Document Embeddings) — improves retrieval quality but adds complexity; can be added later
- **FLARE** — requires token-level probability access not available via Ollama

## User Experience

```
$ pedro research "What are the tradeoffs between symbolic and connectionist AI?"

🪅 Planning 3 sub-questions...
🪅 Executing 3 sub-question(s)...
🪅 Reflecting (pass 1/1)...
🪅 Executing 1 sub-question(s)...
🪅 Synthesizing final answer...

Symbolic AI represents knowledge explicitly as rules and logic...
[Book: ai_foundations.pdf, Page: 42]
```

At `--depth 1` the reflection step is skipped — single decompose → execute → synthesize pass.

## Contracts

### CLI

```
pedro research <question> [--depth N] [--sub-questions N] [--top-k N]
```

### Internal API

```python
# researcher.py

def plan_subquestions(question: str, n: int, client, model) -> list[str]:
    """Generate n sub-questions to answer the main question."""

def reflect(question: str, answer: str, client, model) -> list[str] | None:
    """Return follow-up sub-questions if answer is insufficient, else None."""

def research(
    question: str,
    collection,
    base_url: str,
    llm_model: str,
    embed_model: str,
    depth: int,
    n_subquestions: int,
    top_k: int,
) -> None:
    """Orchestrate the full research loop, printing progress and final answer."""
```

```python
# llm.py — updated signature

def generate_answer(
    question: str,
    chunks: list[dict],
    base_url: str,
    llm_model: str,
    stream: bool = True,   # False for intermediate steps inside researcher
) -> str:
    """Generate and optionally stream an answer. Always returns the full string."""
```

## Parameters

| Env var                   | Default | Description                                 |
| ------------------------- | ------- | ------------------------------------------- |
| `RESEARCH_DEPTH`          | `2`     | Max reflection iterations                   |
| `RESEARCH_N_SUBQUESTIONS` | `3`     | Sub-questions per iteration                 |
| `RAG_TOP_K`               | `5`     | Chunks per sub-question (shared with `ask`) |

## Alternative Approaches Considered

| Approach                         | Description                                                                                   | Why not chosen                                                                               |
| -------------------------------- | --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **HyDE** (Gao et al. 2022)       | Before embedding a query, ask the LLM to write a hypothetical answer, then embed that instead | Improves retrieval quality; deferred for simplicity, can be added as `--hyde` flag           |
| **FLARE** (Jiang et al. 2023)    | Generate the answer token-by-token; pause and retrieve when model confidence is low           | Requires token-level probability access not exposed by Ollama                                |
| **Self-RAG** (Asai et al. 2023)  | Model uses special tokens to decide whether to retrieve, and scores its own outputs           | Requires a fine-tuned model; reflection loop in this ADR approximates the idea via prompting |
| **Graph RAG** (Edge et al. 2024) | Build a knowledge graph from documents; traverse it during retrieval                          | Significant indexing complexity; overkill for a personal book library                        |
| **Multi-vector retrieval**       | Store multiple embeddings per chunk (summary + full text)                                     | Better recall at cost of index size and complexity; worth revisiting at scale                |
