# ADR-0004: Multilingual Search for Deep Research Mode

## Context

The deep research pipeline retrieves only in the language of the input question. When PDF books contain text in multiple languages (e.g., Russian, French, German alongside English), relevant passages in other languages are silently missed.

## Decision

Translate each sub-question into each configured target language before retrieval, merge the results (deduplicated), and feed the richer chunk set into the existing synthesis and reflection loop unchanged.

Three insertion points were considered:

| Option | Where translation happens | Academic name |
|--------|--------------------------|---------------|
| A | Translate seed question → run full research per language | Parallel CLIR |
| B | Translate each subquestion before retrieval | mRAG query translation |
| C | Collect all subquestions, translate, run a final retrieval pass | Retrospective query augmentation |

Option A multiplies total compute by N languages and requires merging two independent research trees. Option C means cross-lingual chunks never enter the reflection loop. Option B is the standard approach in multilingual RAG literature: translate at the retrieval boundary, enrich findings at every iteration including reflection passes.

> **Alternative without translation:** Switching `EMBED_MODEL` to a multilingual model (e.g. `paraphrase-multilingual-minilm-l12-v2`) makes cross-lingual search work natively. Requires re-indexing all PDFs.

## Academic Background

**Query translation vs document translation.**
Saleh & Pecina (ACL 2020, [aclanthology.org/2020.acl-main.613](https://aclanthology.org/2020.acl-main.613/)) compared both strategies in medical CLIR and found query translation consistently outperforms document translation — fewer texts to translate, no loss of retrieval index structure. The finding holds across both SMT and NMT backends.

**Multilingual RAG in practice.**
Chirkova et al. (KnowLLM @ ACL 2024, [aclanthology.org/2024.knowllm-1.15](https://aclanthology.org/2024.knowllm-1.15/)) show that even with strong off-the-shelf multilingual retrievers, task-specific prompt engineering is needed to keep the generator on-language. Without it, retrieved cross-lingual passages cause "language drift" — the model switches output language mid-answer.

**Multilingual dense retrieval (the embedding alternative).**
Asai et al. (NeurIPS 2021, [arxiv:2107.11976](https://arxiv.org/abs/2107.11976)) introduced mDPR, a many-to-many retriever that encodes questions and passages from all languages into a shared dense space, eliminating the translation step entirely. Wang et al. (2024, [arxiv:2402.05672](https://arxiv.org/abs/2402.05672)) extend this to mE5, trained on 1B multilingual pairs — the current state-of-the-art for zero-shot cross-lingual dense retrieval and the practical drop-in for `nomic-embed-text`.

**Benchmark.**
MIRACL (Zhang et al., WSDM 2023, [arxiv:2210.09984](https://arxiv.org/abs/2210.09984)) provides 78k queries with human relevance judgements across 18 languages and is the standard benchmark for evaluating multilingual retrieval quality.

**Takeaway for this project:** query translation at the subquestion level (Option B) is the pragmatic path given the existing monolingual embedding index. Switching to mE5 and re-indexing is the higher-quality long-term alternative.

## Configuration

Two new env vars:

| Var | Default | Purpose |
|-----|---------|---------|
| `RAG_SEARCH_LANGUAGES` | `""` (disabled) | Comma-separated language names, e.g. `Russian,French` |
| `RAG_TRANSLATE_MODEL` | `TINY_MODEL` | Model used for query translation |

Empty `RAG_SEARCH_LANGUAGES` → no behaviour change, no extra queries.

## Implementation

### New prompt — `prompts/translate_question.txt`

```
Translate the following question to {lang}. Output only the translated question, nothing else.

Question: {text}
```

### New functions in `pdf_rag/researcher.py`

```python
def translate_question(text: str, lang: str, client: Client, model: str) -> str:
    prompt = load_prompt("translate_question", text=text, lang=lang)
    response = client.chat(model=model,
                           messages=[{"role": "user", "content": prompt}],
                           stream=False)
    return response["message"]["content"].strip()


def retrieve_multilingual(
    question, collection, embed_model, base_url, top_k,
    languages, translate_model, client, log_fn=None,
) -> list[dict]:
    chunks = query(question, collection, embed_model, base_url, top_k)
    seen = {(c["source_file"], c["page_num"], c["text"][:40]) for c in chunks}
    for lang in languages:
        translated = translate_question(question, lang, client, translate_model)
        if log_fn:
            log_fn(f"  [dim](→ {lang}: {translated})[/dim]")
        for c in query(translated, collection, embed_model, base_url, top_k):
            key = (c["source_file"], c["page_num"], c["text"][:40])
            if key not in seen:
                chunks.append(c)
                seen.add(key)
    return chunks
```

`research()` gains `languages` and `translate_model` parameters; every `query(sq, ...)` call in the sub-question loop is replaced with `retrieve_multilingual(sq, ..., languages, translate_model, client, log_fn)`.

### CLI — `pdf_rag/cli.py`

```python
@click.option("--languages", default=",".join(SEARCH_LANGUAGES))
@click.option("--translate-model", default=TRANSLATE_MODEL)
```

### TUI — `pdf_rag/tui.py`

Import `SEARCH_LANGUAGES`, `TRANSLATE_MODEL` and pass to `research()`. Translated queries surface inline via `log_fn` — no UI changes needed.

## Files Changed

| File | Change |
|------|--------|
| `pdf_rag/config.py` | Add `SEARCH_LANGUAGES`, `TRANSLATE_MODEL` |
| `pdf_rag/researcher.py` | Add `translate_question()`, `retrieve_multilingual()`, update `research()` |
| `pdf_rag/cli.py` | Add `--languages`, `--translate-model` options |
| `pdf_rag/tui.py` | Import and pass new config vars |
| `prompts/translate_question.txt` | New prompt template |
