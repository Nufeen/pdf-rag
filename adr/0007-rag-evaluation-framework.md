# ADR-0007: RAG Evaluation Framework

## Context

Pedro has no systematic way to compare retrieval or generation quality across configurations. Choosing `top_k`, selecting an LLM model, or switching embedding models is currently done by intuition. As the model zoo and use cases grow, we need reproducible, quantitative evidence for these decisions.

## Problem Statement

How do we measure the effect of changing `top_k` and `llm_model` on answer quality without:
1. Requiring cloud APIs (Pedro is fully local)
2. Adding heavy dependencies
3. Blocking the existing pipeline code

## Decision

**Custom `answer_correctness` scorer using Pedro's existing infrastructure**, run via a standalone `eval/` script. No new dependencies.

`answer_correctness` = weighted average of:
- **Factual score** (0–1) — judge LLM prompted with question + answer + ground_truth; float parsed from response
- **Semantic score** (0–1) — cosine similarity between embeddings of answer and ground_truth

Both use infrastructure Pedro already has: `ollama.Client` for LLM calls, `/api/embed` via `requests` for embeddings, `numpy` for cosine similarity.

**Rationale:**
- `answer_correctness` is the most informative single metric when ground truth is available
- RAGAS computes this metric identically internally — adding RAGAS would mean pulling in `langchain`, `langchain-ollama`, `datasets`, and `pandas` for a ~50-line computation
- Full control over the scoring prompt — easy to tune for the domain and judge model
- A manually-maintained `dataset.jsonl` is the simplest starting point; no auto-generation complexity

## Rejected Alternatives

| Option | Reason rejected |
|--------|----------------|
| **RAGAS** | Pulls in `langchain`, `langchain-ollama`, `datasets`, `pandas` to compute a metric Pedro can implement in ~50 lines with existing deps. The `answer_correctness` formula is identical: factual LLM score + semantic cosine similarity. |
| **DeepEval** | Similar heavy dependency tree. No meaningful advantage over a custom scorer for a single metric. |
| **Promptfoo** | Node.js-based; adds a runtime outside the Python stack. |
| **Human evaluation** | Not reproducible. Cannot be automated or run on a matrix of configs. |
| **Perplexity / BLEU / ROUGE** | Wrong signal for RAG — these measure fluency or n-gram overlap, not factual correctness against a ground truth. |
| **Full RAGAS suite** (faithfulness, context recall, etc.) | Requires a strong judge model to be reliable. With a weak local judge the additional metrics add noise, not signal. Can be added later if a better judge becomes available. |

## Academic Background

The `answer_correctness` formula — factual LLM score + semantic cosine score — is grounded in three established lines of work.

**LLM-as-judge (G-Eval, Liu et al., EMNLP 2023; Zheng et al., NeurIPS 2023)**
Liu et al. showed that prompting GPT-4 with chain-of-thought evaluation steps achieves a Spearman correlation of 0.514 with human judgments on summarization — outperforming all prior automatic metrics (arXiv:2303.16634). Zheng et al. validated that strong LLM judges match human preferences at >80% agreement, and introduced the taxonomy of failure modes we need to be aware of with local models: *position bias* (preference for whichever answer appears first), *verbosity bias* (preference for longer responses), and *self-enhancement bias* (a model rating its own outputs higher). GPT-4 was largely immune; smaller models were not (arXiv:2306.05685). This directly motivates our design note that local-judge scores are noisy in absolute terms but remain useful for **relative comparison** across configs where biases affect all runs equally.

**Embedding-based semantic similarity (BERTScore, Zhang et al., ICLR 2020)**
Zhang et al. demonstrated that computing cosine similarity between contextual token embeddings correlates better with human judgment than n-gram metrics (BLEU, ROUGE) on both machine translation and image captioning (arXiv:1904.09675). Our `semantic_score` applies the same principle at sentence level using domain embeddings already present in Pedro — no additional model or library required.

**The RAGAS formula (Es et al., EACL 2024)**
Es et al. defined `answer_correctness` as a weighted combination of factual correctness (LLM-judged) and answer semantic similarity (embedding cosine), with factual correctness weighted at 0.75 (arXiv:2309.15217). We adopt the same formula and default weight. The key difference: RAGAS wraps this in a framework with `langchain`, `datasets`, and `pandas`; we implement it directly using Pedro's existing `ollama.Client` and `/api/embed` endpoint.

**Why BLEU/ROUGE/perplexity are wrong here**
Both G-Eval and BERTScore papers show that n-gram overlap metrics have low correlation with human judgments on tasks requiring semantic correctness — which is exactly the RAG answer evaluation case. Perplexity measures language model confidence, not factual accuracy against a reference.

## Design

### New files

```
eval/
  dataset.jsonl       # Q&A pairs; extend manually
  scorer.py           # factual_score(), semantic_score(), score()
  evaluate.py         # matrix runner + results output
  results/            # auto-created; one CSV per run (gitignored)
```

No changes to `pyproject.toml` — zero new dependencies.

### `eval/dataset.jsonl`

One JSON object per line:

```jsonl
{"question": "...", "ground_truth": "...", "tags": ["topic"]}
```

Tags are optional metadata for filtering subsets of questions.

### `eval/scorer.py`

```python
def factual_score(question, answer, ground_truth, client, judge_model) -> float:
    # Prompts judge LLM via ollama.Client (same pattern as llm.py)
    # Parses first float 0-1 from response via regex
    # Falls back to 0.0 on parse failure

def semantic_score(answer, ground_truth, embed_model, base_url) -> float:
    # Single /api/embed call with input=[answer, ground_truth]
    # Returns numpy cosine similarity between the two vectors

def score(question, answer, ground_truth, client, judge_model, embed_model, base_url,
          factual_weight=0.75) -> float:
    return factual_weight * factual_score(...) + (1 - factual_weight) * semantic_score(...)
```

**Scoring prompt** (`factual_score`):
```
You are a strict evaluator. Given a question, a ground truth answer, and a candidate answer,
rate how factually correct the candidate is compared to the ground truth.
Respond with a single decimal number between 0.0 and 1.0. Nothing else.

Question: {question}
Ground truth: {ground_truth}
Answer: {answer}
```

**Float parsing** — regex `r'0?\.[0-9]+|1\.0|[01]'`; fallback `0.0` on no match.

**Semantic similarity** — both strings embedded in one request (`"input": [answer, ground_truth]`), then `np.dot(a, b) / (norm(a) * norm(b))`. Reuses the same `/api/embed` pattern from `retriever.py`.

### `eval/evaluate.py`

**CLI:**

```
--models      comma-separated model names     (default: LLM_MODEL)
--top-k       comma-separated int values      (default: "5")
--judge       Ollama model for scoring        (default: first --models entry)
--db-path     path to ChromaDB               (default: DB_PATH)
--ollama-url  Ollama base URL                (default: OLLAMA_BASE_URL)
--embed-model fixed for all runs             (default: EMBED_MODEL)
--dataset     path to dataset.jsonl          (default: eval/dataset.jsonl)
```

**Flow:**

```
load dataset.jsonl
collection = _open_collection(db_path)
client = ollama.Client(host=ollama_url)

for model in models:
  for top_k in top_ks:
    for item in dataset:
      chunks = query(item.question, collection, embed_model, base_url, top_k)
      answer = generate_answer(item.question, chunks, base_url, model, stream=False)
      s = score(item.question, answer, item.ground_truth, client, judge, embed_model, base_url)
      store {question, answer, ground_truth, llm_model, top_k, score}

print pivot table:  model | top_k | avg_score
save eval/results/YYYYMMDD_HHMMSS.csv  (stdlib csv module)
```

**Reused functions (no new pipeline code):**

| Function | File |
|----------|------|
| `_open_collection(db_path)` | `pdf_rag/researcher.py` |
| `query(question, collection, embed_model, base_url, top_k)` | `pdf_rag/retriever.py` |
| `generate_answer(question, chunks, base_url, llm_model, stream=False)` | `pdf_rag/llm.py` |

### Usage

```bash
# single model, default top_k=5
python eval/evaluate.py

# full matrix
python eval/evaluate.py --models mistral:7b,llama3:8b --top-k 3,5,10 --judge mistral:7b
```

## Future Considerations

- **`research` pipeline** — extend `evaluate.py` with a `--pipeline research` flag once `ask` baseline is established
- **`embed_model` as a variable** — add `--embed-models` flag; requires re-indexing per embedding model
- **Stronger judge** — swap judge model name only; no code changes needed
- **CI integration** — run eval on a small fixed dataset as a regression check on model/prompt changes
