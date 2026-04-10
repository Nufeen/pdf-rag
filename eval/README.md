# RAG Evaluation

See [ADR-0007](adr/0007-rag-evaluation-framework.md) for design rationale.

## Setup

```bash
# Copy the example dataset and edit with your Q&A pairs
cp eval/dataset.jsonl.example eval/dataset.jsonl
```

Edit `eval/dataset.jsonl` — one JSON object per line:

```jsonl
{"question": "...", "ground_truth": "...", "tags": ["topic"]}
```

## Run Evaluation

```bash
# Single model, default top_k=5
uv run python -m eval.evaluate

# Full matrix: multiple models and top_k values
uv run python -m eval.evaluate --models mistral:7b,llama3:8b --top-k 3,5,10 --judge mistral:7b
```

**CLI options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--models` | `LLM_MODEL` | Comma-separated model names to evaluate |
| `--top-k` | `TOP_K` | Comma-separated top_k values to test |
| `--judge` | first `--models` entry | Model used for factual scoring |
| `--db-path` | `DB_PATH` | Path to ChromaDB |
| `--ollama-url` | `OLLAMA_BASE_URL` | Ollama host URL |
| `--embed-model` | `EMBED_MODEL` | Embedding model for semantic similarity |
| `--dataset` | `eval/dataset.jsonl` | Path to your Q&A dataset |
| `--output-dir` | `eval/results/` | Where to save CSV results |
| `--prompt-dir` | `None` | Directory of `.txt` prompt variants (optional) |

### Prompt Variants

To evaluate different system prompts, use `--prompt-dir`:

```bash
uv run python -m eval.evaluate \
  --models mistral:7b,llama3:8b \
  --top-k 5,10 \
  --prompt-dir eval/prompt_variants/ \
  --judge mistral:7b
```

Starter variants are in `eval/prompt_variants/`:

| File | Style |
|------|-------|
| `baseline.txt` | Exact copy of current `prompts/answer.txt` |
| `concise.txt` | Same rules, 2-4 sentence limit |
| `step-by-step.txt` | Reason over excerpts before synthesizing |

Add new variants by dropping `.txt` files — no code change needed.

See [ADR-0008](adr/0008-prompt-variant-evaluation.md) for design rationale.

### Output

Results are saved to `eval/results/<timestamp>.csv` and a pivot table is printed showing:

| Column | Description |
|--------|-------------|
| `Avg Score` | Weighted average of factual (0.75) and semantic (0.25) scores |
| `Avg Time` | Average time per question (retrieval + generation + scoring) |
| `Q/min` | Questions processed per minute |
