# ADR-0008: Prompt Variant Evaluation

## Context

ADR-0007 established a model × top_k evaluation matrix. Once a winner model and top_k are identified, the next variable to optimise is the system prompt. Different prompt styles (verbosity, reasoning instructions, citation strictness) may have a larger effect on answer quality than model choice at the same parameter scale.

## Problem Statement

How do we compare prompt variants without:
1. Breaking the existing model × top_k evaluation workflow
2. Adding a separate script or separate dataset format
3. Making the result table unreadable

## Decision

**Extend the existing evaluation matrix with an optional prompt dimension.** When `--prompt-dir` is provided the matrix becomes `models × top_k × prompt_variants`. When omitted, behaviour is identical to the current eval (default production prompt, no extra column in results).

Prompt variants are plain `.txt` files in `eval/prompt_variants/`. Adding a new variant means dropping a file — no code change needed.

**Rationale:**
- One script, one dataset, one CSV format — just a wider matrix
- `.txt` files are the natural format for multi-line prompts; easier to read and diff than JSONL
- The `system_prompt` injection point is a one-line addition to `generate_answer()` — minimal surface change to production code
- Sentinel `(None, None)` for the no-prompt-dir case keeps the loop uniform with zero branching in the hot path

## Design

### `pdf_rag/llm.py` — one-line change

Add `system_prompt: str | None = None` to `generate_answer()` after `session_context`.

```python
# before
{"role": "system", "content": _load_system_prompt()},
# after
{"role": "system", "content": system_prompt if system_prompt is not None else _load_system_prompt()},
```

All existing callers are unaffected (`None` is the default).

### `eval/prompt_variants/`

Starter variants — extend by dropping more `.txt` files:

| File | Style |
|------|-------|
| `baseline.txt` | exact copy of current `prompts/answer.txt` |
| `concise.txt` | same rules, 2-4 sentence limit |
| `step-by-step.txt` | reason over excerpts before synthesising |

### `eval/evaluate.py`

**New arg:**
```
--prompt-dir    directory of .txt prompt variants (optional)
```

**Variant resolution in `main()`:**
```python
if args.prompt_dir:
    prompt_variants = [(p.stem, p.read_text().strip())
                       for p in sorted(Path(args.prompt_dir).glob("*.txt"))]
else:
    prompt_variants = [(None, None)]   # sentinel → uses _load_system_prompt()
```

**Extended loop in `run_evaluation()`:**
```
for model in models:
  for top_k in top_ks:
    for (variant_name, system_prompt) in prompt_variants:
      for item in dataset:
        answer = generate_answer(..., system_prompt=system_prompt)
        results.append({..., "prompt_variant": variant_name or "default"})
```

`total` = `len(models) × len(top_ks) × len(prompt_variants) × len(dataset)`

**Pivot table** — `prompt_variant` column added only when values differ from `"default"`:

```
Model           Top-K  Prompt         Avg Score  Avg Time  Count
-----------------------------------------------------------------
mistral:7b      5      baseline        0.723      4.2s      25
mistral:7b      5      concise         0.689      2.9s      25
mistral:7b      5      step-by-step    0.751      6.1s      25
llama3:8b       5      baseline        0.694      3.1s      25
...
```

**CSV** — `prompt_variant` column always present (`"default"` when `--prompt-dir` not used).

### Usage

```bash
# existing behaviour unchanged
python eval/evaluate.py --models mistral:7b,llama3:8b --top-k 3,5,10

# full matrix including prompt variants
python eval/evaluate.py \
  --models mistral:7b,llama3:8b \
  --top-k 5,10 \
  --prompt-dir eval/prompt_variants/ \
  --judge mistral:7b
```

## Future Considerations

- **User message variants** — the citation instruction at the end of the user message in `generate_answer()` is currently hardcoded; a `--user-template-dir` flag could vary that independently
- **Prompt evolution tracking** — store winning prompt back to `prompts/answer.txt` via a `--promote` flag
- **Automated prompt generation** — use an LLM to generate candidate variants from the baseline; feed them into this pipeline
