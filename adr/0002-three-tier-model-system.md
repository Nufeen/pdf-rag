# ADR-0002: Three-Tier Model System for Research Pipeline

## Context

The research pipeline currently uses one model (`RAG_DEEP_MODEL`) for every step. Large quality models produce better final answers but are slow for the many intermediate steps that don't require that quality. The pipeline has a natural split:

- **Orchestration steps** — not shown to the user, just structured outputs: planning sub-questions, per-sub-question answers, intermediate synthesis for reflection, reflection itself
- **Final answer** — streamed to the user: the final synthesis

Using smaller models for orchestration and the quality model only for the final synthesis makes the research command significantly faster without degrading any visible output. The `ask` command is unaffected — it is a single-pass operation and always uses `RAG_DEEP_MODEL`.

## Decision

Introduce three model tiers with independent env vars, all defaulting to `RAG_DEEP_MODEL` so existing single-model behavior is preserved:

```
RAG_DEEP_MODEL   — quality model  → ask command, final research synthesis
RAG_FAST_MODEL  — medium model   → per-sub-question answers, intermediate synthesis
RAG_TINY_MODEL  — 3B model       → orchestration: plan_subquestions, reflect
```

## Model Assignment per Pipeline Step

| Step | Model | Why |
|------|-------|-----|
| `plan_subquestions` | `tiny_model` | Structured list output — just decomposes a question into N strings. No reasoning needed. |
| `reflect` | `tiny_model` | Binary classification + optional list. Either outputs `SUFFICIENT` or a short list of follow-up questions. Trivially simple. |
| `generate_answer` per sub-question (stream=False) | `fast_model` | Reads retrieved context, forms a partial answer. Internal use only — feeds synthesis. |
| `synthesize` intermediate (stream=False) | `fast_model` | Combines findings for the reflection step. Not shown to user. |
| `synthesize` **final** (stream=True) | `llm_model` | User-visible streamed answer. Needs full quality. |

## Why a 3B Model Fits Orchestration

`plan_subquestions` and `reflect` are pure **structured output tasks**:

- `plan_subquestions`: input → list of N strings. Output pattern is fully predictable.
- `reflect`: input → `"SUFFICIENT"` or a list of follow-up questions. Essentially binary classification + list.

Neither produces user-visible output, so quality degradation is invisible. A 3B model (e.g. `qwen2.5:3b`) handles both reliably and fires much faster than a 7B+ model.

The per-sub-question `generate_answer` and intermediate `synthesize` sit in the middle tier — they involve reading context and forming coherent partial answers that feed the final synthesis, so they benefit from slightly more capable models than 3B.

## Configuration

| Env var | Default | Used for |
|---------|---------|----------|
| `RAG_DEEP_MODEL` | `mistral:7b` | `ask`, final synthesis |
| `RAG_FAST_MODEL` | `RAG_DEEP_MODEL` | per-sub-question answers, intermediate synthesis |
| `RAG_TINY_MODEL` | `RAG_FAST_MODEL` | plan sub-questions, reflect |

CLI flags mirror each var: `--model`, `--fast-model`, `--tiny-model` on the `research` command.

## Example

```bash
# Unchanged behavior (all steps use same model)
pedro research "what is entropy?"

# Full three-tier setup
RAG_DEEP_MODEL=llama3:70b RAG_FAST_MODEL=mistral:7b RAG_TINY_MODEL=qwen2.5:3b \
  pedro research "what is entropy?"

# Explicit flags
pedro research "what is entropy?" \
  --model llama3:70b \
  --fast-model mistral:7b \
  --tiny-model qwen2.5:3b
```

## Revision: Reduced to Two Tiers

The three-tier split was later found to be unnecessary complexity. `TINY_MODEL`
and `FAST_MODEL` handled structurally different tasks (orchestration vs. partial
synthesis) but in practice the same model fits both roles well. Having a third
env var and CLI flag added cognitive overhead without measurable benefit.

**Change:** `TINY_MODEL` was removed. Its tasks (plan_subquestions, reflect,
translate_question) now use `FAST_MODEL`. `DEEP_MODEL` was renamed to `LLM_MODEL`
to be more descriptive and less tied to the tier hierarchy.

Current model configuration:

| Env var       | Default       | Used for |
|---------------|---------------|----------|
| `LLM_MODEL`   | `mistral:7b`  | `ask`, final synthesis, model's take |
| `FAST_MODEL`  | `LLM_MODEL`   | sub-question answers, planning, reflection, citations, translation |

CLI flags: `--model`, `--fast-model`
