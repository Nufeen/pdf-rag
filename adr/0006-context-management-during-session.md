# ADR-0006: Context Management During Session

## Context

Each query in Pedro is stateless: chunks are retrieved fresh from the vector DB and no conversation history is passed to the LLM. The session logging system (ADR-0003) stores Q&A pairs for user-side history navigation, but the LLM has no awareness of previous turns.

This limits multi-turn conversations where follow-up questions reference earlier answers (e.g., "What about the opposite approach?" or "Can you elaborate on point 2?").

## Problem Statement

How do we provide conversational context to the LLM without:
1. Exceeding context window limits
2. Suffering from "lost in the middle" degradation
3. Adding significant latency per turn

## Decision

**LLM-summarized rolling context.** After each answer, `FAST_MODEL` compresses the Q&A into a 3–4 sentence running summary. On the next turn the summary is prepended to the retrieved chunks before they reach the LLM.

**Rationale:**
- Bounded token usage — summary length is fixed regardless of session length
- No context window issues
- `FAST_MODEL` is fast enough that the extra call adds ~1–2s
- Fits Pedro's philosophy of simplicity; can be extended with hybrid retrieval later if needed

Rejected alternatives: full conversation history (token explosion, lost-in-the-middle), sliding window (arbitrary cutoff, still grows), hybrid summary + embedded history retrieval (complexity not justified yet).

## Design

### `pdf_rag/context_manager.py` — `SessionContext`

```python
@dataclass
class SessionContext:
    summary: str = ""
    turn_count: int = 0

    def update(self, question, answer, client, model=FAST_MODEL) -> str:
        # calls load_prompt("update_session_summary", ...), updates self.summary
        ...

    def enrich_context(self, chunks_context: str) -> str:
        # prepends "Session context: {summary}\n\n" when summary is non-empty
        ...

    def reset(self) -> None: ...
```

`generate_answer()` in `llm.py` receives an optional `session_context: str = ""`
and prepends it to the chunks context before building the user message.
`run_ask()` and `research()` pass `session_ctx.summary` there and call
`session_ctx.update()` after the answer is produced.

### `prompts/update_session_summary.txt`

Placeholders: `{prev_summary}`, `{question}`, `{answer}`.
First turn passes `"(none)"` for `prev_summary`.

### Wiring

| File | Change |
|------|--------|
| `pdf_rag/llm.py` | `generate_answer()` gains `session_context: str = ""`; prepends to chunks context when non-empty |
| `pdf_rag/researcher.py` | `run_ask()` and `research()` accept `session_ctx: SessionContext \| None`; pass summary to `generate_answer()`; call `update()` after answer |
| `pdf_rag/tui/app.py` | `SessionContext()` initialized in `on_mount()`; passed to both workers (local mode only — server mode is stateless) |

## Future Considerations

- **User-controlled reset** — `/reset` TUI command to clear session context
- **Hybrid retrieval** — embed past Q&A and retrieve relevant turns on demand
- **Export** — include session summary in `/pdf` export
