# ADR-0003: Session Logging and History Navigation

## Context

The TUI stored only the most recent answer (`_last_answer`) with no persistence and no way to revisit earlier questions. Users need:

1. Q&A history navigable with Up/Down arrows in the input field (like shell history)
2. Session logs persisted to disk for later reference and for copying full sessions
3. Ctrl+A to copy the entire current session to clipboard

## Decision

Add a `SessionLog` class that writes one JSONL file per TUI session to `~/.pdf-rag/sessions/`. Keep an in-memory `_history` list that drives Up/Down arrow navigation in the input.

## Session Log Format — JSONL

One file per TUI session, named by timestamp:

```
~/.pdf-rag/sessions/2026-04-02_14-30-00.jsonl
```

Each line is one completed Q&A:

```json
{"ts": "2026-04-02T14:30:00", "mode": "ask", "question": "what is entropy?", "answer": "Entropy is..."}
```

JSONL is chosen because it is easy to append incrementally, parse line by line, and process with standard tools. Each entry is written immediately after the answer stream completes — if the session crashes mid-question, earlier entries are safe.

## History Navigation

`_history: list[dict]` grows as answers complete. `_history_idx: int` tracks cursor position (`-1` = at the new-input end, same as bash `$HISTCMD`).

- **Up** → fill input with the previous question
- **Down** → fill input with the next question, or clear to empty at the end

Textual's `Input` widget does not use Up/Down for anything, so these bindings are safe to intercept with `priority=True`.

## Copy Keys

| Key | Action |
|-----|--------|
| Ctrl+C | Copy last answer (existing) |
| Ctrl+A | Copy full session as formatted plain text |

Full session format:
```
[ask] what is entropy?
Entropy is a measure of disorder...

[research] compare symbolic and connectionist AI
Symbolic AI represents knowledge explicitly...
```

## Architecture

### New module: `pdf_rag/session_log.py`

```python
class SessionLog:
    def __init__(self, sessions_dir: Path) -> None:
        sessions_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.path = sessions_dir / f"{ts}.jsonl"

    def append(self, mode: str, question: str, answer: str) -> None:
        entry = {"ts": datetime.now().isoformat(timespec="seconds"),
                 "mode": mode, "question": question, "answer": answer}
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
```

File I/O is called from worker threads — safe because it doesn't touch Textual's UI layer. List append is GIL-protected.

### Config

```python
SESSIONS_PATH = Path(os.getenv("RAG_SESSIONS_PATH", str(Path.home() / ".pdf-rag" / "sessions")))
```

### TUI changes (`pdf_rag/tui.py`)

New instance fields:
```python
_session_log: SessionLog      # initialised in on_mount
_history: list[dict] = []     # {"mode", "question", "answer"}
_history_idx: int = -1        # -1 = at new-input end
```

New bindings: `up` / `down` (priority, not shown in footer), `ctrl+a`.

After each completed answer in both `_do_ask` and `_do_research`:
```python
self._history.append({"mode": self.mode, "question": question, "answer": answer})
self._history_idx = -1
self._session_log.append(self.mode, question, answer)
```

## Files Changed

| File | Change |
|------|--------|
| `pdf_rag/config.py` | Add `SESSIONS_PATH` env var |
| `pdf_rag/session_log.py` | New — `SessionLog` class |
| `pdf_rag/tui.py` | History fields, Up/Down bindings, Ctrl+A, log append after each answer |
