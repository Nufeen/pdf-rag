# ADR-0001: Interactive terminal mode

## Context

The goal is to drop the user into an interactive shell inspired by opencode/claude: a scrollable output area, a status bar showing the active mode, and a text input at the bottom. All existing CLI commands (`pedro ask`, `pedro research`, `pedro index`) must continue to work unchanged.

## Decision

Add an interactive TUI launched by running `pedro` with no subcommand. The user types questions, gets streamed answers, and toggles between `ask` and `research` modes with Tab.

## Framework: Textual

Textual (by Textualize, built on Rich) was chosen over prompt_toolkit, urwid, and blessed:

- `RichLog` widget handles token-by-token streaming with no full redraws
- `@work(thread=True)` runs sync Ollama SDK calls in background threads
- `call_from_thread()` safely pipes tokens to the UI from worker threads
- Full async/reactive system — status bar updates reactively on mode change
- 120 FPS delta rendering via Rich's segment trees

prompt_toolkit is oriented toward REPL/input, not streaming output. urwid has no native async. blessed/curses require managing everything from primitives with no layout system.

https://pypi.org/project/textual/

Possible alternatives https://github.com/shadawck/awesome-cli-frameworks?tab=readme-ov-file#python

## Layout

```
┌───────────────────────────────────────────────────────────┐
│                                                           │
│  RichLog — scrollable Q&A history + streaming tokens      │
│  (fills available height)                                 │
│                                                           │
├───────────────────────────────────────────────────────────┤
│  STATUS: mode: ask │ model: mistral:7b                    │
├───────────────────────────────────────────────────────────┤
│  > [Input field]                                          │
└───────────────────────────────────────────────────────────┘
  [Tab: toggle mode]  [Ctrl+C: quit]   ← Footer keybindings
```

## Mode Switching

- **Tab** cycles `ask → research → ask`
- Status bar (`Static` widget) updates reactively via Textual's `watch_mode()`
- Both modes share the same `RichLog` history within a session

## Plan

### Callback refactor (backward-compatible)

`generate_answer()` and `synthesize()` previously called `print(token, ...)` directly. An optional `on_token: Callable[[str], None] | None = None` parameter was added to both:

- `None` (default) → falls back to `print(token, end="", flush=True)` — CLI behavior unchanged
- TUI → passes a lambda that routes each token to `RichLog` via `call_from_thread`

`research()` received the same treatment via two callbacks:

- `on_token` — for streaming final answer tokens
- `log_fn` — for step/info/ok status messages (replaces `click.echo` helpers in TUI context)

### New file: `pdf_rag/tui.py`

`PedroApp(App)` with:

- `RichLog` (scrollable history, fills available height)
- `Static` status bar (1 line, dark background)
- `Input` field (3 lines, focused on mount)
- `Footer` (shows Tab / Ctrl+C bindings)
- `mode` reactive property drives status bar updates
- `_do_ask` / `_do_research` worker methods run in threads, pipe output back via `call_from_thread`

### Entry point

```python
@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Local RAG search tool for PDF books."""
    if ctx.invoked_subcommand is None:
        from .tui import PedroApp
        PedroApp().run()
```

### Dependency

`textual>=0.60.0` added to `[project.dependencies]` in `pyproject.toml`.
