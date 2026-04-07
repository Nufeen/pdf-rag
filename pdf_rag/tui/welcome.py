from textual.widgets import RichLog

from ..config import FAST_MODEL, LLM_MODEL, SEARCH_LANGUAGES

_BANNER = """\
[bold green]🌵 pedro[/bold green] — local RAG for your PDF library

  [dim]ask[/dim]       single-pass retrieval + answer
  [dim]research[/dim]  multi-step reasoning with sub-questions and reflection

  [bold]Tab[/bold]        toggle ask / research mode
  [bold]↑ / ↓[/bold]      navigate question history
  [bold]Escape[/bold]     cancel in-progress query
  [bold]Ctrl+Y[/bold]     copy last answer to clipboard
  [bold]Ctrl+A[/bold]     copy full session to clipboard
  [bold]Ctrl+C[/bold]     quit\
"""


def write_welcome(log: RichLog, history_count: int = 0) -> None:
    log.write(_BANNER)
    if SEARCH_LANGUAGES:
        langs = ", ".join(SEARCH_LANGUAGES)
        log.write(f"\n  [dim]multilingual search enabled: {langs}[/dim]")
    log.write(
        f"\n  [dim]models: {LLM_MODEL}"
        + (f"  ·  {FAST_MODEL}" if FAST_MODEL != LLM_MODEL else "")
        + "[/dim]"
    )
    if history_count:
        log.write(f"\n  [dim]↑ {history_count} question(s) restored from previous session[/dim]")
    log.write("")
