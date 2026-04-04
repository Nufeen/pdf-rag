import chromadb
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.reactive import reactive
from textual.widgets import Footer, Input, RichLog, Static
from textual.worker import Worker, get_current_worker

from ..config import (
    DB_PATH,
    COLLECTION_NAME,
    EMBED_MODEL,
    FAST_MODEL,
    DEEP_MODEL,
    OLLAMA_BASE_URL,
    RESEARCH_DEPTH,
    RESEARCH_N_SUBQUESTIONS,
    SEARCH_LANGUAGES,
    SESSIONS_PATH,
    TINY_MODEL,
    TOP_K,
    TRANSLATE_MODEL,
)
from ..llm import generate_answer
from ..researcher import research
from ..retriever import query
from ..session_log import SessionLog
from .welcome import write_welcome

_MODES = ("ask", "research")

CSS = """
Screen {
    layout: vertical;
}

RichLog {
    height: 1fr;
    border: none;
    padding: 0 1;
}

#streaming {
    height: auto;
    max-height: 50vh;
    padding: 0 1;
}

#status {
    height: 1;
    background: $primary-darken-2;
    color: $text;
    padding: 0 1;
}

Input {
    height: 3;
    border: tall $primary;
}
"""


class PedroApp(App):
    CSS = CSS
    BINDINGS = [
        Binding("tab", "toggle_mode", "Toggle mode", show=True, priority=True),
        Binding("up", "history_prev", "Prev question", show=False, priority=True),
        Binding("down", "history_next", "Next question", show=False, priority=True),
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("ctrl+y", "copy_answer", "Copy answer", show=True),
        Binding("ctrl+a", "copy_session", "Copy session", show=True),
        Binding("ctrl+c", "quit", "Quit", show=True),
    ]

    mode: reactive[str] = reactive("ask")
    _current_worker: Worker | None = None
    _last_answer: str = ""
    _history: list[dict] = []
    _history_idx: int = -1

    def compose(self) -> ComposeResult:
        yield RichLog(id="output", wrap=True, markup=True, highlight=False)
        yield Static("", id="streaming")
        yield Static("", id="status")
        yield Input(placeholder="Ask a question… (Tab to switch mode)", id="input")
        yield Footer()

    def on_mount(self) -> None:
        self._session_log = SessionLog(SESSIONS_PATH)
        self._history = SessionLog.load_latest(SESSIONS_PATH)
        self._history_idx = -1
        write_welcome(self.query_one("#output", RichLog), len(self._history))
        self._update_status()
        self.query_one(Input).focus()

    def watch_mode(self, _mode: str) -> None:
        self._update_status()

    def _update_status(self) -> None:
        mode = self.mode
        if mode == "ask":
            models = f"model: [dim]{DEEP_MODEL}[/dim]"
        else:
            models = f"deep: [dim]{DEEP_MODEL}[/dim]  fast: [dim]{FAST_MODEL}[/dim]  tiny: [dim]{TINY_MODEL}[/dim]"
        self.query_one("#status", Static).update(
            f" mode: [bold]{mode}[/bold]  │  {models}"
        )

    def _set_streaming(self, text: str) -> None:
        self.query_one("#streaming", Static).update(text)

    def action_toggle_mode(self) -> None:
        log = self.query_one("#output", RichLog)
        idx = _MODES.index(self.mode)
        self.mode = _MODES[(idx + 1) % len(_MODES)]
        log.write(f"\n[bold cyan]>[/bold cyan] Mode changed to [bold cyan]{self.mode}[/bold cyan]\n")

    def action_history_prev(self) -> None:
        if not self._history:
            return
        self._history_idx = max(
            0,
            len(self._history) - 1 if self._history_idx == -1 else self._history_idx - 1,
        )
        self.query_one(Input).value = self._history[self._history_idx]["question"]

    def action_history_next(self) -> None:
        if self._history_idx == -1:
            return
        self._history_idx += 1
        inp = self.query_one(Input)
        if self._history_idx >= len(self._history):
            self._history_idx = -1
            inp.value = ""
        else:
            inp.value = self._history[self._history_idx]["question"]

    def action_copy_answer(self) -> None:
        if self._last_answer:
            self.copy_to_clipboard(self._last_answer)
            self.notify("Answer copied to clipboard")

    def action_copy_session(self) -> None:
        if not self._history:
            self.notify("No history yet", severity="warning")
            return
        parts = [f"[{e['mode']}] {e['question']}\n{e['answer']}" for e in self._history]
        self.copy_to_clipboard("\n\n".join(parts))
        self.notify(f"Session copied ({len(self._history)} answer(s))")

    def action_cancel(self) -> None:
        if self._current_worker and self._current_worker.is_running:
            self._current_worker.cancel()
            self._set_streaming("")
            self.query_one("#output", RichLog).write("\n[yellow]Cancelled.[/yellow]\n")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        question = event.value.strip()
        if not question:
            return
        event.input.clear()
        log = self.query_one("#output", RichLog)
        log.write(f"\n[bold cyan]>[/bold cyan] {question}\n")
        if self.mode == "ask":
            self._current_worker = self.run_worker(lambda: self._do_ask(question), thread=True)
        else:
            self._current_worker = self.run_worker(lambda: self._do_research(question), thread=True)

    # Workers run in threads (Ollama SDK is synchronous)

    def _do_ask(self, question: str) -> None:
        log = self.query_one("#output", RichLog)
        buf: list[str] = []

        def emit(token: str) -> None:
            if get_current_worker().is_cancelled:
                raise InterruptedError
            buf.append(token)
            self.call_from_thread(self._set_streaming, "".join(buf))

        def log_line(msg: str) -> None:
            self.call_from_thread(log.write, msg)

        collection = _open_collection()
        chunks = query(
            question=question,
            collection=collection,
            embed_model=EMBED_MODEL,
            base_url=OLLAMA_BASE_URL,
            top_k=TOP_K,
        )
        if not chunks:
            log_line("[yellow]No relevant content found. Have you indexed your PDF folder?[/yellow]")
            return

        log_line("\n[dim]Retrieved sources:[/dim]")
        for c in chunks:
            log_line(f"[dim]  - {c['source_file']} (page {c['page_num']}, score: {c['score']:.3f})[/dim]")
        log_line("")

        try:
            generate_answer(
                question=question,
                chunks=chunks,
                base_url=OLLAMA_BASE_URL,
                llm_model=DEEP_MODEL,
                on_token=emit,
            )
            answer = "".join(buf)
            self._last_answer = answer
            self._history.append({"mode": self.mode, "question": question, "answer": answer})
            self._history_idx = -1
            self._session_log.append(self.mode, question, answer)
            self.call_from_thread(self._set_streaming, "")
            self.call_from_thread(log.write, answer)
        except InterruptedError:
            self.call_from_thread(self._set_streaming, "")

    def _do_research(self, question: str) -> None:
        log = self.query_one("#output", RichLog)
        buf: list[str] = []

        def emit(token: str) -> None:
            if get_current_worker().is_cancelled:
                raise InterruptedError
            buf.append(token)
            self.call_from_thread(self._set_streaming, "".join(buf))

        def log_fn(msg: str) -> None:
            self.call_from_thread(log.write, msg)

        collection = _open_collection()
        try:
            pages_by_file = research(
                question=question,
                collection=collection,
                base_url=OLLAMA_BASE_URL,
                llm_model=DEEP_MODEL,
                fast_model=FAST_MODEL,
                tiny_model=TINY_MODEL,
                embed_model=EMBED_MODEL,
                depth=RESEARCH_DEPTH,
                n_subquestions=RESEARCH_N_SUBQUESTIONS,
                top_k=TOP_K,
                languages=SEARCH_LANGUAGES,
                translate_model=TRANSLATE_MODEL,
                log_fn=log_fn,
                on_token=emit,
            )
            answer = "".join(buf)
            self._last_answer = answer
            self._history.append({"mode": self.mode, "question": question, "answer": answer})
            self._history_idx = -1
            self._session_log.append(self.mode, question, answer)
            self.call_from_thread(self._set_streaming, "")
            self.call_from_thread(log.write, answer)
            if pages_by_file:
                self.call_from_thread(log.write, "\n[bold]Sources:[/bold]")
                for filename in sorted(pages_by_file):
                    pages = ", ".join(str(p) for p in sorted(pages_by_file[filename]))
                    self.call_from_thread(log.write, f"[dim]  {filename} — pages {pages}[/dim]")
        except InterruptedError:
            self.call_from_thread(self._set_streaming, "")


def _open_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
