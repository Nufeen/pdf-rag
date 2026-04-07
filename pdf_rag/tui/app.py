from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.reactive import reactive
from textual.widgets import Footer, Input, RichLog, Static
from textual.worker import Worker, get_current_worker

from ..config import (
    DB_PATH,
    EMBED_MODEL,
    FAST_MODEL,
    LLM_MODEL,
    OLLAMA_BASE_URL,
    PDF_EXPORT_PATH,
    RESEARCH_DEPTH,
    RESEARCH_N_SUBQUESTIONS,
    SEARCH_LANGUAGES,
    SESSIONS_PATH,
    SERVER_URL,
    TOP_K,
    TRANSLATE_MODEL,
)
from ..context_manager import SessionContext
from ..researcher import research, run_ask
from ..session_log import SessionLog
from .pdf_export import export_to_pdf
from .stream_client import stream_ask, stream_research
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
        Binding("shift+up", "top_k_up", "top_k +1", show=True, priority=True),
        Binding("shift+down", "top_k_down", "top_k -1", show=True, priority=True),
    ]

    mode: reactive[str] = reactive("ask")
    top_k: reactive[int] = reactive(TOP_K)
    _current_worker: Worker | None = None
    _last_answer: str = ""
    _history: list[dict] = []
    _history_idx: int = -1

    def compose(self) -> ComposeResult:
        yield RichLog(id="output", wrap=True, markup=True, highlight=False)
        yield Static("", id="status")
        yield Input(placeholder="Ask a question… (Tab to switch mode)", id="input")
        yield Footer()

    def on_mount(self) -> None:
        self._session_log = SessionLog(SESSIONS_PATH)
        self._history = SessionLog.load_latest(SESSIONS_PATH)
        self._history_idx = -1
        self._session_ctx = SessionContext()
        write_welcome(self.query_one("#output", RichLog), len(self._history))
        self._update_status()
        self.query_one(Input).focus()

    def watch_mode(self, _mode: str) -> None:
        self._update_status()

    def watch_top_k(self, _: int) -> None:
        self._update_status()

    def _update_status(self) -> None:
        mode = self.mode
        if mode == "ask":
            models = f"model: [dim]{LLM_MODEL}[/dim]"
        else:
            models = f"llm: [dim]{LLM_MODEL}[/dim]  fast: [dim]{FAST_MODEL}[/dim]"
        self.query_one("#status", Static).update(
            f" mode: [bold]{mode}[/bold]  │  {models}  │  top_k: [dim]{self.top_k}[/dim]"
        )

    def action_toggle_mode(self) -> None:
        if self._current_worker and self._current_worker.is_running:
            self._current_worker.cancel()
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

    def _action_export_pdf(self) -> None:
        if not self._last_answer:
            self.notify("No answer to export yet", severity="warning")
            return
        try:
            path = export_to_pdf(
                self._history[-1]["question"] if self._history else "",
                self._last_answer,
                PDF_EXPORT_PATH,
            )
            self.notify(f"Saved to {path}")
        except Exception as exc:
            self.notify(f"PDF export failed: {exc}", severity="error")

    def action_top_k_up(self) -> None:
        self.top_k += 1

    def action_top_k_down(self) -> None:
        if self.top_k > 1:
            self.top_k -= 1

    def _reenable_input(self) -> None:
        inp = self.query_one(Input)
        inp.disabled = False
        inp.focus()

    def action_cancel(self) -> None:
        if self._current_worker and self._current_worker.is_running:
            self._current_worker.cancel()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        question = event.value.strip()
        if not question:
            return
        event.input.clear()
        if question.startswith("/"):
            if question == "/copy":
                self.action_copy_answer()
            elif question == "/pdf":
                self._action_export_pdf()
            else:
                log = self.query_one("#output", RichLog)
                log.write(f"\n[red]Command not found: {question}[/red]\n")
            return
        event.input.disabled = True
        log = self.query_one("#output", RichLog)
        log.write(f"\n[bold cyan]>[/bold cyan] {question}\n")
        if self.mode == "ask":
            self._current_worker = self.run_worker(lambda: self._do_ask(question), thread=True)
        else:
            self._current_worker = self.run_worker(lambda: self._do_research(question), thread=True)

    # Workers run in threads (Ollama SDK is synchronous)

    def _do_ask(self, question: str) -> None:
        log = self.query_one("#output", RichLog)
        line_buf = ""
        full_buf: list[str] = []

        def emit(token: str) -> None:
            nonlocal line_buf
            if get_current_worker().is_cancelled:
                raise InterruptedError
            full_buf.append(token)
            line_buf += token
            while "\n" in line_buf:
                line, line_buf = line_buf.split("\n", 1)
                self.call_from_thread(log.write, line)

        def log_fn(msg: str) -> None:
            self.call_from_thread(log.write, msg)

        def check() -> None:
            if get_current_worker().is_cancelled:
                raise InterruptedError

        try:
            if SERVER_URL:
                stream_ask(
                    server_url=SERVER_URL,
                    question=question,
                    params={"llm_model": LLM_MODEL, "embed_model": EMBED_MODEL, "top_k": self.top_k},
                    on_token=emit,
                    log_fn=log_fn,
                    check=check,
                )
            else:
                run_ask(
                    question=question,
                    db_path=DB_PATH,
                    base_url=OLLAMA_BASE_URL,
                    llm_model=LLM_MODEL,
                    embed_model=EMBED_MODEL,
                    top_k=self.top_k,
                    log_fn=log_fn,
                    on_token=emit,
                    session_ctx=self._session_ctx,
                )
            if line_buf:
                self.call_from_thread(log.write, line_buf)
            answer = "".join(full_buf)
            if answer:
                self._last_answer = answer
                self._history.append({"mode": self.mode, "question": question, "answer": answer})
                self._history_idx = -1
                self._session_log.append(self.mode, question, answer)
        except InterruptedError:
            self.call_from_thread(log.write, "\n[yellow]Cancelled.[/yellow]\n")
        finally:
            self.call_from_thread(self._reenable_input)

    def _do_research(self, question: str) -> None:
        log = self.query_one("#output", RichLog)
        full_buf: list[str] = []
        line_buf = ""

        def check() -> None:
            if get_current_worker().is_cancelled:
                raise InterruptedError

        def emit(token: str) -> None:
            nonlocal line_buf
            check()
            full_buf.append(token)
            line_buf += token
            while "\n" in line_buf:
                line, line_buf = line_buf.split("\n", 1)
                self.call_from_thread(log.write, line)

        def log_fn(msg: str) -> None:
            self.call_from_thread(log.write, msg)

        try:
            if SERVER_URL:
                stream_research(
                    server_url=SERVER_URL,
                    question=question,
                    params={
                        "llm_model": LLM_MODEL,
                        "fast_model": FAST_MODEL,
                        "embed_model": EMBED_MODEL,
                        "depth": RESEARCH_DEPTH,
                        "n_subquestions": RESEARCH_N_SUBQUESTIONS,
                        "top_k": self.top_k,
                        "languages": SEARCH_LANGUAGES,
                        "translate_model": TRANSLATE_MODEL,
                    },
                    on_token=emit,
                    log_fn=log_fn,
                    check=check,
                )
            else:
                research(
                    question=question,
                    db_path=DB_PATH,
                    base_url=OLLAMA_BASE_URL,
                    llm_model=LLM_MODEL,
                    fast_model=FAST_MODEL,
                    embed_model=EMBED_MODEL,
                    depth=RESEARCH_DEPTH,
                    n_subquestions=RESEARCH_N_SUBQUESTIONS,
                    top_k=self.top_k,
                    languages=SEARCH_LANGUAGES,
                    translate_model=TRANSLATE_MODEL,
                    log_fn=log_fn,
                    on_token=emit,
                    check=check,
                    session_ctx=self._session_ctx,
                )
            if line_buf:
                self.call_from_thread(log.write, line_buf)
            answer = "".join(full_buf)
            self._last_answer = answer
            self._history.append({"mode": self.mode, "question": question, "answer": answer})
            self._history_idx = -1
            self._session_log.append(self.mode, question, answer)
        except InterruptedError:
            self.call_from_thread(log.write, "\n[yellow]Cancelled.[/yellow]\n")
        finally:
            self.call_from_thread(self._reenable_input)

