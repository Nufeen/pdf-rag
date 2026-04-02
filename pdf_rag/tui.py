import chromadb
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.reactive import reactive
from textual.widgets import Footer, Input, RichLog, Static
from textual.worker import Worker

from .config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBED_MODEL,
    FAST_MODEL,
    LLM_MODEL,
    OLLAMA_BASE_URL,
    RESEARCH_DEPTH,
    RESEARCH_N_SUBQUESTIONS,
    TINY_MODEL,
    TOP_K,
)
from .llm import generate_answer
from .researcher import research
from .retriever import query

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

#stream {
    height: auto;
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
        Binding("ctrl+c", "quit", "Quit", show=True),
    ]

    mode: reactive[str] = reactive("ask")

    def compose(self) -> ComposeResult:
        yield RichLog(id="output", wrap=True, markup=True, highlight=False)
        yield Static("", id="stream")
        yield Static("", id="status")
        yield Input(placeholder="Ask a question… (Tab to switch mode)", id="input")
        yield Footer()

    def on_mount(self) -> None:
        self._update_status()
        self.query_one(Input).focus()

    def watch_mode(self, _mode: str) -> None:
        self._update_status()

    def _update_status(self) -> None:
        mode = self.mode
        if mode == "ask":
            models = f"model: [dim]{LLM_MODEL}[/dim]"
        else:
            models = f"deep: [dim]{LLM_MODEL}[/dim]  fast: [dim]{FAST_MODEL}[/dim]  tiny: [dim]{TINY_MODEL}[/dim]"
        self.query_one("#status", Static).update(
            f" mode: [bold]{mode}[/bold]  │  {models}"
        )

    def action_toggle_mode(self) -> None:
        log = self.query_one("#output", RichLog)
        idx = _MODES.index(self.mode)
        self.mode = _MODES[(idx + 1) % len(_MODES)]
        log.write(f"\n[bold cyan]>[/bold cyan] Mode changed to [bold cyan]{self.mode}[/bold cyan]\n")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        question = event.value.strip()
        if not question:
            return
        event.input.clear()
        log = self.query_one("#output", RichLog)
        log.write(f"\n[bold cyan]>[/bold cyan] {question}\n")
        if self.mode == "ask":
            self.run_worker(lambda: self._do_ask(question), thread=True)
        else:
            self.run_worker(lambda: self._do_research(question), thread=True)

    # Workers run in threads (Ollama SDK is synchronous)

    def _do_ask(self, question: str) -> None:
        log = self.query_one("#output", RichLog)
        stream = self.query_one("#stream", Static)
        buf: list[str] = []

        def emit(token: str) -> None:
            buf.append(token)
            self.call_from_thread(stream.update, "".join(buf))

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

        generate_answer(
            question=question,
            chunks=chunks,
            base_url=OLLAMA_BASE_URL,
            llm_model=LLM_MODEL,
            on_token=emit,
        )
        answer = "".join(buf)
        self.call_from_thread(log.write, answer)
        self.call_from_thread(log.write, f"[dim]model: {LLM_MODEL}[/dim]")
        self.call_from_thread(stream.update, "")

    def _do_research(self, question: str) -> None:
        log = self.query_one("#output", RichLog)
        stream = self.query_one("#stream", Static)
        buf: list[str] = []

        def emit(token: str) -> None:
            buf.append(token)
            self.call_from_thread(stream.update, "".join(buf))

        def log_fn(msg: str) -> None:
            self.call_from_thread(log.write, msg)

        collection = _open_collection()
        research(
            question=question,
            collection=collection,
            base_url=OLLAMA_BASE_URL,
            llm_model=LLM_MODEL,
            fast_model=FAST_MODEL,
            tiny_model=TINY_MODEL,
            embed_model=EMBED_MODEL,
            depth=RESEARCH_DEPTH,
            n_subquestions=RESEARCH_N_SUBQUESTIONS,
            top_k=TOP_K,
            log_fn=log_fn,
            on_token=emit,
        )
        answer = "".join(buf)
        self.call_from_thread(log.write, answer)
        self.call_from_thread(log.write, f"[dim]model: {LLM_MODEL}[/dim]")
        self.call_from_thread(stream.update, "")


def _open_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
