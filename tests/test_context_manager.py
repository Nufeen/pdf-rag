from unittest.mock import patch

from pdf_rag.context_manager import SessionContext
from tests.conftest import OLLAMA_URL


def test_session_context_initial_state():
    ctx = SessionContext()
    assert ctx.summary == ""
    assert ctx.turn_count == 0


def test_enrich_context_no_summary():
    ctx = SessionContext()
    chunks_ctx = "Retrieved chunks context"
    result = ctx.enrich_context(chunks_ctx)
    assert result == chunks_ctx


def test_enrich_context_with_summary():
    ctx = SessionContext(summary="Previous conversation summary")
    chunks_ctx = "Retrieved chunks context"
    result = ctx.enrich_context(chunks_ctx)
    assert "Session context: Previous conversation summary" in result
    assert chunks_ctx in result


def test_session_context_reset():
    ctx = SessionContext(summary="Some summary", turn_count=5)
    ctx.reset()
    assert ctx.summary == ""
    assert ctx.turn_count == 0


def _make_chat(content: str):
    def _chat(self, model, messages, stream=False, **kwargs):
        return {"message": {"content": content}}
    return _chat


def test_update_sets_summary_and_increments_turn():
    ctx = SessionContext()
    with patch("ollama.Client.chat", _make_chat("Discussed entropy and information theory.")):
        ctx.update("What is entropy?", "Entropy is disorder.", client=__import__("ollama").Client(host=OLLAMA_URL), model="m")
    assert ctx.summary == "Discussed entropy and information theory."
    assert ctx.turn_count == 1


def test_update_uses_previous_summary_in_prompt():
    ctx = SessionContext(summary="Prior: entropy discussed.")
    captured = {}

    def _chat(self, model, messages, stream=False, **kwargs):
        captured["prompt"] = messages[0]["content"]
        return {"message": {"content": "Updated summary."}}

    with patch("ollama.Client.chat", _chat):
        ctx.update("Follow-up?", "More detail.", client=__import__("ollama").Client(host=OLLAMA_URL), model="m")

    assert "Prior: entropy discussed." in captured["prompt"]


def test_update_second_turn_increments_count():
    ctx = SessionContext()
    with patch("ollama.Client.chat", _make_chat("Summary after turn 1.")):
        ctx.update("Q1", "A1", client=__import__("ollama").Client(host=OLLAMA_URL), model="m")
    with patch("ollama.Client.chat", _make_chat("Summary after turn 2.")):
        ctx.update("Q2", "A2", client=__import__("ollama").Client(host=OLLAMA_URL), model="m")
    assert ctx.turn_count == 2
    assert ctx.summary == "Summary after turn 2."
