from unittest.mock import patch

from pdf_rag.researcher import _parse_questions, reflect
from tests.conftest import OLLAMA_URL


# ── _parse_questions ──────────────────────────────────────────────────────────

def test_parse_single_question():
    result = _parse_questions("What is entropy?")
    assert result == ["What is entropy?"]


def test_parse_multiple_questions_by_question_mark():
    result = _parse_questions("What is entropy? What is information?")
    assert result == ["What is entropy?", "What is information?"]


def test_parse_strips_whitespace_around_each_question():
    result = _parse_questions("What is entropy?  What is information?")
    assert all(q == q.strip() for q in result)
    assert "What is information?" in result


def test_parse_multiple_questions_by_newline():
    result = _parse_questions("What is entropy?\nWhat is information?")
    assert "What is entropy?" in result
    assert "What is information?" in result


def test_parse_empty_string_returns_empty_list():
    assert _parse_questions("") == []


def test_parse_trailing_question_mark_only():
    # A line that is just "?" should not produce a question
    result = _parse_questions("?")
    assert result == []


def test_parse_mixed_newline_and_question_mark():
    result = _parse_questions("Q1?\nQ2? Q3?")
    assert len(result) == 3
    assert "Q1?" in result
    assert "Q2?" in result
    assert "Q3?" in result


# ── reflect ───────────────────────────────────────────────────────────────────

def _make_chat(content: str):
    def _chat(self, model, messages, stream=False, **kwargs):
        return {"message": {"content": content}}
    return _chat


def test_reflect_sufficient_uppercase_returns_none():
    with patch("ollama.Client.chat", _make_chat("SUFFICIENT")):
        result = reflect("Q", "A", __import__("ollama").Client(host=OLLAMA_URL), "m")
    assert result is None


def test_reflect_sufficient_mixed_case_returns_none():
    with patch("ollama.Client.chat", _make_chat("Sufficient — the answer is complete.")):
        result = reflect("Q", "A", __import__("ollama").Client(host=OLLAMA_URL), "m")
    assert result is None


def test_reflect_returns_followup_list_when_not_sufficient():
    content = "What about entropy in quantum systems?\nHow does temperature affect entropy?"
    with patch("ollama.Client.chat", _make_chat(content)):
        result = reflect("Q", "A", __import__("ollama").Client(host=OLLAMA_URL), "m")
    assert isinstance(result, list)
    assert len(result) == 2


def test_reflect_empty_response_returns_none():
    with patch("ollama.Client.chat", _make_chat("")):
        result = reflect("Q", "A", __import__("ollama").Client(host=OLLAMA_URL), "m")
    assert result is None
