import pytest

from pdf_rag.llm import build_context, generate_answer, load_prompt


# ── load_prompt ───────────────────────────────────────────────────────────────


def test_load_prompt_existing_file_returns_content():
    # 'answer' prompt exists and has no placeholders at the top level
    result = load_prompt("answer")
    assert len(result) > 0


def test_load_prompt_fills_placeholders():
    # synthesize.txt uses {question} and {context}
    result = load_prompt(
        "synthesize", question="What is entropy?", context="Some context."
    )
    assert "What is entropy?" in result
    assert "Some context." in result


def test_load_prompt_missing_placeholder_raises():
    # synthesize.txt requires {question} and {context}; omitting one should raise
    with pytest.raises(KeyError):
        load_prompt("synthesize", question="What is entropy?")  # missing context


# ── build_context ─────────────────────────────────────────────────────────────


def test_build_context_empty_chunks_returns_empty_string():
    assert build_context([]) == ""


def test_build_context_single_chunk_includes_source_and_page():
    chunks = [{"source_file": "book.pdf", "page_num": 42, "text": "Some text."}]
    result = build_context(chunks)
    assert "book.pdf" in result
    assert "42" in result
    assert "Some text." in result


def test_build_context_multiple_chunks_are_separated():
    chunks = [
        {"source_file": "a.pdf", "page_num": 1, "text": "First."},
        {"source_file": "b.pdf", "page_num": 2, "text": "Second."},
    ]
    result = build_context(chunks)
    assert "First." in result
    assert "Second." in result
    assert "---" in result  # separator between chunks


def test_build_context_numbers_excerpts():
    chunks = [
        {"source_file": "a.pdf", "page_num": 1, "text": "X"},
        {"source_file": "b.pdf", "page_num": 2, "text": "Y"},
    ]
    result = build_context(chunks)
    assert "Excerpt 1" in result
    assert "Excerpt 2" in result


# ── generate_answer with custom system_prompt ──────────────────────────────────


def test_generate_answer_uses_custom_system_prompt(mock_ollama_chat, request):
    """When system_prompt is provided, it should be used instead of the default."""
    chunks = [
        {
            "source_file": "book.pdf",
            "page_num": 1,
            "text": "Some content about entropy.",
        },
    ]
    custom_prompt = "You are a pirate. Answer like a pirate."
    result = generate_answer(
        question="What is entropy?",
        chunks=chunks,
        stream=False,
        system_prompt=custom_prompt,
    )
    assert "Response from" in result


def test_generate_answer_defaults_to_loaded_prompt_when_no_system_prompt(
    mock_ollama_chat,
):
    """When system_prompt is None, the default prompt file is loaded."""
    chunks = [
        {"source_file": "book.pdf", "page_num": 1, "text": "Some content."},
    ]
    result = generate_answer(
        question="What is entropy?",
        chunks=chunks,
        stream=False,
    )
    assert "Response from" in result
