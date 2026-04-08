from unittest.mock import patch

from click.testing import CliRunner

from pdf_rag.cli import cli
from tests.conftest import OLLAMA_URL


def run_research(args, seeded_col):
    runner = CliRunner()
    with patch("chromadb.PersistentClient") as mock_chroma:
        mock_chroma.return_value.get_or_create_collection.return_value = seeded_col
        return runner.invoke(cli, ["research"] + args + ["--ollama-url", OLLAMA_URL])


# ── Stage: Plan ────────────────────────────────────────────────────────────────

def test_plan_stage_present(seeded_collection, mock_embed, mock_ollama_chat):
    result = run_research(
        ["What is entropy?", "--depth", "1", "--sub-questions", "2"],
        seeded_collection,
    )
    assert result.exit_code == 0
    assert "Planning sub-questions" in result.output


def test_plan_stage_lists_user_question(seeded_collection, mock_embed, mock_ollama_chat):
    """Original question always appears as first planned sub-question."""
    result = run_research(
        ["What is entropy?", "--depth", "1", "--sub-questions", "2"],
        seeded_collection,
    )
    assert "What is entropy?" in result.output


def test_plan_stage_multi_question_input(seeded_collection, mock_embed, mock_ollama_chat):
    """Both ?-separated user questions appear in the planning output."""
    result = run_research(
        ["What is entropy? What is information?", "--depth", "1", "--sub-questions", "3"],
        seeded_collection,
    )
    assert result.exit_code == 0
    assert "What is entropy?" in result.output
    assert "What is information?" in result.output


# ── Stage: Execute ─────────────────────────────────────────────────────────────

def test_execute_stage_present(seeded_collection, mock_embed, mock_ollama_chat):
    result = run_research(
        ["What is entropy?", "--depth", "1", "--sub-questions", "1"],
        seeded_collection,
    )
    assert "Executing" in result.output
    assert "sub-question" in result.output


def test_execute_stage_shows_chunk_count(seeded_collection, mock_embed, mock_ollama_chat):
    result = run_research(
        ["What is entropy?", "--depth", "1", "--sub-questions", "1"],
        seeded_collection,
    )
    assert "retrieved" in result.output
    assert "chunks from" in result.output


def test_execute_stage_shows_source_files(seeded_collection, mock_embed, mock_ollama_chat):
    result = run_research(
        ["What is entropy?", "--depth", "1", "--sub-questions", "1"],
        seeded_collection,
    )
    assert "book.pdf" in result.output


# ── Stage: Reflect ─────────────────────────────────────────────────────────────

def test_reflect_stage_absent_at_depth_1(seeded_collection, mock_embed, mock_ollama_chat):
    result = run_research(
        ["What is entropy?", "--depth", "1", "--sub-questions", "1"],
        seeded_collection,
    )
    assert "Reflecting" not in result.output


def test_reflect_stage_present_at_depth_2(seeded_collection, mock_embed, mock_ollama_chat):
    result = run_research(
        ["What is entropy?", "--depth", "2", "--sub-questions", "1"],
        seeded_collection,
    )
    assert result.exit_code == 0
    assert "Reflecting" in result.output


# ── Stage: Synthesize ──────────────────────────────────────────────────────────

def test_synthesize_stage_present(seeded_collection, mock_embed, mock_ollama_chat):
    result = run_research(
        ["What is entropy?", "--depth", "1", "--sub-questions", "1"],
        seeded_collection,
    )
    assert "Synthesizing final answer" in result.output


def test_synthesize_produces_output(seeded_collection, mock_embed, mock_ollama_chat):
    result = run_research(
        ["What is entropy?", "--depth", "1", "--sub-questions", "1"],
        seeded_collection,
    )
    assert "Response from" in result.output


# ── Stage: Sources ─────────────────────────────────────────────────────────────

def test_sources_stage_present(seeded_collection, mock_embed, mock_ollama_chat):
    result = run_research(
        ["What is entropy?", "--depth", "1", "--sub-questions", "1"],
        seeded_collection,
    )
    assert "Sources:" in result.output


def test_sources_stage_lists_pages(seeded_collection, mock_embed, mock_ollama_chat):
    result = run_research(
        ["What is entropy?", "--depth", "1", "--sub-questions", "1"],
        seeded_collection,
    )
    assert "pages" in result.output
