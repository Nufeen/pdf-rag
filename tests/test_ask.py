from unittest.mock import patch

from click.testing import CliRunner

from pdf_rag.cli import cli
from tests.conftest import OLLAMA_URL


def run_ask(args, seeded_col):
    runner = CliRunner()
    with patch("chromadb.PersistentClient") as mock_chroma:
        mock_chroma.return_value.get_or_create_collection.return_value = seeded_col
        return runner.invoke(cli, ["ask"] + args + ["--ollama-url", OLLAMA_URL])


def test_ask_basic(seeded_collection, mock_embed, mock_ollama_chat):
    result = run_ask(["What is entropy?"], seeded_collection)
    assert result.exit_code == 0
    assert "Response from" in result.output


def test_ask_shows_sources(seeded_collection, mock_embed, mock_ollama_chat):
    result = run_ask(["What is entropy?"], seeded_collection)
    assert result.exit_code == 0
    assert "book.pdf" in result.output


def test_ask_no_sources(seeded_collection, mock_embed, mock_ollama_chat):
    result = run_ask(["What is entropy?", "--no-sources"], seeded_collection)
    assert result.exit_code == 0
    assert "book.pdf" not in result.output


def test_ask_top_k(seeded_collection, mock_embed, mock_ollama_chat):
    result = run_ask(["What is entropy?", "--top-k", "1"], seeded_collection)
    assert result.exit_code == 0
    assert "Response from" in result.output
