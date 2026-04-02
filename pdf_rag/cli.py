import click
import chromadb

from .config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBED_MODEL,
    LLM_MODEL,
    OLLAMA_BASE_URL,
    RESEARCH_DEPTH,
    RESEARCH_N_SUBQUESTIONS,
    TOP_K,
)
from .indexer import index_folder
from .llm import generate_answer
from .researcher import research
from .retriever import query


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Local RAG search tool for PDF books."""
    if ctx.invoked_subcommand is None:
        from .tui import PedroApp
        PedroApp().run()


@cli.command()
@click.argument("folder", type=click.Path(exists=True, file_okay=False))
@click.option("--db-path", default=CHROMA_DB_PATH, show_default=True, help="ChromaDB storage path")
@click.option("--embed-model", default=EMBED_MODEL, show_default=True, help="Ollama embedding model")
@click.option("--ollama-url", envvar="OLLAMA_BASE_URL", default=OLLAMA_BASE_URL, show_default=True, help="Ollama base URL")
@click.option("--force", is_flag=True, help="Re-index all files, ignoring existing index")
def index(folder, db_path, embed_model, ollama_url, force):
    """Scan FOLDER and index all PDF files."""
    index_folder(
        folder_path=folder,
        db_path=db_path,
        embed_model=embed_model,
        base_url=ollama_url,
        force=force,
    )


@cli.command()
@click.argument("question")
@click.option("--db-path", default=CHROMA_DB_PATH, show_default=True, help="ChromaDB storage path")
@click.option("--model", default=LLM_MODEL, show_default=True, help="Ollama LLM model")
@click.option("--embed-model", default=EMBED_MODEL, show_default=True, help="Ollama embedding model")
@click.option("--ollama-url", envvar="OLLAMA_BASE_URL", default=OLLAMA_BASE_URL, show_default=True, help="Ollama base URL")
@click.option("--top-k", default=TOP_K, show_default=True, type=int, help="Number of chunks to retrieve")
@click.option("--no-sources", is_flag=True, help="Hide retrieved source list")
def ask(question, db_path, model, embed_model, ollama_url, top_k, no_sources):
    """Ask a QUESTION against the indexed PDF library."""
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    chunks = query(
        question=question,
        collection=collection,
        embed_model=embed_model,
        base_url=ollama_url,
        top_k=top_k,
    )

    if not chunks:
        click.echo("No relevant content found. Have you indexed your PDF folder?")
        return

    if not no_sources:
        click.echo("\nRetrieved sources:")
        for c in chunks:
            click.echo(f"  - {c['source_file']} (page {c['page_num']}, score: {c['score']:.3f})")
        click.echo()

    click.echo("Answer:\n")
    generate_answer(
        question=question,
        chunks=chunks,
        base_url=ollama_url,
        llm_model=model,
    )
    click.echo(click.style(f"model: {model}", fg="bright_black"))


@cli.command("research")
@click.argument("question")
@click.option("--db-path", default=CHROMA_DB_PATH, show_default=True, help="ChromaDB storage path")
@click.option("--model", default=LLM_MODEL, show_default=True, help="Ollama LLM model")
@click.option("--embed-model", default=EMBED_MODEL, show_default=True, help="Ollama embedding model")
@click.option("--ollama-url", envvar="OLLAMA_BASE_URL", default=OLLAMA_BASE_URL, show_default=True, help="Ollama base URL")
@click.option("--depth", default=RESEARCH_DEPTH, show_default=True, type=int, help="Max reflection iterations")
@click.option("--sub-questions", default=RESEARCH_N_SUBQUESTIONS, show_default=True, type=int, help="Sub-questions per iteration")
@click.option("--top-k", default=TOP_K, show_default=True, type=int, help="Chunks retrieved per sub-question")
def research_cmd(question, db_path, model, embed_model, ollama_url, depth, sub_questions, top_k):
    """Deep multi-step research over the indexed PDF library."""
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    research(
        question=question,
        collection=collection,
        base_url=ollama_url,
        llm_model=model,
        embed_model=embed_model,
        depth=depth,
        n_subquestions=sub_questions,
        top_k=top_k,
    )
    click.echo(click.style(f"model: {model}", fg="bright_black"))
