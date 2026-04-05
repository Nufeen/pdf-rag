import click

from .config import (
    DB_PATH,
    DEEP_MODEL,
    EMBED_MODEL,
    FAST_MODEL,
    OLLAMA_BASE_URL,
    RESEARCH_DEPTH,
    RESEARCH_N_SUBQUESTIONS,
    SEARCH_LANGUAGES,
    TINY_MODEL,
    TOP_K,
    TRANSLATE_MODEL,
)
from .indexer import index_folder
from .researcher import research, run_ask


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Local RAG search tool for PDF books."""
    if ctx.invoked_subcommand is None:
        from .tui import PedroApp
        PedroApp().run()


@cli.command()
@click.argument("folder", type=click.Path(exists=True, file_okay=False))
@click.option("--db-path", default=DB_PATH, show_default=True, help="ChromaDB storage path")
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
@click.option("--db-path", default=DB_PATH, show_default=True, help="ChromaDB storage path")
@click.option("--deep-model", default=DEEP_MODEL, show_default=True, help="Ollama LLM model")
@click.option("--embed-model", default=EMBED_MODEL, show_default=True, help="Ollama embedding model")
@click.option("--ollama-url", envvar="OLLAMA_BASE_URL", default=OLLAMA_BASE_URL, show_default=True, help="Ollama base URL")
@click.option("--top-k", default=TOP_K, show_default=True, type=int, help="Number of chunks to retrieve")
@click.option("--no-sources", is_flag=True, help="Hide retrieved source list")
def ask(question, db_path, deep_model, embed_model, ollama_url, top_k, no_sources):
    """Ask a QUESTION against the indexed PDF library."""
    run_ask(
        question=question,
        db_path=db_path,
        base_url=ollama_url,
        llm_model=deep_model,
        embed_model=embed_model,
        top_k=top_k,
        show_sources=not no_sources,
    )
    click.echo(click.style(f"model: {deep_model}", fg="bright_black"))


@cli.command("research")
@click.argument("question")
@click.option("--db-path", default=DB_PATH, show_default=True, help="ChromaDB storage path")
@click.option("--deep-model", default=DEEP_MODEL, show_default=True, help="Quality model for final synthesis")
@click.option("--fast-model", default=FAST_MODEL, show_default=True, help="Model for sub-question answers and intermediate synthesis")
@click.option("--tiny-model", default=TINY_MODEL, show_default=True, help="Model for planning and reflection (3B recommended)")
@click.option("--embed-model", default=EMBED_MODEL, show_default=True, help="Ollama embedding model")
@click.option("--ollama-url", envvar="OLLAMA_BASE_URL", default=OLLAMA_BASE_URL, show_default=True, help="Ollama base URL")
@click.option("--depth", default=RESEARCH_DEPTH, show_default=True, type=int, help="Max reflection iterations")
@click.option("--sub-questions", default=RESEARCH_N_SUBQUESTIONS, show_default=True, type=int, help="Sub-questions per iteration")
@click.option("--top-k", default=TOP_K, show_default=True, type=int, help="Chunks retrieved per sub-question")
@click.option("--languages", default=",".join(SEARCH_LANGUAGES), show_default=True, help="Comma-separated languages for query translation (e.g. Russian,French)")
@click.option("--translate-model", default=TRANSLATE_MODEL, show_default=True, help="Model used for query translation")
def research_cmd(question, db_path, deep_model, fast_model, tiny_model, embed_model, ollama_url, depth, sub_questions, top_k, languages, translate_model):
    """Deep multi-step research over the indexed PDF library."""
    research(
        question=question,
        db_path=db_path,
        base_url=ollama_url,
        llm_model=deep_model,
        fast_model=fast_model,
        tiny_model=tiny_model,
        embed_model=embed_model,
        depth=depth,
        n_subquestions=sub_questions,
        top_k=top_k,
        languages=[l.strip() for l in languages.split(",") if l.strip()],
        translate_model=translate_model,
    )
    models_line = deep_model
    if fast_model != deep_model:
        models_line += f"  ·  {fast_model}"
    if tiny_model != fast_model:
        models_line += f"  ·  {tiny_model}"
    click.echo(click.style(f"models: {models_line}", fg="bright_black"))
