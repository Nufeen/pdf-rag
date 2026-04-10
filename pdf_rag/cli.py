import click

from .config import (
    DB_PATH,
    LLM_MODEL,
    EMBED_MODEL,
    FAST_MODEL,
    OLLAMA_BASE_URL,
    RESEARCH_DEPTH,
    RESEARCH_N_SUBQUESTIONS,
    SEARCH_LANGUAGES,
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
@click.option(
    "--embed-model", default=EMBED_MODEL, show_default=True, help="Ollama embedding model"
)
@click.option(
    "--ollama-url",
    envvar="OLLAMA_BASE_URL",
    default=OLLAMA_BASE_URL,
    show_default=True,
    help="Ollama base URL",
)
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
@click.option("--model", default=LLM_MODEL, show_default=True, help="Ollama LLM model")
@click.option(
    "--embed-model", default=EMBED_MODEL, show_default=True, help="Ollama embedding model"
)
@click.option(
    "--ollama-url",
    envvar="OLLAMA_BASE_URL",
    default=OLLAMA_BASE_URL,
    show_default=True,
    help="Ollama base URL",
)
@click.option(
    "--top-k", default=TOP_K, show_default=True, type=int, help="Number of chunks to retrieve"
)
@click.option("--no-sources", is_flag=True, help="Hide retrieved source list")
def ask(question, db_path, model, embed_model, ollama_url, top_k, no_sources):
    """Ask a QUESTION against the indexed PDF library."""
    run_ask(
        question=question,
        db_path=db_path,
        base_url=ollama_url,
        llm_model=model,
        embed_model=embed_model,
        top_k=top_k,
        show_sources=not no_sources,
    )
    click.echo(click.style(f"model: {model}", fg="bright_black"))


@cli.command("research")
@click.argument("question")
@click.option("--db-path", default=DB_PATH, show_default=True, help="ChromaDB storage path")
@click.option(
    "--model", default=LLM_MODEL, show_default=True, help="LLM model for final synthesis and ask"
)
@click.option(
    "--fast-model",
    default=FAST_MODEL,
    show_default=True,
    help="Model for sub-question answers, planning and reflection",
)
@click.option(
    "--embed-model", default=EMBED_MODEL, show_default=True, help="Ollama embedding model"
)
@click.option(
    "--ollama-url",
    envvar="OLLAMA_BASE_URL",
    default=OLLAMA_BASE_URL,
    show_default=True,
    help="Ollama base URL",
)
@click.option(
    "--depth", default=RESEARCH_DEPTH, show_default=True, type=int, help="Max reflection iterations"
)
@click.option(
    "--sub-questions",
    default=RESEARCH_N_SUBQUESTIONS,
    show_default=True,
    type=int,
    help="Sub-questions per iteration",
)
@click.option(
    "--top-k", default=TOP_K, show_default=True, type=int, help="Chunks retrieved per sub-question"
)
@click.option(
    "--languages",
    default=",".join(SEARCH_LANGUAGES),
    show_default=True,
    help="Comma-separated languages for query translation (e.g. Russian,French)",
)
@click.option(
    "--translate-model",
    default=TRANSLATE_MODEL,
    show_default=True,
    help="Model used for query translation",
)
def research_cmd(
    question,
    db_path,
    model,
    fast_model,
    embed_model,
    ollama_url,
    depth,
    sub_questions,
    top_k,
    languages,
    translate_model,
):
    """Deep multi-step research over the indexed PDF library."""
    research(
        question=question,
        db_path=db_path,
        base_url=ollama_url,
        llm_model=model,
        fast_model=fast_model,
        embed_model=embed_model,
        depth=depth,
        n_subquestions=sub_questions,
        top_k=top_k,
        languages=[lang.strip() for lang in languages.split(",") if lang.strip()],
        translate_model=translate_model,
    )
    models_line = model
    if fast_model != model:
        models_line += f"  ·  {fast_model}"
    click.echo(click.style(f"models: {models_line}", fg="bright_black"))


@cli.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind host")
@click.option("--port", default=8000, show_default=True, type=int, help="Bind port")
def serve(host, port):
    """Start the pedro HTTP server."""
    import uvicorn
    from .server import make_app

    app = make_app(
        db_path=DB_PATH,
        base_url=OLLAMA_BASE_URL,
        llm_model=LLM_MODEL,
        fast_model=FAST_MODEL,
        embed_model=EMBED_MODEL,
        depth=RESEARCH_DEPTH,
        n_subquestions=RESEARCH_N_SUBQUESTIONS,
        top_k=TOP_K,
        languages=SEARCH_LANGUAGES,
        translate_model=TRANSLATE_MODEL,
    )
    click.echo(click.style(f"pedro server → http://{host}:{port}", fg="green", bold=True))
    uvicorn.run(app, host=host, port=port)
