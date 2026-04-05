from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..config import (
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
from ..researcher import research, run_ask


class AskRequest(BaseModel):
    question: str
    llm_model: Optional[str] = None
    embed_model: Optional[str] = None
    top_k: Optional[int] = None
    show_sources: bool = True


class ResearchRequest(BaseModel):
    question: str
    llm_model: Optional[str] = None
    fast_model: Optional[str] = None
    tiny_model: Optional[str] = None
    embed_model: Optional[str] = None
    depth: Optional[int] = None
    n_subquestions: Optional[int] = None
    top_k: Optional[int] = None
    languages: Optional[list[str]] = None
    translate_model: Optional[str] = None


def _sse(kind: str, text: str = "") -> str:
    return f"event: {kind}\ndata: {json.dumps({'text': text})}\n\n"


def make_app(
    db_path: str = DB_PATH,
    base_url: str = OLLAMA_BASE_URL,
    llm_model: str = DEEP_MODEL,
    fast_model: str = FAST_MODEL,
    tiny_model: str = TINY_MODEL,
    embed_model: str = EMBED_MODEL,
    depth: int = RESEARCH_DEPTH,
    n_subquestions: int = RESEARCH_N_SUBQUESTIONS,
    top_k: int = TOP_K,
    languages: list[str] = SEARCH_LANGUAGES,
    translate_model: str = TRANSLATE_MODEL,
) -> FastAPI:
    app = FastAPI(title="pedro")
    _pool = ThreadPoolExecutor(max_workers=4)

    async def _stream(fn, **kwargs) -> AsyncIterator[str]:
        loop = asyncio.get_event_loop()
        q: asyncio.Queue = asyncio.Queue()

        def on_token(t: str) -> None:
            asyncio.run_coroutine_threadsafe(q.put(("token", t)), loop)

        def log_fn(msg: str) -> None:
            asyncio.run_coroutine_threadsafe(q.put(("log", msg)), loop)

        def _done(_) -> None:
            asyncio.run_coroutine_threadsafe(q.put(None), loop)

        future = loop.run_in_executor(_pool, lambda: fn(**kwargs, on_token=on_token, log_fn=log_fn))
        future.add_done_callback(_done)

        while True:
            item = await q.get()
            if item is None:
                break
            kind, text = item
            yield _sse(kind, text)

        yield _sse("done")

    @app.post("/v1/ask")
    async def ask(req: AskRequest) -> StreamingResponse:
        return StreamingResponse(
            _stream(
                run_ask,
                question=req.question,
                db_path=db_path,
                base_url=base_url,
                llm_model=req.llm_model or llm_model,
                embed_model=req.embed_model or embed_model,
                top_k=req.top_k or top_k,
                show_sources=req.show_sources,
            ),
            media_type="text/event-stream",
        )

    @app.post("/v1/research")
    async def research_endpoint(req: ResearchRequest) -> StreamingResponse:
        return StreamingResponse(
            _stream(
                research,
                question=req.question,
                db_path=db_path,
                base_url=base_url,
                llm_model=req.llm_model or llm_model,
                fast_model=req.fast_model or fast_model,
                tiny_model=req.tiny_model or tiny_model,
                embed_model=req.embed_model or embed_model,
                depth=req.depth or depth,
                n_subquestions=req.n_subquestions or n_subquestions,
                top_k=req.top_k or top_k,
                languages=req.languages if req.languages is not None else languages,
                translate_model=req.translate_model or translate_model,
            ),
            media_type="text/event-stream",
        )

    return app
