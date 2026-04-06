from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
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

_MODELS = ["pedro-ask", "pedro-research"]


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


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = True
    # pedro-specific overrides (pass via extra_body in openai SDK)
    top_k: Optional[int] = None
    depth: Optional[int] = None
    n_subquestions: Optional[int] = None
    llm_model: Optional[str] = None
    fast_model: Optional[str] = None
    tiny_model: Optional[str] = None
    embed_model: Optional[str] = None
    languages: Optional[list[str]] = None
    translate_model: Optional[str] = None
    show_sources: bool = True


def _sse_pedro(kind: str, text: str = "") -> str:
    return f"event: {kind}\ndata: {json.dumps({'text': text})}\n\n"


def _oai_chunk(cid: str, created: int, model: str, **delta: Any) -> str:
    return json.dumps({
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    })


def _oai_stop(cid: str, created: int, model: str) -> str:
    return json.dumps({
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    })


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

    async def _run(fn, **kwargs) -> AsyncIterator[tuple[str, str]]:
        """Run fn in a thread and yield raw (kind, text) tuples."""
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
            yield item

    async def _stream_pedro(fn, **kwargs) -> AsyncIterator[str]:
        async for kind, text in _run(fn, **kwargs):
            yield _sse_pedro(kind, text)
        yield _sse_pedro("done")

    async def _stream_openai(fn, model: str, **kwargs) -> AsyncIterator[str]:
        cid = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())
        yield f"data: {_oai_chunk(cid, created, model, role='assistant', content='')}\n\n"
        async for kind, text in _run(fn, **kwargs):
            if kind == "token":
                yield f"data: {_oai_chunk(cid, created, model, content=text)}\n\n"
        yield f"data: {_oai_stop(cid, created, model)}\n\n"
        yield "data: [DONE]\n\n"

    # ── pedro native endpoints ─────────────────────────────────────────────────

    @app.post("/v1/ask")
    async def ask(req: AskRequest) -> StreamingResponse:
        return StreamingResponse(
            _stream_pedro(
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
            _stream_pedro(
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

    # ── OpenAI-compatible endpoints ────────────────────────────────────────────

    @app.get("/v1/models")
    async def list_models() -> JSONResponse:
        now = int(time.time())
        return JSONResponse({
            "object": "list",
            "data": [
                {"id": m, "object": "model", "created": now, "owned_by": "pedro"}
                for m in _MODELS
            ],
        })

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        question = next(
            (m.content for m in reversed(req.messages) if m.role == "user"), ""
        )
        is_research = "research" in req.model

        if is_research:
            fn = research
            kwargs = dict(
                question=question,
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
            )
        else:
            fn = run_ask
            kwargs = dict(
                question=question,
                db_path=db_path,
                base_url=base_url,
                llm_model=req.llm_model or llm_model,
                embed_model=req.embed_model or embed_model,
                top_k=req.top_k or top_k,
                show_sources=req.show_sources,
            )

        if req.stream:
            return StreamingResponse(
                _stream_openai(fn, req.model, **kwargs),
                media_type="text/event-stream",
            )

        # Non-streaming: collect all tokens
        tokens: list[str] = []
        async for kind, text in _run(fn, **kwargs):
            if kind == "token":
                tokens.append(text)
        content = "".join(tokens)
        cid = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        return JSONResponse({
            "id": cid,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    return app
