"""FastAPI application (sub-phase 2.3).

create_app wires the retrieval + synthesis pipeline behind POST /query and a
GET /health probe. Heavy components (BGE-M3, Qdrant, the LLM client) are built
from settings in production but can be injected, so tests run with lightweight
doubles and an in-process Qdrant. Only "naive" retrieval exists in phase 2;
other modes arrive in later phases.

Run locally with: uvicorn --factory mneme.api.app:create_app
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from mneme.api.schemas import HealthResponse, QueryRequest, QueryResponse, Source
from mneme.config import Settings, get_settings
from mneme.rag.synthesize import (
    NOT_FOUND_MESSAGE,
    select_relevant,
    stream_answer_tokens,
    synthesize_answer,
)
from mneme.retrieval.dense import dense_search

if TYPE_CHECKING:
    from mneme_ingest.embed import Embedder
    from qdrant_client import QdrantClient

    from mneme.llm.client import LLMClient

_SNIPPET_CHARS = 240


def _snippet(text: str) -> str:
    return (
        text if len(text) <= _SNIPPET_CHARS else text[:_SNIPPET_CHARS].rstrip() + "..."
    )


def _to_sources(used: list) -> list[Source]:
    return [
        Source(
            rel_path=item.chunk.rel_path,
            heading_path=item.chunk.heading_path,
            snippet=_snippet(item.chunk.text),
            score=item.score,
        )
        for item in used
    ]


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def create_app(
    *,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    qdrant_client: QdrantClient | None = None,
    llm_client: LLMClient | None = None,
    collection: str | None = None,
    top_k: int = 5,
    min_score: float = 0.0,
) -> FastAPI:
    settings = settings or get_settings()
    collection = collection or settings.qdrant_collection

    if embedder is None:
        from mneme_ingest.embed import BGEM3Embedder

        embedder = BGEM3Embedder(device=settings.embed_device)
    if qdrant_client is None:
        from mneme_ingest.vectorstore import make_client

        qdrant_client = make_client(settings.qdrant_url)
    if llm_client is None:
        from mneme.llm.client import OpenAICompatClient

        llm_client = OpenAICompatClient(
            settings.llm_base_url, settings.llm_model, settings.llm_api_key
        )

    app = FastAPI(title="Mneme")

    @app.post("/query", response_model=QueryResponse)
    async def query(request: QueryRequest) -> QueryResponse:
        # Translate infrastructure failures into clear HTTP errors rather than an
        # opaque 500: 503 when the vector store is unreachable, 502 when the LLM
        # call fails (e.g. engine down or model not pulled).
        try:
            retrieved = dense_search(
                qdrant_client, collection, request.question, embedder, top_k=top_k
            )
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"vector store unavailable: {exc}"
            ) from exc
        try:
            answer = await synthesize_answer(
                llm_client, request.question, retrieved, min_score=min_score
            )
        except Exception as exc:
            raise HTTPException(
                status_code=502, detail=f"LLM request failed: {exc}"
            ) from exc
        return QueryResponse(
            answer=answer.text, sources=_to_sources(answer.used), mode="naive"
        )

    @app.get("/query/stream")
    async def query_stream(question: str, mode: str | None = None) -> StreamingResponse:
        # Retrieval runs before the stream opens, so a vector-store failure is a
        # clean 503 rather than a half-open stream. The LLM call is inside the
        # stream, so its failure surfaces as a terminal SSE "error" event.
        try:
            retrieved = dense_search(
                qdrant_client, collection, question, embedder, top_k=top_k
            )
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"vector store unavailable: {exc}"
            ) from exc
        relevant = select_relevant(retrieved, min_score)

        async def events():
            if not relevant:
                yield _sse("token", {"text": NOT_FOUND_MESSAGE})
                yield _sse("sources", {"sources": [], "mode": "naive"})
                return
            try:
                async for token in stream_answer_tokens(llm_client, question, relevant):
                    yield _sse("token", {"text": token})
            except Exception as exc:
                yield _sse("error", {"detail": f"LLM request failed: {exc}"})
                return
            payload = {
                "sources": [s.model_dump() for s in _to_sources(relevant)],
                "mode": "naive",
            }
            yield _sse("sources", payload)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        try:
            qdrant_client.get_collections()
            qdrant_ok = True
        except Exception:
            qdrant_ok = False
        return HealthResponse(
            status="ok",
            qdrant=qdrant_ok,
            llm=settings.llm_model,
            embed=f"dim={embedder.dim}",
        )

    return app


# Convenience for `uvicorn mneme.api.app:app` (builds real components on import).
__all__ = ["create_app", "NOT_FOUND_MESSAGE"]
