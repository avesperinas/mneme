"""FastAPI application (sub-phase 2.3).

create_app wires the retrieval + synthesis pipeline behind POST /query and a
GET /health probe. Heavy components (BGE-M3, Qdrant, the LLM client) are built
from settings in production but can be injected, so tests run with lightweight
doubles and an in-process Qdrant. Only "naive" retrieval exists in phase 2;
other modes arrive in later phases.

Run locally with: uvicorn --factory mneme.api.app:create_app
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException

from mneme.api.schemas import HealthResponse, QueryRequest, QueryResponse, Source
from mneme.config import Settings, get_settings
from mneme.rag.synthesize import NOT_FOUND_MESSAGE, synthesize_answer
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
        sources = [
            Source(
                rel_path=item.chunk.rel_path,
                heading_path=item.chunk.heading_path,
                snippet=_snippet(item.chunk.text),
                score=item.score,
            )
            for item in answer.used
        ]
        return QueryResponse(answer=answer.text, sources=sources, mode="naive")

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
