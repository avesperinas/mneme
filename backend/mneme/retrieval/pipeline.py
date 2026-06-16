"""Retrieval pipeline dispatch (sub-phase 4.2).

Selects the retrieval mode (naive dense vs hybrid) and optionally reranks the
candidates. Keeping every mode reachable from one entry point is what lets the
phase-5 evaluation compare them on equal footing.
"""

from __future__ import annotations

from mneme_ingest.embed import Embedder
from qdrant_client import QdrantClient

from mneme.retrieval.dense import RetrievedChunk, dense_search
from mneme.retrieval.filters import build_filter
from mneme.retrieval.hybrid import hybrid_search
from mneme.retrieval.rerank import Reranker, rerank

MODES = ("naive", "hybrid")


def retrieve(
    mode: str,
    client: QdrantClient,
    collection: str,
    query: str,
    embedder: Embedder,
    *,
    reranker: Reranker | None = None,
    filters: dict | None = None,
    candidate_k: int = 20,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    query_filter = build_filter(filters)
    if mode == "hybrid":
        candidates = hybrid_search(
            client,
            collection,
            query,
            embedder,
            candidate_k=candidate_k,
            top_k=candidate_k,
            query_filter=query_filter,
        )
    elif mode == "naive":
        candidates = dense_search(
            client,
            collection,
            query,
            embedder,
            top_k=candidate_k,
            query_filter=query_filter,
        )
    else:
        raise ValueError(f"unknown retrieval mode: {mode!r}")

    if reranker is not None:
        return rerank(query, candidates, reranker, top_n=top_k)
    return candidates[:top_k]
