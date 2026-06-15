"""Hybrid retrieval (sub-phase 4.1).

Runs dense and sparse queries from a single BGE-M3 pass and fuses the two
rankings with Reciprocal Rank Fusion. RRF combines by rank position, so the
dense and sparse score scales never need to be reconciled. The naive dense path
(dense.py) stays callable on its own for the phase-5 comparison.
"""

from __future__ import annotations

from collections.abc import Iterable

from mneme_ingest.embed import Embedder
from qdrant_client import QdrantClient

from mneme.retrieval.dense import RetrievedChunk, query_dense
from mneme.retrieval.sparse import query_sparse


def reciprocal_rank_fusion(
    rankings: Iterable[list[RetrievedChunk]],
    *,
    k: int = 60,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """Fuse ranked lists: score = sum 1 / (k + rank). Higher is better."""
    scores: dict[str, float] = {}
    chunks: dict[str, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            cid = item.chunk.id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            chunks[cid] = item
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [
        RetrievedChunk(chunk=chunks[cid].chunk, score=score)
        for cid, score in ordered[:top_k]
    ]


def hybrid_search(
    client: QdrantClient,
    collection: str,
    query: str,
    embedder: Embedder,
    *,
    candidate_k: int = 20,
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    dense_vectors, sparse_vectors = embedder.embed_both([query])
    dense_results = query_dense(client, collection, dense_vectors[0], candidate_k)
    sparse_results = query_sparse(client, collection, sparse_vectors[0], candidate_k)
    return reciprocal_rank_fusion([dense_results, sparse_results], k=rrf_k, top_k=top_k)
