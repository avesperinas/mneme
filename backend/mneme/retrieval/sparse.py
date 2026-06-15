"""Sparse lexical retrieval (sub-phase 4.1).

Queries the BGE-M3 sparse vector stored alongside the dense one. Sparse search
matches exact tokens (acronyms, code identifiers) that dense similarity can miss.
"""

from __future__ import annotations

from mneme_ingest.embed import SparseVector
from mneme_ingest.vectorstore import SPARSE_VECTOR, payload_to_chunk
from qdrant_client import QdrantClient, models

from mneme.retrieval.dense import RetrievedChunk


def query_sparse(
    client: QdrantClient,
    collection: str,
    sparse: SparseVector,
    top_k: int,
) -> list[RetrievedChunk]:
    """Sparse search from an already-embedded sparse query vector."""
    response = client.query_points(
        collection,
        query=models.SparseVector(indices=sparse.indices, values=sparse.values),
        using=SPARSE_VECTOR,
        limit=top_k,
        with_payload=True,
    )
    return [
        RetrievedChunk(chunk=payload_to_chunk(point.payload), score=point.score)
        for point in response.points
    ]
