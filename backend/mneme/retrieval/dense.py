"""Naive dense retrieval (sub-phase 2.2).

Embeds the query with the same model used for indexing, runs a dense top-k
cosine search over Qdrant, and returns chunks with scores. Context assembly
prepends each chunk's heading_path to its text, so the heading hierarchy is
visible to the synthesizer (spec 3.3).
"""

from __future__ import annotations

from dataclasses import dataclass

from mneme_ingest.embed import Embedder
from mneme_ingest.models import Chunk
from mneme_ingest.vectorstore import DENSE_VECTOR, payload_to_chunk
from qdrant_client import QdrantClient


@dataclass(slots=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


def query_dense(
    client: QdrantClient,
    collection: str,
    vector: list[float],
    top_k: int,
) -> list[RetrievedChunk]:
    """Dense search from an already-embedded query vector."""
    response = client.query_points(
        collection,
        query=vector,
        using=DENSE_VECTOR,
        limit=top_k,
        with_payload=True,
    )
    return [
        RetrievedChunk(chunk=payload_to_chunk(point.payload), score=point.score)
        for point in response.points
    ]


def dense_search(
    client: QdrantClient,
    collection: str,
    query: str,
    embedder: Embedder,
    *,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    return query_dense(client, collection, embedder.embed([query])[0], top_k)


def format_context(retrieved: list[RetrievedChunk]) -> str:
    """Render retrieved chunks for the prompt, heading_path prepended."""
    blocks = []
    for item in retrieved:
        heading = " > ".join(item.chunk.heading_path)
        location = (
            f"{item.chunk.rel_path} :: {heading}" if heading else item.chunk.rel_path
        )
        blocks.append(f"[{location}]\n{item.chunk.text}")
    return "\n\n".join(blocks)
