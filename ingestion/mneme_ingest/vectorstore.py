"""Qdrant vector store (sub-phase 2.1).

Owns the collection schema and the Chunk <-> Qdrant payload mapping, shared by
indexing (ingestion) and search (retrieval) so both agree on the layout. Dense
vectors are stored under a named vector ("dense") so phase 4 can add a "sparse"
vector without migrating the collection.

Chunk ids are strings ("docid::ordinal"); Qdrant point ids must be uint or UUID,
so the point id is a deterministic uuid5 of the chunk id and the original chunk
id is kept in the payload.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Iterator

from qdrant_client import QdrantClient, models

from mneme_ingest.embed import Embedder, SparseVector
from mneme_ingest.models import Chunk

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"
_POINT_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


def make_client(url: str) -> QdrantClient:
    """Create a Qdrant client. url ':memory:' gives an in-process local store."""
    if url == ":memory:":
        return QdrantClient(":memory:")
    return QdrantClient(url=url)


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


def embedding_text(chunk: Chunk) -> str:
    """Text used to embed a chunk: heading_path prepended for retrieval context."""
    prefix = " > ".join(chunk.heading_path)
    return f"{prefix}\n{chunk.text}" if prefix else chunk.text


def chunk_to_payload(chunk: Chunk) -> dict:
    rel_path = chunk.rel_path
    return {
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "rel_path": rel_path,
        # parent folder (posix), for metadata filtering; "" for vault-root notes
        "folder": rel_path.rsplit("/", 1)[0] if "/" in rel_path else "",
        "heading_path": chunk.heading_path,
        "text": chunk.text,
        "tags": chunk.tags,
        "ordinal": chunk.ordinal,
        "token_count": chunk.token_count,
    }


def payload_to_chunk(payload: dict) -> Chunk:
    return Chunk(
        id=payload["chunk_id"],
        document_id=payload["document_id"],
        rel_path=payload["rel_path"],
        heading_path=list(payload["heading_path"]),
        text=payload["text"],
        tags=list(payload["tags"]),
        ordinal=payload["ordinal"],
        token_count=payload["token_count"],
    )


def ensure_collection(
    client: QdrantClient, collection: str, dim: int, *, recreate: bool = False
) -> None:
    if recreate and client.collection_exists(collection):
        client.delete_collection(collection)
    if not client.collection_exists(collection):
        client.create_collection(
            collection,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=dim, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams()},
        )


def _batches(items: list[Chunk], size: int) -> Iterator[list[Chunk]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _sparse(vector: SparseVector) -> models.SparseVector:
    return models.SparseVector(indices=vector.indices, values=vector.values)


def upsert_chunks(
    client: QdrantClient,
    collection: str,
    chunks: list[Chunk],
    vectors: Iterable[list[float]],
    sparse_vectors: Iterable[SparseVector] | None = None,
) -> None:
    sparse_list = list(sparse_vectors) if sparse_vectors is not None else None
    points = []
    for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
        named = {DENSE_VECTOR: vector}
        if sparse_list is not None:
            named[SPARSE_VECTOR] = _sparse(sparse_list[index])
        points.append(
            models.PointStruct(
                id=point_id(chunk.id),
                vector=named,
                payload=chunk_to_payload(chunk),
            )
        )
    if points:
        client.upsert(collection, points=points)


def index_chunks(
    client: QdrantClient,
    collection: str,
    chunks: list[Chunk],
    embedder: Embedder,
    *,
    recreate: bool = True,
    batch_size: int = 64,
) -> int:
    """Embed (dense + sparse) and upsert chunks; returns points written."""
    ensure_collection(client, collection, embedder.dim, recreate=recreate)
    for batch in _batches(chunks, batch_size):
        texts = [embedding_text(chunk) for chunk in batch]
        dense, sparse = embedder.embed_both(texts)
        upsert_chunks(client, collection, batch, dense, sparse)
    return len(chunks)
