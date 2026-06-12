from pathlib import Path

from mneme_ingest.chunker import chunk_document
from mneme_ingest.embed import Embedder, HashEmbedder
from mneme_ingest.parser import parse_document, split_frontmatter
from mneme_ingest.vectorstore import (
    DENSE_VECTOR,
    embedding_text,
    index_chunks,
    make_client,
    payload_to_chunk,
    point_id,
)

VAULT = Path(__file__).resolve().parent / "fixtures" / "sample_vault"


def _fixture_chunks():
    chunks = []
    for path in sorted(VAULT.rglob("*.md")):
        doc = parse_document(path, VAULT)
        _, body = split_frontmatter(path.read_text())
        chunks.extend(chunk_document(doc, body))
    return chunks


def test_hash_embedder_is_deterministic_and_right_dim():
    embedder = HashEmbedder(dim=16)
    assert isinstance(embedder, Embedder)
    a = embedder.embed(["hello", "world"])
    b = embedder.embed(["hello", "world"])
    assert a == b
    assert len(a[0]) == 16
    assert embedder.embed(["hello"]) != embedder.embed(["different"])


def test_index_holds_one_point_per_chunk():
    chunks = _fixture_chunks()
    client = make_client(":memory:")
    written = index_chunks(client, "mneme", chunks, HashEmbedder(dim=16))
    assert written == len(chunks)
    assert client.count("mneme").count == len(chunks)


def test_payload_round_trips_chunk_fields():
    chunks = _fixture_chunks()
    client = make_client(":memory:")
    index_chunks(client, "mneme", chunks, HashEmbedder(dim=16))

    target = chunks[0]
    stored = client.retrieve(
        "mneme", ids=[point_id(target.id)], with_payload=True, with_vectors=True
    )
    assert len(stored) == 1
    restored = payload_to_chunk(stored[0].payload)
    assert restored == target
    assert len(stored[0].vector[DENSE_VECTOR]) == 16


def test_reindex_does_not_duplicate_points():
    chunks = _fixture_chunks()
    client = make_client(":memory:")
    index_chunks(client, "mneme", chunks, HashEmbedder(dim=16))
    index_chunks(client, "mneme", chunks, HashEmbedder(dim=16))  # recreate=True
    assert client.count("mneme").count == len(chunks)


def test_embedding_text_prepends_heading_path():
    chunks = _fixture_chunks()
    chunk = next(c for c in chunks if c.heading_path == ["System Architecture"])
    text = embedding_text(chunk)
    assert text.startswith("System Architecture\n")
    assert chunk.text in text
