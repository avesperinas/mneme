from pathlib import Path

import pytest
from mneme.retrieval import RetrievedChunk, dense_search, format_context
from mneme_ingest.chunker import chunk_document
from mneme_ingest.embed import HashEmbedder
from mneme_ingest.parser import parse_document, split_frontmatter
from mneme_ingest.vectorstore import index_chunks, make_client

VAULT = Path(__file__).resolve().parent / "fixtures" / "sample_vault"


def _fixture_chunks():
    chunks = []
    for path in sorted(VAULT.rglob("*.md")):
        doc = parse_document(path, VAULT)
        _, body = split_frontmatter(path.read_text())
        chunks.extend(chunk_document(doc, body))
    return chunks


def _indexed_client(embedder):
    client = make_client(":memory:")
    index_chunks(client, "mneme", _fixture_chunks(), embedder)
    return client


# --- mechanics (hash double, no model download) ---------------------------


def test_dense_search_returns_topk_ordered_by_score():
    embedder = HashEmbedder(dim=32)
    client = _indexed_client(embedder)
    results = dense_search(client, "mneme", "anything", embedder, top_k=3)
    assert len(results) == 3
    assert all(isinstance(r, RetrievedChunk) for r in results)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_dense_search_can_return_a_known_chunk():
    embedder = HashEmbedder(dim=32)
    chunks = _fixture_chunks()
    client = _indexed_client(embedder)
    # Embedding the exact embedding-text of a chunk must rank that chunk first.
    from mneme_ingest.vectorstore import embedding_text

    target = chunks[0]
    results = dense_search(client, "mneme", embedding_text(target), embedder, top_k=1)
    assert results[0].chunk.id == target.id


def test_format_context_prepends_heading_path():
    embedder = HashEmbedder(dim=32)
    client = _indexed_client(embedder)
    results = dense_search(client, "mneme", "anything", embedder, top_k=2)
    context = format_context(results)
    for item in results:
        heading = " > ".join(item.chunk.heading_path)
        assert f"{item.chunk.rel_path} :: {heading}" in context
        assert item.chunk.text in context


# --- semantics (real BGE-M3, self-contained, gated on the embed group) -----

SEMANTIC_QUESTIONS = [
    ("How does dense search with embeddings work?", "Retrieval.md"),
    ("What is the overall system architecture?", "projects/Architecture.md"),
    ("What is planned on the roadmap?", "projects/Roadmap.md"),
    ("What did I read in my daily note today?", "Daily Note.md"),
    ("Tell me about the components of the architecture.", "projects/Architecture.md"),
]


def test_five_questions_retrieve_correct_source_note():
    pytest.importorskip("FlagEmbedding")
    from mneme_ingest.embed import BGEM3Embedder

    embedder = BGEM3Embedder(device="cpu")
    client = _indexed_client(embedder)
    for question, expected_rel_path in SEMANTIC_QUESTIONS:
        results = dense_search(client, "mneme", question, embedder, top_k=3)
        rel_paths = {r.chunk.rel_path for r in results}
        assert expected_rel_path in rel_paths, (question, rel_paths)
