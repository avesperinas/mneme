from pathlib import Path

import pytest
from mneme.retrieval import (
    RetrievedChunk,
    dense_search,
    hybrid_search,
    reciprocal_rank_fusion,
)
from mneme_ingest.chunker import chunk_document
from mneme_ingest.embed import HashEmbedder
from mneme_ingest.models import Chunk
from mneme_ingest.parser import parse_document, split_frontmatter
from mneme_ingest.vectorstore import index_chunks, make_client

VAULT = Path(__file__).resolve().parent / "fixtures" / "sample_vault"


def _chunk(cid: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            id=cid,
            document_id="d",
            rel_path="d.md",
            heading_path=["H"],
            text=text,
            tags=[],
            ordinal=0,
            token_count=1,
        ),
        score=0.0,
    )


def _fixture_chunks():
    chunks = []
    for path in sorted(VAULT.rglob("*.md")):
        doc = parse_document(path, VAULT)
        _, body = split_frontmatter(path.read_text())
        chunks.extend(chunk_document(doc, body))
    return chunks


def _indexed_client():
    client = make_client(":memory:")
    index_chunks(client, "mneme", _fixture_chunks(), HashEmbedder(dim=32))
    return client


# --- RRF unit -------------------------------------------------------------


def test_rrf_rewards_agreement_across_rankings():
    a, b, c = _chunk("a", "a"), _chunk("b", "b"), _chunk("c", "c")
    dense = [a, b, c]
    sparse = [b, a, c]
    fused = reciprocal_rank_fusion([dense, sparse], k=60, top_k=3)
    # b is rank 2 then rank 1; a is rank 1 then rank 2 -> a and b lead c
    ids = [r.chunk.id for r in fused]
    assert ids[2] == "c"
    assert set(ids[:2]) == {"a", "b"}
    # fused scores are descending
    scores = [r.score for r in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_deduplicates_chunks_seen_in_both_rankings():
    a = _chunk("a", "a")
    fused = reciprocal_rank_fusion([[a], [a]], top_k=5)
    assert len(fused) == 1
    assert fused[0].chunk.id == "a"


# --- hybrid mechanics + exact-token retrieval (hash double) ---------------


def test_hybrid_recovers_exact_token_match():
    client = _indexed_client()
    # "embeddings" appears verbatim only in Retrieval.md's Dense Search section.
    # The hash double's dense vectors are pseudo-random, so dense alone cannot
    # rely on it; the sparse (word-keyed) signal pulls it to the top.
    embedder = HashEmbedder(dim=32)
    # candidate_k spans the whole fixture so the pseudo-random dense ranking of
    # the hash double cannot drop the target out of the window; the exact-token
    # sparse match is then decisive.
    results = hybrid_search(client, "mneme", "embeddings", embedder, candidate_k=50)
    top = results[0]
    assert top.chunk.rel_path == "Retrieval.md"
    assert "embeddings" in top.chunk.text.lower()


def test_naive_path_still_callable():
    # GUARDRAIL: naive dense search must remain usable for the phase-5 comparison.
    client = _indexed_client()
    results = dense_search(client, "mneme", "anything", HashEmbedder(dim=32), top_k=3)
    assert len(results) == 3


# --- semantics (real BGE-M3, gated on the embed group) --------------------


def test_hybrid_exact_token_with_real_bge():
    pytest.importorskip("FlagEmbedding")
    from mneme_ingest.embed import BGEM3Embedder

    embedder = BGEM3Embedder(device="cpu")
    client = make_client(":memory:")
    index_chunks(client, "mneme", _fixture_chunks(), embedder)
    results = hybrid_search(client, "mneme", "embeddings", embedder, top_k=5)
    rel_paths = {r.chunk.rel_path for r in results}
    assert "Retrieval.md" in rel_paths
