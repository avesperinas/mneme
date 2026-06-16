from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from mneme.api.app import create_app
from mneme.retrieval import FakeReranker, RetrievedChunk, rerank, retrieve
from mneme_ingest.chunker import chunk_document
from mneme_ingest.embed import HashEmbedder
from mneme_ingest.models import Chunk
from mneme_ingest.parser import parse_document, split_frontmatter
from mneme_ingest.vectorstore import index_chunks, make_client

VAULT = Path(__file__).resolve().parent / "fixtures" / "sample_vault"


class FakeLLM:
    async def complete(self, messages, **opts) -> str:
        return "answer"

    async def stream(self, messages, **opts) -> AsyncIterator[str]:
        yield "answer"


def _rc(cid: str, text: str) -> RetrievedChunk:
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


def _indexed():
    chunks = []
    for path in sorted(VAULT.rglob("*.md")):
        doc = parse_document(path, VAULT)
        _, body = split_frontmatter(path.read_text())
        chunks.extend(chunk_document(doc, body))
    client = make_client(":memory:")
    index_chunks(client, "mneme", chunks, HashEmbedder(dim=32))
    return client


# --- rerank unit ----------------------------------------------------------


def test_rerank_reorders_by_overlap_and_truncates():
    candidates = [
        _rc("a", "nothing relevant here"),
        _rc("b", "pizza dough hydration percent"),
        _rc("c", "an unrelated sentence"),
    ]
    out = rerank("hydration for pizza dough", candidates, FakeReranker(), top_n=2)
    assert [r.chunk.id for r in out] == ["b", "a"] or out[0].chunk.id == "b"
    assert out[0].chunk.id == "b"  # most query-word overlap floats to the top
    assert len(out) == 2


def test_rerank_empty_candidates():
    assert rerank("q", [], FakeReranker()) == []


# --- pipeline dispatch ----------------------------------------------------


def test_retrieve_dispatches_naive_and_hybrid():
    client = _indexed()
    embedder = HashEmbedder(dim=32)
    naive = retrieve("naive", client, "mneme", "embeddings", embedder, top_k=3)
    hybrid = retrieve(
        "hybrid", client, "mneme", "embeddings", embedder, candidate_k=50, top_k=3
    )
    assert len(naive) == 3
    assert any(r.chunk.rel_path == "Retrieval.md" for r in hybrid)


def test_retrieve_rejects_unknown_mode():
    client = _indexed()
    with pytest.raises(ValueError):
        retrieve("bogus", client, "mneme", "x", HashEmbedder(dim=32))


def test_retrieve_with_reranker_truncates_to_top_k():
    client = _indexed()
    out = retrieve(
        "hybrid",
        client,
        "mneme",
        "embeddings",
        HashEmbedder(dim=32),
        reranker=FakeReranker(),
        candidate_k=50,
        top_k=3,
    )
    assert len(out) == 3
    assert "embeddings" in out[0].chunk.text.lower()  # reranked to the top


# --- API mode + rerank toggle ---------------------------------------------


def _app(*, default_mode="hybrid", reranker=None, rerank=None):
    client = _indexed()
    return TestClient(
        create_app(
            embedder=HashEmbedder(dim=32),
            qdrant_client=client,
            llm_client=FakeLLM(),
            reranker=reranker,
            collection="mneme",
            default_mode=default_mode,
            rerank=rerank if rerank is not None else False,
            top_k=3,
        )
    )


def test_api_defaults_to_hybrid_mode():
    body = _app().post("/query", json={"question": "embeddings"}).json()
    assert body["mode"] == "hybrid"
    assert len(body["sources"]) >= 1


def test_api_mode_naive_override():
    body = (
        _app().post("/query", json={"question": "embeddings", "mode": "naive"}).json()
    )
    assert body["mode"] == "naive"


def test_api_with_injected_reranker_runs():
    client = _app(reranker=FakeReranker(), rerank=True)
    body = client.post("/query", json={"question": "embeddings"}).json()
    assert body["mode"] == "hybrid"
    assert len(body["sources"]) >= 1


# --- real cross-encoder (gated on the embed group) ------------------------


def test_real_reranker_prefers_relevant_passage():
    pytest.importorskip("FlagEmbedding")
    from mneme.retrieval.rerank import BGEReranker

    reranker = BGEReranker(device="cpu")
    scores = reranker.score(
        "how much hydration for pizza dough",
        ["Use around 65 percent hydration for pizza dough.", "The sky is blue."],
    )
    assert scores[0] > scores[1]
