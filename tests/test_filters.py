from collections.abc import AsyncIterator
from pathlib import Path

from fastapi.testclient import TestClient
from mneme.api.app import create_app
from mneme.retrieval import retrieve
from mneme.retrieval.filters import build_filter
from mneme_ingest.chunker import chunk_document
from mneme_ingest.embed import HashEmbedder
from mneme_ingest.parser import parse_document, split_frontmatter
from mneme_ingest.vectorstore import index_chunks, make_client

VAULT = Path(__file__).resolve().parent / "fixtures" / "sample_vault"


class FakeLLM:
    async def complete(self, messages, **opts) -> str:
        return "answer"

    async def stream(self, messages, **opts) -> AsyncIterator[str]:
        yield "answer"


def _indexed():
    chunks = []
    for path in sorted(VAULT.rglob("*.md")):
        doc = parse_document(path, VAULT)
        _, body = split_frontmatter(path.read_text())
        chunks.extend(chunk_document(doc, body))
    client = make_client(":memory:")
    index_chunks(client, "mneme", chunks, HashEmbedder(dim=32))
    return client


# --- build_filter unit ----------------------------------------------------


def test_build_filter_none_when_empty():
    assert build_filter(None) is None
    assert build_filter({}) is None


def test_build_filter_builds_conditions():
    assert build_filter({"tags": "core"}) is not None
    assert build_filter({"folder": "projects"}) is not None


# --- filtered retrieval ---------------------------------------------------


def test_tag_filter_returns_only_tagged_chunks():
    client = _indexed()
    results = retrieve(
        "naive",
        client,
        "mneme",
        "anything",
        HashEmbedder(dim=32),
        filters={"tags": ["planning"]},
        top_k=10,
    )
    assert results  # planning is on Roadmap
    assert all("planning" in r.chunk.tags for r in results)
    assert all(r.chunk.rel_path == "projects/Roadmap.md" for r in results)


def test_folder_filter_returns_only_that_folder():
    client = _indexed()
    results = retrieve(
        "hybrid",
        client,
        "mneme",
        "anything",
        HashEmbedder(dim=32),
        filters={"folder": "projects"},
        candidate_k=50,
        top_k=10,
    )
    assert results
    assert all(r.chunk.rel_path.startswith("projects/") for r in results)


def test_no_filter_returns_across_folders():
    client = _indexed()
    results = retrieve(
        "naive", client, "mneme", "anything", HashEmbedder(dim=32), top_k=10
    )
    folders = {r.chunk.rel_path.rsplit("/", 1)[0] for r in results}
    assert len(folders) > 1  # unfiltered spans root and projects/


# --- API filters field ----------------------------------------------------


def test_api_query_honors_tag_filter():
    client = TestClient(
        create_app(
            embedder=HashEmbedder(dim=32),
            qdrant_client=_indexed(),
            llm_client=FakeLLM(),
            collection="mneme",
            rerank=False,
            top_k=10,
        )
    )
    body = client.post(
        "/query",
        json={"question": "anything", "filters": {"tags": ["planning"]}},
    ).json()
    assert body["sources"]
    assert all(s["rel_path"] == "projects/Roadmap.md" for s in body["sources"])
