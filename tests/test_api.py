from collections.abc import AsyncIterator
from pathlib import Path

from fastapi.testclient import TestClient
from mneme.api.app import create_app
from mneme.rag import NOT_FOUND_MESSAGE
from mneme_ingest.chunker import chunk_document
from mneme_ingest.embed import HashEmbedder
from mneme_ingest.parser import parse_document, split_frontmatter
from mneme_ingest.vectorstore import index_chunks, make_client

VAULT = Path(__file__).resolve().parent / "fixtures" / "sample_vault"


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    async def complete(self, messages, **opts) -> str:
        return self.reply

    async def stream(self, messages, **opts) -> AsyncIterator[str]:
        yield self.reply


def _fixture_chunks():
    chunks = []
    for path in sorted(VAULT.rglob("*.md")):
        doc = parse_document(path, VAULT)
        _, body = split_frontmatter(path.read_text())
        chunks.extend(chunk_document(doc, body))
    return chunks


def _client(*, chunks, llm_reply="Grounded answer [Retrieval.md]."):
    embedder = HashEmbedder(dim=32)
    qdrant = make_client(":memory:")
    index_chunks(qdrant, "mneme", chunks, embedder)
    app = create_app(
        embedder=embedder,
        qdrant_client=qdrant,
        llm_client=FakeLLM(llm_reply),
        collection="mneme",
        top_k=3,
    )
    return TestClient(app)


def test_query_returns_answer_with_sources():
    client = _client(chunks=_fixture_chunks())
    resp = client.post("/query", json={"question": "how does dense search work?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Grounded answer [Retrieval.md]."
    assert body["mode"] == "naive"
    assert len(body["sources"]) == 3
    source = body["sources"][0]
    assert set(source) == {"rel_path", "heading_path", "snippet", "score"}
    assert source["heading_path"]  # non-empty heading path carried through


def test_query_out_of_domain_says_not_found_with_no_sources():
    # Empty collection -> retrieval finds nothing -> not found, no fabrication.
    client = _client(chunks=[])
    resp = client.post("/query", json={"question": "what is the capital of France?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == NOT_FOUND_MESSAGE
    assert body["sources"] == []


def test_health_reports_components():
    client = _client(chunks=_fixture_chunks())
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["qdrant"] is True
    assert body["llm"]
    assert body["embed"].startswith("dim=")
