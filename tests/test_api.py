import json
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
        # split into several tokens so streaming is genuinely incremental
        parts = self.reply.split(" ")
        for index, part in enumerate(parts):
            yield part if index == len(parts) - 1 else part + " "


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


def test_cors_allows_configured_frontend_origin():
    app = create_app(
        embedder=HashEmbedder(dim=32),
        qdrant_client=make_client(":memory:"),
        llm_client=FakeLLM("x"),
        collection="mneme",
        cors_origins=["http://localhost:5173"],
    )
    resp = TestClient(app).options(
        "/query",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_health_reports_components():
    client = _client(chunks=_fixture_chunks())
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["qdrant"] is True
    assert body["llm"]
    assert body["embed"].startswith("dim=")


class _BrokenQdrant:
    def query_points(self, *a, **k):
        raise ConnectionError("could not connect to qdrant")

    def get_collections(self):
        raise ConnectionError("could not connect to qdrant")


class _BrokenLLM:
    async def complete(self, messages, **opts):
        raise RuntimeError("model not pulled")

    async def stream(self, messages, **opts):
        yield ""


def test_unreachable_qdrant_returns_503():
    app = create_app(
        embedder=HashEmbedder(dim=32),
        qdrant_client=_BrokenQdrant(),
        llm_client=FakeLLM("unused"),
        collection="mneme",
    )
    resp = TestClient(app).post("/query", json={"question": "anything?"})
    assert resp.status_code == 503
    assert "vector store unavailable" in resp.json()["detail"]


def test_failing_llm_returns_502():
    embedder = HashEmbedder(dim=32)
    qdrant = make_client(":memory:")
    index_chunks(qdrant, "mneme", _fixture_chunks(), embedder)
    app = create_app(
        embedder=embedder,
        qdrant_client=qdrant,
        llm_client=_BrokenLLM(),
        collection="mneme",
        top_k=3,
    )
    resp = TestClient(app).post("/query", json={"question": "how does search work?"})
    assert resp.status_code == 502
    assert "LLM request failed" in resp.json()["detail"]


# --- streaming (3.1) ------------------------------------------------------


def _parse_sse(text: str) -> list[dict]:
    events = []
    for block in text.strip().split("\n\n"):
        event = {}
        for line in block.splitlines():
            if line.startswith("event:"):
                event["event"] = line[len("event:") :].strip()
            elif line.startswith("data:"):
                event["data"] = line[len("data:") :].strip()
        if event:
            events.append(event)
    return events


class _StreamBrokenLLM:
    async def complete(self, messages, **opts):
        raise RuntimeError("engine down")

    async def stream(self, messages, **opts):
        raise RuntimeError("engine down")
        yield ""  # unreachable; makes this an async generator


def test_stream_yields_tokens_then_sources():
    client = _client(
        chunks=_fixture_chunks(), llm_reply="Hybrid search fuses rankings."
    )
    resp = client.get("/query/stream", params={"question": "how does hybrid work?"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(resp.text)
    tokens = [json.loads(e["data"])["text"] for e in events if e["event"] == "token"]
    assert len(tokens) > 1  # genuinely incremental, not one blob
    assert "".join(tokens) == "Hybrid search fuses rankings."
    # the sources event is last, after all tokens
    assert events[-1]["event"] == "sources"
    payload = json.loads(events[-1]["data"])
    assert payload["mode"] == "naive"
    assert len(payload["sources"]) == 3


def test_stream_out_of_domain_emits_not_found_and_no_sources():
    client = _client(chunks=[])
    resp = client.get("/query/stream", params={"question": "capital of France?"})
    events = _parse_sse(resp.text)
    tokens = [json.loads(e["data"])["text"] for e in events if e["event"] == "token"]
    assert tokens == [NOT_FOUND_MESSAGE]
    assert json.loads(events[-1]["data"])["sources"] == []


def test_stream_llm_failure_emits_error_event():
    embedder = HashEmbedder(dim=32)
    qdrant = make_client(":memory:")
    index_chunks(qdrant, "mneme", _fixture_chunks(), embedder)
    app = create_app(
        embedder=embedder,
        qdrant_client=qdrant,
        llm_client=_StreamBrokenLLM(),
        collection="mneme",
        top_k=3,
    )
    resp = TestClient(app).get(
        "/query/stream", params={"question": "how does it work?"}
    )
    assert resp.status_code == 200  # stream opened; failure is in-band
    events = _parse_sse(resp.text)
    assert any(e["event"] == "error" for e in events)
