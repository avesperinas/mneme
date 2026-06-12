import json

import httpx
import pytest
from mneme.llm import LLMClient, OpenAICompatClient


def _make_client(handler) -> OpenAICompatClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport)
    return OpenAICompatClient("http://engine:8000/v1", "test-model", client=http)


async def test_complete_returns_message_content():
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://engine:8000/v1/chat/completions"
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["stream"] is False
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "I am a test."}}]}
        )

    client = _make_client(handler)
    try:
        out = await client.complete([{"role": "user", "content": "hi"}])
    finally:
        await client.aclose()
    assert out == "I am a test."


async def test_stream_yields_deltas_until_done():
    sse = (
        'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
        'data: {"choices":[{"delta":{}}]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, content=sse.encode())

    client = _make_client(handler)
    messages = [{"role": "user", "content": "hi"}]
    try:
        tokens = [tok async for tok in client.stream(messages)]
    finally:
        await client.aclose()
    assert tokens == ["Hello", " world"]


async def test_complete_raises_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found"})

    client = _make_client(handler)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await client.complete([{"role": "user", "content": "hi"}])
    finally:
        await client.aclose()


def test_openai_compat_client_satisfies_protocol():
    client = OpenAICompatClient("http://engine:8000/v1", "m")
    assert isinstance(client, LLMClient)
