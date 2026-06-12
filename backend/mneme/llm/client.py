"""LLMClient abstraction and its single OpenAI-compatible implementation.

This is the only module permitted to know how to talk to a serving engine.
Both vLLM (gpu profile) and Ollama (cpu profile) expose the same
/v1/chat/completions contract, so one client serves both; engine selection is
config-only via the base URL. No other module may import an engine SDK or call
an engine endpoint directly (see spec section 3.2).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Protocol, TypedDict, runtime_checkable

import httpx


class Message(TypedDict):
    role: str
    content: str


@runtime_checkable
class LLMClient(Protocol):
    async def complete(self, messages: list[Message], **opts: object) -> str: ...

    def stream(self, messages: list[Message], **opts: object) -> AsyncIterator[str]: ...


class OpenAICompatClient:
    """Talks to any OpenAI-compatible /v1/chat/completions endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "not-needed",
        *,
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _payload(
        self, messages: list[Message], stream: bool, opts: dict[str, object]
    ) -> dict[str, object]:
        return {"model": self._model, "messages": messages, "stream": stream, **opts}

    async def complete(self, messages: list[Message], **opts: object) -> str:
        resp = await self._client.post(
            f"{self._base_url}/chat/completions",
            json=self._payload(messages, False, opts),
            headers=self._headers(),
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    async def stream(
        self, messages: list[Message], **opts: object
    ) -> AsyncIterator[str]:
        async with self._client.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            json=self._payload(messages, True, opts),
            headers=self._headers(),
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                delta = json.loads(data)["choices"][0]["delta"].get("content")
                if delta:
                    yield delta

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
