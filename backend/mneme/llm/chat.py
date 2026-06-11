"""One-shot chat CLI: `just chat "your prompt"`.

Sends a single user message through the LLMClient and streams the reply to
stdout. Runs on the host and targets the published engine port (localhost),
resolved from config.yaml for the active profile.
"""

from __future__ import annotations

import asyncio
import sys

import httpx

from mneme.config import get_settings
from mneme.llm.client import Message, OpenAICompatClient


async def _run(prompt: str) -> int:
    settings = get_settings()
    client = OpenAICompatClient(
        settings.llm_base_url, settings.llm_model, settings.llm_api_key
    )
    messages: list[Message] = [{"role": "user", "content": prompt}]
    try:
        async for token in client.stream(messages):
            print(token, end="", flush=True)
        print()
        return 0
    except httpx.ConnectError:
        print(
            f"\nCould not reach the LLM at {settings.llm_base_url}. "
            "Is the stack up? Try `make run`.",
            file=sys.stderr,
        )
        return 1
    except httpx.HTTPStatusError as exc:
        detail = ""
        if exc.response.status_code == 404:
            detail = f" Model '{settings.llm_model}' may not be pulled yet."
        print(
            f"\nLLM request failed ({exc.response.status_code}).{detail}",
            file=sys.stderr,
        )
        return 1
    finally:
        await client.aclose()


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print('usage: just chat "your prompt"', file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(asyncio.run(_run(" ".join(argv))))


if __name__ == "__main__":
    main()
