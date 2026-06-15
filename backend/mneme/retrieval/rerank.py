"""Cross-encoder reranking (sub-phase 4.2).

Takes the top-N candidates from dense or hybrid retrieval and rescores each
(query, chunk) pair with BGE-reranker-v2-m3, returning the top-K. A cross-encoder
reads the query and passage together, so it is more precise than the bi-encoder
similarity used for the first-stage recall, at the cost of running per candidate.

The real reranker is lazy-loaded behind the optional `embed` group; FakeReranker
is a deterministic, dependency-free double (scores by query/passage word overlap)
for tests and offline runs.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from mneme.retrieval.dense import RetrievedChunk


@runtime_checkable
class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class FakeReranker:
    """Deterministic reranker scoring by shared-word count. For tests."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        terms = set(re.findall(r"\w+", query.lower()))
        return [
            float(len(terms & set(re.findall(r"\w+", passage.lower()))))
            for passage in passages
        ]


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class BGEReranker:
    """BGE-reranker-v2-m3 cross-encoder.

    Uses transformers' AutoModelForSequenceClassification directly (the model's
    documented usage) rather than FlagEmbedding's reranker wrapper, which calls a
    tokenizer method that recent transformers releases removed. transformers and
    torch ship with the embed group.
    """

    def __init__(
        self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "auto"
    ) -> None:
        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise ImportError(
                "Reranking requires the embed group: `uv sync --group embed`."
            ) from exc
        self._torch = torch
        self._device = _resolve_device(device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.to(self._device)
        self._model.eval()

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        torch = self._torch
        pairs = [[query, passage] for passage in passages]
        with torch.no_grad():
            inputs = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            logits = self._model(**inputs, return_dict=True).logits.view(-1).float()
        return logits.tolist()


def rerank(
    query: str,
    candidates: list[RetrievedChunk],
    reranker: Reranker,
    *,
    top_n: int = 5,
) -> list[RetrievedChunk]:
    """Rescore candidates with the cross-encoder and return the top_n."""
    if not candidates:
        return []
    scores = reranker.score(query, [c.chunk.text for c in candidates])
    ordered = sorted(zip(candidates, scores), key=lambda cs: cs[1], reverse=True)
    return [
        RetrievedChunk(chunk=item.chunk, score=float(score))
        for item, score in ordered[:top_n]
    ]
