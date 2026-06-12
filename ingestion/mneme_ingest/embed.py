"""Embeddings (sub-phase 2.1).

An Embedder turns text into dense vectors. The real implementation is BGE-M3
(BAAI/bge-m3) via FlagEmbedding, which also yields sparse vectors that phase 4
will use; it is loaded lazily and lives behind the optional `embed` dependency
group, so the default dev/test environment stays light. HashEmbedder is a
deterministic, dependency-free double for tests and offline pipeline checks (not
for real retrieval quality).

The same Embedder is used to index chunks (ingestion) and to embed queries
(retrieval), so dense vectors are always produced by the same model.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Protocol, runtime_checkable

BGE_M3_DENSE_DIM = 1024


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Deterministic pseudo-embedding from a hash. For tests and offline runs."""

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        values: list[float] = []
        counter = 0
        while len(values) < self.dim:
            digest = hashlib.sha256(f"{counter}:{text}".encode()).digest()
            for offset in range(0, len(digest), 4):
                if len(values) >= self.dim:
                    break
                (raw,) = struct.unpack("<I", digest[offset : offset + 4])
                values.append((raw / 0xFFFFFFFF) * 2.0 - 1.0)
            counter += 1
        return values

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class BGEM3Embedder:
    """BGE-M3 dense embeddings via FlagEmbedding (installed via the embed group)."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "auto") -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except (
            ImportError
        ) as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "BGE-M3 requires FlagEmbedding. Install the embed group: "
                "`uv sync --group embed`."
            ) from exc

        resolved = _resolve_device(device)
        self._model = BGEM3FlagModel(
            model_name, use_fp16=(resolved == "cuda"), devices=resolved
        )
        self.dim = BGE_M3_DENSE_DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        output = self._model.encode(
            texts,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return [vector.tolist() for vector in output["dense_vecs"]]
