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
import re
import struct
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

BGE_M3_DENSE_DIM = 1024
_SPARSE_DIM = 100_000  # index space for the hash-based sparse double


@dataclass(slots=True)
class SparseVector:
    indices: list[int]
    values: list[float]


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    # Dense and sparse in one pass (BGE-M3 produces both); used for hybrid.
    def embed_both(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[SparseVector]]: ...


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

    def _sparse(self, text: str) -> SparseVector:
        # Word-keyed sparse, so exact tokens shared by query and chunk overlap.
        weights: dict[int, float] = {}
        for word in set(re.findall(r"\w+", text.lower())):
            index = int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "little")
            weights[index % _SPARSE_DIM] = 1.0
        return SparseVector(indices=list(weights), values=list(weights.values()))

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_both(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[SparseVector]]:
        return [self._vector(t) for t in texts], [self._sparse(t) for t in texts]


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

    def embed_both(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[SparseVector]]:
        output = self._model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = [vector.tolist() for vector in output["dense_vecs"]]
        sparse = [
            SparseVector(
                indices=[int(token) for token in weights],
                values=[float(weight) for weight in weights.values()],
            )
            for weights in output["lexical_weights"]
        ]
        return dense, sparse
