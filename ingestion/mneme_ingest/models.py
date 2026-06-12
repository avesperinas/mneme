"""Core ingestion data contracts (spec section 3.3).

Document is produced by the parser (1.1); Chunk by the structural chunker (1.2).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Document:
    id: str  # stable hash of the vault-relative path
    rel_path: str
    title: str
    frontmatter: dict
    tags: list[str]
    wikilinks: list[str]  # outgoing link targets (note titles/paths), deduped
    mtime: float


@dataclass(slots=True)
class Chunk:
    id: str  # f"{document_id}::{ordinal}"
    document_id: str
    rel_path: str
    heading_path: list[str]  # ancestor headings, always non-empty
    text: str
    tags: list[str]
    ordinal: int
    token_count: int


@dataclass(slots=True)
class Heading:
    """A single ATX heading; positions feed the structural chunker (1.2)."""

    level: int  # 1-6
    text: str
    line: int  # 0-based line index within the document body
