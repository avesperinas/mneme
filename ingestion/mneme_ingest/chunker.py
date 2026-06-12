"""Structure-aware chunker (sub-phase 1.2).

Splits a document along its heading hierarchy. Each heading's content becomes
an independent segment carrying the full ancestor heading_path; content is never
merged across sibling sections. A segment that exceeds the token target is split
recursively on sentence boundaries (never mid-sentence) with bounded, sentence-
granular overlap. Notes with no headings fall back to the same recursive split
under heading_path = [title]; fixed-size character slicing is never used.

The heading line itself is not included in the chunk text: it lives in
heading_path and is prepended at retrieval time (spec 3.3).

token_count uses a lightweight, deterministic, dependency-free counter by
default. It is injectable so phase 2 can substitute the embedding model's
tokenizer for an exact count if needed.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from mneme_ingest.models import Chunk, Document
from mneme_ingest.parser import parse_headings

DEFAULT_TARGET_TOKENS = 400
DEFAULT_OVERLAP_TOKENS = 50

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n\s*\n")

TokenCounter = Callable[[str], int]


def count_tokens(text: str) -> int:
    """Approximate token count: words plus standalone punctuation marks."""
    return len(_TOKEN_RE.findall(text))


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]


def _overlap_tail(
    sentences: list[str], overlap_tokens: int, count: TokenCounter
) -> list[str]:
    """Trailing whole sentences whose cumulative tokens stay within overlap."""
    tail: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        cost = count(sentence)
        if total + cost > overlap_tokens:
            break
        tail.insert(0, sentence)
        total += cost
    return tail


def _pack_sentences(
    sentences: list[str],
    target_tokens: int,
    overlap_tokens: int,
    count: TokenCounter,
) -> list[str]:
    chunks: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        cost = count(sentence)
        if current and current_tokens + cost > target_tokens:
            chunks.append(current)
            current = _overlap_tail(current, overlap_tokens, count)
            current_tokens = sum(count(s) for s in current)
        current.append(sentence)
        current_tokens += cost
    if current:
        chunks.append(current)
    return [" ".join(c) for c in chunks]


def _split_segment(
    text: str, target_tokens: int, overlap_tokens: int, count: TokenCounter
) -> list[str]:
    if count(text) <= target_tokens:
        return [text]
    sentences = _split_sentences(text)
    if not sentences:
        return [text]
    return _pack_sentences(sentences, target_tokens, overlap_tokens, count)


def _segments(body: str, title: str) -> list[tuple[list[str], str]]:
    """Ordered (heading_path, text) segments built from the heading hierarchy."""
    lines = body.splitlines()
    headings = parse_headings(body)
    segments: list[tuple[list[str], str]] = []

    first = headings[0].line if headings else len(lines)
    preamble = "\n".join(lines[:first]).strip()
    if preamble:
        segments.append(([title], preamble))

    stack = []
    for index, heading in enumerate(headings):
        while stack and stack[-1].level >= heading.level:
            stack.pop()
        stack.append(heading)
        heading_path = [h.text for h in stack]
        start = heading.line + 1
        end = headings[index + 1].line if index + 1 < len(headings) else len(lines)
        text = "\n".join(lines[start:end]).strip()
        if text:
            segments.append((heading_path, text))
    return segments


def chunk_document(
    doc: Document,
    body: str,
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    token_counter: TokenCounter = count_tokens,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    ordinal = 0
    for heading_path, text in _segments(body, doc.title):
        for piece in _split_segment(text, target_tokens, overlap_tokens, token_counter):
            chunks.append(
                Chunk(
                    id=f"{doc.id}::{ordinal}",
                    document_id=doc.id,
                    rel_path=doc.rel_path,
                    heading_path=list(heading_path),
                    text=piece,
                    tags=list(doc.tags),
                    ordinal=ordinal,
                    token_count=token_counter(piece),
                )
            )
            ordinal += 1
    return chunks
