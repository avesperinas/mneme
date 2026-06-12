"""Obsidian markdown parser (sub-phase 1.1).

Parses a note into a Document: YAML frontmatter, #tags, outgoing [[wikilinks]]
(including [[note#heading]], [[note|alias]], folder paths, and ![[embeds]]),
the title, and a stable id. Heading hierarchy is exposed separately for the
chunker. Code spans and fenced code blocks are stripped before scanning for
tags and links so that examples inside code are never mistaken for real ones.

Link targets are extracted as written; resolving them to document ids (and
recording unresolved ones) is the graph stage's job (1.3).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml

from mneme_ingest.models import Document, Heading

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)
_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
# A tag is '#' followed by a letter then word/slash/hyphen chars, not preceded
# by a word char or slash (so mid-word '#' and '## headings' never match).
_TAG_RE = re.compile(r"(?<![\w/])#([A-Za-z][\w/-]*)")
_WIKILINK_RE = re.compile(r"\[\[([^\[\]]+?)\]\]")
_H1_RE = re.compile(r"^#\s+(.+?)\s*#*\s*$", re.MULTILINE)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def document_id(rel_path: str) -> str:
    """Stable 16-char id from the vault-relative (posix) path."""
    return hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:16]


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Return (frontmatter, body). Frontmatter is {} when absent or malformed."""
    if text.startswith("---"):
        match = _FRONTMATTER_RE.match(text)
        if match:
            meta = yaml.safe_load(match.group(1))
            if not isinstance(meta, dict):
                meta = {}
            return meta, text[match.end() :]
    return {}, text


def strip_code(body: str) -> str:
    """Remove fenced blocks and inline code so they are ignored when scanning."""
    body = _FENCED_CODE_RE.sub("", body)
    return _INLINE_CODE_RE.sub("", body)


def _frontmatter_tags(meta: dict) -> list[str]:
    raw = meta.get("tags")
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = re.split(r"[,\s]+", raw)
    elif isinstance(raw, list | tuple):
        parts = [str(t) for t in raw if t is not None]
    else:
        return []
    return [p.strip().lstrip("#") for p in parts if p and p.strip()]


def extract_tags(meta: dict, clean_body: str) -> list[str]:
    """Tags from frontmatter and body, normalized, deduped, sorted.

    Wikilinks are removed first so a heading anchor like [[#Section]] is not
    mistaken for a #tag.
    """
    body_tags = _TAG_RE.findall(_WIKILINK_RE.sub(" ", clean_body))
    return sorted(set(_frontmatter_tags(meta) + body_tags))


def extract_wikilinks(clean_body: str) -> list[str]:
    """Outgoing note targets, deduped in first-seen order.

    Drops the alias (after '|') and the heading anchor (after '#'). Pure
    intra-note heading links like [[#Section]] resolve to an empty target and
    are skipped. Embeds ![[Note]] are treated as outgoing links.
    """
    seen: set[str] = set()
    out: list[str] = []
    for inner in _WIKILINK_RE.findall(clean_body):
        target = inner.split("|", 1)[0].split("#", 1)[0].strip()
        if target and target not in seen:
            seen.add(target)
            out.append(target)
    return out


def parse_headings(body: str) -> list[Heading]:
    """ATX headings in document order, ignoring those inside fenced code."""
    headings: list[Heading] = []
    in_fence = False
    for index, line in enumerate(body.splitlines()):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _HEADING_RE.match(line)
        if match:
            headings.append(
                Heading(
                    level=len(match.group(1)),
                    text=match.group(2).strip(),
                    line=index,
                )
            )
    return headings


def _title(meta: dict, clean_body: str, path: Path) -> str:
    if meta.get("title"):
        return str(meta["title"])
    match = _H1_RE.search(clean_body)
    if match:
        return match.group(1).strip()
    return path.stem


def parse_document(path: Path, vault_root: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    meta, body = split_frontmatter(text)
    clean = strip_code(body)
    rel_path = path.relative_to(vault_root).as_posix()
    return Document(
        id=document_id(rel_path),
        rel_path=rel_path,
        title=_title(meta, clean, path),
        frontmatter=meta,
        tags=extract_tags(meta, clean),
        wikilinks=extract_wikilinks(clean),
        mtime=path.stat().st_mtime,
    )


def parse_vault(vault_root: Path) -> list[Document]:
    """Parse every .md note under vault_root, sorted by path for determinism."""
    return [
        parse_document(path, vault_root) for path in sorted(vault_root.rglob("*.md"))
    ]
