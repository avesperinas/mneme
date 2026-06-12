"""Wikilink graph extraction (sub-phase 1.3).

Builds a directed note-to-note graph from the raw wikilink targets the parser
extracted. Targets are resolved to document ids using Obsidian-style filename
resolution (by vault-relative path, else by basename); targets that match no
document are recorded as unresolved and logged, never dropped silently. The
graph persists to SQLite so phase 6 can load it for graph-aware retrieval.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from mneme_ingest.models import Document

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GraphNode:
    id: str
    rel_path: str
    title: str


@dataclass(slots=True)
class Edge:
    source_id: str
    target_id: str
    target_raw: str  # the link text as written, for traceability


@dataclass(slots=True)
class UnresolvedLink:
    source_id: str
    source_rel_path: str
    target_raw: str


@dataclass(slots=True)
class LinkGraph:
    nodes: list[GraphNode]
    edges: list[Edge]
    unresolved: list[UnresolvedLink]


def _strip_md(rel_path: str) -> str:
    return rel_path[:-3] if rel_path.lower().endswith(".md") else rel_path


def _normalize_target(target: str) -> str:
    text = target.strip().replace("\\", "/")
    if text.startswith("./"):
        text = text[2:]
    return _strip_md(text)


def _resolve(
    target: str, by_path: dict[str, str], by_name: dict[str, str]
) -> str | None:
    norm = _normalize_target(target).lower()
    if norm in by_path:
        return by_path[norm]
    if "/" not in norm and norm in by_name:
        return by_name[norm]
    return None


def build_link_graph(documents: Iterable[Document]) -> LinkGraph:
    docs = list(documents)
    nodes = [GraphNode(id=d.id, rel_path=d.rel_path, title=d.title) for d in docs]

    # Resolution maps. Sort first so a basename collision resolves deterministically
    # to the lexicographically first path (first writer wins).
    by_path: dict[str, str] = {}
    by_name: dict[str, str] = {}
    for doc in sorted(docs, key=lambda d: d.rel_path):
        path_key = _strip_md(doc.rel_path).lower()
        by_path.setdefault(path_key, doc.id)
        by_name.setdefault(path_key.rsplit("/", 1)[-1], doc.id)

    edges: list[Edge] = []
    unresolved: list[UnresolvedLink] = []
    for doc in docs:
        for target in doc.wikilinks:
            target_id = _resolve(target, by_path, by_name)
            if target_id is None:
                unresolved.append(UnresolvedLink(doc.id, doc.rel_path, target))
            else:
                edges.append(Edge(doc.id, target_id, target))

    if unresolved:
        logger.warning("Found %d unresolved wikilink(s):", len(unresolved))
        for link in unresolved:
            logger.warning("  %s -> [[%s]]", link.source_rel_path, link.target_raw)

    return LinkGraph(nodes=nodes, edges=edges, unresolved=unresolved)


_SCHEMA = """
CREATE TABLE nodes (
    id TEXT PRIMARY KEY,
    rel_path TEXT NOT NULL,
    title TEXT NOT NULL
);
CREATE TABLE edges (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    target_raw TEXT NOT NULL
);
CREATE TABLE unresolved_links (
    source_id TEXT NOT NULL,
    source_rel_path TEXT NOT NULL,
    target_raw TEXT NOT NULL
);
CREATE INDEX idx_edges_source ON edges (source_id);
CREATE INDEX idx_edges_target ON edges (target_id);
"""


def save_graph(graph: LinkGraph, db_path: Path) -> None:
    """Persist the graph to SQLite, rebuilding the tables from scratch."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            "DROP TABLE IF EXISTS nodes;"
            "DROP TABLE IF EXISTS edges;"
            "DROP TABLE IF EXISTS unresolved_links;"
        )
        conn.executescript(_SCHEMA)
        conn.executemany(
            "INSERT INTO nodes VALUES (?, ?, ?)",
            [(n.id, n.rel_path, n.title) for n in graph.nodes],
        )
        conn.executemany(
            "INSERT INTO edges VALUES (?, ?, ?)",
            [(e.source_id, e.target_id, e.target_raw) for e in graph.edges],
        )
        conn.executemany(
            "INSERT INTO unresolved_links VALUES (?, ?, ?)",
            [(u.source_id, u.source_rel_path, u.target_raw) for u in graph.unresolved],
        )
        conn.commit()
    finally:
        conn.close()


def load_graph(db_path: Path) -> LinkGraph:
    conn = sqlite3.connect(db_path)
    try:
        nodes = [
            GraphNode(*row)
            for row in conn.execute("SELECT id, rel_path, title FROM nodes")
        ]
        edges = [
            Edge(*row)
            for row in conn.execute(
                "SELECT source_id, target_id, target_raw FROM edges"
            )
        ]
        unresolved = [
            UnresolvedLink(*row)
            for row in conn.execute(
                "SELECT source_id, source_rel_path, target_raw FROM unresolved_links"
            )
        ]
        return LinkGraph(nodes=nodes, edges=edges, unresolved=unresolved)
    finally:
        conn.close()
