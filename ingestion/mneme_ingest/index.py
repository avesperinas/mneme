"""Index CLI (sub-phase 1.4).

Runs the full ingestion pipeline over an Obsidian vault:
parse -> chunk -> graph, then prints a stats report.

    python -m mneme_ingest.index --vault <path>

The run is robust: a single malformed note is logged and skipped rather than
crashing the whole index. The link graph is optionally persisted to SQLite with
--graph-db. Chunk embedding and upsert are phase 2.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from mneme_ingest.chunker import (
    DEFAULT_OVERLAP_TOKENS,
    DEFAULT_TARGET_TOKENS,
    chunk_document,
)
from mneme_ingest.graph import LinkGraph, build_link_graph, save_graph
from mneme_ingest.models import Chunk, Document
from mneme_ingest.parser import parse_document, split_frontmatter

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IndexResult:
    documents: list[Document]
    chunks: list[Chunk]
    graph: LinkGraph
    skipped: list[Path]


def run_index(
    vault: Path,
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    graph_db: Path | None = None,
) -> IndexResult:
    documents: list[Document] = []
    chunks: list[Chunk] = []
    skipped: list[Path] = []

    for path in sorted(vault.rglob("*.md")):
        try:
            doc = parse_document(path, vault)
            _, body = split_frontmatter(path.read_text(encoding="utf-8"))
            doc_chunks = chunk_document(
                doc,
                body,
                target_tokens=target_tokens,
                overlap_tokens=overlap_tokens,
            )
        except Exception as exc:  # never let one bad note abort the whole index
            logger.warning("skipping %s: %s", path, exc)
            skipped.append(path)
            continue
        documents.append(doc)
        chunks.extend(doc_chunks)

    graph = build_link_graph(documents)
    if graph_db is not None:
        save_graph(graph, graph_db)

    return IndexResult(documents=documents, chunks=chunks, graph=graph, skipped=skipped)


def format_report(result: IndexResult, vault: Path) -> str:
    total_tokens = sum(c.token_count for c in result.chunks)
    lines = [
        "Mneme ingestion report",
        f"  vault:            {vault}",
        f"  documents:        {len(result.documents)}",
        f"  chunks:           {len(result.chunks)}",
        f"  tokens:           {total_tokens}",
        f"  links (resolved): {len(result.graph.edges)}",
        f"  unresolved links: {len(result.graph.unresolved)}",
    ]
    if result.skipped:
        lines.append(f"  skipped notes:    {len(result.skipped)}")
    if result.graph.unresolved:
        lines.append("  unresolved:")
        lines.extend(
            f"    {link.source_rel_path} -> [[{link.target_raw}]]"
            for link in result.graph.unresolved
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m mneme_ingest.index",
        description="Parse, chunk, and graph an Obsidian vault.",
    )
    parser.add_argument(
        "--vault",
        type=Path,
        default=os.environ.get("VAULT_PATH"),
        help="vault directory (defaults to $VAULT_PATH)",
    )
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    parser.add_argument("--overlap-tokens", type=int, default=DEFAULT_OVERLAP_TOKENS)
    parser.add_argument(
        "--graph-db",
        type=Path,
        default=None,
        help="optional SQLite path to persist the link graph",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="embed chunks and upsert them to Qdrant (otherwise report only)",
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.environ.get("QDRANT_URL"),
        help="Qdrant URL (':memory:' for a local store); defaults to $QDRANT_URL",
    )
    parser.add_argument(
        "--collection",
        default=os.environ.get("QDRANT_COLLECTION", "mneme"),
    )
    parser.add_argument(
        "--embedder",
        choices=["bge", "hash"],
        default="bge",
        help="bge = real BGE-M3 (needs the embed group); hash = offline double",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    if args.vault is None:
        parser.error("provide --vault or set VAULT_PATH")
    if not args.vault.is_dir():
        parser.error(f"vault path not found: {args.vault}")

    result = run_index(
        args.vault,
        target_tokens=args.target_tokens,
        overlap_tokens=args.overlap_tokens,
        graph_db=args.graph_db,
    )
    print(format_report(result, args.vault))

    if args.index:
        if not args.qdrant_url:
            parser.error("--index requires --qdrant-url or $QDRANT_URL")

        from mneme_ingest.embed import BGEM3Embedder, HashEmbedder
        from mneme_ingest.vectorstore import index_chunks, make_client

        embedder = (
            HashEmbedder()
            if args.embedder == "hash"
            else BGEM3Embedder(device=os.environ.get("EMBED_DEVICE", "auto"))
        )
        client = make_client(args.qdrant_url)
        written = index_chunks(client, args.collection, result.chunks, embedder)
        print(
            f"  indexed:          {written} points -> "
            f"{args.collection} @ {args.qdrant_url}"
        )


if __name__ == "__main__":
    main()
