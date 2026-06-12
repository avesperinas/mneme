import logging
from pathlib import Path

from mneme_ingest.graph import build_link_graph, load_graph, save_graph
from mneme_ingest.parser import parse_vault

VAULT = Path(__file__).resolve().parent / "fixtures" / "sample_vault"


def _edges_as_relpaths(graph):
    by_id = {n.id: n.rel_path for n in graph.nodes}
    return sorted((by_id[e.source_id], by_id[e.target_id]) for e in graph.edges)


def test_node_count_equals_document_count():
    docs = parse_vault(VAULT)
    graph = build_link_graph(docs)
    assert len(graph.nodes) == len(docs)


def test_resolved_edges_match_wikilinks():
    docs = parse_vault(VAULT)
    graph = build_link_graph(docs)

    # Resolution works both by path (projects/Roadmap) and by basename (Architecture).
    assert _edges_as_relpaths(graph) == sorted(
        [
            ("Daily Note.md", "projects/Architecture.md"),
            ("Daily Note.md", "Retrieval.md"),
            ("Retrieval.md", "projects/Architecture.md"),
            ("projects/Architecture.md", "Retrieval.md"),
            ("projects/Architecture.md", "projects/Roadmap.md"),
            ("projects/Roadmap.md", "projects/Architecture.md"),
        ]
    )

    # edge count == resolved wikilinks == total links minus unresolved
    total_links = sum(len(d.wikilinks) for d in docs)
    assert len(graph.edges) == total_links - len(graph.unresolved)


def test_unresolved_links_recorded_separately():
    graph = build_link_graph(parse_vault(VAULT))
    by_id = {n.id: n.rel_path for n in graph.nodes}
    assert len(graph.unresolved) == 1
    only = graph.unresolved[0]
    assert by_id[only.source_id] == "Daily Note.md"
    assert only.target_raw == "Nonexistent Note"


def test_unresolved_links_are_logged_not_silent(caplog):
    with caplog.at_level(logging.WARNING, logger="mneme_ingest.graph"):
        build_link_graph(parse_vault(VAULT))
    assert "unresolved" in caplog.text.lower()
    assert "Nonexistent Note" in caplog.text


def test_graph_round_trips_through_sqlite(tmp_path):
    graph = build_link_graph(parse_vault(VAULT))
    db = tmp_path / "graph.db"
    save_graph(graph, db)
    loaded = load_graph(db)

    assert {(n.id, n.rel_path, n.title) for n in loaded.nodes} == {
        (n.id, n.rel_path, n.title) for n in graph.nodes
    }
    assert _edges_as_relpaths(loaded) == _edges_as_relpaths(graph)
    assert [(u.source_rel_path, u.target_raw) for u in loaded.unresolved] == [
        (u.source_rel_path, u.target_raw) for u in graph.unresolved
    ]


def test_save_graph_is_idempotent(tmp_path):
    graph = build_link_graph(parse_vault(VAULT))
    db = tmp_path / "graph.db"
    save_graph(graph, db)
    save_graph(graph, db)  # rebuilding over an existing db must not duplicate
    loaded = load_graph(db)
    assert len(loaded.edges) == len(graph.edges)
