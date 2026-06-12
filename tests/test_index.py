from pathlib import Path

import pytest
from mneme_ingest.graph import load_graph
from mneme_ingest.index import format_report, main, run_index

VAULT = Path(__file__).resolve().parent / "fixtures" / "sample_vault"


def test_run_index_over_fixture_vault():
    result = run_index(VAULT)
    assert len(result.documents) == 4
    assert len(result.chunks) > 0
    assert len(result.graph.edges) == 6
    assert len(result.graph.unresolved) == 1
    assert result.skipped == []


def test_report_numbers_are_internally_consistent():
    result = run_index(VAULT)
    total_links = sum(len(d.wikilinks) for d in result.documents)
    # every wikilink is either a resolved edge or recorded unresolved
    assert len(result.graph.edges) + len(result.graph.unresolved) == total_links
    # reported tokens equal the sum over chunks
    report = format_report(result, VAULT)
    total_tokens = sum(c.token_count for c in result.chunks)
    assert f"tokens:           {total_tokens}" in report
    assert f"documents:        {len(result.documents)}" in report
    assert "Nonexistent Note" in report  # unresolved listed for visibility


def test_run_index_persists_graph_when_requested(tmp_path):
    db = tmp_path / "graph.db"
    result = run_index(VAULT, graph_db=db)
    assert db.exists()
    loaded = load_graph(db)
    assert len(loaded.edges) == len(result.graph.edges)


def test_run_index_skips_malformed_note_without_crashing(tmp_path):
    (tmp_path / "good.md").write_text("# Good\n\nLinks to [[good]].\n")
    # invalid UTF-8 bytes -> read_text raises, note must be skipped not fatal
    (tmp_path / "bad.md").write_bytes(b"\xff\xfe not valid utf-8")

    result = run_index(tmp_path)
    assert len(result.documents) == 1
    assert result.documents[0].rel_path == "good.md"
    assert len(result.skipped) == 1
    assert result.skipped[0].name == "bad.md"


def test_main_prints_report(capsys):
    main(["--vault", str(VAULT)])
    out = capsys.readouterr().out
    assert "Mneme ingestion report" in out
    assert "documents:        4" in out


def test_main_errors_on_missing_vault(tmp_path):
    with pytest.raises(SystemExit):
        main(["--vault", str(tmp_path / "does-not-exist")])


def test_main_indexes_to_qdrant_when_url_given(capsys):
    main(
        [
            "--vault",
            str(VAULT),
            "--index",
            "--qdrant-url",
            ":memory:",
            "--embedder",
            "hash",
        ]
    )
    out = capsys.readouterr().out
    assert "indexed:          7 points" in out
