from pathlib import Path

from mneme_ingest.parser import (
    document_id,
    parse_document,
    parse_headings,
    parse_vault,
    split_frontmatter,
)

VAULT = Path(__file__).resolve().parent / "fixtures" / "sample_vault"


def _by_rel_path():
    return {doc.rel_path: doc for doc in parse_vault(VAULT)}


def test_parse_vault_finds_all_notes():
    docs = _by_rel_path()
    assert set(docs) == {
        "Daily Note.md",
        "Retrieval.md",
        "projects/Architecture.md",
        "projects/Roadmap.md",
    }


def test_frontmatter_and_tags_extracted():
    doc = _by_rel_path()["projects/Architecture.md"]
    assert doc.title == "System Architecture"
    assert doc.frontmatter["status"] == "active"
    # frontmatter [design, core] plus body #design/system, deduped and sorted
    assert doc.tags == ["core", "design", "design/system"]


def test_wikilink_forms_normalized_to_note_targets():
    doc = _by_rel_path()["projects/Architecture.md"]
    # [[Retrieval]], [[projects/Roadmap|alias]], [[#Components]] (skipped),
    # [[Retrieval#Dense Search]] (deduped) -> two distinct targets.
    assert doc.wikilinks == ["Retrieval", "projects/Roadmap"]


def test_code_blocks_and_spans_are_ignored():
    doc = _by_rel_path()["Retrieval.md"]
    assert doc.wikilinks == ["Architecture"]  # FakeLink / AlsoFake were in code
    assert doc.tags == ["core"]  # #fake-tag / #alsofake were in code


def test_title_falls_back_to_first_h1():
    doc = _by_rel_path()["Daily Note.md"]
    assert doc.title == "Daily Note"
    assert doc.frontmatter == {}


def test_embeds_and_unresolved_links_are_outgoing_targets():
    doc = _by_rel_path()["Daily Note.md"]
    # plain link, ![[embed]], and an as-yet unresolved target, in order
    assert doc.wikilinks == ["Architecture", "Retrieval", "Nonexistent Note"]
    assert doc.tags == ["journal"]


def test_flat_note_without_headings():
    doc = _by_rel_path()["projects/Roadmap.md"]
    assert doc.title == "Roadmap"
    assert doc.tags == ["planning"]
    assert doc.wikilinks == ["Architecture"]


def test_document_id_is_stable_and_path_derived():
    docs = _by_rel_path()
    assert docs["Retrieval.md"].id == document_id("Retrieval.md")
    # deterministic across calls
    assert document_id("a/b.md") == document_id("a/b.md")
    assert document_id("a/b.md") != document_id("a/c.md")


def test_parse_headings_captures_hierarchy_ignoring_code():
    doc_path = VAULT / "projects" / "Architecture.md"
    _, body = split_frontmatter(doc_path.read_text())
    headings = [(h.level, h.text) for h in parse_headings(body)]
    assert headings == [
        (1, "System Architecture"),
        (2, "Overview"),
        (3, "Components"),
    ]


def test_parse_headings_skips_fenced_code_comments():
    doc_path = VAULT / "Retrieval.md"
    _, body = split_frontmatter(doc_path.read_text())
    headings = [(h.level, h.text) for h in parse_headings(body)]
    # the '# this ...' line inside the python fence must not be a heading
    assert headings == [(1, "Retrieval"), (2, "Dense Search")]


def test_parse_document_directly():
    doc = parse_document(VAULT / "Retrieval.md", VAULT)
    assert doc.rel_path == "Retrieval.md"
    assert doc.mtime > 0
