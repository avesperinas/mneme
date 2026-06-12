from pathlib import Path

from mneme_ingest.chunker import (
    _split_sentences,
    chunk_document,
    count_tokens,
)
from mneme_ingest.models import Document
from mneme_ingest.parser import parse_document, parse_vault, split_frontmatter

VAULT = Path(__file__).resolve().parent / "fixtures" / "sample_vault"


def _doc_and_body(rel: str) -> tuple[Document, str]:
    path = VAULT / rel
    doc = parse_document(path, VAULT)
    _, body = split_frontmatter(path.read_text())
    return doc, body


def _synthetic(body: str, *, title: str = "T") -> Document:
    return Document(
        id="docid",
        rel_path="x.md",
        title=title,
        frontmatter={},
        tags=["t"],
        wikilinks=[],
        mtime=0.0,
    )


# --- structure ------------------------------------------------------------


def test_chunks_follow_heading_hierarchy():
    doc, body = _doc_and_body("projects/Architecture.md")
    chunks = chunk_document(doc, body)
    assert [c.heading_path for c in chunks] == [
        ["System Architecture"],
        ["System Architecture", "Overview"],
        ["System Architecture", "Overview", "Components"],
    ]
    # ids are contiguous and id-format follows the contract
    assert [c.ordinal for c in chunks] == [0, 1, 2]
    assert chunks[1].id == "docid::1".replace("docid", doc.id)
    assert all(c.token_count > 0 for c in chunks)


def test_no_chunk_crosses_a_section_boundary():
    doc, body = _doc_and_body("projects/Architecture.md")
    chunks = chunk_document(doc, body)
    intro = next(c for c in chunks if c.heading_path == ["System Architecture"])
    components = next(c for c in chunks if c.heading_path[-1] == "Components")
    assert "Intro linking" in intro.text
    assert "Details" not in intro.text  # content from a sibling section
    assert "Details" in components.text
    assert "Intro linking" not in components.text


def test_every_chunk_has_non_empty_heading_path():
    for doc in parse_vault(VAULT):
        _, body = split_frontmatter((VAULT / doc.rel_path).read_text())
        for chunk in chunk_document(doc, body):
            assert chunk.heading_path
            assert all(part for part in chunk.heading_path)


def test_flat_note_uses_title_as_heading_path():
    doc, body = _doc_and_body("projects/Roadmap.md")
    chunks = chunk_document(doc, body)
    assert chunks
    assert all(c.heading_path == ["Roadmap"] for c in chunks)


# --- splitting, overlap, sentence safety ----------------------------------


def _sentences(n: int) -> list[str]:
    return [f"Word{i} alpha beta gamma." for i in range(n)]


def test_long_section_splits_with_sentence_overlap():
    sentences = _sentences(10)
    body = "# Section\n\n" + " ".join(sentences)
    doc = _synthetic(body)
    chunks = chunk_document(doc, body, target_tokens=12, overlap_tokens=6)

    assert len(chunks) > 1
    assert all(c.heading_path == ["Section"] for c in chunks)
    assert all(c.token_count <= 12 for c in chunks)

    # consecutive chunks overlap by exactly the shared boundary sentence
    for a, b in zip(chunks, chunks[1:]):
        assert _split_sentences(a.text)[-1] == _split_sentences(b.text)[0]

    # every original sentence is covered, none lost
    covered = {s for c in chunks for s in _split_sentences(c.text)}
    assert covered == set(sentences)


def test_never_splits_mid_sentence():
    sentences = _sentences(8)
    body = "# S\n\n" + " ".join(sentences)
    doc = _synthetic(body)
    chunks = chunk_document(doc, body, target_tokens=12, overlap_tokens=0)
    joined = [c.text for c in chunks]
    for sentence in sentences:
        assert any(sentence in text for text in joined)


def test_oversized_single_sentence_stays_whole():
    sentence = " ".join(["Word"] * 19) + " end."
    body = "# S\n\n" + sentence
    doc = _synthetic(body)
    chunks = chunk_document(doc, body, target_tokens=12, overlap_tokens=4)
    # a sentence longer than the target is emitted whole, never cut
    assert len(chunks) == 1
    assert chunks[0].text == sentence
    assert chunks[0].token_count > 12


def test_no_heading_fallback_is_recursive_not_blind():
    sentences = _sentences(10)
    body = " ".join(sentences)  # no headings at all
    doc = _synthetic(body, title="Flat")
    chunks = chunk_document(doc, body, target_tokens=12, overlap_tokens=0)
    assert len(chunks) > 1
    assert all(c.heading_path == ["Flat"] for c in chunks)
    # split on sentence boundaries, so each original sentence stays intact
    covered = {s for c in chunks for s in _split_sentences(c.text)}
    assert covered == set(sentences)


def test_token_count_matches_counter():
    doc, body = _doc_and_body("Retrieval.md")
    for chunk in chunk_document(doc, body):
        assert chunk.token_count == count_tokens(chunk.text)
        assert chunk.id == f"{doc.id}::{chunk.ordinal}"
        assert chunk.tags == doc.tags
