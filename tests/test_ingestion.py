import pytest
from src.ingestion.document_loader import load_document, load_all_documents
from src.ingestion.chunker import chunk_text, _split_by_section, _estimate_tokens


def test_load_document_parses_frontmatter(tmp_path):
    doc_path = tmp_path / "test-policy.md"
    doc_path.write_text(
        "---\ntitle: Test Policy\nversion: \"1.0\"\neffective_date: 2024-01-01\ndepartment: Ops\n---\n\n"
        "# 1. Purpose\nThis is a test.\n"
    )
    doc = load_document(doc_path)
    assert doc.title == "Test Policy"
    assert doc.version == "1.0"
    assert doc.department == "Ops"
    assert "This is a test" in doc.text
    assert "---" not in doc.text  # frontmatter stripped from body


def test_load_document_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_document(tmp_path / "does-not-exist.md")


def test_load_document_handles_no_frontmatter(tmp_path):
    doc_path = tmp_path / "plain.md"
    doc_path.write_text("Just plain text, no frontmatter.")
    doc = load_document(doc_path)
    assert doc.title == "plain"  # falls back to filename
    assert "Just plain text" in doc.text


def test_load_all_documents_finds_real_corpus():
    docs = load_all_documents("data/documents")
    assert len(docs) == 10  # the 10 synthetic policy docs written for this project
    titles = {d.title for d in docs}
    assert "Procurement Policy" in titles
    assert "SLA Policy" in titles


def test_split_by_section_finds_numbered_headers():
    text = "# 1. First\nContent one.\n# 2. Second\nContent two."
    sections = _split_by_section(text)
    assert len(sections) == 2
    assert sections[0][0] == "1. First"
    assert sections[1][0] == "2. Second"


def test_split_by_section_handles_no_headers():
    sections = _split_by_section("Just some text with no headers at all.")
    assert len(sections) == 1
    assert sections[0][0] is None


def test_chunk_text_produces_one_chunk_per_short_section():
    text = "# 1. First\nShort content.\n# 2. Second\nAlso short."
    chunks = chunk_text(text)
    assert len(chunks) == 2
    assert chunks[0].section_title == "1. First"
    assert chunks[1].section_title == "2. Second"


def test_chunk_text_splits_oversized_section():
    long_content = "word " * 500  # well over 400-token estimate
    text = f"# 1. Long Section\n{long_content}"
    chunks = chunk_text(text, chunk_size_tokens=100, overlap_tokens=10)
    assert len(chunks) > 1
    assert all(c.section_title == "1. Long Section" for c in chunks)


def test_chunk_text_on_real_procurement_policy_produces_multiple_sections():
    from src.ingestion.document_loader import load_document
    doc = load_document("data/documents/procurement-policy.md")
    chunks = chunk_text(doc.text)
    assert len(chunks) >= 5  # the document has 7 numbered sections
    section_titles = {c.section_title for c in chunks}
    assert any("Secondary Approval" in (t or "") for t in section_titles)
