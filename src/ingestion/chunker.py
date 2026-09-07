"""
Splits document text into chunks for embedding. Chunking strategy: section-aware first
(split on markdown headers, since every synthetic policy doc is authored with numbered
sections), then size-bounded within a section for any section that's still too long.

Why section-aware rather than fixed-size sliding window: these policy documents are
short (1-2KB each) and each numbered section is a self-contained unit of policy meaning
(e.g. "Section 4.2 Secondary Approval"). Splitting mid-section with a generic sliding
window risks cutting a rule in half across two chunks and losing the citation's meaning
("Section 4.2" should retrieve the whole rule, not an arbitrary 400-token slice of it).
A fixed-size window is the right default for long-form prose; it's the wrong default
for short, structured policy documents like these -- see docs/rag-design.md.
"""
import re
from dataclasses import dataclass

CHUNK_SIZE_TOKENS = 400  # from config/retrieval.yaml -- kept in sync manually for now
CHUNK_OVERLAP_TOKENS = 60

SECTION_HEADER_PATTERN = re.compile(r"^(#{1,3}\s+.+)$", re.MULTILINE)


@dataclass
class Chunk:
    text: str
    section_title: str | None
    chunk_index: int


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (chars / 4) -- good enough for chunk-size decisions,
    not meant to match the actual embedding model's tokenizer exactly."""
    return len(text) // 4


def _split_by_section(text: str) -> list[tuple[str | None, str]]:
    """Returns (section_title, section_text) pairs. Text before the first header
    (if any) gets section_title=None."""
    headers = list(SECTION_HEADER_PATTERN.finditer(text))
    if not headers:
        return [(None, text)]

    sections = []
    if headers[0].start() > 0:
        sections.append((None, text[: headers[0].start()].strip()))

    for i, header_match in enumerate(headers):
        title = header_match.group(1).lstrip("#").strip()
        start = header_match.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        section_text = text[start:end].strip()
        sections.append((title, f"{header_match.group(1)}\n{section_text}"))

    return sections


def _split_oversized_section(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Fallback sliding-window split for any single section still too long after
    section-aware splitting -- rare for these documents, but a real long policy
    document could have an oversized section, so this path needs to exist."""
    max_chars = max_tokens * 4
    overlap_chars = overlap_tokens * 4
    if len(text) <= max_chars:
        return [text]

    pieces = []
    start = 0
    while start < len(text):
        end = start + max_chars
        pieces.append(text[start:end])
        start = end - overlap_chars
    return pieces


def chunk_text(
    text: str,
    chunk_size_tokens: int = CHUNK_SIZE_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    sections = _split_by_section(text)
    chunks = []
    idx = 0

    for section_title, section_text in sections:
        if not section_text:
            continue
        if _estimate_tokens(section_text) <= chunk_size_tokens:
            chunks.append(Chunk(text=section_text, section_title=section_title, chunk_index=idx))
            idx += 1
        else:
            for piece in _split_oversized_section(section_text, chunk_size_tokens, overlap_tokens):
                chunks.append(Chunk(text=piece, section_title=section_title, chunk_index=idx))
                idx += 1

    return chunks


if __name__ == "__main__":
    from src.ingestion.document_loader import load_all_documents

    for doc in load_all_documents():
        chunks = chunk_text(doc.text)
        print(f"{doc.document_id}: {len(chunks)} chunks")
        for c in chunks:
            print(f"  [{c.chunk_index}] section={c.section_title!r} ({_estimate_tokens(c.text)} est. tokens)")
