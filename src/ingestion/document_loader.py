"""
Loads a document and its frontmatter metadata. Supports markdown with YAML frontmatter
(what data/documents/*.md actually use) and plain text/PDF as a fallback. Markdown +
frontmatter was chosen over actual .pdf/.docx for the synthetic corpus specifically
because it's easy to diff/version in git and easy to author realistically -- see
docs/rag-design.md for that reasoning.
"""
from dataclasses import dataclass
from pathlib import Path
import re

try:
    import yaml
except ImportError:  # pyyaml is in requirements.txt; guarded for clarity if run standalone
    yaml = None


@dataclass
class LoadedDocument:
    document_id: str  # derived from filename, stable across re-indexing
    title: str
    version: str | None
    effective_date: str | None
    department: str | None
    text: str
    source_path: str


FRONTMATTER_PATTERN = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


def load_document(path: str | Path) -> LoadedDocument:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found at {path}")

    if path.suffix.lower() == ".pdf":
        return _load_pdf(path)

    raw = path.read_text(encoding="utf-8")
    match = FRONTMATTER_PATTERN.match(raw)

    if match and yaml:
        frontmatter = yaml.safe_load(match.group(1)) or {}
        body = match.group(2).strip()
    else:
        frontmatter = {}
        body = raw.strip()

    return LoadedDocument(
        document_id=path.stem,
        title=frontmatter.get("title", path.stem),
        version=frontmatter.get("version"),
        effective_date=str(frontmatter.get("effective_date")) if frontmatter.get("effective_date") else None,
        department=frontmatter.get("department"),
        text=body,
        source_path=str(path),
    )


def _load_pdf(path: Path) -> LoadedDocument:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return LoadedDocument(
        document_id=path.stem,
        title=path.stem.replace("-", " ").replace("_", " ").title(),
        version=None,
        effective_date=None,
        department=None,
        text=text.strip(),
        source_path=str(path),
    )


def load_all_documents(directory: str | Path = "data/documents") -> list[LoadedDocument]:
    """Loads every policy document in the directory -- explicitly excludes README.md
    and any other file starting with README, since those are directory documentation,
    not corpus content, and shouldn't be indexed or counted as a document."""
    directory = Path(directory)
    docs = []
    for path in sorted(directory.glob("*.md")) + sorted(directory.glob("*.pdf")):
        if path.stem.upper().startswith("README"):
            continue
        docs.append(load_document(path))
    return docs


if __name__ == "__main__":
    docs = load_all_documents()
    for d in docs:
        print(f"{d.document_id}: '{d.title}' v{d.version} ({len(d.text)} chars)")
