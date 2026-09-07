"""
Ties the ingestion pipeline together: load -> chunk -> embed -> upsert into the vector
store. Deletes a document's existing chunks before re-adding them (delete_document_chunks)
so re-running indexing on an updated document doesn't leave stale chunks from a longer
previous version -- this is what makes re-indexing safe (FEATURES.md Tier 2).
"""
from dataclasses import dataclass

from src.ingestion.document_loader import LoadedDocument, load_all_documents, load_document
from src.ingestion.chunker import chunk_text
from src.retrieval.embeddings import embed_texts
from src.retrieval.vector_store import upsert_chunks, delete_document_chunks, get_collection

COLLECTION_NAME = "operations_policy_documents"


@dataclass
class IndexResult:
    document_id: str
    title: str
    chunk_count: int


def index_document(doc: LoadedDocument) -> IndexResult:
    chunks = chunk_text(doc.text)
    if not chunks:
        return IndexResult(document_id=doc.document_id, title=doc.title, chunk_count=0)

    delete_document_chunks(COLLECTION_NAME, doc.document_id)

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)
    ids = [f"{doc.document_id}::{c.chunk_index}" for c in chunks]
    metadatas = [
        {
            "document_id": doc.document_id,
            "title": doc.title,
            "version": doc.version or "",
            "section_title": c.section_title or "",
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]

    upsert_chunks(COLLECTION_NAME, ids, texts, embeddings, metadatas)
    return IndexResult(document_id=doc.document_id, title=doc.title, chunk_count=len(chunks))


def index_all_documents(directory: str = "data/documents") -> list[IndexResult]:
    docs = load_all_documents(directory)
    return [index_document(doc) for doc in docs]


def list_indexed_documents() -> list[dict]:
    """Groups chunks back up by document_id for a document-level listing (GET /documents),
    rather than exposing raw chunk records to the API consumer."""
    collection = get_collection(COLLECTION_NAME)
    result = collection.get(include=["metadatas"])

    by_doc: dict[str, dict] = {}
    for metadata in result.get("metadatas", []):
        doc_id = metadata["document_id"]
        if doc_id not in by_doc:
            by_doc[doc_id] = {
                "document_id": doc_id,
                "title": metadata["title"],
                "version": metadata["version"] or None,
                "chunk_count": 0,
            }
        by_doc[doc_id]["chunk_count"] += 1

    return list(by_doc.values())


if __name__ == "__main__":
    results = index_all_documents()
    for r in results:
        print(f"Indexed '{r.title}' ({r.document_id}): {r.chunk_count} chunks")
