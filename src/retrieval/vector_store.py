"""
Chroma in local persistent mode -- not a hosted vector DB. Chosen for the same reason
as the local embedding model: this project's document volume (a handful of policy
documents, dozens of chunks) doesn't need a hosted service, and local persistence
keeps the whole system runnable with `docker compose up` and no external account.
See docs/rag-design.md for the reasoning and what would change at real scale.
"""
import os
from functools import lru_cache


@lru_cache(maxsize=1)
def get_client():
    import chromadb
    persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma")
    return chromadb.PersistentClient(path=persist_dir)


def get_collection(name: str = "operations_policy_documents"):
    client = get_client()
    return client.get_or_create_collection(name=name)


def upsert_chunks(
    collection_name: str,
    ids: list[str],
    texts: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
) -> None:
    collection = get_collection(collection_name)
    collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)


def delete_document_chunks(collection_name: str, document_id: str) -> None:
    """Removes all chunks for a document before re-indexing it -- this is what makes
    re-indexing (Tier 2 in FEATURES.md) safe: without this, an updated document with
    fewer chunks than its previous version would leave stale chunks behind."""
    collection = get_collection(collection_name)
    collection.delete(where={"document_id": document_id})


def count_chunks(collection_name: str) -> int:
    return get_collection(collection_name).count()
