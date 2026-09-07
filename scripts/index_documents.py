"""Convenience entry point: `python scripts/index_documents.py`."""
from dotenv import load_dotenv
from src.ingestion.index_documents import index_all_documents

if __name__ == "__main__":
    load_dotenv()
    results = index_all_documents()
    total_chunks = sum(r.chunk_count for r in results)
    print(f"Indexed {len(results)} documents, {total_chunks} chunks total.")
    for r in results:
        print(f"  '{r.title}' ({r.document_id}): {r.chunk_count} chunks")
