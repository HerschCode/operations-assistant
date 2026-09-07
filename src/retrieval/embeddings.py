"""
Local embedding model (sentence-transformers), not an API call -- deliberate for a
project this size: no per-embedding cost, no network dependency for indexing, and the
model (all-MiniLM-L6-v2) is small enough to run comfortably on CPU. A hosted embedding
API would be the better choice at a much larger document volume; not needed here.
See docs/rag-design.md.
"""
from functools import lru_cache

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.tolist()


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
