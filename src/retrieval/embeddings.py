"""
Local embedding model, not an API call -- deliberate for a project this size: no
per-embedding cost, no network dependency for indexing, and the model
(all-MiniLM-L6-v2) is small enough to run comfortably on CPU. A hosted embedding API
would be the better choice at a much larger document volume; not needed here. See
docs/rag-design.md.

Runs the same all-MiniLM-L6-v2 model via ONNX Runtime (chromadb's own
DefaultEmbeddingFunction) rather than sentence-transformers+PyTorch. Both load the
identical model; ONNX Runtime is chromadb's own transitive dependency already (no new
package required) and has a small fraction of PyTorch's footprint -- switched after a
live deploy on Render's free tier (512MB RAM) OOM-crash-looped under the
sentence-transformers+torch stack, something no local test could have caught since
every test here mocks embed_texts/embed_query rather than loading a real model.
"""
from functools import lru_cache


@lru_cache(maxsize=1)
def get_embedding_model():
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    return DefaultEmbeddingFunction()


def embed_texts(texts: list[str]) -> list[list[float]]:
    import numpy as np
    model = get_embedding_model()
    embeddings = model(texts)
    # chromadb's own upsert validation wants numpy arrays (or a list of them), not
    # plain Python lists -- DefaultEmbeddingFunction's return type varies by chromadb
    # version, so normalize explicitly rather than assuming its shape.
    return [np.asarray(e, dtype=float) for e in embeddings]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
