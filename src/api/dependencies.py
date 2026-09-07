"""Shared clients/config the routes need -- centralized so routes.py doesn't each
reinvent how to reach operations-performance's API or the vector store."""
import os
import httpx


def get_ops_performance_client() -> httpx.Client:
    base_url = os.environ.get("OPS_PERFORMANCE_API_URL", "http://localhost:8000")
    return httpx.Client(base_url=base_url, timeout=15.0)


def check_ops_performance_reachable() -> bool:
    try:
        with get_ops_performance_client() as client:
            response = client.get("/health")
            return response.status_code == 200
    except Exception:
        return False


def check_vector_store_reachable() -> bool:
    """Real check lands in Phase 15 once src/retrieval/vector_store.py exists.
    Returns False honestly rather than faking a health check that doesn't check anything."""
    try:
        from src.retrieval.vector_store import get_collection  # deferred import: may not exist yet
        get_collection()
        return True
    except Exception:
        return False
