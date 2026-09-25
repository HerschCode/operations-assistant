"""Shared clients/config the routes need -- centralized so routes.py doesn't each
reinvent how to reach operations-performance's API or the vector store."""
import os
import httpx

from src.tools import upstream_auth


def get_ops_performance_client() -> httpx.Client:
    base_url = os.environ.get("OPS_PERFORMANCE_API_URL", "http://localhost:8000")
    return httpx.Client(base_url=base_url, timeout=15.0)


# Deliberately short (15s) and separate from src/tools/client.py's 60s tool-call
# timeout: /health is a liveness probe -- this project's own Dockerfile HEALTHCHECK
# and Render's infra-level health checks have short timeouts of their own (5s in the
# Dockerfile), so a slow /health would make THIS container flap unhealthy, which is
# worse than /health honestly reporting "operations-performance isn't reachable right
# now" when it's mid-cold-start. A real chat question calling a tool gets the patient
# 60s timeout instead, because failing a real answer over a slow-to-wake dependency is
# a worse trade than a slightly-stale reachable=false flag on a health endpoint nobody
# but infrastructure looks at directly.


def check_ops_performance_reachable() -> bool:
    try:
        with get_ops_performance_client() as client:
            # A private Cloud Run service answers 403 to any call without an ID token, /health
            # included, so under AUTH_MODE=google_id_token this probe must present one too.
            # In the default api_key mode outbound_headers() is empty: the probe is unchanged.
            headers = upstream_auth.outbound_headers(str(client.base_url))
            response = client.get("/health", headers=headers)
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
