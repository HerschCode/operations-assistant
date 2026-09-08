"""
Per-client API keys with named roles, closing the gap this file's own prior
docstring named explicitly: "Not built: per-client keys, key rotation, scopes/roles."

Configuration: API_KEYS env var, comma-separated `name:key:role` triples, e.g.
    API_KEYS=analyst:abc123:reader,ci-pipeline:def456:reader,admin:ghi789:admin
A malformed entry (wrong number of `:`-separated parts) is skipped rather than
crashing the whole app on a typo -- one bad entry in a long list shouldn't take
every other configured key down with it.

Backward compatible: the original single-secret API_KEY env var still works if set
(mapped to a synthetic client named "default" with role "admin"), so an existing
deployment's already-configured API_KEY doesn't silently stop working when this
ships -- this project's live Render deployment is exactly that case.

Rotation: revoke a compromised key by removing its entry from API_KEYS and
redeploying -- no code change needed, same operational model the single-key
version already had.

Roles: unlike operations-performance (read-only throughout), this project has a
real write endpoint -- POST /documents (document upload/indexing) -- gated to the
"admin" role via require_role("admin") on that route specifically. /chat,
/investigate, and /documents (GET) remain open to any valid key, matching their
original behavior.

/health and /demo/* are intentionally exempt (see src/api/main.py / routes.py) --
health checks are conventionally unauthenticated, and /demo/chat is this project's
public portfolio demo surface, guarded by src/api/rate_limit.py instead of a key a
site visitor can't reasonably be expected to have.
"""
import os
import secrets
from dataclasses import dataclass

from fastapi import Header, HTTPException, Request


@dataclass(frozen=True)
class ApiClient:
    name: str
    role: str


def _load_configured_keys() -> dict[str, ApiClient]:
    """Returns {key: ApiClient}. Rebuilt from env on every call rather than cached
    at import time -- matches this module's original get_configured_api_key()
    reading live, and lets tests that monkeypatch env vars work without a module
    reload for every case."""
    keys: dict[str, ApiClient] = {}

    raw = os.environ.get("API_KEYS", "")
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":")
        if len(parts) != 3:
            continue
        name, key, role = parts
        if key:
            keys[key] = ApiClient(name=name, role=role)

    legacy_key = os.environ.get("API_KEY")
    if legacy_key and legacy_key not in keys:
        keys[legacy_key] = ApiClient(name="default", role="admin")

    return keys


def get_configured_api_key() -> str | None:
    """Kept for backward compatibility with anything checking whether legacy-style
    single-key auth is configured at all."""
    return os.environ.get("API_KEY")


async def require_api_key(request: Request, x_api_key: str = Header(default=None)) -> None:
    configured = _load_configured_keys()

    if not configured:
        # No keys configured at all -- fail open for local development (matches
        # this project's existing pattern of falling back to permissive defaults
        # locally). A real deployment MUST set API_KEY or API_KEYS.
        return

    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")

    matched: ApiClient | None = None
    for key, client in configured.items():
        if secrets.compare_digest(x_api_key, key):
            matched = client
            break

    if matched is None:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")

    request.state.api_client = matched


def require_role(role: str):
    """Dependency factory for role-gated routes: Depends(require_role("admin")).
    Must run after require_api_key has already populated request.state.api_client
    (FastAPI evaluates Depends() in the order listed on the route). When no keys
    are configured at all (fail-open local dev), request.state.api_client is never
    set, so this is a no-op too -- consistent with require_api_key's own fail-open
    behavior rather than becoming a stricter gate than the base auth check."""
    async def _check_role(request: Request) -> None:
        client: ApiClient | None = getattr(request.state, "api_client", None)
        if client is not None and client.role != role:
            raise HTTPException(status_code=403, detail=f"This endpoint requires the '{role}' role")
    return _check_role
