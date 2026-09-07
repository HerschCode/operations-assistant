"""
API-key authentication -- closes the "No API authentication exists" gap named in
docs/security-notes.md and docs/production-readiness-review.md.

Deliberately simple: a single shared secret in API_KEY, checked against the
X-API-Key request header via constant-time comparison (secrets.compare_digest --
a naive `==` string comparison leaks timing information about how many leading
characters matched, which is a real (if narrow) side-channel for guessing a secret).

Not built: per-client keys, key rotation, scopes/roles. Named explicitly as a
reasonable next step in docs/security-notes.md, not silently out of scope.

/health is intentionally exempt (see src/api/main.py) -- health checks are
conventionally unauthenticated so a load balancer or orchestrator can probe
liveness without needing a credential, a standard and deliberate exception, not an
oversight.
"""
import os
import secrets

from fastapi import Header, HTTPException


def get_configured_api_key() -> str | None:
    return os.environ.get("API_KEY")


async def require_api_key(x_api_key: str = Header(default=None)) -> None:
    configured_key = get_configured_api_key()

    if not configured_key:
        # No API_KEY configured -- fail open for local development (matches this
        # project's existing pattern of falling back to permissive defaults locally,
        # e.g. API_ALLOWED_ORIGINS defaulting rather than crashing). A real deployment
        # MUST set API_KEY; docs/security-notes.md says so directly.
        return

    if not x_api_key or not secrets.compare_digest(x_api_key, configured_key):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")
