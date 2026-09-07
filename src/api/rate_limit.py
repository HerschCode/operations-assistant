"""
A minimal in-memory, per-IP sliding-window rate limiter -- built specifically for
`/demo/chat` (routes.py), the one endpoint in this project deliberately exposed
without an API key so a portfolio visitor can try the agent directly in a browser.
Every other endpoint stays behind src/api/auth.py's API-key check; this one instead
needs a cost/abuse guard, since it triggers a real Groq/Gemini call per request.

Deliberately NOT built as anything more sophisticated: no Redis, no distributed
counter. A single-process in-memory limiter is honest about its real limitation --
it resets on restart and doesn't coordinate across multiple instances -- and is
genuinely sufficient for a single free-tier portfolio deployment. Documented as a
known limitation rather than silently implied to scale, matching this project's
existing pattern (see docs/security-notes.md) of naming what a mechanism doesn't
cover.
"""
import time
from collections import defaultdict, deque

# 5 requests per 10 minutes per IP -- generous enough for a recruiter trying a few
# real questions, tight enough that a scripted loop can't run up the Groq/Gemini bill
# on a public, unauthenticated endpoint.
WINDOW_SECONDS = 600
MAX_REQUESTS_PER_WINDOW = 5

_requests: dict[str, deque] = defaultdict(deque)


def is_allowed(client_id: str, now: float | None = None) -> bool:
    now = now if now is not None else time.time()
    window = _requests[client_id]

    while window and window[0] <= now - WINDOW_SECONDS:
        window.popleft()

    if len(window) >= MAX_REQUESTS_PER_WINDOW:
        return False

    window.append(now)
    return True


def reset() -> None:
    """Test-only: clears all tracked clients between test cases."""
    _requests.clear()
