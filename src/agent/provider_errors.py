"""
Detects "the upstream model provider rate-limited us" across whichever provider
run_agent happened to dispatch to (Groq/Anthropic/Gemini -- see src/agent/providers.py),
so the API layer can return a distinct, honest status for that case instead of a
blanket 502 "agent failed."

Found via a real load test (scripts/load_test_live.py) against the actual deployed
service: a handful of truly concurrent /demo/chat requests tripped Groq's own
free-tier rate limit, and the response was a 502 indistinguishable from a genuine
agent failure -- a caller has no way to know "retry shortly" vs "something is
actually broken." No test caught this because every existing test mocks the
provider client entirely, so a rate-limit response from the SDK never gets
exercised.
"""
import groq
import anthropic


def is_rate_limit_error(exc: Exception) -> bool:
    if isinstance(exc, (groq.RateLimitError, anthropic.RateLimitError)):
        return True
    # google-genai doesn't expose a dedicated RateLimitError class -- its ClientError
    # carries the real HTTP status instead, so check that generically.
    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status_code == 429:
        return True
    return False
