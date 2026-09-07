"""
Was deliberately empty (see git history / PLAN.md) until operations-performance
exposed GET /metrics/conformance -- that endpoint now exists (built alongside this
file), so process-specific tooling is unblocked. rework/variant equivalents still
don't have API endpoints on the other side yet, so this file covers conformance only
for now -- add more here as operations-performance's API grows, not speculatively.
"""
from src.tools.client import get

GET_CONFORMANCE_SCHEMA = {
    "name": "get_conformance",
    "description": (
        "Get the process conformance rate -- what fraction of cases follow the expected "
        "Procure-to-Pay sequence, and a breakdown of deviation types (skipped steps, "
        "unexpected activities, disallowed repeats) for cases that don't."
    ),
    "input_schema": {"type": "object", "properties": {}, "required": []},
}


def get_conformance() -> dict:
    return get("/metrics/conformance")


ALL_PROCESS_TOOLS = [(GET_CONFORMANCE_SCHEMA, get_conformance)]
