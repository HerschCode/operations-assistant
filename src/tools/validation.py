"""
Input validation for everything the agent passes into a tool. The agent's tool
arguments come from an LLM's tool_use block -- untrusted input in the same sense as
any user-facing form field, even though the "user" here is a model, not a person
typing directly. Every tool that takes a parameter validates it here before using it,
rather than trusting the LLM to only ever produce well-formed input.
"""
import re

CASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,50}$")
MAX_QUERY_LENGTH = 500


def validate_case_id(case_id: str) -> str:
    """case_id gets interpolated directly into a URL path
    (f"/orders/{case_id}/risk" in prediction.py) -- an unvalidated value here is a
    path-traversal / request-smuggling risk against operations-performance's API
    (e.g. a case_id of "../../../admin" or one containing a slash). Restricting to a
    plain alphanumeric+dash+underscore pattern closes that off rather than relying on
    the downstream API to reject it gracefully."""
    if not case_id or not CASE_ID_PATTERN.match(case_id):
        raise ValueError(
            f"Invalid case_id: must be 1-50 alphanumeric/dash/underscore characters, got {case_id!r}"
        )
    return case_id


def validate_query(query: str) -> str:
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(f"query exceeds max length of {MAX_QUERY_LENGTH} characters")
    return query.strip()


def clamp_int(value: int, max_value: int, min_value: int = 1) -> int:
    """Clamps rather than raising -- an out-of-range top_n/min_volume from the model
    isn't malicious, just needs bounding to whatever config/tools.yaml's
    max_result_rows says for that tool, so the agent's context doesn't get flooded by
    an unbounded result set."""
    return max(min_value, min(value, max_value))
