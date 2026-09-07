"""
Renamed in intent (not on disk, to avoid an import-path change mid-project) from a
generic 'database' stub to what it actually holds: operational/pipeline-health tools,
distinct from the business analytics in analytics.py. Lets the agent check "is the
underlying data fresh" when investigating something -- e.g. before trusting a metric,
confirm the last pipeline run actually succeeded recently.
"""
from src.tools.client import get
from src.tools.validation import clamp_int

GET_PIPELINE_STATUS_SCHEMA = {
    "name": "get_pipeline_status",
    "description": "Check the status and timing of recent data pipeline runs, to confirm the underlying data is current before trusting a metric.",
    "input_schema": {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "Number of recent runs to return, default 5.", "default": 5}
        },
        "required": [],
    },
}


def get_pipeline_status(limit: int = 5) -> list:
    limit = clamp_int(limit, max_value=200, min_value=1)  # matches operations-performance's own cap
    return get("/observability/pipeline-runs", params={"limit": limit})


ALL_PIPELINE_TOOLS = [(GET_PIPELINE_STATUS_SCHEMA, get_pipeline_status)]
