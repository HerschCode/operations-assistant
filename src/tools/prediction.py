"""Wraps operations-performance's SLA-risk prediction endpoint as an agent tool."""
from src.tools.client import get
from src.tools.validation import validate_case_id

PREDICT_SLA_RISK_SCHEMA = {
    "name": "predict_sla_risk",
    "description": "Predict whether a specific procurement case is at risk of breaching its SLA, with a risk level and probability.",
    "input_schema": {
        "type": "object",
        "properties": {
            "case_id": {"type": "string", "description": "The case ID to assess, e.g. 'C1023'."}
        },
        "required": ["case_id"],
    },
}


def predict_sla_risk(case_id: str) -> dict:
    # case_id is interpolated directly into a URL path below -- validated first so a
    # malformed or path-traversal-shaped value (e.g. containing '/' or '..') can't
    # reach operations-performance's API at all, rather than trusting that API to
    # reject it gracefully.
    case_id = validate_case_id(case_id)
    return get(f"/orders/{case_id}/risk")


ALL_PREDICTION_TOOLS = [(PREDICT_SLA_RISK_SCHEMA, predict_sla_risk)]
