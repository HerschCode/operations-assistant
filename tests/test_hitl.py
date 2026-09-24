"""
Tests for Phase 5 HITL: propose_intervention tool, HITL LangGraph agent,
and the /chat/hitl + /interventions/* API endpoints.

The LangGraph client is injected as a fake that deterministically calls
propose_intervention, matching the pattern used in test_langchain_agent.py.
No real LLM API keys or network calls are needed.
"""
import json
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from src.tools.interventions import (
    InterventionRecord,
    _STORE,
    approve_intervention,
    get_intervention,
    propose_intervention,
    reject_intervention,
)
from src.agent.hitl_agent import get_graph, start_hitl_turn, resume_hitl_turn, HITLAgentState
from src.api.main import app


# ── intervention store unit tests ─────────────────────────────────────────────

def setup_function():
    _STORE.clear()


def test_propose_intervention_stores_record():
    result = propose_intervention("escalate_case", "C1023", "8 days over SLA")
    iv_id = result["intervention_id"]
    assert iv_id.startswith("inv_")
    assert result["status"] == "pending_approval"
    record = get_intervention(iv_id)
    assert record is not None
    assert record.action == "escalate_case"
    assert record.target == "C1023"
    assert record.status == "pending_approval"


def test_propose_intervention_rejects_invalid_action():
    with pytest.raises(ValueError, match="Unknown action"):
        propose_intervention("delete_database", "prod", "test")


def test_propose_intervention_normalises_invalid_priority():
    result = propose_intervention("flag_supplier", "ACME", "low ratings", priority="critical")
    iv_id = result["intervention_id"]
    assert get_intervention(iv_id).priority == "normal"


def test_approve_intervention_marks_executed():
    result = propose_intervention("notify_manager", "SLA breach", "breach escalation")
    iv_id = result["intervention_id"]
    approved = approve_intervention(iv_id)
    assert approved["status"] == "executed"
    assert get_intervention(iv_id).status == "executed"
    assert "notified" in approved["execution_note"].lower() or "notify" in approved["execution_note"].lower()


def test_reject_intervention_marks_rejected():
    result = propose_intervention("flag_supplier", "ACME", "repeated delays")
    iv_id = result["intervention_id"]
    rejected = reject_intervention(iv_id, reason="Not enough evidence")
    assert rejected["status"] == "rejected"
    assert "Not enough evidence" in rejected["execution_note"]


def test_double_approve_raises():
    result = propose_intervention("escalate_case", "C999", "test")
    iv_id = result["intervention_id"]
    approve_intervention(iv_id)
    with pytest.raises(ValueError, match="already"):
        approve_intervention(iv_id)


def test_approve_unknown_raises():
    with pytest.raises(KeyError):
        approve_intervention("inv_doesnotexist")


# ── HITL agent graph tests (injected fake client) ─────────────────────────────

def _ai_msg(content=None, tool_calls=None):
    return AIMessage(content=content or "", tool_calls=tool_calls or [])


def _tc(name, args, call_id="c1"):
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _fake_client_propose_then_done(intervention_id_capture):
    """Fake client: first call proposes an intervention; after approval, says done."""
    responses = [
        # Round 1: call propose_intervention
        _ai_msg(tool_calls=[_tc(
            "propose_intervention",
            {"action": "escalate_case", "target": "C1023", "reason": "8 days over SLA"},
        )]),
    ]
    call_count = [0]

    class _Fake:
        def invoke(self, messages):
            idx = call_count[0]
            call_count[0] += 1
            if idx < len(responses):
                return responses[idx]
            # Final call after approval — return text answer
            return _ai_msg(content="Case C1023 has been escalated to management.")

    return _Fake()


def test_start_hitl_turn_pauses_on_propose_intervention():
    _STORE.clear()
    client = _fake_client_propose_then_done([])
    result = start_hitl_turn("Escalate case C1023", client=client)
    assert result["status"] == "awaiting_approval"
    assert "intervention" in result
    assert result["intervention"]["action"] == "escalate_case"
    assert result["intervention"]["target"] == "C1023"
    iv_id = result["intervention"]["intervention_id"]
    assert get_intervention(iv_id) is not None
    assert get_intervention(iv_id).status == "pending_approval"
    # thread_id stored on record so approval endpoint can resume
    assert get_intervention(iv_id).thread_id == result["thread_id"]


def test_resume_hitl_turn_approved():
    _STORE.clear()
    # The fake client is registered in _CLIENT_REGISTRY by start_hitl_turn and
    # reused on resume — no separate patching needed
    client = _fake_client_propose_then_done([])
    start = start_hitl_turn("Escalate case C1023", client=client)
    thread_id = start["thread_id"]

    result = resume_hitl_turn(thread_id, decision="approved")

    assert result["status"] == "done"
    iv_id = start["intervention"]["intervention_id"]
    assert get_intervention(iv_id).status == "executed"


def test_resume_hitl_turn_rejected():
    _STORE.clear()
    client = _fake_client_propose_then_done([])
    start = start_hitl_turn("Escalate case C1023", client=client)
    thread_id = start["thread_id"]

    result = resume_hitl_turn(thread_id, decision="Not enough data to escalate")

    assert result["status"] == "done"
    iv_id = start["intervention"]["intervention_id"]
    assert get_intervention(iv_id).status == "rejected"
    assert "Not enough data" in (get_intervention(iv_id).execution_note or "")


def test_start_hitl_turn_no_propose_completes_normally():
    """When the model doesn't call propose_intervention, the turn completes normally."""
    _STORE.clear()

    class _DirectClient:
        def invoke(self, messages):
            return _ai_msg(content="The SLA target is 5 business days.")

    result = start_hitl_turn("What is the SLA target?", client=_DirectClient())
    assert result["status"] == "done"
    assert result["response"].answer == "The SLA target is 5 business days."


# ── API endpoint tests ─────────────────────────────────────────────────────────

@pytest.fixture
def api_client():
    return TestClient(app)


def test_get_intervention_not_found(api_client):
    resp = api_client.get("/interventions/inv_doesnotexist")
    assert resp.status_code == 404


def test_get_intervention_found(api_client):
    _STORE.clear()
    result = propose_intervention("flag_supplier", "ACME", "three late deliveries", priority="high")
    iv_id = result["intervention_id"]
    resp = api_client.get(f"/interventions/{iv_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["intervention_id"] == iv_id
    assert data["action"] == "flag_supplier"
    assert data["status"] == "pending_approval"


def test_approve_endpoint_not_found(api_client):
    resp = api_client.post("/interventions/inv_nope/approve")
    assert resp.status_code == 404


def test_approve_endpoint_already_resolved(api_client):
    _STORE.clear()
    result = propose_intervention("escalate_case", "C42", "test")
    iv_id = result["intervention_id"]
    approve_intervention(iv_id)  # resolve it directly
    resp = api_client.post(f"/interventions/{iv_id}/approve")
    assert resp.status_code == 409


def test_approve_endpoint_no_thread(api_client):
    """Intervention exists but was not started via HITL — no thread_id to resume."""
    _STORE.clear()
    result = propose_intervention("escalate_case", "C43", "test")
    iv_id = result["intervention_id"]
    # Don't set thread_id (simulates calling propose_intervention directly, not via HITL graph)
    resp = api_client.post(f"/interventions/{iv_id}/approve")
    assert resp.status_code == 409


def test_reject_endpoint_with_reason(api_client):
    _STORE.clear()
    # Create an intervention manually so we can test the reject-404/409 paths
    result = propose_intervention("notify_manager", "SLA", "breach")
    iv_id = result["intervention_id"]

    # No thread_id set → 409
    resp = api_client.post(f"/interventions/{iv_id}/reject", json={"reason": "Not needed"})
    assert resp.status_code == 409
