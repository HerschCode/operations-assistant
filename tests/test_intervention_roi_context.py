"""ROI context on a proposal comes from operations-performance, never from the model."""
from unittest.mock import patch

from src.tools import interventions


def _summary():
    return {
        "label": "SIMULATION: effects are assumptions",
        "assumptions": {"breach_cost": 400, "capacity_pct": 20, "types": {}},
        "break_even_effect_simulated": 0.081,
    }


def test_proposal_carries_roi_context_from_p1():
    with patch("src.tools.client.get", return_value=_summary()) as get:
        out = interventions.propose_intervention("escalate_case", "C1001", "SLA at risk")
    get.assert_called_once_with("/roi/summary")
    ctx = out["roi_context"]
    assert ctx["break_even_effect"] == 0.081
    assert ctx["breach_cost"] == 400
    assert ctx["label"].startswith("SIMULATION")
    assert interventions.get_intervention(out["intervention_id"]).roi_context == ctx


def test_proposal_still_works_when_p1_is_down():
    with patch("src.tools.client.get", side_effect=RuntimeError("P1 unreachable")):
        out = interventions.propose_intervention("flag_supplier", "Acme", "late deliveries")
    assert out["status"] == "pending_approval"
    assert out["roi_context"] is None


def test_unexpected_shape_gives_no_context():
    with patch("src.tools.client.get", return_value=[{"not": "a summary"}]):
        out = interventions.propose_intervention("notify_manager", "Ops", "backlog")
    assert out["roi_context"] is None

