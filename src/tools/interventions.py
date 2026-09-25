"""
Intervention store and propose_intervention tool.

propose_intervention is the only write-like tool in the system — it proposes an
action that must be approved by a human before it executes. Calling the tool adds
a record to the in-memory store with status "pending_approval"; nothing is executed
until approve_intervention() is called (via the HITL agent's LangGraph interrupt
path or the POST /interventions/{id}/approve endpoint).

Execution is intentionally a logged mock: operations-performance exposes only
read-only analytics endpoints, so the "action" would call a write-back API (a
notification service, an escalation endpoint) in a real deployment. The architecture
(propose → interrupt → approve → execute) is the portfolio demonstration; the mock
execution is honest about the boundary.
"""
import time
import uuid
from dataclasses import dataclass

VALID_ACTIONS = {
    "escalate_case",
    "flag_supplier",
    "notify_manager",
    "request_approval",
    "mark_exception",
}
VALID_PRIORITIES = {"low", "normal", "high", "urgent"}

PROPOSE_INTERVENTION_SCHEMA = {
    "name": "propose_intervention",
    "description": (
        "Propose an operational intervention action for human approval. Use this when "
        "you have identified a problem that warrants a concrete action (escalating a case, "
        "flagging a supplier, notifying a manager) but must not act unilaterally. "
        "The action will not execute until a human approves it at POST /interventions/{id}/approve. "
        "Valid actions: escalate_case, flag_supplier, notify_manager, request_approval, mark_exception."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "The action type. One of: escalate_case, flag_supplier, notify_manager, request_approval, mark_exception.",
            },
            "target": {
                "type": "string",
                "description": "The entity to act on (case ID, supplier name, team name, etc.).",
            },
            "reason": {
                "type": "string",
                "description": "Why this intervention is needed. Be specific — this is what the human reviewer sees.",
            },
            "priority": {
                "type": "string",
                "description": "Urgency: low, normal, high, or urgent. Default: normal.",
                "default": "normal",
            },
        },
        "required": ["action", "target", "reason"],
    },
}


@dataclass
class InterventionRecord:
    intervention_id: str
    action: str
    target: str
    reason: str
    priority: str
    status: str          # "pending_approval" | "rejected" | "executed"
    created_at: float
    thread_id: str | None = None   # HITL thread to resume on approval
    initiated_by: str | None = None  # API client name that proposed the intervention
    approved_by: str | None = None   # API client name that approved or rejected it
    resolved_at: float | None = None
    execution_note: str | None = None


_STORE: dict[str, InterventionRecord] = {}


def propose_intervention(action: str, target: str, reason: str, priority: str = "normal") -> dict:
    if action not in VALID_ACTIONS:
        raise ValueError(
            f"Unknown action {action!r}. Valid actions: {sorted(VALID_ACTIONS)}"
        )
    if priority not in VALID_PRIORITIES:
        priority = "normal"

    intervention_id = f"inv_{uuid.uuid4().hex[:8]}"
    record = InterventionRecord(
        intervention_id=intervention_id,
        action=action,
        target=target,
        reason=reason[:500],
        priority=priority,
        status="pending_approval",
        created_at=time.time(),
    )
    _STORE[intervention_id] = record
    return {
        "intervention_id": intervention_id,
        "status": "pending_approval",
        "action": action,
        "target": target,
        "reason": reason,
        "priority": priority,
        "message": (
            f"Intervention '{action}' on '{target}' has been proposed and is awaiting human approval. "
            f"A reviewer must call POST /interventions/{intervention_id}/approve to execute it, "
            f"or POST /interventions/{intervention_id}/reject to cancel."
        ),
    }


def get_intervention(intervention_id: str) -> InterventionRecord | None:
    return _STORE.get(intervention_id)


def set_intervention_thread(intervention_id: str, thread_id: str) -> None:
    if intervention_id in _STORE:
        _STORE[intervention_id].thread_id = thread_id


def set_intervention_initiated_by(intervention_id: str, initiated_by: str) -> None:
    if intervention_id in _STORE:
        _STORE[intervention_id].initiated_by = initiated_by


def set_intervention_approved_by(intervention_id: str, approved_by: str) -> None:
    if intervention_id in _STORE:
        _STORE[intervention_id].approved_by = approved_by


def approve_intervention(intervention_id: str) -> dict:
    record = _STORE.get(intervention_id)
    if record is None:
        raise KeyError(f"Intervention {intervention_id!r} not found")
    if record.status != "pending_approval":
        raise ValueError(f"Intervention {intervention_id!r} is already {record.status!r}")

    execution_note = _execute_action(record)
    record.status = "executed"
    record.resolved_at = time.time()
    record.execution_note = execution_note
    return {"intervention_id": intervention_id, "status": "executed", "execution_note": execution_note}


def reject_intervention(intervention_id: str, reason: str = "") -> dict:
    record = _STORE.get(intervention_id)
    if record is None:
        raise KeyError(f"Intervention {intervention_id!r} not found")
    if record.status != "pending_approval":
        raise ValueError(f"Intervention {intervention_id!r} is already {record.status!r}")

    record.status = "rejected"
    record.resolved_at = time.time()
    record.execution_note = f"Rejected. {reason}".strip() if reason else "Rejected."
    return {
        "intervention_id": intervention_id,
        "status": "rejected",
        "execution_note": record.execution_note,
    }


def _execute_action(record: InterventionRecord) -> str:
    """Execute the approved intervention.

    Tries to enrich the execution note with live context from operations-performance
    (P1). Falls back to a plain description if P1 is unreachable — the action is
    still recorded, just without the supporting data. This is the point at which the
    trilogy loop closes: P2 (assistant) acting on a human-approved decision by calling
    back into P1 (performance) to confirm the state of the entity being acted on.
    """
    descriptions = {
        "escalate_case": f"Case {record.target} escalated to operations management.",
        "flag_supplier": f"Supplier {record.target} flagged for procurement team review.",
        "notify_manager": f"Operations manager notified regarding {record.target}.",
        "request_approval": f"Additional approval requested for {record.target}.",
        "mark_exception": f"Exception recorded for {record.target} per policy.",
    }
    base = descriptions.get(record.action, f"Action '{record.action}' executed on '{record.target}'.")
    context = _fetch_execution_context(record)
    return f"{base} {context}".strip() if context else base


def _fetch_execution_context(record: InterventionRecord) -> str:
    """Fetch supporting data from operations-performance to enrich the execution note.

    Returns an empty string if P1 is unreachable or the entity isn't found —
    the execution always succeeds even when the enrichment fails.
    """
    try:
        from src.tools.client import get as ops_get  # noqa: PLC0415
        if record.action == "escalate_case":
            result = ops_get(f"/orders/{record.target}/risk")
            parts = []
            prob = result.get("breach_probability")
            days = result.get("days_in_progress")
            if prob is not None:
                parts.append(f"SLA breach probability: {prob:.0%}")
            if days is not None:
                parts.append(f"days in progress: {days}")
            return f"({', '.join(parts)})" if parts else ""
        if record.action == "flag_supplier":
            rows = ops_get("/suppliers/performance")
            match = next(
                (r for r in (rows if isinstance(rows, list) else [])
                 if r.get("supplier_name") == record.target),
                None,
            )
            if match:
                score = match.get("performance_score")
                late = match.get("late_delivery_rate_pct")
                parts = []
                if score is not None:
                    parts.append(f"performance score: {score}")
                if late is not None:
                    parts.append(f"late delivery rate: {late}%")
                return f"({', '.join(parts)})" if parts else ""
    except Exception:
        pass
    return ""
