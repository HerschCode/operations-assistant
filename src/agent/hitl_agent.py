"""
Human-in-the-loop LangGraph agent.

Extends the base LangGraph agent (langgraph_agent.py) with a `propose_intervention`
tool and an `approval_gate` node. When the model calls `propose_intervention`, the
graph pauses at `approval_gate` via LangGraph's interrupt() mechanism and waits for
a human to approve or reject via the POST /interventions/{id}/approve|reject endpoints.
Once the human decides, the graph resumes and the model generates a final answer that
reflects the decision.

Architecture:

  call_model ──→ _should_continue_hitl ──┬──→ run_tools ──┐
                                         ├──→ approval_gate ──→ call_model
                                         └──→ END

  approval_gate calls interrupt({intervention data}) to pause the graph.
  Resuming with Command(resume="approved"|"rejected") continues from that node.

The graph is compiled with a MemorySaver checkpointer so state survives across
the two HTTP requests (initial /chat/hitl and the subsequent /approve or /reject).
Each request uses a thread_id as the checkpointer key.

Thread safety: MemorySaver is in-process and not process-safe. For a multi-worker
deployment, replace it with LangGraph's SqliteSaver or PostgresSaver. The
interface is identical; only the constructor changes.
"""
from __future__ import annotations

import uuid
from typing import Annotated, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from src.agent.agent import AgentResponse, ToolCallRecord, _extract_citations
from langchain_core.runnables import RunnableConfig

from src.agent.langchain_tools import build_langchain_tools
from src.agent.prompts import SYSTEM_PROMPT
from src.observability.logging_config import get_logger
from src.tools.interventions import (
    PROPOSE_INTERVENTION_SCHEMA,
    approve_intervention,
    get_intervention,
    propose_intervention,
    reject_intervention,
    set_intervention_thread,
)
from src.tools.registry import call_tool

logger = get_logger("agent.hitl")

# Tool registry for HITL graph: all standard tools + propose_intervention
_HITL_TOOL_FUNCTIONS = {
    **__import__("src.tools.registry", fromlist=["TOOL_FUNCTIONS"]).TOOL_FUNCTIONS,
    "propose_intervention": propose_intervention,
}
_HITL_TOOL_SCHEMAS = (
    __import__("src.tools.registry", fromlist=["TOOL_SCHEMAS"]).TOOL_SCHEMAS
    + [PROPOSE_INTERVENTION_SCHEMA]
)


# model_client is NOT stored in LangGraph state — MemorySaver can't serialize Python
# objects. Instead, the client is kept in _CLIENT_REGISTRY keyed by thread_id and
# looked up in _call_model via the RunnableConfig that LangGraph passes to each node.
_CLIENT_REGISTRY: dict[str, Any] = {}


class HITLAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_calls: list[ToolCallRecord]
    budget_remaining: int
    # intervention_id set by run_tools when propose_intervention is called;
    # cleared by approval_gate after the human decides
    pending_intervention_id: str | None


def _call_model(state: HITLAgentState, config: RunnableConfig) -> dict:
    thread_id = (config.get("configurable") or {}).get("thread_id", "")
    client = _CLIENT_REGISTRY.get(thread_id) or _default_client()
    return {"messages": [client.invoke(state["messages"])]}


def _run_tools(state: HITLAgentState) -> dict:
    import concurrent.futures
    import time

    last_msg = state["messages"][-1]
    pending = (last_msg.tool_calls or [])[: state["budget_remaining"]]

    def _run_one(tc: dict) -> tuple:
        t0 = time.monotonic()
        try:
            fn = _HITL_TOOL_FUNCTIONS.get(tc["name"])
            if fn is None:
                raise ValueError(f"Unknown tool: {tc['name']}")
            result = fn(**tc["args"])
            ms = round((time.monotonic() - t0) * 1000, 1)
            record = ToolCallRecord(name=tc["name"], input=tc["args"], result=result)
            msg = ToolMessage(content=str(result), tool_call_id=tc["id"])
            logger.info("tool ok", extra={"tool": tc["name"], "ms": ms})
        except Exception as exc:
            ms = round((time.monotonic() - t0) * 1000, 1)
            record = ToolCallRecord(name=tc["name"], input=tc["args"], error=str(exc))
            msg = ToolMessage(content=f"Error: {exc}", tool_call_id=tc["id"])
            logger.warning("tool fail", extra={"tool": tc["name"], "ms": ms, "err": str(exc)})
        return msg, record

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(len(pending), 1)) as pool:
        pairs = [f.result() for f in [pool.submit(_run_one, tc) for tc in pending]]

    new_messages = [msg for msg, _ in pairs]
    new_records = [rec for _, rec in pairs]
    remaining = max(0, state["budget_remaining"] - len(pending))

    # If propose_intervention was called and succeeded, capture the intervention_id
    pending_id = state.get("pending_intervention_id")
    for rec in new_records:
        if rec.name == "propose_intervention" and rec.error is None and isinstance(rec.result, dict):
            pending_id = rec.result.get("intervention_id")
            break

    return {
        "messages": new_messages,
        "tool_calls": list(state["tool_calls"]) + new_records,
        "budget_remaining": remaining,
        "pending_intervention_id": pending_id,
    }


def _approval_gate(state: HITLAgentState) -> dict:
    intervention_id = state.get("pending_intervention_id")
    if not intervention_id:
        return {}

    record = get_intervention(intervention_id)

    # Pause the graph — execution resumes when Command(resume=...) is passed
    decision: str = interrupt(
        {
            "intervention_id": intervention_id,
            "action": record.action if record else "unknown",
            "target": record.target if record else "unknown",
            "reason": record.reason if record else "",
            "priority": record.priority if record else "normal",
        }
    )

    # Apply the human's decision
    if decision == "approved":
        result = approve_intervention(intervention_id)
        outcome = (
            f"[SYSTEM] Intervention {intervention_id} was APPROVED and executed: "
            f"{result['execution_note']}"
        )
    else:
        reject_reason = decision if decision not in ("approved", "rejected") else ""
        result = reject_intervention(intervention_id, reason=reject_reason)
        outcome = (
            f"[SYSTEM] Intervention {intervention_id} was REJECTED: "
            f"{result['execution_note']}"
        )

    return {
        "messages": [SystemMessage(content=outcome)],
        "pending_intervention_id": None,
    }


def _should_continue(state: HITLAgentState) -> str:
    if state.get("pending_intervention_id"):
        return "approval_gate"
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls and state["budget_remaining"] > 0:
        return "run_tools"
    return END


# Shared checkpointer — one MemorySaver per process. Replace with SqliteSaver or
# PostgresSaver for multi-worker deployments; interface is identical.
_CHECKPOINTER = MemorySaver()
_GRAPH: Any = None


def _build_graph() -> Any:
    g = StateGraph(HITLAgentState)
    g.add_node("call_model", _call_model)
    g.add_node("run_tools", _run_tools)
    g.add_node("approval_gate", _approval_gate)
    g.set_entry_point("call_model")
    g.add_conditional_edges(
        "call_model",
        _should_continue,
        {"run_tools": "run_tools", "approval_gate": "approval_gate", END: END},
    )
    g.add_edge("run_tools", "call_model")
    g.add_edge("approval_gate", "call_model")
    return g.compile(checkpointer=_CHECKPOINTER)


def get_graph() -> Any:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()
    return _GRAPH


def _default_client() -> Any:
    import os
    from langchain_groq import ChatGroq
    from langchain_core.utils.function_calling import convert_to_openai_tool
    # Build LangChain tool specs from our HITL schemas
    tools = [
        {"type": "function", "function": {
            "name": s["name"],
            "description": s["description"],
            "parameters": s["input_schema"],
        }}
        for s in _HITL_TOOL_SCHEMAS
    ]
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0.0,
    ).bind_tools(tools)


def start_hitl_turn(
    question: str,
    thread_id: str | None = None,
    client: Any | None = None,
    history: list[dict] | None = None,
) -> dict:
    """Start a new HITL agent turn. Returns either a completed AgentResponse or a
    dict with status='awaiting_approval' when the graph pauses at propose_intervention.

    Args:
        question: The user's question.
        thread_id: LangGraph thread ID for checkpointing (generated if not given).
        client: Injectable LangChain model (for testing).
        history: Prior conversation turns.

    Returns:
        dict with one of:
          {"status": "done", "thread_id": ..., "response": AgentResponse}
          {"status": "awaiting_approval", "thread_id": ..., "intervention": {...}}
    """
    thread_id = thread_id or str(uuid.uuid4())[:12]
    config = {"configurable": {"thread_id": thread_id}}

    if client is None:
        client = _default_client()
    # Register client by thread_id so _call_model can find it on resume
    _CLIENT_REGISTRY[thread_id] = client

    system = SystemMessage(content=SYSTEM_PROMPT)
    past: list = []
    for turn in (history or []):
        past.append(HumanMessage(content=turn["question"]))
        past.append(AIMessage(content=turn["answer"]))

    initial_state: HITLAgentState = {
        "messages": [system] + past + [HumanMessage(content=question)],
        "tool_calls": [],
        "budget_remaining": 6,
        "pending_intervention_id": None,
    }

    graph = get_graph()

    try:
        graph.invoke(initial_state, config)
    except Exception:
        pass  # interrupt() raises internally; get_state() gives us what we need

    state = graph.get_state(config)

    # Check if graph is paused at approval_gate
    if state.next:
        interrupts = []
        for task in state.tasks:
            interrupts.extend(getattr(task, "interrupts", []))

        if interrupts:
            interrupt_value = interrupts[0].value
            intervention_id = interrupt_value.get("intervention_id", "")
            set_intervention_thread(intervention_id, thread_id)
            logger.info("hitl paused", extra={"thread_id": thread_id, "intervention": intervention_id})
            return {
                "status": "awaiting_approval",
                "thread_id": thread_id,
                "intervention": interrupt_value,
            }

    # Graph completed — extract final answer
    return {"status": "done", "thread_id": thread_id, "response": _extract_response(state)}


def resume_hitl_turn(thread_id: str, decision: str) -> dict:
    """Resume a paused HITL turn with the human's decision.

    Args:
        thread_id: The thread_id from start_hitl_turn.
        decision: "approved" or "rejected" (or a rejection reason string).

    Returns:
        dict with {"status": "done", "thread_id": ..., "response": AgentResponse}
    """
    config = {"configurable": {"thread_id": thread_id}}
    graph = get_graph()
    graph.invoke(Command(resume=decision), config)
    state = graph.get_state(config)
    logger.info("hitl resumed", extra={"thread_id": thread_id, "decision": decision})
    return {"status": "done", "thread_id": thread_id, "response": _extract_response(state)}


def _extract_response(state) -> AgentResponse:
    messages = state.values.get("messages", [])
    answer = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
            answer = msg.content
            break
    if not answer:
        answer = "The agent completed the task but did not produce a text response."

    tool_calls: list[ToolCallRecord] = state.values.get("tool_calls", [])
    tools_used = list(dict.fromkeys(tc.name for tc in tool_calls))
    budget_exceeded = state.values.get("budget_remaining", 1) == 0

    return AgentResponse(
        answer=answer,
        tool_calls=tool_calls,
        tools_used=tools_used,
        citations=_extract_citations(tool_calls),
        budget_exceeded=budget_exceeded,
    )
