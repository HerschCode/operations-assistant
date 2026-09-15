"""
LangGraph execution path — a StateGraph agent replacing the manual
`while tool_calls: call_model → run_tools` Python loop with an explicit
directed graph whose control flow is declared as nodes and edges, not
imperative code.

Key architectural difference from langchain_agent.py's manual loop:
  - State is a typed `AgentState` dict, not mutable local variables
  - The routing decision ("do I call another tool?") is a named conditional
    edge (`_should_continue`) — inspectable, composable, not buried in a loop body
  - The graph is compiled once and reused, not rebuilt per request

This is the foundation for:
  - Human-in-the-loop approval (add an interrupt_before on run_tools)
  - Parallel tool execution (a Fan-Out → Fan-In sub-graph replacing run_tools)
  - Persistent checkpointing / durable execution (inject a LangGraph Checkpointer)
None of those are built here — stated as known next steps, not silently absent.

Uses ChatGroq (already in requirements) so no new cloud credential is required
beyond what the other Groq paths already use.
"""

from __future__ import annotations

import time
import uuid
from typing import Annotated, Any

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from typing_extensions import TypedDict

from src.agent.agent import AgentResponse, ToolCallRecord, _extract_citations
from src.agent.langchain_tools import build_langchain_tools
from src.agent.prompts import SYSTEM_PROMPT
from src.tools.registry import call_tool
from src.tools.client import OpsPerformanceUnavailable
from src.observability.logging_config import get_logger

logger = get_logger("agent.langgraph")


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_calls: list[ToolCallRecord]
    budget_remaining: int
    # The bound LangChain chat model is carried in state so nodes are pure functions
    # of state (no shared mutable global) — which is what makes graph checkpointing
    # and replay work correctly when added later.
    model_client: Any
    # Optional SSE event callback; None in the normal (non-streaming) path so that
    # graph nodes stay pure functions even when streaming is active.
    event_cb: Any


def _call_model(state: AgentState) -> dict:
    response = state["model_client"].invoke(state["messages"])
    return {"messages": [response]}


def _run_tools(state: AgentState) -> dict:
    last_msg = state["messages"][-1]
    pending = (last_msg.tool_calls or [])[: state["budget_remaining"]]
    new_messages: list = []
    new_tool_calls = state["tool_calls"][:]
    event_cb = state.get("event_cb")

    for tc in pending:
        t0 = time.monotonic()
        if event_cb:
            event_cb({"type": "tool_start", "tool": tc["name"]})
        try:
            result = call_tool(tc["name"], **tc["args"])
            duration_ms = round((time.monotonic() - t0) * 1000, 1)
            new_tool_calls.append(ToolCallRecord(name=tc["name"], input=tc["args"], result=result))
            new_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            if event_cb:
                event_cb({"type": "tool_done", "tool": tc["name"], "ok": True})
            logger.info("tool ok", extra={"tool": tc["name"], "ms": duration_ms})
        except (OpsPerformanceUnavailable, Exception) as exc:
            duration_ms = round((time.monotonic() - t0) * 1000, 1)
            new_tool_calls.append(ToolCallRecord(name=tc["name"], input=tc["args"], error=str(exc)))
            new_messages.append(ToolMessage(content=f"Error: {exc}", tool_call_id=tc["id"]))
            if event_cb:
                event_cb({"type": "tool_done", "tool": tc["name"], "ok": False, "error": str(exc)})
            logger.warning("tool fail", extra={"tool": tc["name"], "ms": duration_ms, "err": str(exc)})

    remaining = max(0, state["budget_remaining"] - len(pending))
    return {"messages": new_messages, "tool_calls": new_tool_calls, "budget_remaining": remaining}


def _should_continue(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    has_tools = hasattr(last_msg, "tool_calls") and bool(last_msg.tool_calls)
    if has_tools and state["budget_remaining"] > 0:
        return "run_tools"
    return END


def _build_graph() -> Any:
    g = StateGraph(AgentState)
    g.add_node("call_model", _call_model)
    g.add_node("run_tools", _run_tools)
    g.set_entry_point("call_model")
    g.add_conditional_edges(
        "call_model",
        _should_continue,
        {"run_tools": "run_tools", END: END},
    )
    g.add_edge("run_tools", "call_model")
    return g.compile()


_GRAPH: Any = None


def _get_graph() -> Any:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()
    return _GRAPH


def _default_client(model: str) -> Any:
    import os
    from langchain_groq import ChatGroq
    tools = build_langchain_tools()
    return ChatGroq(model=model, api_key=os.environ["GROQ_API_KEY"], temperature=0.0).bind_tools(tools)


def run_agent_langgraph(
    question: str,
    config: dict,
    client: Any | None = None,
    history: list[dict] | None = None,
    event_cb=None,
) -> AgentResponse:
    """`client` is an already-tool-bound LangChain chat model when injected
    for testing; this function builds one from ChatGroq otherwise — same
    injectable-client pattern every other provider in this project uses."""
    turn_id = str(uuid.uuid4())[:8]
    if client is None:
        client = _default_client(config["model"])

    system = SystemMessage(content=SYSTEM_PROMPT)
    past: list = []
    for turn in (history or []):
        past.append(HumanMessage(content=turn["question"]))
        past.append(AIMessage(content=turn["answer"]))

    initial_state: AgentState = {
        "messages": [system] + past + [HumanMessage(content=question)],
        "tool_calls": [],
        "budget_remaining": config.get("max_tool_calls_per_turn", 6),
        "model_client": client,
        "event_cb": event_cb,
    }

    logger.info("langgraph run start", extra={"turn_id": turn_id})
    final_state = _get_graph().invoke(initial_state)

    # Last AIMessage with non-empty text content is the answer.
    answer = ""
    for msg in reversed(final_state["messages"]):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
            answer = msg.content
            break

    tool_calls: list[ToolCallRecord] = final_state["tool_calls"]
    tools_used = list(dict.fromkeys(tc.name for tc in tool_calls))
    budget_exceeded = final_state["budget_remaining"] == 0

    if not answer:
        answer = (
            "I gathered some information but reached the tool-call limit before "
            "I could fully answer. Here is what I found so far, though it may be incomplete."
        )

    logger.info(
        "langgraph run done",
        extra={"turn_id": turn_id, "tools_used": tools_used, "budget_exceeded": budget_exceeded},
    )
    return AgentResponse(
        answer=answer,
        tool_calls=tool_calls,
        tools_used=tools_used,
        citations=_extract_citations(tool_calls),
        budget_exceeded=budget_exceeded,
    )
