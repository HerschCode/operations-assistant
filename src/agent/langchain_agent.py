"""
LangChain-orchestrated agent loop -- an additional, named orchestration path
alongside src/agent/providers.py's hand-rolled Groq/Gemini loops and
src/agent/agent.py's Anthropic loop, not a replacement for any of them. Reuses
every real tool (src/agent/langchain_tools.py wraps the exact same functions
src/tools/registry.py already exposes), the same SYSTEM_PROMPT, and the same
AgentResponse/ToolCallRecord/_extract_citations this project's other providers
already use -- so a caller gets an identical response shape regardless of which
orchestration layer answered, and there's exactly one place each tool's real
behavior is defined.

Uses ChatGroq's `.bind_tools()` + a manual round loop (not LangChain's
AgentExecutor or LangGraph) -- consistent with this project's stated position on
adopting a heavier agent framework only when a genuine need appears
(FUTURE_IMPROVEMENTS.md), not speculatively. This still genuinely demonstrates
LangChain's core abstractions (ChatModel, tool binding, typed messages), it just
doesn't pull in the full agentic-framework surface area for a single-turn
tool-calling loop that doesn't need it.
"""
import time
import uuid

from src.tools.registry import call_tool
from src.tools.client import OpsPerformanceUnavailable
from src.agent.prompts import SYSTEM_PROMPT
from src.agent.langchain_tools import build_langchain_tools
from src.observability.logging_config import get_logger

logger = get_logger("agent")


def _default_langchain_client(model: str):
    import os
    from langchain_groq import ChatGroq
    return ChatGroq(model=model, api_key=os.environ["GROQ_API_KEY"], temperature=0.0)


def run_agent_langchain(question: str, config: dict, client=None, history: list[dict] | None = None, event_cb=None):
    """`client` is an already-tool-bound LangChain chat model (i.e. the result of
    `.bind_tools(...)`) when injected for testing -- constructing the real
    ChatGroq + bind_tools pair is left to this function when client is None, same
    injectable-client pattern every other provider in this project uses."""
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

    from src.agent.agent import AgentResponse, ToolCallRecord, _extract_citations

    tools = build_langchain_tools()
    if client is None:
        raw_client = _default_langchain_client(config["model"])
        client = raw_client.bind_tools(tools)

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for turn in history or []:
        messages.append(HumanMessage(content=turn["content"]) if turn["role"] == "user" else AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=question))

    tool_calls: list[ToolCallRecord] = []
    max_rounds = config["max_tool_calls_per_turn"]
    turn_id = str(uuid.uuid4())[:8]
    turn_start = time.monotonic()

    logger.info("agent turn started", extra={"turn_id": turn_id, "provider": "langchain", "question_length": len(question)})

    for round_num in range(max_rounds + 1):
        api_call_start = time.monotonic()
        ai_message: AIMessage = client.invoke(messages)
        api_call_duration_ms = round((time.monotonic() - api_call_start) * 1000, 1)
        messages.append(ai_message)

        requested_tool_calls = ai_message.tool_calls or []
        logger.info(
            "model round complete",
            extra={
                "turn_id": turn_id, "round": round_num, "api_call_duration_ms": api_call_duration_ms,
                "tools_requested": [tc["name"] for tc in requested_tool_calls],
            },
        )

        if not requested_tool_calls:
            tools_used = list(dict.fromkeys(tc.name for tc in tool_calls))
            logger.info(
                "agent turn complete",
                extra={
                    "turn_id": turn_id, "total_duration_ms": round((time.monotonic() - turn_start) * 1000, 1),
                    "rounds": round_num + 1, "tools_used": tools_used, "budget_exceeded": False,
                },
            )
            return AgentResponse(
                answer=ai_message.content or "", tool_calls=tool_calls, tools_used=tools_used,
                citations=_extract_citations(tool_calls),
            )

        if round_num == max_rounds:
            tools_used = list(dict.fromkeys(tc.name for tc in tool_calls))
            logger.warning(
                "agent turn hit tool-call budget",
                extra={
                    "turn_id": turn_id, "total_duration_ms": round((time.monotonic() - turn_start) * 1000, 1),
                    "rounds": round_num + 1, "tools_used": tools_used, "budget_exceeded": True,
                },
            )
            return AgentResponse(
                answer=(
                    "I gathered some information but reached the tool-call limit for this "
                    "question before I could fully answer it. Here's what I found so far, "
                    "though it may be incomplete."
                ),
                tool_calls=tool_calls, tools_used=tools_used,
                citations=_extract_citations(tool_calls), budget_exceeded=True,
            )

        for tc in requested_tool_calls:
            t0 = time.monotonic()
            try:
                result = call_tool(tc["name"], **tc["args"])
                duration_ms = round((time.monotonic() - t0) * 1000, 1)
                tool_calls.append(ToolCallRecord(name=tc["name"], input=tc["args"], result=result))
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
                logger.info(
                    "tool call succeeded",
                    extra={"turn_id": turn_id, "tool_name": tc["name"], "duration_ms": duration_ms},
                )
            except (OpsPerformanceUnavailable, Exception) as exc:
                duration_ms = round((time.monotonic() - t0) * 1000, 1)
                tool_calls.append(ToolCallRecord(name=tc["name"], input=tc["args"], error=str(exc)))
                messages.append(ToolMessage(content=f"Error: {exc}", tool_call_id=tc["id"]))
                logger.warning(
                    "tool call failed",
                    extra={"turn_id": turn_id, "tool_name": tc["name"], "duration_ms": duration_ms, "error": str(exc)},
                )

    raise RuntimeError("Agent loop exited without producing a response")
