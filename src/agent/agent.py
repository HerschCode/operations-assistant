"""
The tool-selection loop. Calls the configured model provider with the full tool registry,
executes whatever tools the model requests via src/tools/registry.py, feeds results back,
and repeats until the model produces a final text answer or the tool-call budget
(config/agent.yaml:max_tool_calls_per_turn) is exhausted.

`run_agent` itself is a thin dispatcher -- config["provider"] (default "anthropic") picks
which implementation runs. The Anthropic path (`_run_agent_anthropic`, below) is the
original implementation, unchanged, so every test in tests/test_agent.py that passes a
fake client keeps passing exactly as before. Groq and Gemini live in
src/agent/providers.py since their SDKs return differently-shaped responses and need
their own message-formatting and tool-result round-tripping -- see that file's docstring
for why they aren't forced into this same function body.

Every provider's client is injectable (client=...) specifically so this is testable
without a real API key or network call.
"""
import os
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.tools.registry import TOOL_SCHEMAS, call_tool
from src.tools.client import OpsPerformanceUnavailable
from src.agent.prompts import SYSTEM_PROMPT
from src.observability.logging_config import get_logger

logger = get_logger("agent")

DEFAULT_CONFIG = {
    "provider": "anthropic",
    "model": "claude-sonnet-4-6",
    "max_tokens": 1500,
    "temperature": 0.0,
    "max_tool_calls_per_turn": 6,
    "tool_timeout_seconds": 15,
}


def load_agent_config(path: str = "config/agent.yaml") -> dict:
    config = dict(DEFAULT_CONFIG)
    p = Path(path)
    if p.exists():
        with open(p) as f:
            raw = yaml.safe_load(f) or {}
        config.update({k: v for k, v in raw.items() if k in DEFAULT_CONFIG})
        # 'investigation' is a nested override section, not a top-level default --
        # kept separately rather than added to DEFAULT_CONFIG (which represents the
        # flat, single-turn chat config), but still needs to survive the filter above
        # or investigation.py's config.get("investigation", {}) always sees nothing.
        if "investigation" in raw:
            config["investigation"] = raw["investigation"]
    # Deployment picks a provider/model via env vars rather than editing the checked-in
    # YAML, so config/agent.yaml's committed default (which tests/test_agent.py's
    # Anthropic-shaped fake clients rely on) never has to change for a real deployment
    # to run Groq or Gemini instead.
    if os.environ.get("AGENT_PROVIDER"):
        config["provider"] = os.environ["AGENT_PROVIDER"]
    if os.environ.get("AGENT_MODEL"):
        config["model"] = os.environ["AGENT_MODEL"]
    return config


@dataclass
class ToolCallRecord:
    name: str
    input: dict
    result: object = None
    error: str | None = None


@dataclass
class AgentResponse:
    answer: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    budget_exceeded: bool = False
    # Real token usage summed across every model round this turn made (a multi-tool
    # turn calls the model more than once) -- None for providers/paths that don't
    # report it. Populated from the provider's own real usage field (e.g. Groq's
    # response.usage), never estimated -- src/evaluation/cost_estimator.py's
    # estimate_tokens() heuristic is for pre-flight planning before a real call
    # exists, not for a turn that already has genuine numbers.
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def _default_client():
    from anthropic import Anthropic
    return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def _extract_citations(tool_calls: list[ToolCallRecord]) -> list[dict]:
    citations = []
    for tc in tool_calls:
        if tc.error:
            continue
        if tc.name == "search_policy_documents":
            if isinstance(tc.result, dict) and tc.result.get("found"):
                for r in tc.result["results"]:
                    citations.append({"kind": "document", "reference": r["citation"]})
        else:
            citations.append({"kind": "data", "reference": tc.name})
    return citations


def _execute_tool_calls(
    tool_use_blocks: list, tool_calls: list[ToolCallRecord], turn_id: str, event_cb=None,
) -> list[dict]:
    tool_results_content = []
    for block in tool_use_blocks:
        t0 = time.monotonic()
        if event_cb:
            event_cb({"type": "tool_start", "tool": block.name})
        try:
            result = call_tool(block.name, **(block.input or {}))
            duration_ms = round((time.monotonic() - t0) * 1000, 1)
            tool_calls.append(ToolCallRecord(name=block.name, input=block.input, result=result))
            tool_results_content.append({
                "type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result, default=str),
            })
            if event_cb:
                event_cb({"type": "tool_done", "tool": block.name, "ok": True})
            logger.info(
                "tool call succeeded",
                extra={"turn_id": turn_id, "tool_name": block.name, "duration_ms": duration_ms},
            )
        except (OpsPerformanceUnavailable, Exception) as exc:
            duration_ms = round((time.monotonic() - t0) * 1000, 1)
            tool_calls.append(ToolCallRecord(name=block.name, input=block.input, error=str(exc)))
            tool_results_content.append({
                "type": "tool_result", "tool_use_id": block.id,
                "content": f"Error: {exc}", "is_error": True,
            })
            if event_cb:
                event_cb({"type": "tool_done", "tool": block.name, "ok": False, "error": str(exc)})
            logger.warning(
                "tool call failed",
                extra={"turn_id": turn_id, "tool_name": block.name, "duration_ms": duration_ms, "error": str(exc)},
            )
    return tool_results_content


def run_agent(
    question: str, config_path: str = "config/agent.yaml", client=None,
    config_override: dict | None = None, history: list[dict] | None = None,
    event_cb=None,
) -> AgentResponse:
    config = config_override if config_override is not None else load_agent_config(config_path)
    provider = config.get("provider", "anthropic")
    if provider == "groq":
        from src.agent.providers import run_agent_groq
        return run_agent_groq(question, config, client=client, history=history, event_cb=event_cb)
    if provider == "gemini":
        from src.agent.providers import run_agent_gemini
        return run_agent_gemini(question, config, client=client, history=history, event_cb=event_cb)
    if provider == "langchain":
        from src.agent.langchain_agent import run_agent_langchain
        return run_agent_langchain(question, config, client=client, history=history, event_cb=event_cb)
    if provider == "langgraph":
        from src.agent.langgraph_agent import run_agent_langgraph
        return run_agent_langgraph(question, config, client=client, history=history, event_cb=event_cb)
    return _run_agent_anthropic(question, config, client=client, history=history, event_cb=event_cb)


def _run_agent_anthropic(
    question: str, config: dict, client=None, history: list[dict] | None = None, event_cb=None,
) -> AgentResponse:
    client = client or _default_client()
    # Prior turns (simple text Q/A pairs, not raw tool-call scaffolding -- see
    # src/agent/conversation_store.py's docstring for why) give the model
    # conversational context; this turn still gathers fresh tool evidence rather than
    # trusting anything from a prior turn is still current.
    messages = list(history) if history else []
    messages.append({"role": "user", "content": question})
    tool_calls: list[ToolCallRecord] = []
    max_rounds = config["max_tool_calls_per_turn"]
    turn_id = str(uuid.uuid4())[:8]
    turn_start = time.monotonic()

    logger.info("agent turn started", extra={"turn_id": turn_id, "question_length": len(question)})

    for round_num in range(max_rounds + 1):
        api_call_start = time.monotonic()
        response = client.messages.create(
            model=config["model"],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )
        api_call_duration_ms = round((time.monotonic() - api_call_start) * 1000, 1)
        messages.append({"role": "assistant", "content": response.content})

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        logger.info(
            "model round complete",
            extra={
                "turn_id": turn_id, "round": round_num, "api_call_duration_ms": api_call_duration_ms,
                "tools_requested": [b.name for b in tool_use_blocks],
            },
        )

        if not tool_use_blocks:
            answer = "".join(b.text for b in response.content if b.type == "text")
            tools_used = list(dict.fromkeys(tc.name for tc in tool_calls))
            logger.info(
                "agent turn complete",
                extra={
                    "turn_id": turn_id, "total_duration_ms": round((time.monotonic() - turn_start) * 1000, 1),
                    "rounds": round_num + 1, "tools_used": tools_used, "budget_exceeded": False,
                },
            )
            return AgentResponse(
                answer=answer, tool_calls=tool_calls, tools_used=tools_used,
                citations=_extract_citations(tool_calls),
            )

        if round_num == max_rounds:
            # Budget exhausted and the model still wants to call tools -- stop here
            # rather than looping forever. Return what's been gathered so far with the
            # budget flag set, so the API layer can surface that honestly rather than
            # silently truncating.
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

        tool_results_content = _execute_tool_calls(tool_use_blocks, tool_calls, turn_id, event_cb=event_cb)
        messages.append({"role": "user", "content": tool_results_content})

    # unreachable given the range(max_rounds + 1) loop above always returns inside it,
    # but kept as an explicit safety net rather than relying on that being obviously true
    raise RuntimeError("Agent loop exited without producing a response")
