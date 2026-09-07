"""
Investigation mode: run the normal tool-calling agent loop to gather evidence, then a
second, separate call asking the model to compile that evidence into a strict JSON
report matching InvestigationReport's fixed shape (Summary -> Evidence -> Root Causes
-> Policy -> Recommendations -> Limitations). Two calls, not one, because forcing the
report JSON schema on every response (including simple ones) would be wasteful and
awkward -- investigation mode is explicitly opted into via a separate endpoint/function.
"""
import json
import re
from dataclasses import dataclass, field

from src.agent.agent import run_agent, AgentResponse, load_agent_config
from src.agent.prompts import INVESTIGATION_COMPILE_PROMPT

INVESTIGATION_REPORT_KEYS = [
    "executive_summary", "problem", "evidence", "root_causes",
    "relevant_policy", "recommendations", "limitations",
]


@dataclass
class InvestigationReport:
    executive_summary: str
    problem: str
    evidence: list[str] = field(default_factory=list)
    root_causes: list[str] = field(default_factory=list)
    relevant_policy: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    limitations: str = ""


@dataclass
class InvestigationResult:
    report: InvestigationReport
    tools_used: list[str]
    citations: list[dict]
    parse_failed: bool = False


def _strip_code_fences(text: str) -> str:
    """The model is asked not to use markdown code fences around the JSON, but LLMs
    add them anyway often enough that this is worth handling defensively rather than
    letting json.loads fail on a well-formed-but-fenced response."""
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*\n(.*)\n```$", text, re.DOTALL)
    return match.group(1).strip() if match else text


def _fallback_report(agent_response: AgentResponse, reason: str) -> InvestigationReport:
    """When the compile step fails to produce parseable JSON, don't silently drop the
    investigation -- return the agent's working answer as the executive summary and
    say plainly in limitations that structured compilation failed. A degraded-but-
    honest report beats no report or a crash."""
    return InvestigationReport(
        executive_summary=agent_response.answer,
        problem="(structured compilation failed -- see limitations)",
        limitations=f"Could not compile a structured report: {reason}. Raw answer is shown as the executive summary.",
    )


def _parse_report(raw_text: str, agent_response: AgentResponse) -> tuple[InvestigationReport, bool]:
    try:
        cleaned = _strip_code_fences(raw_text)
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return _fallback_report(agent_response, f"invalid JSON ({exc})"), True

    missing = [k for k in INVESTIGATION_REPORT_KEYS if k not in data]
    if missing:
        return _fallback_report(agent_response, f"missing keys: {missing}"), True

    return InvestigationReport(
        executive_summary=data["executive_summary"],
        problem=data["problem"],
        evidence=data.get("evidence") or [],
        root_causes=data.get("root_causes") or [],
        relevant_policy=data.get("relevant_policy") or [],
        recommendations=data.get("recommendations") or [],
        limitations=data.get("limitations") or "",
    ), False


def run_investigation(
    question: str, config_path: str = "config/agent.yaml", client=None
) -> InvestigationResult:
    from src.agent.agent import _default_client

    config = load_agent_config(config_path)
    # Investigation mode gets a larger tool-call budget than a normal chat turn --
    # config/agent.yaml's investigation.max_tool_calls, falling back to the base
    # budget if that section isn't present (e.g. in a stripped-down test config).
    investigation_config = dict(config)
    investigation_config["max_tool_calls_per_turn"] = config.get("investigation", {}).get(
        "max_tool_calls", config["max_tool_calls_per_turn"]
    )

    client = client or _default_client()

    agent_response = run_agent(question, client=client, config_override=investigation_config)

    tool_summary = ", ".join(tc.name for tc in agent_response.tool_calls) or "none"
    compile_prompt = INVESTIGATION_COMPILE_PROMPT.format(
        question=question, working_answer=agent_response.answer, tool_summary=tool_summary,
    )

    compile_response = client.messages.create(
        model=config["model"],
        max_tokens=config["max_tokens"],
        temperature=0.0,  # compilation should be deterministic even if chat isn't
        messages=[{"role": "user", "content": compile_prompt}],
    )
    raw_text = "".join(b.text for b in compile_response.content if b.type == "text")

    report, parse_failed = _parse_report(raw_text, agent_response)
    return InvestigationResult(
        report=report, tools_used=agent_response.tools_used,
        citations=agent_response.citations, parse_failed=parse_failed,
    )
