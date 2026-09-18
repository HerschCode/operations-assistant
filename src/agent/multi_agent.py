"""
Two-agent orchestration: a Researcher agent (the existing single-agent tool-calling
loop, unchanged) followed by an independent Reviewer agent that never sees the
Researcher's reasoning -- only the question, the evidence the tools actually returned,
and the draft answer.

Why a second agent rather than another prompt line telling the first agent to "double
check": the Researcher wrote the draft, so asking it to grade its own work reuses the
same blind spots that produced the error. A separate call with a different role, a
different system prompt, and *only the evidence* as its source of truth is the standard
actor/critic pattern -- and the one place this project's own evaluation already
established mechanical grounding checks are needed (src/evaluation/evaluate_answers.py).

The Reviewer's contract (JSON, parsed defensively):
    {"verdict": "approve" | "revise", "issues": [str, ...], "revised_answer": str | null}

Design decisions worth stating:
- **The Reviewer can never make things worse silently.** A "revise" verdict only replaces
  the draft if it supplies a non-empty revised_answer; any reviewer failure (API error,
  malformed JSON, unsupported provider) returns the Researcher's answer unchanged with
  reviewer_verdict="error". A reviewer outage must not take down a chat turn.
- **One revision pass, not a loop.** Unbounded critic/actor loops are where multi-agent
  systems get expensive and non-terminating; this is a single review, deliberately.
- **Providers:** groq and anthropic (the two this project actually runs and tests).
  Gemini/LangChain paths raise NotImplementedError from the reviewer call, which the
  orchestrator catches and reports as reviewer_verdict="error" -- stated, not hidden.
"""
import json
import os
from dataclasses import dataclass, field

from src.agent.agent import AgentResponse, load_agent_config, run_agent
from src.agent.investigation import _strip_code_fences
from src.observability.logging_config import get_logger

logger = get_logger("agent.multi")

MAX_EVIDENCE_CHARS_PER_TOOL = 1500

REVIEWER_SYSTEM_PROMPT = """You are an independent compliance reviewer for an operations assistant.
You did NOT write the draft answer below and you have no access to how it was produced.
Your only sources of truth are the QUESTION and the EVIDENCE (raw tool results).

Check the draft answer against the evidence:
1. Is every factual claim and every number in the draft actually supported by the evidence?
2. Does it cite the sources it relies on?
3. If the evidence does not contain enough to answer, does the draft say so plainly
   (rather than guessing or filling the gap from general knowledge)?

Respond with ONLY a JSON object, no prose, no code fences:
{"verdict": "approve" | "revise", "issues": ["short description of each problem"],
 "revised_answer": "a corrected answer using ONLY the evidence, or null if approving"}

Approve if the draft is fully supported. Revise only for a real problem -- do not rewrite
for style. A revised_answer must not introduce any claim absent from the evidence."""


@dataclass
class ReviewResult:
    verdict: str  # "approve" | "revise" | "error"
    issues: list[str] = field(default_factory=list)
    revised: bool = False
    error: str | None = None


@dataclass
class MultiAgentResponse:
    answer: str  # final answer (revised if the reviewer supplied a valid revision)
    draft_answer: str  # the Researcher's original answer, always preserved
    agent_response: AgentResponse  # the Researcher's full response (tool calls, citations, ...)
    review: ReviewResult


def _evidence_digest(response: AgentResponse) -> str:
    """Condenses this turn's real tool results into text the Reviewer can check
    against. Truncated per tool so one large result can't crowd out the rest."""
    parts = []
    for tc in response.tool_calls:
        if tc.error is not None:
            parts.append(f"[{tc.name}] ERROR: {tc.error}")
        elif tc.result is not None:
            text = json.dumps(tc.result, default=str) if not isinstance(tc.result, str) else tc.result
            if len(text) > MAX_EVIDENCE_CHARS_PER_TOOL:
                text = text[:MAX_EVIDENCE_CHARS_PER_TOOL] + " ...[truncated]"
            parts.append(f"[{tc.name}] {text}")
    return "\n".join(parts) if parts else "(no tools were called; there is no evidence)"


def _reviewer_user_prompt(question: str, evidence: str, draft: str) -> str:
    return f"QUESTION:\n{question}\n\nEVIDENCE:\n{evidence}\n\nDRAFT ANSWER:\n{draft}"


def _call_reviewer_llm(system: str, user: str, config: dict, client=None) -> str:
    """Single tool-free completion for the Reviewer. groq + anthropic only."""
    provider = config.get("provider", "anthropic")
    if provider == "groq":
        if client is None:
            from src.agent.providers import _default_groq_client
            client = _default_groq_client()
        resp = client.chat.completions.create(
            model=config["model"],
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=config.get("max_tokens", 1500),
        )
        return resp.choices[0].message.content or ""
    if provider == "anthropic":
        if client is None:
            from src.agent.agent import _default_client
            client = _default_client()
        resp = client.messages.create(
            model=config["model"], max_tokens=config.get("max_tokens", 1500), temperature=0.0,
            system=system, messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")
    raise NotImplementedError(f"reviewer agent supports groq and anthropic, not {provider!r}")


def _parse_review(raw: str) -> tuple[ReviewResult, str | None]:
    """Defensive parse: models sometimes wrap JSON in fences or add a preamble.
    Returns (ReviewResult, revised_answer_or_None)."""
    text = _strip_code_fences(raw.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in reviewer output")
    data = json.loads(text[start:end + 1])
    verdict = data.get("verdict")
    if verdict not in ("approve", "revise"):
        raise ValueError(f"unexpected verdict {verdict!r}")
    issues = [str(i) for i in (data.get("issues") or [])]
    revised = data.get("revised_answer")
    revised_answer = revised.strip() if isinstance(revised, str) and revised.strip() else None
    return ReviewResult(verdict=verdict, issues=issues), revised_answer


def review_answer(question: str, response: AgentResponse, config: dict, reviewer_client=None) -> tuple[ReviewResult, str | None]:
    """Runs the Reviewer. Returns (ReviewResult, revised_answer_or_None). Never raises:
    any failure becomes verdict="error" so the caller keeps the Researcher's answer."""
    try:
        raw = _call_reviewer_llm(
            REVIEWER_SYSTEM_PROMPT,
            _reviewer_user_prompt(question, _evidence_digest(response), response.answer),
            config, client=reviewer_client,
        )
        return _parse_review(raw)
    except Exception as exc:  # noqa: BLE001 -- reviewer failure must never fail the turn
        logger.warning("reviewer agent failed; keeping researcher answer", extra={"error": str(exc)})
        return ReviewResult(verdict="error", error=f"{type(exc).__name__}: {exc}"), None


def run_multi_agent(
    question: str, history: list[dict] | None = None, researcher_client=None, reviewer_client=None,
    config_override: dict | None = None,
) -> MultiAgentResponse:
    config = config_override or load_agent_config()
    draft = run_agent(question, client=researcher_client, history=history, config_override=config_override)

    review, revised_answer = review_answer(question, draft, config, reviewer_client=reviewer_client)

    final = draft.answer
    if review.verdict == "revise" and revised_answer:
        final = revised_answer
        review.revised = True

    return MultiAgentResponse(answer=final, draft_answer=draft.answer, agent_response=draft, review=review)
