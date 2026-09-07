"""
Groundedness / citation-correctness checking -- the piece of Phase 19 that was left
explicitly undone in docs/evaluation.md ("no citation-correctness scoring, no
groundedness scoring"). This doesn't need a real model to build or run: it's a
mechanical check of whether a claim actually traces back to something a tool
returned, which is exactly the property "grounding" is supposed to guarantee
(src/agent/prompts.py's SYSTEM_PROMPT instructs this; this module is how you'd
verify the instruction was actually followed, on a real transcript once one exists).

Two checks, deliberately mechanical rather than another LLM call judging the first
one (which would just move the trust problem, not solve it):
1. Citation existence: does every citation in the response correspond to a real
   tool_result the agent actually received in this turn (not a fabricated reference)?
2. Numeric groundedding: do numbers mentioned in the final answer text actually appear
   in one of the tool results (a crude but genuinely useful signal -- an answer citing
   "23.4%" that appears nowhere in any tool result is a real red flag worth surfacing).
"""
import re
from dataclasses import dataclass, field

from src.agent.agent import AgentResponse


@dataclass
class GroundednessCheck:
    all_citations_traceable: bool
    untraceable_citations: list[str] = field(default_factory=list)
    numbers_in_answer: list[str] = field(default_factory=list)
    ungrounded_numbers: list[str] = field(default_factory=list)


def _numbers_in_text(text: str) -> list[str]:
    """Extracts number-like tokens (including %, $, decimals) worth checking for
    grounding -- deliberately permissive regex since false positives here just mean
    an extra number gets checked, which is cheap, vs. false negatives silently
    missing a fabricated figure.

    The decimal group is `(?:\\.\\d+)?`, not `\\.?\\d*` -- the first version let an
    optional bare '.' match with zero trailing digits, which swept up an ordinary
    sentence-ending period right after a number (e.g. '$10,000.' at the end of a
    sentence, matched as one token instead of '$10,000' + '.'). Found by this
    module's own test suite, not by inspection."""
    return re.findall(r"\$?\d[\d,]*(?:\.\d+)?%?", text)


def _tool_results_text(response: AgentResponse) -> str:
    """Flattens every tool result this turn into one searchable string -- a real
    number-appears-somewhere-in-the-evidence check, not a citation-format check."""
    parts = []
    for tc in response.tool_calls:
        if tc.error is None and tc.result is not None:
            parts.append(str(tc.result))
    return " ".join(parts)


def _normalize_number(token: str) -> str:
    """Strips formatting ($ , %) so a prose number like '$10,000' or '18.4%' can be
    compared against a tool result's raw representation (e.g. 10000 or 18.4), which
    won't carry that formatting. Found necessary by running this module's own tests:
    the first version compared '18.4%' against a tool result containing bare 18.4
    and incorrectly flagged a genuinely grounded number as ungrounded."""
    return token.replace("$", "").replace(",", "").replace("%", "")


def check_groundedness(response: AgentResponse) -> GroundednessCheck:
    evidence_text = _tool_results_text(response)

    # citation traceability: every citation's reference should correspond to
    # something that actually came from a real tool call this turn (a data citation
    # names the tool; a document citation names a title that should appear in a
    # search_policy_documents result)
    known_tool_names = {tc.name for tc in response.tool_calls if tc.error is None}
    untraceable = []
    for citation in response.citations:
        if citation["kind"] == "data" and citation["reference"] not in known_tool_names:
            untraceable.append(citation["reference"])
        elif citation["kind"] == "document" and citation["reference"] not in evidence_text:
            untraceable.append(citation["reference"])

    numbers_in_answer = _numbers_in_text(response.answer)
    ungrounded = [
        n for n in numbers_in_answer if _normalize_number(n) not in evidence_text
    ]

    return GroundednessCheck(
        all_citations_traceable=len(untraceable) == 0,
        untraceable_citations=untraceable,
        numbers_in_answer=numbers_in_answer,
        ungrounded_numbers=ungrounded,
    )


if __name__ == "__main__":
    print(
        "This module checks a single AgentResponse's groundedness -- import "
        "check_groundedness() and call it with a real response once a live agent run "
        "exists. See tests/test_groundedness.py for worked examples against mocked "
        "responses."
    )
