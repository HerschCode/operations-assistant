"""Tests for the Researcher + Reviewer two-agent orchestration (src/agent/multi_agent.py).
The Researcher (run_agent) is stubbed with a canned AgentResponse; the Reviewer's LLM is a
fake client -- so these exercise the orchestration logic (verdict handling, failure
isolation, evidence digest), not any real model."""
import json
from types import SimpleNamespace

import pytest

from src.agent import multi_agent as ma
from src.agent.agent import AgentResponse, ToolCallRecord


def _response(answer="The SLA target is 10 business days.", tool_calls=None):
    return AgentResponse(answer=answer, tool_calls=tool_calls or [
        ToolCallRecord(name="search_policy_documents", input={"query": "sla"},
                       result={"found": True, "results": [{"citation": "SLA Policy, Section 2", "text": "10 business days"}]}),
    ])


class FakeGroq:
    """chat.completions.create(...) -> object with choices[0].message.content"""
    def __init__(self, content=None, raises=None):
        self.content, self.raises, self.calls = content, raises, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises:
            raise self.raises
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])


GROQ_CONFIG = {"provider": "groq", "model": "test-model", "max_tokens": 500, "max_tool_calls_per_turn": 4}


@pytest.fixture
def stub_researcher(monkeypatch):
    def _install(resp):
        monkeypatch.setattr(ma, "run_agent", lambda *a, **k: resp)
    return _install


def test_approve_keeps_the_researchers_answer(stub_researcher):
    stub_researcher(_response())
    reviewer = FakeGroq(json.dumps({"verdict": "approve", "issues": [], "revised_answer": None}))
    out = ma.run_multi_agent("What is the SLA?", reviewer_client=reviewer, config_override=GROQ_CONFIG)
    assert out.answer == out.draft_answer == "The SLA target is 10 business days."
    assert out.review.verdict == "approve" and out.review.revised is False


def test_revise_with_a_revision_replaces_the_answer_and_preserves_the_draft(stub_researcher):
    stub_researcher(_response(answer="The SLA target is 14 business days."))
    reviewer = FakeGroq(json.dumps({
        "verdict": "revise", "issues": ["14 is not in the evidence"],
        "revised_answer": "The SLA target is 10 business days (SLA Policy, Section 2).",
    }))
    out = ma.run_multi_agent("What is the SLA?", reviewer_client=reviewer, config_override=GROQ_CONFIG)
    assert out.answer.startswith("The SLA target is 10")
    assert out.draft_answer == "The SLA target is 14 business days."  # original always preserved
    assert out.review.verdict == "revise" and out.review.revised is True
    assert out.review.issues == ["14 is not in the evidence"]


def test_revise_without_a_revised_answer_keeps_the_draft(stub_researcher):
    """A 'revise' verdict that supplies nothing to revise TO must not blank the answer."""
    stub_researcher(_response())
    reviewer = FakeGroq(json.dumps({"verdict": "revise", "issues": ["vague"], "revised_answer": "   "}))
    out = ma.run_multi_agent("q", reviewer_client=reviewer, config_override=GROQ_CONFIG)
    assert out.answer == out.draft_answer
    assert out.review.revised is False


def test_fenced_json_is_parsed(stub_researcher):
    stub_researcher(_response())
    payload = json.dumps({"verdict": "approve", "issues": [], "revised_answer": None})
    reviewer = FakeGroq(f"```json\n{payload}\n```")
    out = ma.run_multi_agent("q", reviewer_client=reviewer, config_override=GROQ_CONFIG)
    assert out.review.verdict == "approve"


def test_json_with_a_preamble_is_parsed(stub_researcher):
    stub_researcher(_response())
    payload = json.dumps({"verdict": "approve", "issues": [], "revised_answer": None})
    reviewer = FakeGroq(f"Here is my review: {payload} Thanks.")
    assert ma.run_multi_agent("q", reviewer_client=reviewer, config_override=GROQ_CONFIG).review.verdict == "approve"


@pytest.mark.parametrize("bad_output", [
    "not json at all",
    "{}",
    json.dumps({"verdict": "maybe"}),
    "{ broken json",
])
def test_malformed_reviewer_output_degrades_to_error_and_keeps_the_answer(stub_researcher, bad_output):
    stub_researcher(_response())
    out = ma.run_multi_agent("q", reviewer_client=FakeGroq(bad_output), config_override=GROQ_CONFIG)
    assert out.review.verdict == "error"
    assert out.answer == out.draft_answer  # a broken reviewer must never change or blank the answer


def test_reviewer_api_failure_never_fails_the_turn(stub_researcher):
    stub_researcher(_response())
    out = ma.run_multi_agent("q", reviewer_client=FakeGroq(raises=RuntimeError("groq 503")), config_override=GROQ_CONFIG)
    assert out.review.verdict == "error" and "groq 503" in out.review.error
    assert out.answer == "The SLA target is 10 business days."


def test_unsupported_provider_is_reported_not_hidden(stub_researcher):
    stub_researcher(_response())
    out = ma.run_multi_agent("q", config_override={**GROQ_CONFIG, "provider": "gemini"})
    assert out.review.verdict == "error"
    assert "groq and anthropic" in out.review.error


def test_researcher_failure_propagates(monkeypatch):
    """Only the *reviewer* is failure-isolated. If the Researcher itself fails there is no
    draft to fall back to, so the error must surface to the caller unchanged."""
    def boom(*a, **k):
        raise ConnectionError("upstream down")
    monkeypatch.setattr(ma, "run_agent", boom)
    with pytest.raises(ConnectionError):
        ma.run_multi_agent("q", reviewer_client=FakeGroq("{}"), config_override=GROQ_CONFIG)


def test_reviewer_sees_evidence_and_draft_but_not_the_researchers_system_prompt(stub_researcher):
    stub_researcher(_response(answer="DRAFT-MARKER"))
    reviewer = FakeGroq(json.dumps({"verdict": "approve", "issues": [], "revised_answer": None}))
    ma.run_multi_agent("QUESTION-MARKER", reviewer_client=reviewer, config_override=GROQ_CONFIG)
    sent = reviewer.calls[0]["messages"]
    assert sent[0]["content"] == ma.REVIEWER_SYSTEM_PROMPT  # its own role, not the researcher's
    assert "QUESTION-MARKER" in sent[1]["content"]
    assert "DRAFT-MARKER" in sent[1]["content"]
    assert "10 business days" in sent[1]["content"]  # real tool evidence reached the reviewer
    assert reviewer.calls[0]["temperature"] == 0.0


def test_evidence_digest_truncates_and_reports_tool_errors():
    big = "x" * (ma.MAX_EVIDENCE_CHARS_PER_TOOL * 3)
    resp = AgentResponse(answer="a", tool_calls=[
        ToolCallRecord(name="get_sla_metrics", input={}, result=big),
        ToolCallRecord(name="get_bottlenecks", input={}, result=None, error="API unreachable"),
    ])
    digest = ma._evidence_digest(resp)
    assert "[truncated]" in digest
    assert len(digest) < len(big)
    assert "[get_bottlenecks] ERROR: API unreachable" in digest


def test_evidence_digest_says_so_when_no_tools_were_called():
    assert "no evidence" in ma._evidence_digest(AgentResponse(answer="hi"))


def test_anthropic_reviewer_path_uses_system_param_and_text_blocks(stub_researcher):
    stub_researcher(_response())
    payload = json.dumps({"verdict": "approve", "issues": [], "revised_answer": None})
    calls = []

    class FakeAnthropic:
        def __init__(self):
            self.messages = SimpleNamespace(create=self._create)

        def _create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=payload)])

    cfg = {"provider": "anthropic", "model": "claude-test", "max_tokens": 300, "max_tool_calls_per_turn": 4}
    out = ma.run_multi_agent("q", reviewer_client=FakeAnthropic(), config_override=cfg)
    assert out.review.verdict == "approve"
    assert calls[0]["system"] == ma.REVIEWER_SYSTEM_PROMPT
    assert calls[0]["messages"][0]["role"] == "user"
