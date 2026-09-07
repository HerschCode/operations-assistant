from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import json

from src.agent.investigation import (
    run_investigation, _parse_report, _strip_code_fences, InvestigationReport,
)
from src.agent.agent import AgentResponse, ToolCallRecord


VALID_REPORT_JSON = json.dumps({
    "executive_summary": "SLA breaches increased due to secondary approval delays.",
    "problem": "Why did SLA breaches increase this quarter?",
    "evidence": ["Breach rate rose from 11.7% to 18.4%", "Credit Review stage duration up 42%"],
    "root_causes": ["Increased manual review volume for high-value orders"],
    "relevant_policy": ["Procurement Policy, Section 4.2"],
    "recommendations": ["Automate low-risk credit screening"],
    "limitations": "Based on last quarter's data only; no supplier-level breakdown performed.",
})


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def fake_response(*blocks):
    return SimpleNamespace(content=list(blocks))


def test_strip_code_fences_removes_json_fence():
    fenced = "```json\n{\"a\": 1}\n```"
    assert _strip_code_fences(fenced) == '{"a": 1}'


def test_strip_code_fences_leaves_unfenced_text_alone():
    plain = '{"a": 1}'
    assert _strip_code_fences(plain) == plain


def test_parse_report_handles_valid_json():
    agent_response = AgentResponse(answer="working answer", tool_calls=[], tools_used=[], citations=[])
    report, failed = _parse_report(VALID_REPORT_JSON, agent_response)
    assert failed is False
    assert report.executive_summary.startswith("SLA breaches increased")
    assert len(report.evidence) == 2
    assert report.root_causes == ["Increased manual review volume for high-value orders"]


def test_parse_report_falls_back_on_invalid_json():
    agent_response = AgentResponse(answer="my working answer", tool_calls=[], tools_used=[], citations=[])
    report, failed = _parse_report("this is not JSON at all", agent_response)
    assert failed is True
    assert report.executive_summary == "my working answer"
    assert "Could not compile" in report.limitations


def test_parse_report_falls_back_on_missing_keys():
    agent_response = AgentResponse(answer="fallback answer", tool_calls=[], tools_used=[], citations=[])
    incomplete = json.dumps({"executive_summary": "x", "problem": "y"})  # missing most keys
    report, failed = _parse_report(incomplete, agent_response)
    assert failed is True
    assert "missing keys" in report.limitations


def test_parse_report_handles_fenced_valid_json():
    agent_response = AgentResponse(answer="working answer", tool_calls=[], tools_used=[], citations=[])
    fenced = f"```json\n{VALID_REPORT_JSON}\n```"
    report, failed = _parse_report(fenced, agent_response)
    assert failed is False
    assert report.executive_summary.startswith("SLA breaches increased")


@patch("src.agent.investigation.run_agent")
def test_run_investigation_uses_investigation_budget(mock_run_agent):
    mock_run_agent.return_value = AgentResponse(
        answer="Found the cause.", tool_calls=[ToolCallRecord(name="get_sla_metrics", input={}, result={})],
        tools_used=["get_sla_metrics"], citations=[{"kind": "data", "reference": "get_sla_metrics"}],
    )
    client = MagicMock()
    client.messages.create.return_value = fake_response(text_block(VALID_REPORT_JSON))

    result = run_investigation("why did SLA breach increase", client=client)

    # config/agent.yaml's investigation.max_tool_calls should have been passed through,
    # not the base max_tool_calls_per_turn -- this is exactly what the load_agent_config
    # fix (passing through the 'investigation' key) needed to make true
    call_kwargs = mock_run_agent.call_args
    passed_config = call_kwargs.kwargs["config_override"]
    assert passed_config["max_tool_calls_per_turn"] == 12  # config/agent.yaml's investigation value
    assert result.report.executive_summary.startswith("SLA breaches increased")
    assert result.parse_failed is False


@patch("src.agent.investigation.run_agent")
def test_run_investigation_returns_degraded_report_on_bad_compile_output(mock_run_agent):
    mock_run_agent.return_value = AgentResponse(
        answer="Here is what I found.", tool_calls=[], tools_used=[], citations=[],
    )
    client = MagicMock()
    client.messages.create.return_value = fake_response(text_block("not valid json"))

    result = run_investigation("some question", client=client)
    assert result.parse_failed is True
    assert result.report.executive_summary == "Here is what I found."
