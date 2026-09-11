"""
Tests src/agent/langchain_agent.py's tool-loop logic with a fake client (same
injectable-client pattern as tests/test_providers.py) -- no real API key or
network call needed. Uses LangChain's own real AIMessage class for fidelity
(not a SimpleNamespace stand-in), since its `.tool_calls` shape is what the
loop actually depends on.
"""
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from src.agent.langchain_agent import run_agent_langchain


DEFAULT_TEST_CONFIG = {
    "model": "openai/gpt-oss-120b", "max_tokens": 1500, "temperature": 0.0,
    "max_tool_calls_per_turn": 6, "tool_timeout_seconds": 15,
}


def _ai_message(content=None, tool_calls=None):
    return AIMessage(content=content or "", tool_calls=tool_calls or [])


def _tool_call(name, args, call_id="call_1"):
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _fake_client(responses):
    client = MagicMock()
    client.invoke.side_effect = responses
    return client


def test_run_agent_langchain_no_tools_needed_returns_immediately():
    client = _fake_client([_ai_message(content="The answer is 42.")])
    result = run_agent_langchain("what is the answer", DEFAULT_TEST_CONFIG, client=client)
    assert result.answer == "The answer is 42."
    assert result.tool_calls == []
    assert client.invoke.call_count == 1


@patch("src.agent.langchain_agent.call_tool")
def test_run_agent_langchain_calls_tool_then_answers(mock_call_tool):
    mock_call_tool.return_value = {"mean_hours": 48.0}
    client = _fake_client([
        _ai_message(tool_calls=[_tool_call("get_cycle_time", {})]),
        _ai_message(content="Mean cycle time is 48 hours."),
    ])
    result = run_agent_langchain("what is the cycle time", DEFAULT_TEST_CONFIG, client=client)
    assert result.answer == "Mean cycle time is 48 hours."
    assert result.tools_used == ["get_cycle_time"]
    mock_call_tool.assert_called_once_with("get_cycle_time")


@patch("src.agent.langchain_agent.call_tool")
def test_run_agent_langchain_stops_at_budget(mock_call_tool):
    mock_call_tool.return_value = {"ok": True}
    config = {**DEFAULT_TEST_CONFIG, "max_tool_calls_per_turn": 1}
    client = _fake_client([
        _ai_message(tool_calls=[_tool_call("get_cycle_time", {})]),
        _ai_message(tool_calls=[_tool_call("get_cycle_time", {})]),
    ])
    result = run_agent_langchain("loop forever", config, client=client)
    assert result.budget_exceeded is True


@patch("src.agent.langchain_agent.call_tool")
def test_run_agent_langchain_handles_tool_error_gracefully(mock_call_tool):
    mock_call_tool.side_effect = RuntimeError("downstream unavailable")
    client = _fake_client([
        _ai_message(tool_calls=[_tool_call("get_cycle_time", {})]),
        _ai_message(content="I couldn't retrieve that data."),
    ])
    result = run_agent_langchain("what is the cycle time", DEFAULT_TEST_CONFIG, client=client)
    assert result.tool_calls[0].error == "downstream unavailable"
    assert result.answer == "I couldn't retrieve that data."


def test_run_agent_dispatches_to_langchain_when_configured():
    from src.agent.agent import run_agent
    client = _fake_client([_ai_message(content="42")])
    result = run_agent(
        "q", client=client, config_override={**DEFAULT_TEST_CONFIG, "provider": "langchain"},
    )
    assert result.answer == "42"
