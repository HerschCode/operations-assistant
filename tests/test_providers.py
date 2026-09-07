"""
Tests for src/agent/providers.py's Groq and Gemini implementations, mirroring
tests/test_agent.py's fake-client pattern so the tool-loop logic is exercised without a
real API key or network call. Also confirms load_agent_config's env-var override (used
to switch providers at deploy time without editing the checked-in config/agent.yaml).
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import json

import pytest

from src.agent.agent import load_agent_config
from src.agent.providers import (
    run_agent_groq, run_agent_gemini, _tool_schemas_to_openai, _tool_schemas_to_gemini,
)


# --- config env-var override -------------------------------------------------------

def test_load_agent_config_provider_defaults_to_anthropic(monkeypatch):
    monkeypatch.delenv("AGENT_PROVIDER", raising=False)
    config = load_agent_config("config/agent.yaml")
    assert config["provider"] == "anthropic"


def test_load_agent_config_env_var_overrides_provider(monkeypatch):
    monkeypatch.setenv("AGENT_PROVIDER", "groq")
    monkeypatch.setenv("AGENT_MODEL", "llama-3.3-70b-versatile")
    config = load_agent_config("config/agent.yaml")
    assert config["provider"] == "groq"
    assert config["model"] == "llama-3.3-70b-versatile"


# --- schema conversion --------------------------------------------------------------

def test_tool_schemas_to_openai_preserves_name_and_parameters():
    schemas = [{"name": "get_cycle_time", "description": "desc", "input_schema": {"type": "object", "properties": {}}}]
    converted = _tool_schemas_to_openai(schemas)
    assert converted[0]["type"] == "function"
    assert converted[0]["function"]["name"] == "get_cycle_time"
    assert converted[0]["function"]["parameters"] == schemas[0]["input_schema"]


def test_tool_schemas_to_gemini_wraps_in_function_declarations():
    schemas = [{"name": "get_bottlenecks", "description": "desc", "input_schema": {"type": "object", "properties": {}}}]
    converted = _tool_schemas_to_gemini(schemas)
    assert converted[0]["function_declarations"][0]["name"] == "get_bottlenecks"


# --- Groq loop ------------------------------------------------------------------------

def _groq_message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _groq_tool_call(name, args, call_id="call_1"):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def _groq_response(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _groq_client(responses):
    client = MagicMock()
    client.chat.completions.create.side_effect = responses
    return client


DEFAULT_TEST_CONFIG = {
    "model": "llama-3.3-70b-versatile", "max_tokens": 1500, "temperature": 0.0,
    "max_tool_calls_per_turn": 6, "tool_timeout_seconds": 15,
}


def test_run_agent_groq_no_tools_needed_returns_immediately():
    client = _groq_client([_groq_response(_groq_message(content="The answer is 42."))])
    result = run_agent_groq("what is the answer", DEFAULT_TEST_CONFIG, client=client)
    assert result.answer == "The answer is 42."
    assert result.tool_calls == []
    assert client.chat.completions.create.call_count == 1


@patch("src.agent.providers.call_tool")
def test_run_agent_groq_calls_tool_then_answers(mock_call_tool):
    mock_call_tool.return_value = {"mean_hours": 48.0}
    client = _groq_client([
        _groq_response(_groq_message(tool_calls=[_groq_tool_call("get_cycle_time", {})])),
        _groq_response(_groq_message(content="Mean cycle time is 48 hours.")),
    ])
    result = run_agent_groq("what is the cycle time", DEFAULT_TEST_CONFIG, client=client)
    assert result.answer == "Mean cycle time is 48 hours."
    assert result.tools_used == ["get_cycle_time"]
    mock_call_tool.assert_called_once_with("get_cycle_time")


@patch("src.agent.providers.call_tool")
def test_run_agent_groq_stops_at_budget(mock_call_tool):
    mock_call_tool.return_value = {"ok": True}
    config = {**DEFAULT_TEST_CONFIG, "max_tool_calls_per_turn": 1}
    client = _groq_client([
        _groq_response(_groq_message(tool_calls=[_groq_tool_call("get_cycle_time", {})])),
        _groq_response(_groq_message(tool_calls=[_groq_tool_call("get_cycle_time", {})])),
    ])
    result = run_agent_groq("loop forever", config, client=client)
    assert result.budget_exceeded is True


@patch("src.agent.providers.call_tool")
def test_run_agent_groq_handles_tool_error_gracefully(mock_call_tool):
    mock_call_tool.side_effect = RuntimeError("downstream unavailable")
    client = _groq_client([
        _groq_response(_groq_message(tool_calls=[_groq_tool_call("get_cycle_time", {})])),
        _groq_response(_groq_message(content="I couldn't retrieve that data.")),
    ])
    result = run_agent_groq("what is the cycle time", DEFAULT_TEST_CONFIG, client=client)
    assert result.tool_calls[0].error == "downstream unavailable"
    assert result.answer == "I couldn't retrieve that data."


# --- Gemini loop ------------------------------------------------------------------------

def _gemini_part(text=None, function_call=None):
    return SimpleNamespace(text=text, function_call=function_call)


def _gemini_function_call(name, args):
    return SimpleNamespace(name=name, args=args)


def _gemini_response(*parts):
    return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=list(parts)))])


def _gemini_client(responses):
    client = MagicMock()
    client.send_message.side_effect = responses
    return client


def test_run_agent_gemini_no_tools_needed_returns_immediately():
    client = _gemini_client([_gemini_response(_gemini_part(text="The answer is 42."))])
    result = run_agent_gemini("what is the answer", DEFAULT_TEST_CONFIG, client=client)
    assert result.answer == "The answer is 42."
    assert result.tool_calls == []
    assert client.send_message.call_count == 1


@patch("src.agent.providers.call_tool")
def test_run_agent_gemini_calls_tool_then_answers(mock_call_tool):
    mock_call_tool.return_value = {"mean_hours": 48.0}
    client = _gemini_client([
        _gemini_response(_gemini_part(function_call=_gemini_function_call("get_cycle_time", {}))),
        _gemini_response(_gemini_part(text="Mean cycle time is 48 hours.")),
    ])
    result = run_agent_gemini("what is the cycle time", DEFAULT_TEST_CONFIG, client=client)
    assert result.answer == "Mean cycle time is 48 hours."
    assert result.tools_used == ["get_cycle_time"]
    mock_call_tool.assert_called_once_with("get_cycle_time")


@patch("src.agent.providers.call_tool")
def test_run_agent_gemini_stops_at_budget(mock_call_tool):
    mock_call_tool.return_value = {"ok": True}
    config = {**DEFAULT_TEST_CONFIG, "max_tool_calls_per_turn": 1}
    client = _gemini_client([
        _gemini_response(_gemini_part(function_call=_gemini_function_call("get_cycle_time", {}))),
        _gemini_response(_gemini_part(function_call=_gemini_function_call("get_cycle_time", {}))),
    ])
    result = run_agent_gemini("loop forever", config, client=client)
    assert result.budget_exceeded is True


@patch("src.agent.providers.call_tool")
def test_run_agent_gemini_handles_tool_error_gracefully(mock_call_tool):
    mock_call_tool.side_effect = RuntimeError("downstream unavailable")
    client = _gemini_client([
        _gemini_response(_gemini_part(function_call=_gemini_function_call("get_cycle_time", {}))),
        _gemini_response(_gemini_part(text="I couldn't retrieve that data.")),
    ])
    result = run_agent_gemini("what is the cycle time", DEFAULT_TEST_CONFIG, client=client)
    assert result.tool_calls[0].error == "downstream unavailable"
    assert result.answer == "I couldn't retrieve that data."


# --- dispatcher -------------------------------------------------------------------------

def test_run_agent_dispatches_to_groq_when_configured():
    from src.agent.agent import run_agent
    client = _groq_client([_groq_response(_groq_message(content="42"))])
    result = run_agent(
        "q", client=client, config_override={**DEFAULT_TEST_CONFIG, "provider": "groq"},
    )
    assert result.answer == "42"


def test_run_agent_dispatches_to_gemini_when_configured():
    from src.agent.agent import run_agent
    client = _gemini_client([_gemini_response(_gemini_part(text="42"))])
    result = run_agent(
        "q", client=client, config_override={**DEFAULT_TEST_CONFIG, "provider": "gemini"},
    )
    assert result.answer == "42"
