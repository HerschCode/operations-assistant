"""
Phase 23: deliberately break things and confirm the system degrades honestly rather
than crashing or silently producing wrong answers. Uses httpx.MockTransport to exercise
src/tools/client.py's REAL request/response/error-handling code -- not mocking this
module's own functions (that's what test_tools.py's normal-path tests do), actually
running httpx.Client.get() against a fake transport. This is meaningfully closer to
testing the real network path than anything before Phase 23, short of running both
services for real (Phase 24).
"""
import json as json_module
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
import httpx
import pytest

from src.tools.client import get, get_text, OpsPerformanceUnavailable
from src.tools.analytics import get_cycle_time
from src.agent.agent import run_agent, AgentResponse


def _mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# HTTP-layer failure modes (real httpx.Client, fake transport)
# ---------------------------------------------------------------------------

def test_500_server_error_raises_ops_performance_unavailable():
    def handler(request):
        return httpx.Response(500, text="Internal Server Error")

    with pytest.raises(OpsPerformanceUnavailable, match="500"):
        get("/metrics/cycle-time", client=_mock_client(handler))


def test_503_service_unavailable_raises_ops_performance_unavailable():
    def handler(request):
        return httpx.Response(503, text="Service Unavailable")

    with pytest.raises(OpsPerformanceUnavailable, match="503"):
        get("/metrics/sla", client=_mock_client(handler))


def test_malformed_json_response_raises_ops_performance_unavailable_not_json_decode_error():
    """This is the bug Phase 23 found: a 200 response with a body that isn't valid
    JSON previously crashed with a raw json.JSONDecodeError bubbling all the way up
    through the tool and into the agent loop, instead of the clean
    OpsPerformanceUnavailable every other failure mode produces. Fixed in client.py's
    get() by catching json.JSONDecodeError explicitly. This test is what would catch
    a regression if that handling were ever accidentally removed."""
    def handler(request):
        return httpx.Response(200, text="this is not json at all {{{")

    with pytest.raises(OpsPerformanceUnavailable, match="unparseable"):
        get("/metrics/cycle-time", client=_mock_client(handler))


def test_empty_response_body_raises_ops_performance_unavailable():
    def handler(request):
        return httpx.Response(200, text="")

    with pytest.raises(OpsPerformanceUnavailable, match="unparseable"):
        get("/metrics/cycle-time", client=_mock_client(handler))


def test_connection_refused_raises_ops_performance_unavailable():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(OpsPerformanceUnavailable, match="Could not reach"):
        get("/metrics/cycle-time", client=_mock_client(handler))


def test_timeout_raises_ops_performance_unavailable():
    def handler(request):
        raise httpx.TimeoutException("request timed out", request=request)

    with pytest.raises(OpsPerformanceUnavailable, match="Could not reach"):
        get("/metrics/cycle-time", client=_mock_client(handler))


def test_404_with_non_json_body_falls_back_to_generic_detail():
    """A 404 handler tries to read a JSON 'detail' field from the error body -- if
    the error body itself isn't valid JSON (a genuinely plausible failure of a
    failing service), that inner .json() call is wrapped in a bare except in client.py
    specifically so a broken error body doesn't prevent reporting the 404 at all."""
    def handler(request):
        return httpx.Response(404, text="not json")

    with pytest.raises(OpsPerformanceUnavailable, match="not found"):
        get("/orders/UNKNOWN/risk", client=_mock_client(handler))


def test_get_text_handles_server_error():
    def handler(request):
        return httpx.Response(500)

    with pytest.raises(OpsPerformanceUnavailable, match="500"):
        get_text("/reports/management", client=_mock_client(handler))


# ---------------------------------------------------------------------------
# Tool-level: does a real HTTP failure survive being wrapped by a tool function
# ---------------------------------------------------------------------------

@patch("src.tools.analytics.get")
def test_tool_function_propagates_ops_performance_unavailable(mock_get):
    mock_get.side_effect = OpsPerformanceUnavailable("simulated outage")
    with pytest.raises(OpsPerformanceUnavailable):
        get_cycle_time()


# ---------------------------------------------------------------------------
# Agent-level: a tool failing mid-turn should degrade to an honest answer, not crash
# ---------------------------------------------------------------------------

def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_use_block(name, input_, block_id="t1"):
    return SimpleNamespace(type="tool_use", name=name, input=input_, id=block_id)


def fake_response(*blocks):
    return SimpleNamespace(content=list(blocks))


@patch("src.agent.agent.call_tool")
def test_agent_turn_survives_downstream_api_outage(mock_call_tool):
    mock_call_tool.side_effect = OpsPerformanceUnavailable(
        "Could not reach operations-performance API at http://localhost:8000/metrics/cycle-time: connection refused"
    )

    client = MagicMock()
    client.messages.create.side_effect = [
        fake_response(tool_use_block("get_cycle_time", {})),
        fake_response(text_block("I couldn't retrieve the cycle time data right now -- the operations data service appears to be unavailable.")),
    ]

    result = run_agent("what is the cycle time", client=client)

    assert result.tool_calls[0].error is not None
    assert "connection refused" in result.tool_calls[0].error
    assert "couldn't retrieve" in result.answer.lower()
    # the turn completed with an honest answer, not an unhandled exception


@patch("src.agent.agent.call_tool")
def test_agent_turn_survives_all_tools_failing(mock_call_tool):
    """Every single tool call in the turn fails -- confirms the loop still produces
    a final answer rather than crashing when nothing succeeds at all."""
    mock_call_tool.side_effect = OpsPerformanceUnavailable("simulated total outage")

    client = MagicMock()
    client.messages.create.side_effect = [
        fake_response(
            tool_use_block("get_cycle_time", {}, "t1"),
            tool_use_block("get_sla_metrics", {}, "t2"),
        ),
        fake_response(text_block("Both data sources are unavailable right now.")),
    ]

    result = run_agent("give me a full status update", client=client)
    assert len(result.tool_calls) == 2
    assert all(tc.error is not None for tc in result.tool_calls)
    assert result.answer  # still produced something, not an exception


# ---------------------------------------------------------------------------
# Malformed / extreme input at the API boundary
# ---------------------------------------------------------------------------

def test_chat_request_rejects_overly_long_question():
    from fastapi.testclient import TestClient
    from src.api.main import app

    client = TestClient(app)
    response = client.post("/chat", json={"question": "x" * 3000})  # over the 2000 char limit
    assert response.status_code == 422


def test_investigate_request_rejects_overly_long_question():
    from fastapi.testclient import TestClient
    from src.api.main import app

    client = TestClient(app)
    response = client.post("/investigate", json={"question": "x" * 3000})
    assert response.status_code == 422


def test_chat_request_rejects_missing_question_field():
    from fastapi.testclient import TestClient
    from src.api.main import app

    client = TestClient(app)
    response = client.post("/chat", json={})
    assert response.status_code == 422
