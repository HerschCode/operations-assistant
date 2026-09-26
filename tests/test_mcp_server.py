"""
Tests for src/mcp_server.py.

Checks:
  - All 9 tools are registered with the expected names and required params
  - Tool calls delegate to the underlying implementation (not re-test the
    implementation itself -- that's test_tools.py's job)
  - OpsPerformanceUnavailable propagates as an exception so MCPServer can
    convert it to an MCP error response
  - Validation still runs (case_id, query) -- the MCP layer adds no bypass
"""
import json
import pytest
from unittest.mock import patch

from mcp.server.mcpserver.exceptions import ToolError

from src.mcp_server import mcp
from src.tools.client import OpsPerformanceUnavailable

EXPECTED_TOOL_NAMES = {
    "get_cycle_time",
    "get_bottlenecks",
    "get_sla_metrics",
    "get_supplier_performance",
    "get_management_report",
    "predict_sla_risk",
    "get_pipeline_status",
    "search_policy_documents",
    "get_conformance",
    "propose_intervention",
}


@pytest.mark.asyncio
async def test_all_ten_tools_registered():
    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert names == EXPECTED_TOOL_NAMES


@pytest.mark.asyncio
async def test_required_params_are_marked_required():
    tools = await mcp.list_tools()
    schemas = {t.name: t.input_schema for t in tools}

    assert "case_id" in schemas["predict_sla_risk"].get("required", [])
    assert "query" in schemas["search_policy_documents"].get("required", [])


@pytest.mark.asyncio
async def test_optional_params_have_no_required_entry():
    tools = await mcp.list_tools()
    schemas = {t.name: t.input_schema for t in tools}

    for name in ("get_cycle_time", "get_bottlenecks", "get_sla_metrics",
                 "get_supplier_performance", "get_pipeline_status"):
        req = schemas[name].get("required", [])
        assert req == [], f"{name} should have no required params, got {req}"


@pytest.mark.asyncio
async def test_get_cycle_time_delegates_to_implementation():
    with patch("src.mcp_server._get_cycle_time", return_value={"mean_hours": 48.0}) as mock_fn:
        result = await mcp.call_tool("get_cycle_time", {})
    mock_fn.assert_called_once_with(segment=None)
    assert json.loads(result.content[0].text)["mean_hours"] == 48.0


@pytest.mark.asyncio
async def test_get_cycle_time_passes_segment():
    with patch("src.mcp_server._get_cycle_time", return_value=[]) as mock_fn:
        await mcp.call_tool("get_cycle_time", {"segment": "category"})
    mock_fn.assert_called_once_with(segment="category")


@pytest.mark.asyncio
async def test_predict_sla_risk_delegates():
    payload = {"case_id": "C1023", "risk_level": "HIGH", "probability": 0.82}
    with patch("src.mcp_server._predict_sla_risk", return_value=payload) as mock_fn:
        result = await mcp.call_tool("predict_sla_risk", {"case_id": "C1023"})
    mock_fn.assert_called_once_with(case_id="C1023")
    assert json.loads(result.content[0].text)["risk_level"] == "HIGH"


@pytest.mark.asyncio
async def test_predict_sla_risk_validates_case_id():
    # validate_case_id in src/tools/prediction.py rejects path-traversal patterns;
    # _tool_errors() converts the ValueError to ToolError so MCP clients get a
    # typed error response rather than an opaque crash
    with pytest.raises(ToolError, match="Invalid case_id"):
        await mcp.call_tool("predict_sla_risk", {"case_id": "../../../admin"})


@pytest.mark.asyncio
async def test_search_policy_documents_validates_empty_query():
    with pytest.raises(ToolError, match="query must not be empty"):
        await mcp.call_tool("search_policy_documents", {"query": "   "})


@pytest.mark.asyncio
async def test_ops_performance_unavailable_becomes_tool_error():
    # OpsPerformanceUnavailable is an anticipated failure (downstream API down);
    # _tool_errors() converts it to ToolError so MCP clients get a typed error
    with patch("src.mcp_server._get_sla_metrics", side_effect=OpsPerformanceUnavailable("down")):
        with pytest.raises(ToolError, match="down"):
            await mcp.call_tool("get_sla_metrics", {})


@pytest.mark.asyncio
async def test_get_management_report_returns_text_directly():
    report = "## Finding\n\nCycle time p90 exceeds target."
    with patch("src.mcp_server._get_management_report", return_value=report):
        result = await mcp.call_tool("get_management_report", {})
    # get_management_report returns plain text, not JSON
    assert result.content[0].text == report


@pytest.mark.asyncio
async def test_get_bottlenecks_passes_top_n():
    with patch("src.mcp_server._get_bottlenecks", return_value=[]) as mock_fn:
        await mcp.call_tool("get_bottlenecks", {"top_n": 5})
    mock_fn.assert_called_once_with(top_n=5)


@pytest.mark.asyncio
async def test_get_conformance_no_args():
    payload = {"conformance_rate": 0.87, "deviations": []}
    with patch("src.mcp_server._get_conformance", return_value=payload):
        result = await mcp.call_tool("get_conformance", {})
    assert json.loads(result.content[0].text)["conformance_rate"] == 0.87


@pytest.mark.asyncio
async def test_propose_intervention_delegates():
    fake_result = {
        "intervention_id": "inv_abc12345",
        "status": "pending_approval",
        "action": "escalate_case",
        "target": "C1001",
        "reason": "SLA at risk",
        "priority": "high",
        "message": "awaiting approval",
    }
    with patch("src.mcp_server._propose_intervention", return_value=fake_result) as mock_fn:
        result = await mcp.call_tool(
            "propose_intervention",
            {"action": "escalate_case", "target": "C1001", "reason": "SLA at risk", "priority": "high"},
        )
    mock_fn.assert_called_once_with(action="escalate_case", target="C1001", reason="SLA at risk", priority="high")
    assert json.loads(result.content[0].text)["intervention_id"] == "inv_abc12345"


@pytest.mark.asyncio
async def test_propose_intervention_appends_roi():
    fake_result = {
        "intervention_id": "inv_roi1234",
        "status": "pending_approval",
        "action": "flag_supplier",
        "target": "Acme",
        "reason": "Late deliveries [ROI: prevents 2h delay]",
        "priority": "normal",
        "message": "awaiting approval",
    }
    with patch("src.mcp_server._propose_intervention", return_value=fake_result) as mock_fn:
        await mcp.call_tool(
            "propose_intervention",
            {"action": "flag_supplier", "target": "Acme", "reason": "Late deliveries", "roi_estimate": "prevents 2h delay"},
        )
    call_kwargs = mock_fn.call_args.kwargs
    assert "ROI: prevents 2h delay" in call_kwargs["reason"]


@pytest.mark.asyncio
async def test_propose_intervention_invalid_action_raises():
    with pytest.raises(ToolError, match="Unknown action"):
        await mcp.call_tool(
            "propose_intervention",
            {"action": "nuke_everything", "target": "C1001", "reason": "test"},
        )


@pytest.mark.asyncio
async def test_propose_intervention_required_params():
    tools = await mcp.list_tools()
    schemas = {t.name: t.input_schema for t in tools}
    required = schemas["propose_intervention"].get("required", [])
    assert "action" in required
    assert "target" in required
    assert "reason" in required
