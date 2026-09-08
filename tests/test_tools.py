from unittest.mock import patch, MagicMock
import httpx
import pytest

from src.tools.client import get, get_text, OpsPerformanceUnavailable
from src.tools.analytics import get_cycle_time, get_bottlenecks, get_management_report
from src.tools.prediction import predict_sla_risk
from src.tools.database import get_pipeline_status
from src.tools.documents import search_policy_documents
from src.tools.process import get_conformance
from src.tools.registry import ALL_TOOLS, TOOL_SCHEMAS, TOOL_FUNCTIONS, call_tool


def _mock_client(handler) -> httpx.Client:
    """Builds a real httpx.Client backed by a MockTransport -- request construction,
    status-code checking, and response parsing all run for real, only the actual
    socket I/O is replaced. Strictly more of the real code path exercised than
    patching httpx.get away entirely, which is what these four tests did before
    client.py became injectable (Phase 23)."""
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_get_returns_parsed_json():
    def handler(request):
        return httpx.Response(200, json={"mean_hours": 48.0})

    result = get("/metrics/cycle-time", client=_mock_client(handler))
    assert result == {"mean_hours": 48.0}


def test_get_raises_ops_performance_unavailable_on_404():
    def handler(request):
        return httpx.Response(404, json={"detail": "No case found"})

    with pytest.raises(OpsPerformanceUnavailable, match="No case found"):
        get("/orders/UNKNOWN/risk", client=_mock_client(handler))


def test_get_raises_on_connection_error():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(OpsPerformanceUnavailable, match="Could not reach"):
        get("/metrics/cycle-time", client=_mock_client(handler))


def test_get_text_returns_raw_text_not_json():
    def handler(request):
        return httpx.Response(200, text="# Management Report\n\nSome markdown.")

    result = get_text("/reports/management", client=_mock_client(handler))
    assert result.startswith("# Management Report")


@patch("src.tools.analytics.get")
def test_get_cycle_time_calls_correct_endpoint(mock_get):
    mock_get.return_value = {"mean_hours": 48.0, "case_count": 100}
    result = get_cycle_time()
    mock_get.assert_called_once_with("/metrics/cycle-time")
    assert result["case_count"] == 100


@patch("src.tools.analytics.get")
def test_get_bottlenecks_passes_top_n_param(mock_get):
    mock_get.return_value = []
    get_bottlenecks(top_n=5)
    mock_get.assert_called_once_with("/metrics/bottlenecks", params={"top_n": 5})


@patch("src.tools.analytics.get_text")
def test_get_management_report_uses_text_not_json(mock_get_text):
    mock_get_text.return_value = "# Report"
    result = get_management_report()
    assert result == "# Report"


@patch("src.tools.prediction.get")
def test_predict_sla_risk_builds_correct_path(mock_get):
    mock_get.return_value = {"case_id": "C1", "breach_probability": 0.8, "risk_level": "HIGH"}
    result = predict_sla_risk(case_id="C1")
    mock_get.assert_called_once_with("/orders/C1/risk")
    assert result["risk_level"] == "HIGH"


@patch("src.tools.database.get")
def test_get_pipeline_status_default_limit(mock_get):
    mock_get.return_value = []
    get_pipeline_status()
    mock_get.assert_called_once_with("/observability/pipeline-runs", params={"limit": 5})


@patch("src.tools.documents.hybrid_search")
def test_search_policy_documents_wraps_results(mock_search):
    mock_result = MagicMock()
    mock_result.citation = "Procurement Policy, Section 4.2"
    mock_result.text = "Orders above $10,000 require secondary approval."
    mock_result.similarity_score = 0.85
    mock_search.return_value = [mock_result]

    result = search_policy_documents(query="secondary approval")
    assert result["found"] is True
    assert result["results"][0]["citation"] == "Procurement Policy, Section 4.2"


@patch("src.tools.documents.hybrid_search")
def test_search_policy_documents_reports_not_found(mock_search):
    mock_search.return_value = []
    result = search_policy_documents(query="something unrelated")
    assert result["found"] is False
    assert result["results"] == []


def test_registry_has_no_duplicate_tool_names():
    names = [schema["name"] for schema in TOOL_SCHEMAS]
    assert len(names) == len(set(names)), "duplicate tool name found in registry"


def test_registry_every_tool_has_a_callable():
    for schema, fn in ALL_TOOLS:
        assert callable(fn)
        assert schema["name"] in TOOL_FUNCTIONS


def test_call_tool_raises_on_unknown_name():
    with pytest.raises(ValueError, match="Unknown tool"):
        call_tool("not_a_real_tool")


@patch("src.tools.analytics.get")
def test_call_tool_dispatches_correctly(mock_get):
    mock_get.return_value = {"mean_hours": 10.0}
    result = call_tool("get_cycle_time")
    assert result == {"mean_hours": 10.0}


@patch("src.tools.process.get")
def test_get_conformance_calls_correct_endpoint(mock_get):
    mock_get.return_value = {"total_cases": 100, "conformant_cases": 80, "conformance_rate_pct": 80.0, "deviation_breakdown": {}}
    result = get_conformance()
    mock_get.assert_called_once_with("/metrics/conformance")
    assert result["conformance_rate_pct"] == 80.0


def test_registry_includes_conformance_tool():
    assert "get_conformance" in TOOL_FUNCTIONS
    assert any(s["name"] == "get_conformance" for s in TOOL_SCHEMAS)
