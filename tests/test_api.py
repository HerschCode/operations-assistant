from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.api.main import app
from src.agent.agent import AgentResponse

client = TestClient(app)


def test_health_endpoint_responds():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


@patch("src.api.routes.list_indexed_documents")
def test_documents_endpoint(mock_list):
    mock_list.return_value = [{"document_id": "procurement-policy", "title": "Procurement Policy", "version": "2.3", "chunk_count": 7}]
    response = client.get("/documents")
    assert response.status_code == 200
    assert response.json()[0]["document_id"] == "procurement-policy"


@patch("src.api.routes.run_agent")
def test_chat_endpoint_returns_answer_and_citations(mock_run_agent):
    mock_run_agent.return_value = AgentResponse(
        answer="Orders above $10,000 need secondary approval.",
        tool_calls=[], tools_used=["search_policy_documents"],
        citations=[{"kind": "document", "reference": "Procurement Policy, Section 4.2"}],
    )
    response = client.post("/chat", json={"question": "what approval is needed for large orders"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Orders above $10,000 need secondary approval."
    assert body["citations"][0]["kind"] == "document"
    assert "conversation_id" in body


@patch("src.api.routes.run_agent")
def test_chat_endpoint_generates_conversation_id_when_not_provided(mock_run_agent):
    mock_run_agent.return_value = AgentResponse(answer="ok", tool_calls=[], tools_used=[], citations=[])
    response = client.post("/chat", json={"question": "test"})
    assert response.json()["conversation_id"]


@patch("src.api.routes.run_agent")
def test_chat_endpoint_echoes_provided_conversation_id(mock_run_agent):
    mock_run_agent.return_value = AgentResponse(answer="ok", tool_calls=[], tools_used=[], citations=[])
    response = client.post("/chat", json={"question": "test", "conversation_id": "abc123"})
    assert response.json()["conversation_id"] == "abc123"


@patch("src.api.routes.run_agent")
def test_chat_endpoint_returns_502_on_agent_failure(mock_run_agent):
    mock_run_agent.side_effect = RuntimeError("Anthropic API unreachable")
    response = client.post("/chat", json={"question": "test"})
    assert response.status_code == 502


@patch("src.api.routes.run_agent")
def test_chat_endpoint_returns_503_on_provider_rate_limit(mock_run_agent):
    """Found by a real load test (scripts/load_test_live.py) against the actual
    deployed service -- a provider rate-limit was previously indistinguishable from
    a genuine agent failure (both 502). This is the regression guard."""
    import httpx, groq
    response_obj = httpx.Response(status_code=429, request=httpx.Request("POST", "https://api.groq.com"))
    mock_run_agent.side_effect = groq.RateLimitError("rate limited", response=response_obj, body=None)
    response = client.post("/chat", json={"question": "test"})
    assert response.status_code == 503
    assert response.headers["retry-after"] == "10"


def test_chat_endpoint_rejects_empty_question():
    response = client.post("/chat", json={"question": ""})
    assert response.status_code == 422  # pydantic min_length=1 validation


@patch("src.api.routes.run_investigation")
def test_investigate_endpoint_returns_structured_report(mock_run_investigation):
    from src.agent.investigation import InvestigationResult, InvestigationReport as InvReportDC

    mock_run_investigation.return_value = InvestigationResult(
        report=InvReportDC(
            executive_summary="Breaches rose due to manual review load.",
            problem="Why did SLA breaches increase?",
            evidence=["Breach rate 11.7% -> 18.4%"],
            root_causes=["Manual review volume increased"],
            relevant_policy=["Procurement Policy, Section 4.2"],
            recommendations=["Automate low-risk screening"],
            limitations="Last quarter only.",
        ),
        tools_used=["get_sla_metrics", "search_policy_documents"],
        citations=[],
        parse_failed=False,
    )
    response = client.post("/investigate", json={"question": "why did SLA breaches increase"})
    assert response.status_code == 200
    body = response.json()
    assert body["report"]["executive_summary"] == "Breaches rose due to manual review load."
    assert body["report"]["relevant_policy"][0]["reference"] == "Procurement Policy, Section 4.2"
    assert body["report"]["relevant_policy"][0]["kind"] == "document"
    assert "get_sla_metrics" in body["tools_used"]


@patch("src.api.routes.run_investigation")
def test_investigate_endpoint_returns_502_on_failure(mock_run_investigation):
    mock_run_investigation.side_effect = RuntimeError("model unavailable")
    response = client.post("/investigate", json={"question": "why"})
    assert response.status_code == 502


def test_upload_document_rejects_unsupported_file_type():
    response = client.post("/documents", files={"file": ("test.exe", b"binary content", "application/octet-stream")})
    assert response.status_code == 415


def test_upload_document_rejects_empty_file():
    response = client.post("/documents", files={"file": ("test.md", b"", "text/markdown")})
    assert response.status_code == 400


def test_upload_document_rejects_oversized_file():
    from src.api.routes import MAX_UPLOAD_BYTES
    oversized = b"x" * (MAX_UPLOAD_BYTES + 1)
    response = client.post("/documents", files={"file": ("test.md", oversized, "text/markdown")})
    assert response.status_code == 413


@patch("src.api.routes.index_document")
def test_upload_document_indexes_valid_markdown(mock_index_document):
    from src.ingestion.index_documents import IndexResult
    mock_index_document.return_value = IndexResult(document_id="test-policy", title="Test Policy", chunk_count=2)

    content = b"---\ntitle: Test Policy\nversion: \"1.0\"\n---\n\n# 1. Purpose\nA test document.\n"
    response = client.post("/documents", files={"file": ("test-policy.md", content, "text/markdown")})

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == "test-policy"
    assert body["chunk_count"] == 2
    mock_index_document.assert_called_once()


@patch("src.api.routes.index_document")
def test_upload_document_returns_422_on_processing_failure(mock_index_document):
    mock_index_document.side_effect = ValueError("something went wrong parsing this")
    response = client.post("/documents", files={"file": ("test.md", b"some content", "text/markdown")})
    assert response.status_code == 422


@patch("src.api.routes.run_agent")
def test_chat_endpoint_passes_prior_history_on_second_call(mock_run_agent):
    from src.agent.conversation_store import reset_all
    reset_all()
    mock_run_agent.return_value = AgentResponse(answer="first answer", tool_calls=[], tools_used=[], citations=[])

    first = client.post("/chat", json={"question": "first question"})
    convo_id = first.json()["conversation_id"]

    mock_run_agent.return_value = AgentResponse(answer="second answer", tool_calls=[], tools_used=[], citations=[])
    client.post("/chat", json={"question": "second question", "conversation_id": convo_id})

    second_call_kwargs = mock_run_agent.call_args_list[1].kwargs
    assert len(second_call_kwargs["history"]) == 2  # first question + first answer
    assert second_call_kwargs["history"][0]["content"] == "first question"
