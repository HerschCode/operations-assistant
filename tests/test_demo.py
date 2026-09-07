from unittest.mock import patch
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.rate_limit import reset as reset_rate_limit
from src.agent.agent import AgentResponse

client = TestClient(app)


def setup_function():
    reset_rate_limit()


def test_demo_page_serves_html_without_auth():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Operations Assistant" in response.text


@patch("src.api.routes.run_agent")
def test_demo_chat_works_without_api_key(mock_run_agent):
    mock_run_agent.return_value = AgentResponse(
        answer="Mean cycle time is 48 hours.", tool_calls=[], tools_used=["get_cycle_time"], citations=[],
    )
    response = client.post("/demo/chat", json={"question": "what is the cycle time"})
    assert response.status_code == 200
    assert response.json()["answer"] == "Mean cycle time is 48 hours."
    assert response.json()["conversation_id"] == "demo"


@patch("src.api.routes.run_agent")
def test_demo_chat_enforces_rate_limit(mock_run_agent):
    mock_run_agent.return_value = AgentResponse(answer="ok", tool_calls=[], tools_used=[], citations=[])
    for _ in range(5):
        response = client.post(
            "/demo/chat", json={"question": "q"}, headers={"X-Forwarded-For": "9.9.9.9"},
        )
        assert response.status_code == 200

    sixth = client.post("/demo/chat", json={"question": "q"}, headers={"X-Forwarded-For": "9.9.9.9"})
    assert sixth.status_code == 429


@patch("src.api.routes.run_agent")
def test_demo_chat_rate_limit_is_per_client_not_global(mock_run_agent):
    """A real deployment sits behind Render's reverse proxy -- request.client.host
    would be the proxy's own address for every visitor, which is exactly what would
    make the limit global instead of per-visitor if X-Forwarded-For weren't used."""
    mock_run_agent.return_value = AgentResponse(answer="ok", tool_calls=[], tools_used=[], citations=[])
    for _ in range(5):
        client.post("/demo/chat", json={"question": "q"}, headers={"X-Forwarded-For": "1.1.1.1"})

    exhausted = client.post("/demo/chat", json={"question": "q"}, headers={"X-Forwarded-For": "1.1.1.1"})
    assert exhausted.status_code == 429

    different_visitor = client.post("/demo/chat", json={"question": "q"}, headers={"X-Forwarded-For": "2.2.2.2"})
    assert different_visitor.status_code == 200


@patch("src.api.routes.run_agent")
def test_demo_chat_returns_502_on_agent_failure(mock_run_agent):
    mock_run_agent.side_effect = RuntimeError("model unavailable")
    response = client.post("/demo/chat", json={"question": "q"})
    assert response.status_code == 502
