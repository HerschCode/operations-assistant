import importlib
from unittest.mock import patch
from fastapi.testclient import TestClient

from src.agent.agent import AgentResponse


def _client_with_api_key(monkeypatch, key: str | None):
    if key is None:
        monkeypatch.delenv("API_KEY", raising=False)
    else:
        monkeypatch.setenv("API_KEY", key)
    import src.api.main as main_module
    importlib.reload(main_module)
    return TestClient(main_module.app)


def test_health_never_requires_a_key(monkeypatch):
    client = _client_with_api_key(monkeypatch, "secret-456")
    assert client.get("/health").status_code == 200


def test_chat_fails_open_when_no_key_configured(monkeypatch):
    client = _client_with_api_key(monkeypatch, None)
    with patch("src.api.routes.run_agent") as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(answer="ok", tool_calls=[], tools_used=[], citations=[])
        response = client.post("/chat", json={"question": "test"})
    assert response.status_code == 200


def test_chat_rejects_missing_key_when_configured(monkeypatch):
    client = _client_with_api_key(monkeypatch, "secret-456")
    response = client.post("/chat", json={"question": "test"})
    assert response.status_code == 401


def test_chat_rejects_wrong_key(monkeypatch):
    client = _client_with_api_key(monkeypatch, "secret-456")
    response = client.post("/chat", json={"question": "test"}, headers={"X-API-Key": "wrong"})
    assert response.status_code == 401


def test_chat_accepts_correct_key(monkeypatch):
    client = _client_with_api_key(monkeypatch, "secret-456")
    with patch("src.api.routes.run_agent") as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(answer="ok", tool_calls=[], tools_used=[], citations=[])
        response = client.post("/chat", json={"question": "test"}, headers={"X-API-Key": "secret-456"})
    assert response.status_code == 200
