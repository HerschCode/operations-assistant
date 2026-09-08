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


def _client_with_api_keys(monkeypatch, keys_env: str | None):
    monkeypatch.delenv("API_KEY", raising=False)
    if keys_env is None:
        monkeypatch.delenv("API_KEYS", raising=False)
    else:
        monkeypatch.setenv("API_KEYS", keys_env)
    import src.api.main as main_module
    importlib.reload(main_module)
    return TestClient(main_module.app)


def test_multiple_clients_each_have_their_own_working_key(monkeypatch):
    client = _client_with_api_keys(monkeypatch, "analyst:key-a:reader,admin:key-b:admin")
    with patch("src.api.routes.run_agent") as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(answer="ok", tool_calls=[], tools_used=[], citations=[])
        r1 = client.post("/chat", json={"question": "q"}, headers={"X-API-Key": "key-a"})
        r2 = client.post("/chat", json={"question": "q"}, headers={"X-API-Key": "key-b"})
    assert r1.status_code == 200
    assert r2.status_code == 200


def test_revoking_one_clients_key_does_not_affect_another(monkeypatch):
    client = _client_with_api_keys(monkeypatch, "analyst:key-a:reader")  # key-b removed = revoked
    with patch("src.api.routes.run_agent") as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(answer="ok", tool_calls=[], tools_used=[], citations=[])
        still_works = client.post("/chat", json={"question": "q"}, headers={"X-API-Key": "key-a"})
        revoked = client.post("/chat", json={"question": "q"}, headers={"X-API-Key": "key-b"})
    assert still_works.status_code == 200
    assert revoked.status_code == 401


def test_document_upload_requires_admin_role(monkeypatch):
    """POST /documents is this project's one real write endpoint -- the actual,
    non-speculative use of require_role(). A reader-role key can chat but can't
    upload/index a new document."""
    client = _client_with_api_keys(monkeypatch, "analyst:key-a:reader,admin:key-b:admin")

    reader_attempt = client.post(
        "/documents", headers={"X-API-Key": "key-a"},
        files={"file": ("test.md", b"# Test doc", "text/markdown")},
    )
    assert reader_attempt.status_code == 403

    with patch("src.api.routes.index_document") as mock_index, \
         patch("src.api.routes.load_document") as mock_load:
        from src.ingestion.document_loader import LoadedDocument
        from src.ingestion.index_documents import IndexResult
        mock_load.return_value = LoadedDocument(
            document_id="test", title="Test", version=None, effective_date=None,
            department=None, text="# Test doc", source_path="test.md",
        )
        mock_index.return_value = IndexResult(document_id="test", title="Test", chunk_count=1)
        admin_attempt = client.post(
            "/documents", headers={"X-API-Key": "key-b"},
            files={"file": ("test.md", b"# Test doc", "text/markdown")},
        )
    assert admin_attempt.status_code == 200


def test_legacy_api_key_still_works_alongside_api_keys(monkeypatch):
    monkeypatch.setenv("API_KEY", "legacy-secret")
    monkeypatch.setenv("API_KEYS", "analyst:key-a:reader")
    import src.api.main as main_module
    importlib.reload(main_module)
    client = TestClient(main_module.app)
    with patch("src.api.routes.run_agent") as mock_run_agent:
        mock_run_agent.return_value = AgentResponse(answer="ok", tool_calls=[], tools_used=[], citations=[])
        legacy = client.post("/chat", json={"question": "q"}, headers={"X-API-Key": "legacy-secret"})
        new_style = client.post("/chat", json={"question": "q"}, headers={"X-API-Key": "key-a"})
    assert legacy.status_code == 200
    assert new_style.status_code == 200
