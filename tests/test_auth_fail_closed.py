"""With no API keys configured, protected routes refuse requests unless local development opts out."""
from fastapi.testclient import TestClient

from src.api.main import app


def test_no_keys_configured_fails_closed(monkeypatch):
    monkeypatch.delenv("ALLOW_UNAUTHENTICATED", raising=False)
    r = TestClient(app).get("/documents")
    assert r.status_code == 503
    assert "not configured" in r.json()["detail"]


def test_explicit_opt_out_allows_local_development(monkeypatch):
    monkeypatch.setenv("ALLOW_UNAUTHENTICATED", "1")
    r = TestClient(app).get("/documents")
    assert r.status_code != 503


def test_opt_out_needs_the_exact_value(monkeypatch):
    monkeypatch.setenv("ALLOW_UNAUTHENTICATED", "true")
    assert TestClient(app).get("/documents").status_code == 503


def test_configured_key_still_required(monkeypatch):
    monkeypatch.delenv("ALLOW_UNAUTHENTICATED", raising=False)
    monkeypatch.setenv("API_KEY", "k-123")
    client = TestClient(app)
    assert client.get("/documents").status_code == 401
    assert client.get("/documents", headers={"X-API-Key": "k-123"}).status_code != 401


def test_health_and_demo_stay_public(monkeypatch):
    monkeypatch.delenv("ALLOW_UNAUTHENTICATED", raising=False)
    assert TestClient(app).get("/health").status_code != 503
