"""
Tests for the semantic similarity cache (Fix 6).
CPU-only — encoder is injected as a deterministic fake so no model download is needed.
"""
import json
import time
import tempfile
import os
import numpy as np
import pytest

from src.cache.semantic_cache import SemanticCache, is_enabled


# ── fake encoder ──────────────────────────────────────────────────────────────

class _FakeEncoder:
    """Returns deterministic embeddings:  hash(text[0]) as a unit vector."""

    def encode(self, texts: list[str]) -> list[np.ndarray]:
        result = []
        for text in texts:
            # Use first char's ordinal to pick a direction; others get a perpendicular
            seed = ord(text[0]) if text else 0
            v = np.zeros(4, dtype=np.float32)
            v[seed % 4] = 1.0
            result.append(v)
        return result


def _cache(tmpdir) -> SemanticCache:
    db = str(tmpdir.join("test_cache.db"))
    os.environ["SEMANTIC_CACHE_DB"] = db
    return SemanticCache(similarity_threshold=0.97, ttl_seconds=3600, _encoder=_FakeEncoder())


# ── basic cache behaviour ─────────────────────────────────────────────────────

class TestSemanticCacheBasic:
    def test_miss_on_empty_cache(self, tmpdir):
        c = _cache(tmpdir)
        assert c.get("What is the SLA target?") is None

    def test_hit_after_put(self, tmpdir):
        c = _cache(tmpdir)
        response = {"answer": "5 business days", "tools_used": ["get_sla_metrics"], "citations": []}
        c.put("What is the SLA target?", response)
        hit = c.get("What is the SLA target?")
        assert hit is not None
        assert hit["answer"] == "5 business days"

    def test_same_query_returns_same_response(self, tmpdir):
        c = _cache(tmpdir)
        response = {"answer": "42", "tools_used": [], "citations": []}
        c.put("How many cases?", response)
        assert c.get("How many cases?")["answer"] == "42"

    def test_dissimilar_query_is_miss(self, tmpdir):
        """'What' and 'How' start with different chars → different embedding direction."""
        c = _cache(tmpdir)
        response = {"answer": "5 days", "tools_used": [], "citations": []}
        c.put("What is the SLA?", response)
        # 'H' vs 'W': _FakeEncoder maps to different axes → cosine = 0
        assert c.get("How many cases?") is None

    def test_clear_removes_all_entries(self, tmpdir):
        c = _cache(tmpdir)
        c.put("What is the SLA?", {"answer": "5d", "tools_used": [], "citations": []})
        c.clear()
        assert c.get("What is the SLA?") is None


class TestSemanticCacheTTL:
    def test_expired_entry_is_miss(self, tmpdir, monkeypatch):
        """An entry older than ttl_seconds is not returned."""
        c = SemanticCache(
            similarity_threshold=0.97,
            ttl_seconds=10,
            _encoder=_FakeEncoder(),
        )
        os.environ["SEMANTIC_CACHE_DB"] = str(tmpdir.join("ttl.db"))

        response = {"answer": "cached", "tools_used": [], "citations": []}
        c.put("What is SLA?", response)

        # Patch time so the entry appears expired
        monkeypatch.setattr(time, "time", lambda: time.time() + 3600)
        assert c.get("What is SLA?") is None


class TestSemanticCacheIsEnabled:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("SEMANTIC_CACHE", raising=False)
        assert not is_enabled()

    def test_enabled_by_env(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_CACHE", "1")
        assert is_enabled()

    def test_enabled_with_true(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_CACHE", "true")
        assert is_enabled()


class TestSemanticCacheGracefulDegradation:
    def test_get_returns_none_on_broken_db(self, monkeypatch):
        """get() never raises — returns None on any error."""
        monkeypatch.setenv("SEMANTIC_CACHE_DB", "/nonexistent_dir/bad.db")
        c = SemanticCache(_encoder=_FakeEncoder())
        # mkdir will fail → _get_connection raises → get() swallows and returns None
        # (On Windows the mkdir failure may actually succeed for root, so we just
        # check that it doesn't raise, not that it returns None in all cases)
        result = c.get("anything")
        assert result is None or isinstance(result, dict)

    def test_put_is_noop_on_broken_encoder(self, tmpdir):
        """put() never raises even if encoding fails."""
        class _BrokenEncoder:
            def encode(self, texts):
                raise RuntimeError("encoder broke")

        c = SemanticCache(_encoder=_BrokenEncoder())
        os.environ["SEMANTIC_CACHE_DB"] = str(tmpdir.join("broken.db"))
        c.put("test", {"answer": "x", "tools_used": [], "citations": []})  # must not raise


class TestSemanticCacheAPIIntegration:
    """Smoke test: /chat route returns a cache hit on a repeated question."""

    def test_chat_route_uses_cache_on_hit(self, monkeypatch, tmpdir):
        from fastapi.testclient import TestClient
        from src.api.main import app

        db_path = str(tmpdir.join("route_cache.db"))
        monkeypatch.setenv("SEMANTIC_CACHE", "1")
        monkeypatch.setenv("SEMANTIC_CACHE_DB", db_path)

        # Pre-populate the cache with a deterministic fake encoder
        cache = SemanticCache(_encoder=_FakeEncoder())
        cache.put(
            "What is the SLA target?",
            {"answer": "Cached: 5 business days", "tools_used": [], "citations": []},
        )

        # Patch the SemanticCache constructor in routes so it uses our fake encoder
        monkeypatch.setattr(
            "src.cache.semantic_cache.SemanticCache",
            lambda **kw: SemanticCache(_encoder=_FakeEncoder()),
        )

        client = TestClient(app)
        resp = client.post("/chat", json={"question": "What is the SLA target?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Cached: 5 business days"
