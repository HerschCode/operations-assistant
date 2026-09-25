"""
Semantic similarity cache for LLM responses.

Stores (query, response) pairs in SQLite, keyed by embedding similarity.
On a new query, the cache embeds it and compares against stored embeddings;
if the nearest neighbour exceeds `similarity_threshold` (default 0.97) *and*
is within `ttl_seconds` (default 3600), the cached response is returned without
calling the LLM.

Design choices:
- SQLite: matches the summaries store (src/agent/conversation_store.py) — no
  new infrastructure for a portfolio deployment; path is env-configurable.
- sentence-transformers loaded lazily: the import is deferred to the first put()
  or get() call, so the cache can be constructed at app startup without forcing
  the embedding model to load until an actual request arrives.
- Thread safety: a new sqlite3 connection is opened per call (same pattern as
  conversation_store.py) — connections are not shared across threads.
- Graceful degradation: get() returns None on any error; put() is a no-op on
  failure. The /chat route treats a None cache result as a miss and proceeds
  with the LLM call normally.

The cache is intentionally NOT applied to:
- /chat/hitl (HITL turns have stateful side-effects from propose_intervention
  and must not return a stale cached answer)
- /investigate (investigation results depend on volatile operational data)

Activation: set SEMANTIC_CACHE=1. Path: SEMANTIC_CACHE_DB (default data/cache.db).
Model: SEMANTIC_CACHE_MODEL (default all-MiniLM-L6-v2, same as retrieval layer).
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np

_DEFAULT_DB = "data/cache.db"
_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_DEFAULT_THRESHOLD = 0.97
_DEFAULT_TTL = 3600


def _db_path() -> str:
    return os.environ.get("SEMANTIC_CACHE_DB", _DEFAULT_DB)


def _model_name() -> str:
    return os.environ.get("SEMANTIC_CACHE_MODEL", _DEFAULT_MODEL)


def _get_connection() -> sqlite3.Connection:
    path = Path(_db_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS semantic_cache (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            query_hash  TEXT UNIQUE,
            query       TEXT NOT NULL,
            embedding   BLOB NOT NULL,
            response    TEXT NOT NULL,
            created_at  REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


class SemanticCache:
    """Embedding-similarity cache with SQLite persistence.

    Usage:
        cache = SemanticCache()
        hit = cache.get("What is the SLA target?")
        if hit is not None:
            return hit
        response = call_llm(question)
        cache.put("What is the SLA target?", response)
        return response
    """

    def __init__(
        self,
        similarity_threshold: float = _DEFAULT_THRESHOLD,
        ttl_seconds: int = _DEFAULT_TTL,
        _encoder: Any = None,  # injectable for tests
    ) -> None:
        self.threshold = similarity_threshold
        self.ttl = ttl_seconds
        self._encoder = _encoder  # None = lazy load from env

    def _get_encoder(self) -> Any:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            self._encoder = SentenceTransformer(_model_name())
        return self._encoder

    def _embed(self, text: str) -> np.ndarray:
        enc = self._get_encoder()
        vec = enc.encode([text])[0]
        return np.array(vec, dtype=np.float32)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b)) / denom

    def get(self, query: str) -> dict | None:
        """Return cached response dict if a similar recent query exists, else None."""
        try:
            cutoff = time.time() - self.ttl
            conn = _get_connection()
            rows = conn.execute(
                "SELECT embedding, response FROM semantic_cache WHERE created_at > ?",
                (cutoff,),
            ).fetchall()
            conn.close()

            if not rows:
                return None

            q_emb = self._embed(query)
            best_sim = 0.0
            best_resp: str | None = None

            for emb_bytes, resp_json in rows:
                emb = np.frombuffer(emb_bytes, dtype=np.float32)
                sim = self._cosine(q_emb, emb)
                if sim > best_sim:
                    best_sim = sim
                    best_resp = resp_json

            if best_sim >= self.threshold and best_resp is not None:
                return json.loads(best_resp)
        except Exception:
            pass
        return None

    def put(self, query: str, response: dict) -> None:
        """Store query + response in the cache."""
        try:
            q_emb = self._embed(query)
            emb_bytes = q_emb.tobytes()
            query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
            now = time.time()
            conn = _get_connection()
            conn.execute(
                """INSERT OR REPLACE INTO semantic_cache
                   (query_hash, query, embedding, response, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (query_hash, query, emb_bytes, json.dumps(response), now),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def clear(self) -> None:
        """Remove all cached entries (useful in tests and cache invalidation)."""
        try:
            conn = _get_connection()
            conn.execute("DELETE FROM semantic_cache")
            conn.commit()
            conn.close()
        except Exception:
            pass


def is_enabled() -> bool:
    return os.environ.get("SEMANTIC_CACHE", "").strip() in ("1", "true", "yes")
