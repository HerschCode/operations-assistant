"""
SQLite-backed conversation history, keyed by conversation_id. Replaces the original
pure in-memory dict (still available in git history) -- that version lost every
conversation on ANY process restart, not just a redeploy, which was a real gap named
in FUTURE_IMPROVEMENTS.md as "if this ever needs to survive a restart."

Still explicitly NOT durable across a Render redeploy (the free tier's disk is
ephemeral -- see docs/deployment.md's note on this same limitation for the Chroma
index) and NOT shared across multiple API instances -- a real multi-instance
deployment needing that would back this with Postgres/Redis instead. Named here as
a known limitation, not hidden. What this DOES fix: an ordinary process
crash/restart (the far more common case single-instance) no longer silently drops
every in-flight conversation.

A new sqlite3 connection is opened per call rather than one shared connection kept
across calls -- sqlite3 connections aren't safe to share across threads by default,
and FastAPI runs sync route handlers in a real thread pool (not just async tasks on
one thread), so a shared connection would need its own locking to be safe. Opening
per-call avoids that entirely; SQLite's own file-level locking handles the rest, and
this project's conversation volume doesn't need connection pooling.

Stores only simple {"role", "content"} text turns, not the raw tool-call scaffolding
a turn generates internally -- same reasoning as before: each new turn re-gathers
whatever tool evidence it needs fresh, rather than trusting a previous turn's tool
results might still be current.
"""
import os
import sqlite3
from pathlib import Path

MAX_TURNS_RETAINED = 10  # per conversation -- bounds how much prior context gets replayed

_DB_PATH = Path(os.environ.get("CONVERSATION_DB_PATH", "data/conversations.db"))


def _get_connection() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            turn_order INTEGER NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_turns_conversation ON turns(conversation_id, turn_order)")
    return conn


def get_history(conversation_id: str) -> list[dict]:
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT role, content FROM turns WHERE conversation_id = ? ORDER BY turn_order",
            (conversation_id,),
        ).fetchall()
        return [{"role": role, "content": content} for role, content in rows]
    finally:
        conn.close()


def append_turn(conversation_id: str, question: str, answer: str) -> None:
    conn = _get_connection()
    try:
        next_order = conn.execute(
            "SELECT COALESCE(MAX(turn_order), -1) + 1 FROM turns WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO turns (conversation_id, role, content, turn_order) VALUES (?, 'user', ?, ?)",
            (conversation_id, question, next_order),
        )
        conn.execute(
            "INSERT INTO turns (conversation_id, role, content, turn_order) VALUES (?, 'assistant', ?, ?)",
            (conversation_id, answer, next_order + 1),
        )
        conn.commit()

        # Trim to the most recent MAX_TURNS_RETAINED*2 messages, same retention
        # policy as the original in-memory version.
        keep_count = MAX_TURNS_RETAINED * 2
        conn.execute(
            """
            DELETE FROM turns
            WHERE conversation_id = ? AND id NOT IN (
                SELECT id FROM turns WHERE conversation_id = ? ORDER BY turn_order DESC LIMIT ?
            )
            """,
            (conversation_id, conversation_id, keep_count),
        )
        conn.commit()
    finally:
        conn.close()


def clear(conversation_id: str) -> None:
    conn = _get_connection()
    try:
        conn.execute("DELETE FROM turns WHERE conversation_id = ?", (conversation_id,))
        conn.commit()
    finally:
        conn.close()


def reset_all() -> None:
    """Test-only: clears every conversation. Not exposed via the API."""
    conn = _get_connection()
    try:
        conn.execute("DELETE FROM turns")
        conn.commit()
    finally:
        conn.close()
