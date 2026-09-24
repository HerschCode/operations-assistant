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

MAX_TURNS_RETAINED = 10   # per conversation -- bounds how much prior context gets replayed

# Summarization thresholds: when stored turns hit SUMMARIZE_THRESHOLD, the oldest
# SUMMARIZE_BATCH_SIZE rows are collapsed into a summary and deleted. This keeps the
# live turn window small while preserving key context across longer conversations.
SUMMARIZE_THRESHOLD = 8
SUMMARIZE_BATCH_SIZE = 4

# Default path exported for test introspection (see test_conversation_store.py's
# disk-persistence test). _get_connection reads the env var on each call so test
# fixtures can override it without reloading this module.
_DB_PATH = Path("data/conversations.db")


def _get_connection() -> sqlite3.Connection:
    db_path = Path(os.environ.get("CONVERSATION_DB_PATH", str(_DB_PATH)))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            conversation_id TEXT PRIMARY KEY,
            summary TEXT NOT NULL
        )
    """)
    return conn


def get_history(conversation_id: str) -> list[dict]:
    summary = get_summary(conversation_id)
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT role, content FROM turns WHERE conversation_id = ? ORDER BY turn_order",
            (conversation_id,),
        ).fetchall()
        history = [{"role": role, "content": content} for role, content in rows]
    finally:
        conn.close()

    if summary:
        # Prepend the summary as a synthetic exchange so the model has context for
        # turns that have already been collapsed. Kept as user/assistant alternating
        # to match the Anthropic messages format the agent uses.
        prefix = [
            {"role": "user", "content": "(Earlier conversation context)"},
            {"role": "assistant", "content": f"Summary of earlier turns: {summary}"},
        ]
        return prefix + history
    return history


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


def get_summary(conversation_id: str) -> str | None:
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT summary FROM summaries WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def save_summary(conversation_id: str, summary: str) -> None:
    conn = _get_connection()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO summaries (conversation_id, summary) VALUES (?, ?)",
            (conversation_id, summary),
        )
        conn.commit()
    finally:
        conn.close()


def get_turns_to_summarize(conversation_id: str) -> list[dict]:
    """Return the oldest SUMMARIZE_BATCH_SIZE rows if total count >= SUMMARIZE_THRESHOLD.

    Each returned dict has keys: id, role, content. Returns [] when the conversation
    is short enough that no summarization is needed yet.
    """
    conn = _get_connection()
    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM turns WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()[0]
        if total < SUMMARIZE_THRESHOLD:
            return []
        rows = conn.execute(
            "SELECT id, role, content FROM turns WHERE conversation_id = ? "
            "ORDER BY turn_order LIMIT ?",
            (conversation_id, SUMMARIZE_BATCH_SIZE),
        ).fetchall()
        return [{"id": row[0], "role": row[1], "content": row[2]} for row in rows]
    finally:
        conn.close()


def delete_turns(turn_ids: list[int]) -> None:
    if not turn_ids:
        return
    conn = _get_connection()
    try:
        placeholders = ",".join("?" * len(turn_ids))
        conn.execute(f"DELETE FROM turns WHERE id IN ({placeholders})", turn_ids)
        conn.commit()
    finally:
        conn.close()


def reset_all() -> None:
    """Test-only: clears every conversation and summary. Not exposed via the API."""
    conn = _get_connection()
    try:
        conn.execute("DELETE FROM turns")
        conn.execute("DELETE FROM summaries")
        conn.commit()
    finally:
        conn.close()
