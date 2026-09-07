"""
In-memory conversation history, keyed by conversation_id. Deliberately simple and
explicitly NOT production-durable -- resets on process restart, not shared across
multiple API instances. A real deployment needing that would back this with Redis
or a DB table; named here as a known limitation, not hidden.

Stores only simple {"role", "content"} text turns, not the raw tool-call scaffolding
a turn generates internally. This is a deliberate design choice: each new turn in a
conversation re-gathers whatever tool evidence it needs fresh, rather than trusting
a previous turn's tool results might still be current -- prior turns provide
conversational CONTEXT (what was already discussed), not cached DATA.
"""
from collections import defaultdict

_conversations: dict[str, list[dict]] = defaultdict(list)

MAX_TURNS_RETAINED = 10  # per conversation -- bounds how much prior context gets replayed


def get_history(conversation_id: str) -> list[dict]:
    return list(_conversations[conversation_id])


def append_turn(conversation_id: str, question: str, answer: str) -> None:
    _conversations[conversation_id].append({"role": "user", "content": question})
    _conversations[conversation_id].append({"role": "assistant", "content": answer})
    # keep only the most recent MAX_TURNS_RETAINED*2 messages (question+answer pairs)
    _conversations[conversation_id] = _conversations[conversation_id][-(MAX_TURNS_RETAINED * 2):]


def clear(conversation_id: str) -> None:
    _conversations.pop(conversation_id, None)


def reset_all() -> None:
    """Test-only: clears every conversation. Not exposed via the API."""
    _conversations.clear()
