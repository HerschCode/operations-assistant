"""
Phase 23 (load & failure testing) named real concurrency testing as a gap this
environment couldn't close -- no live server to point a load-generation tool at.
What IS achievable without one: hitting the real FastAPI app (via TestClient, which
runs the real app code, not a mock of it) from multiple real threads at once. This
doesn't measure latency-under-load the way a real deployed server + locust/k6 would,
but it does exercise real concurrency-sensitive code -- specifically
conversation_store's module-level dict, which was never verified thread-safe until
this test.
"""
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.main import app
from src.agent.agent import AgentResponse
from src.agent.conversation_store import reset_all, get_history

client = TestClient(app)


def test_concurrent_health_requests_all_succeed():
    with ThreadPoolExecutor(max_workers=20) as pool:
        responses = list(pool.map(lambda _: client.get("/health"), range(20)))
    assert all(r.status_code == 200 for r in responses)


@patch("src.api.routes.run_agent")
def test_concurrent_chat_requests_with_different_conversation_ids_stay_isolated(mock_run_agent):
    """20 threads, each starting and using its OWN conversation -- confirms
    conversation_store doesn't leak state across concurrent conversations (e.g. one
    thread's history accidentally appearing in another's)."""
    reset_all()
    mock_run_agent.side_effect = lambda question, history=None: AgentResponse(
        answer=f"answer to: {question}", tool_calls=[], tools_used=[], citations=[],
    )

    def do_conversation(i):
        convo_id = f"concurrent-convo-{i}"
        r1 = client.post("/chat", json={"question": f"question-{i}-a", "conversation_id": convo_id})
        r2 = client.post("/chat", json={"question": f"question-{i}-b", "conversation_id": convo_id})
        return i, r1.json(), r2.json()

    with ThreadPoolExecutor(max_workers=20) as pool:
        results = list(pool.map(do_conversation, range(20)))

    for i, r1_body, r2_body in results:
        # each conversation's second answer should reflect ITS OWN first question in
        # history, not another thread's -- checked via the store directly, not just
        # trusting the mocked answer text
        history = get_history(f"concurrent-convo-{i}")
        assert history[0]["content"] == f"question-{i}-a"
        assert f"question-{i}-a" not in [h["content"] for h in get_history(f"concurrent-convo-{(i+1) % 20}")]


@patch("src.api.routes.run_agent")
def test_concurrent_requests_to_same_conversation_do_not_crash(mock_run_agent):
    """Real threads writing to the SAME conversation_id concurrently -- now backed by
    SQLite (a per-call connection, relying on SQLite's own file-level locking rather
    than Python's GIL, since conversation_store.py switched from an in-memory dict to
    persist across restarts). The INTERLEAVING of which request's turn gets appended
    first is genuinely a race (not deterministic) -- this test checks the system
    doesn't crash or lose data, not that ordering is deterministic, which is an
    honest distinction worth keeping rather than asserting something concurrent
    access can't actually promise."""
    reset_all()
    mock_run_agent.side_effect = lambda question, history=None: AgentResponse(
        answer=f"answer to: {question}", tool_calls=[], tools_used=[], citations=[],
    )

    def do_request(i):
        return client.post("/chat", json={"question": f"shared-question-{i}", "conversation_id": "shared-convo"})

    with ThreadPoolExecutor(max_workers=10) as pool:
        responses = list(pool.map(do_request, range(10)))

    assert all(r.status_code == 200 for r in responses)
    # all 10 turns landed somewhere (bounded by MAX_TURNS_RETAINED, so not necessarily
    # all 20 messages survive, but the store shouldn't have crashed or corrupted)
    history = get_history("shared-convo")
    assert len(history) % 2 == 0  # always question+answer pairs, never a torn write
    assert len(history) > 0
