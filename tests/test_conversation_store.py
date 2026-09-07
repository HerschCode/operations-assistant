from src.agent.conversation_store import get_history, append_turn, clear, reset_all, MAX_TURNS_RETAINED


def setup_function():
    reset_all()


def test_get_history_empty_for_new_conversation():
    assert get_history("new-convo") == []


def test_append_turn_stores_question_and_answer():
    append_turn("c1", "What is the cycle time?", "It's 48 hours.")
    history = get_history("c1")
    assert history == [
        {"role": "user", "content": "What is the cycle time?"},
        {"role": "assistant", "content": "It's 48 hours."},
    ]


def test_conversations_are_isolated_by_id():
    append_turn("c1", "q1", "a1")
    append_turn("c2", "q2", "a2")
    assert get_history("c1") == [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
    assert get_history("c2") == [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]


def test_history_is_bounded_to_max_turns():
    for i in range(MAX_TURNS_RETAINED + 5):
        append_turn("c1", f"q{i}", f"a{i}")
    history = get_history("c1")
    assert len(history) == MAX_TURNS_RETAINED * 2
    assert history[0]["content"] == f"q{5}"  # oldest 5 turns dropped


def test_clear_removes_a_conversation():
    append_turn("c1", "q", "a")
    clear("c1")
    assert get_history("c1") == []
