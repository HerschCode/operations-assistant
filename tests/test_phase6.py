"""
Phase 6 tests: structured outputs, prompt caching, conversation summarization.

No real API keys or network calls needed -- all LLM clients are injected fakes.
"""
import os
import pytest
from dataclasses import dataclass, field
from unittest.mock import patch

# ── shared fake Anthropic response primitives ─────────────────────────────────

@dataclass
class _TextBlock:
    type: str = "text"
    text: str = ""


@dataclass
class _ToolUseBlock:
    type: str = "tool_use"
    id: str = "tu_1"
    name: str = "compile_report"
    input: dict = field(default_factory=dict)


@dataclass
class _Usage:
    input_tokens: int = 100
    output_tokens: int = 50
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class _FakeResponse:
    content: list
    usage: _Usage = field(default_factory=_Usage)
    stop_reason: str = "end_turn"


# ── structured outputs ────────────────────────────────────────────────────────

class TestCompileWithTool:
    def _valid_report_input(self):
        return {
            "executive_summary": "High-value orders exceed SLA by 8 days on average.",
            "problem": "SLA breach rate for high-value POs is 45%.",
            "evidence": ["Cycle time: 14.2 days", "SLA target: 5 days"],
            "root_causes": ["Manual approval bottleneck"],
            "relevant_policy": ["Procurement Policy §4.2"],
            "recommendations": ["Automate approval for repeat suppliers"],
            "limitations": "Analysis covers only Q3 data.",
        }

    def _fake_client(self, tool_input):
        class _Client:
            class messages:
                @staticmethod
                def create(**kwargs):
                    return _FakeResponse(
                        content=[_ToolUseBlock(input=tool_input)],
                        usage=_Usage(),
                    )
        return _Client()

    def _text_only_client(self, text):
        class _Client:
            class messages:
                @staticmethod
                def create(**kwargs):
                    return _FakeResponse(content=[_TextBlock(text=text)])
        return _Client()

    def _raising_client(self):
        class _Client:
            class messages:
                @staticmethod
                def create(**kwargs):
                    raise AttributeError("tool_choice not supported")
        return _Client()

    def test_compile_with_tool_returns_report(self):
        from src.agent.investigation import _compile_with_tool, _report_from_dict
        data = self._valid_report_input()
        result, failed = _compile_with_tool(
            self._fake_client(data),
            {"model": "test", "max_tokens": 1000},
            "compile this",
        )
        assert not failed
        assert result is not None
        report = _report_from_dict(result)
        assert report.executive_summary == data["executive_summary"]
        assert report.evidence == data["evidence"]
        assert report.root_causes == data["root_causes"]

    def test_compile_with_tool_fallback_on_text_only(self):
        from src.agent.investigation import _compile_with_tool
        result, failed = _compile_with_tool(
            self._text_only_client("plain text"),
            {"model": "test", "max_tokens": 1000},
            "compile this",
        )
        assert failed
        assert result is None

    def test_compile_with_tool_fallback_on_exception(self):
        from src.agent.investigation import _compile_with_tool
        result, failed = _compile_with_tool(
            self._raising_client(),
            {"model": "test", "max_tokens": 1000},
            "compile this",
        )
        assert failed
        assert result is None

    def test_run_investigation_uses_tool_path(self, monkeypatch):
        """run_investigation() produces a clean InvestigationResult when the tool path succeeds."""
        from src.agent.investigation import run_investigation
        from src.agent.agent import AgentResponse, ToolCallRecord

        tool_input = {
            "executive_summary": "Cycle time is 12 days.",
            "problem": "SLA breach for electronics.",
            "evidence": ["avg cycle: 12 days"],
            "root_causes": ["supplier delay"],
            "relevant_policy": [],
            "recommendations": ["flag supplier"],
            "limitations": "Q3 only.",
        }

        # Fake agent that returns a minimal AgentResponse
        def fake_run_agent(question, client=None, config_override=None):
            return AgentResponse(answer="Cycle time is high.", tool_calls=[], tools_used=[])

        # Fake client: returns a tool_use block for compile, nothing for agent loop
        client = self._fake_client(tool_input)

        monkeypatch.setattr("src.agent.investigation.run_agent", fake_run_agent)

        result = run_investigation("Why is SLA breached?", client=client)
        assert not result.parse_failed
        assert result.report.executive_summary == "Cycle time is 12 days."
        assert result.report.evidence == ["avg cycle: 12 days"]


# ── prompt caching ────────────────────────────────────────────────────────────

class TestPromptCaching:
    def _cache_client(self, cache_read=200):
        """Fake client that returns a text answer with cache_read_input_tokens."""
        class _Client:
            class messages:
                @staticmethod
                def create(**kwargs):
                    return _FakeResponse(
                        content=[_TextBlock(text="The SLA target is 5 days.")],
                        usage=_Usage(input_tokens=500, output_tokens=30, cache_read_input_tokens=cache_read),
                    )
        return _Client()

    def test_cache_read_tokens_accumulated(self):
        from src.agent.agent import _run_agent_anthropic
        with patch.dict(os.environ, {"PROMPT_CACHE": "1"}):
            result = _run_agent_anthropic(
                "What is the SLA target?",
                config={"model": "test", "max_tokens": 500, "temperature": 0.0, "max_tool_calls_per_turn": 3},
                client=self._cache_client(cache_read=200),
            )
        assert result.cache_read_input_tokens == 200
        assert result.answer == "The SLA target is 5 days."

    def test_no_cache_when_env_not_set(self):
        from src.agent.agent import _run_agent_anthropic
        env = {k: v for k, v in os.environ.items() if k != "PROMPT_CACHE"}
        with patch.dict(os.environ, env, clear=True):
            result = _run_agent_anthropic(
                "What is the SLA target?",
                config={"model": "test", "max_tokens": 500, "temperature": 0.0, "max_tool_calls_per_turn": 3},
                client=self._cache_client(cache_read=200),
            )
        # cache_read_input_tokens should be None when caching is off
        assert result.cache_read_input_tokens is None

    def test_cached_system_param_is_list(self):
        """When PROMPT_CACHE=1, system is passed as a list with cache_control."""
        captured = {}

        class _CapturingClient:
            class messages:
                @staticmethod
                def create(**kwargs):
                    captured.update(kwargs)
                    return _FakeResponse(
                        content=[_TextBlock(text="ok")],
                        usage=_Usage(),
                    )

        from src.agent.agent import _run_agent_anthropic
        with patch.dict(os.environ, {"PROMPT_CACHE": "1"}):
            _run_agent_anthropic(
                "Q?",
                config={"model": "m", "max_tokens": 100, "temperature": 0.0, "max_tool_calls_per_turn": 1},
                client=_CapturingClient(),
            )

        assert isinstance(captured.get("system"), list)
        assert captured["system"][0].get("cache_control") == {"type": "ephemeral"}
        assert "extra_headers" in captured
        assert "anthropic-beta" in captured["extra_headers"]


# ── conversation summarization ────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Each test gets its own SQLite DB so tests don't share state."""
    db_file = str(tmp_path / "test_conv.db")
    monkeypatch.setenv("CONVERSATION_DB_PATH", db_file)
    yield db_file


class TestSummarizer:
    def _fake_anthropic(self, summary_text: str):
        class _Client:
            class messages:
                @staticmethod
                def create(**kwargs):
                    return _FakeResponse(content=[_TextBlock(text=summary_text)])
        return _Client()

    def test_summarize_turns_returns_text(self):
        from src.agent.summarizer import summarize_turns
        turns = [
            {"role": "user", "content": "What is the cycle time for electronics?"},
            {"role": "assistant", "content": "The cycle time is 14.2 days per current data."},
        ]
        result = summarize_turns(turns, client=self._fake_anthropic("Electronics cycle time is 14.2 days."))
        assert result == "Electronics cycle time is 14.2 days."

    def test_summarize_turns_empty_returns_empty(self):
        from src.agent.summarizer import summarize_turns
        assert summarize_turns([]) == ""

    def test_summarize_turns_client_error_returns_fallback(self):
        from src.agent.summarizer import summarize_turns

        class _BrokenClient:
            class messages:
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("network error")

        result = summarize_turns(
            [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}],
            client=_BrokenClient(),
        )
        assert result.startswith("[summary unavailable:")


class TestConversationStore:
    def _add_turns(self, conv_id: str, n_exchanges: int):
        from src.agent.conversation_store import append_turn
        for i in range(n_exchanges):
            append_turn(conv_id, f"Q{i}", f"A{i}")

    def test_get_and_save_summary_roundtrip(self):
        from src.agent.conversation_store import get_summary, save_summary
        assert get_summary("conv1") is None
        save_summary("conv1", "Cycle time was discussed.")
        assert get_summary("conv1") == "Cycle time was discussed."

    def test_get_history_prepends_summary(self):
        from src.agent.conversation_store import get_history, save_summary, append_turn
        append_turn("c1", "What is SLA?", "SLA is 5 days.")
        save_summary("c1", "Earlier we discussed cycle time.")
        history = get_history("c1")
        # prefix (2) + live turn (2) = 4
        assert len(history) == 4
        assert history[0]["role"] == "user"
        assert "Earlier conversation context" in history[0]["content"]
        assert "Earlier we discussed" in history[1]["content"]
        assert history[2]["content"] == "What is SLA?"

    def test_get_history_no_summary_no_prefix(self):
        from src.agent.conversation_store import get_history, append_turn
        append_turn("c2", "Q", "A")
        history = get_history("c2")
        assert len(history) == 2
        assert history[0]["content"] == "Q"

    def test_get_turns_to_summarize_below_threshold(self):
        from src.agent.conversation_store import get_turns_to_summarize, SUMMARIZE_THRESHOLD
        self._add_turns("c3", (SUMMARIZE_THRESHOLD // 2) - 1)
        assert get_turns_to_summarize("c3") == []

    def test_get_turns_to_summarize_at_threshold(self):
        from src.agent.conversation_store import (
            get_turns_to_summarize, SUMMARIZE_THRESHOLD, SUMMARIZE_BATCH_SIZE,
        )
        self._add_turns("c4", SUMMARIZE_THRESHOLD // 2)  # exactly at threshold
        turns = get_turns_to_summarize("c4")
        assert len(turns) == SUMMARIZE_BATCH_SIZE
        assert all("role" in t and "content" in t and "id" in t for t in turns)

    def test_delete_turns_removes_rows(self):
        from src.agent.conversation_store import (
            get_turns_to_summarize, delete_turns, SUMMARIZE_THRESHOLD,
            get_history,
        )
        self._add_turns("c5", SUMMARIZE_THRESHOLD // 2)
        to_delete = get_turns_to_summarize("c5")
        ids = [t["id"] for t in to_delete]
        delete_turns(ids)
        remaining = get_history("c5")
        remaining_ids = set()
        # After deletion the oldest batch is gone; total rows < threshold now
        assert len(remaining) < SUMMARIZE_THRESHOLD

    def test_reset_all_clears_summaries(self):
        from src.agent.conversation_store import get_summary, save_summary, reset_all
        save_summary("cx", "some text")
        reset_all()
        assert get_summary("cx") is None
