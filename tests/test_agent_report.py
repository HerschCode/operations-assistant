from src.evaluation.evaluate_agent import AgentEvalResult
from src.evaluation.generate_agent_report import _investigate, build_report_rows, render_markdown_table


def make_summary(results: list[AgentEvalResult]) -> dict:
    passed = sum(1 for r in results if r.tool_selection_correct)
    return {
        "total": len(results), "passed": passed,
        "pass_rate_pct": round((passed / len(results)) * 100, 1) if results else 0.0,
        "results": results,
    }


def test_investigate_returns_none_for_correct_result():
    r = AgentEvalResult("q", "data", ["get_cycle_time"], ["get_cycle_time"], True, "answer")
    assert _investigate(r) is None


def test_investigate_flags_missing_tool():
    r = AgentEvalResult("q", "combined", ["get_sla_metrics", "search_policy_documents"], ["get_sla_metrics"], False, "answer")
    investigation = _investigate(r)
    assert "missing expected tool" in investigation
    assert "search_policy_documents" in investigation


def test_investigate_flags_unexpected_extra_tool():
    r = AgentEvalResult("q", "data", ["get_cycle_time"], ["get_cycle_time", "get_bottlenecks"], False, "answer")
    investigation = _investigate(r)
    assert "unexpected tool" in investigation
    assert "get_bottlenecks" in investigation


def test_investigate_directs_to_manual_review_for_unanswerable():
    r = AgentEvalResult("q", "unanswerable", [], [], False, "I don't have that information.")
    investigation = _investigate(r)
    assert investigation is not None
    assert "manually" in investigation.lower()


def test_build_report_rows_pairs_each_result_with_investigation():
    results = [
        AgentEvalResult("q1", "data", ["get_cycle_time"], ["get_cycle_time"], True, "a1"),
        AgentEvalResult("q2", "combined", ["get_sla_metrics", "search_policy_documents"], ["get_sla_metrics"], False, "a2"),
    ]
    rows = build_report_rows(make_summary(results))
    assert rows[0].investigation is None
    assert rows[1].investigation is not None


def test_render_markdown_table_includes_all_rows_and_summary_line():
    results = [
        AgentEvalResult("What is cycle time?", "data", ["get_cycle_time"], ["get_cycle_time"], True, "a"),
        AgentEvalResult("Why SLA breach and policy?", "combined", ["get_sla_metrics", "search_policy_documents"], ["get_sla_metrics"], False, "a"),
    ]
    table = render_markdown_table(make_summary(results))
    assert "1/2 (50.0%)" in table
    assert "What is cycle time?" in table
    assert "Why SLA breach and policy?" in table
    assert "PASS" in table and "FAIL" in table
    assert "missing expected tool" in table
