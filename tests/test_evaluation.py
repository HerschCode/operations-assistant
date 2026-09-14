from unittest.mock import patch

from src.evaluation.evaluate_agent import _tool_selection_correct, load_questions, run_evaluation
from src.agent.agent import AgentResponse


def test_load_questions_reads_real_file():
    questions = load_questions()
    assert len(questions) == 25
    categories = {q["category"] for q in questions}
    assert categories == {"data", "document", "combined", "multi_step", "unanswerable", "adversarial"}


def test_tool_selection_correct_when_expected_tools_subset_of_actual():
    assert _tool_selection_correct(["get_sla_metrics"], ["get_sla_metrics", "search_policy_documents"], "combined")


def test_tool_selection_incorrect_when_expected_tool_missing():
    assert not _tool_selection_correct(["get_sla_metrics", "search_policy_documents"], ["get_sla_metrics"], "combined")


def test_tool_selection_always_correct_for_unanswerable_category():
    # scored separately by answer text inspection, not tool-set comparison
    assert _tool_selection_correct(["anything"], [], "unanswerable")


def test_tool_selection_always_correct_for_adversarial_category():
    assert _tool_selection_correct(["anything"], [], "adversarial")


@patch("src.evaluation.evaluate_agent.run_agent")
def test_run_evaluation_scores_all_questions(mock_run_agent):
    mock_run_agent.return_value = AgentResponse(
        answer="test answer", tool_calls=[], tools_used=["get_cycle_time"], citations=[],
    )
    summary = run_evaluation()
    assert summary["total"] == 25
    assert mock_run_agent.call_count == 25
