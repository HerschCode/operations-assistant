from src.agent.agent import AgentResponse, ToolCallRecord
from src.evaluation.evaluate_answers import check_groundedness, _numbers_in_text


def test_numbers_in_text_extracts_percentages_and_dollars():
    numbers = _numbers_in_text("The breach rate is 18.4% and the order was worth $10,000.")
    assert "18.4%" in numbers
    assert "$10,000" in numbers


def test_groundedness_passes_when_number_appears_in_tool_result():
    response = AgentResponse(
        answer="The breach rate is 18.4%.",
        tool_calls=[ToolCallRecord(name="get_sla_metrics", input={}, result={"breach_rate_pct": 18.4})],
        tools_used=["get_sla_metrics"],
        citations=[{"kind": "data", "reference": "get_sla_metrics"}],
    )
    check = check_groundedness(response)
    assert check.all_citations_traceable
    assert check.ungrounded_numbers == []


def test_groundedness_flags_fabricated_number_not_in_any_tool_result():
    response = AgentResponse(
        answer="The breach rate is 99.9%, an alarming figure.",
        tool_calls=[ToolCallRecord(name="get_sla_metrics", input={}, result={"breach_rate_pct": 18.4})],
        tools_used=["get_sla_metrics"],
        citations=[{"kind": "data", "reference": "get_sla_metrics"}],
    )
    check = check_groundedness(response)
    assert "99.9%" in check.ungrounded_numbers


def test_groundedness_flags_citation_to_a_tool_that_was_never_called():
    response = AgentResponse(
        answer="Per supplier data, this looks fine.",
        tool_calls=[ToolCallRecord(name="get_sla_metrics", input={}, result={"breach_rate_pct": 18.4})],
        tools_used=["get_sla_metrics"],
        citations=[{"kind": "data", "reference": "get_supplier_performance"}],  # never actually called
    )
    check = check_groundedness(response)
    assert not check.all_citations_traceable
    assert "get_supplier_performance" in check.untraceable_citations


def test_groundedness_flags_document_citation_not_in_retrieved_text():
    response = AgentResponse(
        answer="Per the Exception Handling Procedure, this is allowed.",
        tool_calls=[ToolCallRecord(
            name="search_policy_documents", input={"query": "x"},
            result={"found": True, "results": [{"citation": "Procurement Policy, Section 4.2", "text": "...", "similarity_score": 0.8}]},
        )],
        tools_used=["search_policy_documents"],
        # cites a DIFFERENT document than what was actually retrieved -- should be flagged
        citations=[{"kind": "document", "reference": "Exception Handling Procedure, Section 3"}],
    )
    check = check_groundedness(response)
    assert not check.all_citations_traceable


def test_groundedness_ignores_errored_tool_calls_as_evidence():
    """A failed tool call's error message shouldn't count as 'evidence' that grounds
    a number -- only successful results should."""
    response = AgentResponse(
        answer="The rate is 42%.",
        tool_calls=[ToolCallRecord(name="get_sla_metrics", input={}, error="42 seconds elapsed before timeout")],
        tools_used=[], citations=[],
    )
    check = check_groundedness(response)
    assert "42%" in check.ungrounded_numbers
