import json
from pathlib import Path

import pytest

from src.evaluation.claim_support import answer_supported, key_terms, sentence_support

CHUNKS = ["# 5. Manual Credit Review\nOrders from a new supplier (fewer than 3 prior completed orders) route to Manual "
          "Credit Review. Manual Credit Review is performed by the Vendor Risk team.",
          "| Category | Target (business days) |\n|---|---|\n| 3-way match | 10 |\n| Consignment | 14 |",
          "SLA targets are reviewed quarterly by Operations leadership."]


@pytest.mark.parametrize("answer", [
    "Manual Credit Review is performed by the Vendor Risk team.",
    "The SLA target for Consignment orders is **14 business days**.",
    "A supplier needs at least three prior completed orders to avoid Manual Credit Review.",
    "SLA targets are reviewed quarterly by Operations leadership.",
])
def test_correct_answers_pass(answer):
    assert answer_supported(answer, CHUNKS, 0.65)[0]


@pytest.mark.parametrize("answer,why", [
    ("Manual Credit Review is performed by the Finance team.", "missing_key_terms"),
    ("The SLA target for Consignment orders is **21 business days**.", "missing_number_claims"),
    ("SLA targets are reviewed monthly by Operations leadership.", "missing_key_terms"),
])
def test_planted_wrong_facts_are_blocked_with_a_reason(answer, why):
    ok, detail = answer_supported(answer, CHUNKS, 0.65)
    assert not ok and any(d[why] for d in detail)


def test_section_numbers_are_not_evidence_for_a_claim():
    # "5" appears only as the heading "# 5. Manual Credit Review", never as "5 business days"
    assert not sentence_support("Approval takes 5 business days.", CHUNKS)["numbers_supported"]


def test_list_markers_are_not_claims_and_sentence_start_is_not_a_key_term():
    assert sentence_support("1. Manual Credit Review is performed by the Vendor Risk team.", CHUNKS)["numbers_supported"]
    assert "since" not in key_terms("Since the PO exceeds the limit, Procurement leadership approves.")


def test_off_topic_answer_is_blocked_and_empty_answer_is_blocked():
    assert not answer_supported("The Eiffel Tower was completed in 1889 for the World's Fair.", CHUNKS, 0.65)[0]
    assert not answer_supported("", CHUNKS, 0.65)[0]


def test_polarity_flip_is_a_known_limitation():
    """Documented limit: a lexical check cannot see Yes/No or included/excluded flips."""
    assert answer_supported("Yes. SLA targets are reviewed quarterly by Operations leadership.", CHUNKS, 0.65)[0]


def test_committed_labeled_eval_support_gate_beats_nli_on_held_out_questions():
    d = json.loads(Path("reports/gate_labeled_eval.json").read_text(encoding="utf-8"))
    g = d["gates"]
    nli = g["nli@0.05 (deployed)"]["held_out_even_ids"]
    sup = g[f"support@{d['chosen_support_recall']}"]["held_out_even_ids"]
    assert sup["correct_pass_rate"] > nli["correct_pass_rate"] + 0.3
    assert sup["off_context_pass_rate"] <= nli["off_context_pass_rate"]
    assert sup["wrong_fact_pass_rate"] <= nli["wrong_fact_pass_rate"]
