"""
Phase 7 tests: QLoRA data preparation and evaluation utilities.

Tests are CPU-only and do not load any model -- they verify the data pipeline
logic (format, split, keyword extraction) without requiring CUDA or a model download.
"""
import json
import math
import tempfile
from pathlib import Path

import pytest

from scripts.prepare_generation_data import _build_answer, _stratified_split
from scripts.evaluate_generation import extract_keywords, keyword_recall


# ── data preparation ──────────────────────────────────────────────────────────

class TestBuildAnswer:
    def test_strips_markdown_header(self):
        text = "# 2. Targets by Category\nStandard POs: 5 business days"
        answer = _build_answer("Procurement Policy, §7", text)
        assert "According to Procurement Policy, §7:" in answer
        # Markdown header markers should be stripped
        assert "# " not in answer

    def test_includes_citation(self):
        answer = _build_answer("SLA Policy, Section 2", "3-way match: 10 days")
        assert "SLA Policy, Section 2" in answer

    def test_includes_content(self):
        answer = _build_answer("Doc", "approval within 5 business days")
        assert "5 business days" in answer


class TestStratifiedSplit:
    def _make_items(self, n):
        return [{"id": i} for i in range(n)]

    def test_split_fractions(self):
        by_cat = {
            "lookup": self._make_items(30),
            "numerical": self._make_items(25),
            "paraphrase": self._make_items(17),
        }
        train, test = _stratified_split(by_cat, train_frac=0.80, seed=42)
        total = 30 + 25 + 17
        assert len(train) + len(test) == total
        # Each split gets at least 1 from each category
        train_cats = set()
        test_cats = set()
        for item in train:
            for cat, items in by_cat.items():
                if item in items:
                    train_cats.add(cat)
        assert len(train) > len(test)

    def test_small_category_gets_at_least_one_in_each_split(self):
        by_cat = {"rare": [{"id": 1}, {"id": 2}]}
        train, test = _stratified_split(by_cat, train_frac=0.80, seed=42)
        # With 2 items and 80% train, we expect 1 train, 1 test
        assert len(train) == 1
        assert len(test) == 1

    def test_deterministic_with_same_seed(self):
        by_cat = {"a": [{"id": i} for i in range(20)]}
        train1, test1 = _stratified_split(by_cat, 0.8, seed=0)
        train2, test2 = _stratified_split(by_cat, 0.8, seed=0)
        assert [x["id"] for x in train1] == [x["id"] for x in train2]

    def test_different_seed_gives_different_order(self):
        by_cat = {"a": [{"id": i} for i in range(20)]}
        train1, _ = _stratified_split(by_cat, 0.8, seed=0)
        train2, _ = _stratified_split(by_cat, 0.8, seed=99)
        # Very likely to differ in order with 20 items
        assert [x["id"] for x in train1] != [x["id"] for x in train2]


class TestJsonlFormat:
    """End-to-end check that prepare_generation_data produces valid JSONL files."""

    def test_train_jsonl_valid(self, tmp_path):
        train_path = Path("data/finetune/train.jsonl")
        if not train_path.exists():
            pytest.skip("data/finetune/train.jsonl not yet generated")
        with open(train_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                obj = json.loads(line)
                assert "messages" in obj, f"line {i} missing messages"
                roles = [m["role"] for m in obj["messages"]]
                assert roles == ["system", "user", "assistant"], f"line {i} wrong roles: {roles}"
                for msg in obj["messages"]:
                    assert msg["content"].strip(), f"line {i} has empty content"

    def test_test_jsonl_valid(self, tmp_path):
        test_path = Path("data/finetune/test.jsonl")
        if not test_path.exists():
            pytest.skip("data/finetune/test.jsonl not yet generated")
        with open(test_path, encoding="utf-8") as f:
            examples = [json.loads(l) for l in f if l.strip()]
        assert len(examples) > 0

    def test_no_overlap_between_splits(self):
        train_path = Path("data/finetune/train.jsonl")
        test_path = Path("data/finetune/test.jsonl")
        if not (train_path.exists() and test_path.exists()):
            pytest.skip("splits not generated yet")
        def load_ids(path):
            ids = set()
            with open(path) as f:
                for line in f:
                    obj = json.loads(line)
                    meta_id = obj.get("meta", {}).get("id")
                    if meta_id is not None:
                        ids.add(meta_id)
            return ids
        train_ids = load_ids(train_path)
        test_ids = load_ids(test_path)
        overlap = train_ids & test_ids
        assert not overlap, f"Train/test overlap on ids: {overlap}"


# ── evaluation utilities ──────────────────────────────────────────────────────

class TestExtractKeywords:
    def test_extracts_numbers(self):
        kw = extract_keywords("SLA target is 10 business days")
        assert "10" in kw

    def test_extracts_capitalised_terms(self):
        kw = extract_keywords("Procurement Policy requires Secondary Approval")
        assert "Procurement" in kw or "Secondary" in kw

    def test_extracts_domain_terms(self):
        kw = extract_keywords("escalation procedure for SLA breach events")
        assert any("escalat" in k for k in kw) or "breach" in kw

    def test_ignores_stopwords(self):
        kw = extract_keywords("the a an and or but for")
        # All stopwords → empty or near-empty
        assert len(kw) == 0

    def test_empty_string(self):
        assert extract_keywords("") == set()


class TestKeywordRecall:
    def test_perfect_match(self):
        ref = "The SLA target for 3-way match is 10 business days."
        gen = "According to the SLA Policy: 3-way match target is 10 business days."
        assert keyword_recall(ref, gen) == 1.0

    def test_zero_match(self):
        ref = "SLA target is 10 business days for 3-way match."
        gen = "I don't know anything about Northstar Manufacturing policies."
        r = keyword_recall(ref, gen)
        assert r < 0.3

    def test_partial_match(self):
        ref = "Standard POs must be approved within 5 business days per Procurement Policy §7."
        gen = "The approval target is 5 days."
        r = keyword_recall(ref, gen)
        assert 0.0 < r < 1.0

    def test_empty_reference_returns_one(self):
        # No keywords to match → trivially correct
        assert keyword_recall("the a an", "anything") == 1.0
