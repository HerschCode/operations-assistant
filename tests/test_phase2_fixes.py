"""
Tests for Fix 2: agent eval improvements (quota-proof, kappa computation).
CPU-only — no agent calls, no model loading.
"""
import json
import tempfile
from pathlib import Path

import pytest

from scripts.compute_kappa import cohen_kappa, weighted_kappa, _kappa_label


# ── Cohen's kappa ─────────────────────────────────────────────────────────────

class TestCohenKappa:
    def test_perfect_agreement(self):
        assert cohen_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 1.0

    def test_zero_kappa_chance_agreement(self):
        # When both raters use each category equally, chance agreement is 1/K
        # Complete disagreement pattern: systematic offset
        rater_a = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5]
        rater_b = [2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1]
        k = cohen_kappa(rater_a, rater_b)
        assert k < 0  # systematic disagreement → negative kappa

    def test_symmetric(self):
        a = [1, 2, 3, 4, 5, 3, 2, 1]
        b = [1, 2, 3, 4, 4, 3, 2, 2]
        assert cohen_kappa(a, b) == pytest.approx(cohen_kappa(b, a), abs=1e-9)

    def test_empty_returns_nan(self):
        import math
        assert math.isnan(cohen_kappa([], []))


class TestWeightedKappa:
    def test_perfect_agreement(self):
        assert weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 1.0

    def test_weighted_higher_than_unweighted_for_off_by_one(self):
        # Off-by-one errors should be penalised less than larger gaps
        a = [1, 2, 3, 4, 5]
        b = [2, 3, 4, 5, 4]  # all off-by-one
        assert weighted_kappa(a, b) > cohen_kappa(a, b)

    def test_larger_disagreement_lowers_weighted_kappa(self):
        a = [1, 2, 3, 4, 5]
        b_close = [2, 3, 4, 5, 4]   # off by 1
        b_far = [5, 1, 5, 1, 5]     # far off
        assert weighted_kappa(a, b_close) > weighted_kappa(a, b_far)


class TestKappaLabel:
    def test_almost_perfect(self):
        assert _kappa_label(0.85) == "almost perfect"
        assert _kappa_label(1.00) == "almost perfect"

    def test_substantial(self):
        assert _kappa_label(0.65) == "substantial"

    def test_moderate(self):
        assert _kappa_label(0.45) == "moderate"

    def test_fair(self):
        assert _kappa_label(0.25) == "fair"

    def test_slight(self):
        assert _kappa_label(0.05) == "slight"

    def test_poor(self):
        assert _kappa_label(-0.1) == "poor"


# ── Full compute_kappa end-to-end (temp files) ────────────────────────────────

class TestComputeKappaEndToEnd:
    def _make_results(self, rows: list[dict]) -> dict:
        return {
            "summary": {"n_questions": len(rows), "n_evaluated": len(rows)},
            "rows": rows,
        }

    def test_runs_with_sufficient_overlap(self, tmp_path, monkeypatch):
        import sys
        from scripts import compute_kappa as ck

        # Build 25 rows with judge scores
        rows = [
            {"id": f"Q{i:03d}", "category": "data", "llm_judge_score": (i % 5) + 1}
            for i in range(1, 26)
        ]
        results_file = tmp_path / "results.json"
        results_file.write_text(json.dumps(self._make_results(rows)), encoding="utf-8")

        labels_file = tmp_path / "labels.json"
        labels = [{"id": f"Q{i:03d}", "human_score": (i % 5) + 1} for i in range(1, 26)]
        labels_file.write_text(json.dumps(labels), encoding="utf-8")

        monkeypatch.setattr(sys, "argv", [
            "compute_kappa", str(labels_file),
            "--results", str(results_file),
            "--min-overlap", "20",
        ])
        ck.main()  # should not raise
        assert (tmp_path / "kappa_results.json").exists()

    def test_fails_with_insufficient_overlap(self, tmp_path, monkeypatch):
        import sys
        from scripts import compute_kappa as ck

        rows = [{"id": "Q001", "category": "data", "llm_judge_score": 4}]
        results_file = tmp_path / "results.json"
        results_file.write_text(json.dumps(self._make_results(rows)), encoding="utf-8")

        labels_file = tmp_path / "labels.json"
        labels_file.write_text(json.dumps([{"id": "Q001", "human_score": 4}]), encoding="utf-8")

        monkeypatch.setattr(sys, "argv", [
            "compute_kappa", str(labels_file),
            "--results", str(results_file),
            "--min-overlap", "20",
        ])
        with pytest.raises(SystemExit) as exc_info:
            ck.main()
        assert exc_info.value.code == 1
