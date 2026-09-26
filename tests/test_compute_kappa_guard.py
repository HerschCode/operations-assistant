"""compute_kappa must only accept human judgements of real answers."""
import json

import pytest

from scripts.compute_kappa import _load_human_scores


def test_compute_kappa_refuses_oracle_labels(tmp_path):
    f = tmp_path / "labels.json"
    f.write_text(json.dumps([{"id": "Q1", "human_score": 5, "note": "[oracle] expected"}]), encoding="utf-8")
    with pytest.raises(ValueError):
        _load_human_scores(f)
