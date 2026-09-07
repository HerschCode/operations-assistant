import pytest
from src.tools.validation import validate_case_id, validate_query, clamp_int


def test_validate_case_id_accepts_normal_ids():
    assert validate_case_id("C1023") == "C1023"
    assert validate_case_id("case_001-a") == "case_001-a"


def test_validate_case_id_rejects_empty():
    with pytest.raises(ValueError):
        validate_case_id("")


def test_validate_case_id_rejects_path_traversal_attempt():
    with pytest.raises(ValueError):
        validate_case_id("../../../admin")


def test_validate_case_id_rejects_embedded_slash():
    with pytest.raises(ValueError):
        validate_case_id("C1/other")


def test_validate_case_id_rejects_overly_long_input():
    with pytest.raises(ValueError):
        validate_case_id("C" * 51)


def test_validate_query_rejects_empty():
    with pytest.raises(ValueError):
        validate_query("")
    with pytest.raises(ValueError):
        validate_query("   ")


def test_validate_query_rejects_overly_long_input():
    with pytest.raises(ValueError):
        validate_query("x" * 501)


def test_validate_query_strips_whitespace():
    assert validate_query("  hello  ") == "hello"


def test_clamp_int_within_range_unchanged():
    assert clamp_int(10, max_value=50) == 10


def test_clamp_int_above_max_clamped_down():
    assert clamp_int(10000, max_value=50) == 50


def test_clamp_int_below_min_clamped_up():
    assert clamp_int(-5, max_value=50, min_value=1) == 1
