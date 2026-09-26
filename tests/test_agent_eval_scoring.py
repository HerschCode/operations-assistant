"""Regression: the v2 agent eval read tool arguments from a field that does not exist, so every argument check
scored False (arg_correctness_rate 0.0) regardless of what the agent did."""
from scripts.evaluate_agent_v2 import _score_arg_correctness
from src.agent.agent import ToolCallRecord


def test_arguments_are_read_from_the_agents_real_tool_call_records():
    rec = ToolCallRecord(name="get_cycle_time", input={"segment": "category"})
    calls = [{"name": rec.name, "args": getattr(rec, "input", None) or {}}]
    assert _score_arg_correctness({"get_cycle_time": {"segment": "category"}}, calls) is True
    assert _score_arg_correctness({"get_cycle_time": {"segment": "supplier_id"}}, calls) is False


def test_eval_script_does_not_read_a_nonexistent_args_attribute():
    src = open("scripts/evaluate_agent_v2.py", encoding="utf-8").read()
    assert 'getattr(tc, "input"' in src and "total_tokens" not in src.split("def main")[-1].split('"tokens"')[1][:80]
