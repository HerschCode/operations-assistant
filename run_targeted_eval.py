"""Run only the 4 previously-failing questions to verify fixes."""
import os
from pathlib import Path

env_file = Path(__file__).parent / ".env"
for line in env_file.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

os.environ["AGENT_MODEL"] = "openai/gpt-oss-20b"

import time
from src.agent.agent import run_agent, load_agent_config
from src.evaluation.evaluate_agent import load_questions, _tool_selection_correct

questions = load_questions("data/evaluation/agent_questions.json")

# Previously-failing question indices (0-based): Q13, Q14, Q15, Q23
target_indices = [12, 13, 14, 22]
targets = [(i, questions[i]) for i in target_indices]

print(f"Running {len(targets)} targeted questions\n")
from groq import RateLimitError, APITimeoutError, APIConnectionError

for n, (idx, q) in enumerate(targets):
    if n > 0:
        print("  [sleeping 20s between questions...]")
        time.sleep(20)
    print(f"\n[Q{idx+1}] ({q['category']}) {q['question']}")
    print(f"  expected: {q['expected_tools']}")

    error_kind = "unknown"
    response = None
    for attempt in range(4):
        try:
            response = run_agent(q["question"])
            break
        except RateLimitError:
            error_kind = "rate limit"
            wait = 15 * (attempt + 1)
            print(f"  [rate limit] sleeping {wait}s...")
            time.sleep(wait)
        except (APITimeoutError, APIConnectionError) as exc:
            error_kind = "timeout/connection"
            wait = 10 * (attempt + 1)
            print(f"  [timeout] sleeping {wait}s...")
            time.sleep(wait)
    else:
        from src.agent.agent import AgentResponse
        response = AgentResponse(answer=f"[eval error: {error_kind}]")

    actual = response.tools_used
    correct = _tool_selection_correct(q["expected_tools"], actual, q["category"])
    status = "PASS" if correct else "FAIL"
    print(f"  actual:   {actual}")
    print(f"  -> {status}")
    if not correct:
        missing = set(q["expected_tools"]) - set(actual)
        print(f"     missing: {missing}")

print("\nDone.")
