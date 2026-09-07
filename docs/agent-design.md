# Agent Design

## Two calls, not one: chat vs. investigation
`run_agent()` (Phase 17) is the tool-calling loop: gather evidence, produce a final text answer.
`run_investigation()` (Phase 18) reuses that exact loop for evidence-gathering, then makes a
**second, separate** call asking the model to compile the gathered evidence into strict JSON
matching `InvestigationReport`'s fixed shape. Forcing every response into that structured format
would be wasteful and awkward for a simple factual question -- investigation mode is opted into
explicitly (`POST /investigate`), not the default behavior of `POST /chat`.

## Tool-call budget, and what happens when it's hit
`config/agent.yaml`'s `max_tool_calls_per_turn` (6 for chat, 12 for investigation via the nested
`investigation.max_tool_calls` section) caps how many rounds of tool-calling the loop will run.
If the model still wants to call tools when the budget is exhausted, the loop stops and returns
what's been gathered so far with `budget_exceeded=True` -- rather than looping forever, and rather
than silently truncating without saying so. `tests/test_agent.py`'s
`test_run_agent_stops_at_tool_call_budget` verifies this doesn't infinite-loop.

## Config passthrough bug, caught by a test that specifically checked for it
`load_agent_config()` originally filtered the YAML file's contents down to only the flat keys in
`DEFAULT_CONFIG` -- which silently dropped the nested `investigation:` section
`run_investigation()` needs. This wasn't caught by casual inspection; it was caught by
`test_run_investigation_uses_investigation_budget`, which specifically asserts the budget passed
to `run_agent`'s `config_override` equals `config/agent.yaml`'s actual `investigation.max_tool_calls`
value (12), not the base value (6). A test that only checks "does it run without error" would have
passed with the bug still present -- this is why the test asserts the *specific* value, not just
success.

## config_override, not monkeypatching
The first draft of `run_investigation()` needed a different tool-call budget than a normal chat
turn, and initially did this by temporarily replacing `agent.load_agent_config` with a lambda,
then restoring it in a `finally` block. That's fragile (breaks under concurrent requests, awkward
to reason about, awkward to test) and was replaced before shipping with a proper
`config_override: dict | None` parameter on `run_agent()` itself. Worth naming as a real design
correction made during development, not just a clean design arrived at on the first attempt.

## Testability: the Anthropic client is always injectable
Every function that calls the Anthropic API (`run_agent`, `run_investigation`) takes `client=None`
and falls back to a real client only when none is provided. Every test in `tests/test_agent.py`
and `tests/test_investigation.py` passes a mocked client using `unittest.mock.MagicMock` with
`SimpleNamespace` response objects (not `MagicMock` all the way down -- `MagicMock().type` doesn't
behave like a real attribute, which would make `block.type == "tool_use"` checks pass or fail
unpredictably). This is what makes 60+ tests runnable with no API key and no network call.

## Structured-output parsing: defensive, not optimistic
`_parse_report()` in `investigation.py` handles three real failure modes, not just the happy path:
a response wrapped in markdown code fences despite being asked not to (`_strip_code_fences`),
genuinely invalid JSON, and valid JSON missing required keys. All three fall back to
`_fallback_report()`, which surfaces the agent's raw working answer as the executive summary and
says plainly in `limitations` that structured compilation failed -- a degraded-but-honest report,
not a crash and not a silently wrong one.

## What's NOT built yet
- **Conversational context now exists** (built in a later session) -- `src/agent/conversation_store.py`
  keeps simple text Q/A pairs per `conversation_id`, replayed as context on the next `/chat` call.
  Deliberately stores only text turns, not raw tool-call scaffolding: each new turn still gathers
  fresh tool evidence rather than trusting a prior turn's results are still current. In-memory only
  (not durable across restarts, not shared across instances) -- named as a real limitation, not
  hidden.
- **Guardrails/prompt-injection testing** -- the `agent_questions.json` evaluation set includes one
  adversarial question, but no automated pass/fail scoring of the *answer text* for injection
  resistance exists yet (Phase 20)
- **LangGraph** -- deliberately not used; the tool-loop is a plain Python `for` loop because it's
  simple enough not to need a framework at this scale (see FEATURES.md's Tier 3 note on this)
