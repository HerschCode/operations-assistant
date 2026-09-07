# Failure Modes

Phase 23: deliberately break things and confirm the system degrades honestly rather than crashing
or, worse, silently producing a wrong answer. `tests/test_failure_scenarios.py` is the artifact --
this doc explains what it covers and, just as importantly, what it doesn't.

## A real bug found by this phase
`client.py:get()` never caught `json.JSONDecodeError`. A 200 response with a malformed or
non-JSON body (a genuinely plausible failure -- a proxy error page returned with a 200, a
truncated response, a misconfigured endpoint) would crash with a raw `JSONDecodeError` bubbling
all the way up through the tool function and into the agent loop, instead of the clean
`OpsPerformanceUnavailable` every *other* failure mode already produced. Found while writing
`test_malformed_json_response_raises_ops_performance_unavailable_not_json_decode_error`, fixed by
catching `json.JSONDecodeError` explicitly in `get()`. The test now guards against this regressing.

## How this phase tests more of the real code path than earlier phases
`src/tools/client.py`'s `get()`/`get_text()` now accept an optional `httpx.Client` (same
injectable-client pattern `agent.py` already used for the Anthropic client). Failure tests build a
real `httpx.Client` backed by `httpx.MockTransport` -- request construction, status-code handling,
and response parsing all run for real; only the actual socket I/O is replaced by a handler
function that returns a canned response or raises a transport-level error. This is a meaningfully
different (and stronger) kind of test than `test_tools.py`'s normal-path tests, which mock this
module's own functions away entirely. It's still not the real network (Phase 24 is), but it's
closer than anything before this phase.

**This refactor required fixing three tests in `test_tools.py`** that had patched `httpx.get`
directly -- they broke the moment `client.py` stopped calling the bare module-level function.
Fixed by rewriting them against the same `MockTransport` pattern, which is a genuine improvement:
they now exercise more of the real request path than the mocks they replaced.

## Scenarios covered
- **HTTP-layer**: 500, 503, malformed JSON body, empty body, connection refused, timeout, a 404
  whose own error body isn't valid JSON either (the fallback-to-generic-detail path)
- **Tool-level**: confirms `OpsPerformanceUnavailable` propagates cleanly out of a tool function
- **Agent-level**: a single tool failing mid-turn, and *every* tool in a multi-tool turn failing --
  both confirm the agent loop still produces an honest final answer rather than an unhandled
  exception. `AgentResponse.tool_calls[i].error` is populated and the model's final text
  acknowledges the failure (in the test, because the mocked final response says so -- a real model
  would need the same behavior verified in Phase 24 against a live API)
- **Input boundary**: `POST /chat` and `POST /investigate` both reject a question over 2000
  characters and a missing `question` field with a 422, via Pydantic's existing
  `ChatRequest`/`InvestigateRequest` validation -- confirming that validation was actually wired
  up correctly, not just declared in the schema and never exercised

## What this phase does NOT cover (honestly)
- **Real concurrency / load testing** -- this environment has no way to spin up a real server and
  hit it with concurrent requests to observe actual behavior under load (connection pool exhaustion,
  race conditions in the (currently nonexistent) conversation-context handling, etc.). Everything
  here tests one request's failure handling in isolation, not the system's behavior under many
  simultaneous requests. This is a real gap, not a stylistic choice -- it needs an environment with
  a runnable server and a load-generation tool (locust, k6, even a simple asyncio script) to close.
- **The vector store being unavailable mid-request** -- `src/api/dependencies.py`'s
  `check_vector_store_reachable` exists for `/health`, but no test here simulates Chroma itself
  failing during an actual `search_policy_documents` call the way the HTTP failure tests do for
  `operations-performance`'s API. A reasonable next addition, following the same MockTransport-style
  pattern if Chroma's client supports comparable injection, or a simpler approach of monkeypatching
  `get_collection` to raise.
- **A real LLM's behavior under a real tool failure** -- the agent-level tests here use a *scripted*
  mocked response that already says the right thing ("I couldn't retrieve that data"). They prove
  the code *can* produce and return that answer correctly; they don't prove a real model, given a
  tool_result with `is_error: true`, will actually choose to respond that way rather than
  something less honest. That's a Phase 24 question, needing a real API key.
