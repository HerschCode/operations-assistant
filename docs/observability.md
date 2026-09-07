# Observability

## Reused directly from operations-performance
`src/observability/logging_config.py` is copied byte-for-byte from
`operations-performance/src/observability/logging_config.py` -- same JSON-structured logging,
same reasoning (machine-parseable log lines, not free text). `src/api/middleware.py` is the same
per-request logging + `X-Request-ID` pattern too. Shared infrastructure that has no reason to
differ between the two projects stays identical rather than being reimplemented with subtle
differences -- confirmed identical with a `diff` when copied, not just visually similar.

## What's genuinely different here: agent-loop instrumentation
A pipeline run (`operations-performance`) is one linear sequence of stages. An agent turn
(`run_agent` in this project) is a variable number of rounds, each potentially calling multiple
tools, and the actual latency users experience is dominated by LLM API calls and tool calls, not
by data processing. `src/agent/agent.py` is instrumented accordingly:

- **`turn_id`** (not `request_id` -- deliberately distinct from the HTTP middleware's ID, since
  one HTTP request maps to exactly one agent turn but the turn itself has internal structure worth
  tracking under its own identifier) ties together every log line for one question: turn start,
  each model round, each tool call, turn completion.
- **Per-round logging** (`"model round complete"`) captures which tools the model requested and
  how long that specific API call took -- this is what would let you answer "was this turn slow
  because of the LLM or because of a slow tool" after the fact.
- **Per-tool-call logging** (`"tool call succeeded"` / `"tool call failed"`) captures duration and,
  on failure, the actual error -- this is what makes `docs/tool-design.md`'s
  `OpsPerformanceUnavailable` exception visible in logs, not just handled silently in code.
- **Turn completion** logs total duration, round count, tools used, and whether the tool-call
  budget was hit (`budget_exceeded`) -- the three things you'd actually want to know when looking
  back at "why did this investigation take so long" or "why did this answer seem incomplete."

## A real testing lesson from writing this (not a code bug)
The first version of the logging test used pytest's `caplog` fixture, which failed -- not because
the logging code was wrong (the JSON lines were visibly correct in the test's captured stdout
output), but because `configure_logging()` clears the root logger's handlers and installs its own
`StreamHandler` writing to stdout, which also strips out whatever handler `caplog` had attached.
Fixed by testing against `capsys` (raw stdout capture) instead, parsing the JSON lines directly --
which is actually the more honest test anyway, since raw stdout is exactly where these logs land
in production (a container's stdout, picked up by whatever log aggregator sits in front of it).

## What's NOT done
- No log aggregation/shipping configured (Cloud Logging, CloudWatch, etc.) -- these are structured
  JSON lines to stdout, ready to be picked up by whatever the deployment environment provides, not
  wired to a specific destination
- No metrics/tracing beyond structured logs (no OpenTelemetry, no dashboards) -- Tier 3 territory
  for a project this size
- No persisted turn history the way `operations-performance`'s `pipeline_runs` table persists
  pipeline run history -- an equivalent `agent_turns` table (or reusing the same Postgres instance)
  would be a reasonable next addition if this needed to answer "how has agent latency trended over
  the last week" rather than just "what happened in this one turn, right now"
