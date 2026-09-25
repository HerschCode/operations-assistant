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

## OpenTelemetry tracing (Phase 3)

`src/observability/spans.py` adds optional OTLP tracing spans to the key pipeline operations:

| Span | Attributes |
|---|---|
| `retrieve` | `question`, `top_k`, `method`, `n_chunks_returned` |
| `faithfulness_gate` | `faithfulness_score`, `gated`, `threshold` |
| `tool:<name>` | `tool.name`, `tool.arg.*` per argument |
| `llm_call` | `model`, `prompt_tokens`, `completion_tokens`, `cost_usd` |
| `agent` | `question`, `provider`, `model` |

All spans degrade to no-ops when `OTEL_EXPORTER=none` (the default) or when `opentelemetry-sdk`
is not installed — the wrapped code runs unchanged, no import errors.

**Backend selection** via `OTEL_EXPORTER` env var:

```bash
# Local Arize Phoenix (no Docker, no cloud)
pip install arize-phoenix openinference-instrumentation opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc
python -m phoenix.server.main serve   # starts on :4317
OTEL_EXPORTER=phoenix python -m src.api.main

# OTLP/gRPC to any collector
OTEL_EXPORTER=otlp OTEL_EXPORTER_OTLP_ENDPOINT=http://otelcol:4317 python -m src.api.main

# Console (development / smoke-test)
OTEL_EXPORTER=console python -m src.api.main
```

## Semantic cache (Fix 6)

`src/cache/semantic_cache.py` adds an embedding-similarity cache for `/chat` responses.
Activate with `SEMANTIC_CACHE=1`. Repeated or near-identical questions (cosine ≥ 0.97
by default) return the stored answer without an LLM call.

Key properties:
- **SQLite persistence** via `data/cache.db` (configurable via `SEMANTIC_CACHE_DB`)
- **Lazy encoder load** — sentence-transformers loads on first use, not at startup
- **TTL** — entries expire after `SEMANTIC_CACHE_TTL_SECONDS` (default 3600)
- **Graceful degradation** — `get()` and `put()` are no-ops on any error; a broken
  cache never blocks a request
- **Not applied to `/chat/hitl`** — HITL turns have stateful side-effects
  (`propose_intervention`) that cannot safely be returned from a cache

## Production upgrade paths

### Postgres instead of SQLite
Conversation summaries (`src/agent/conversation_store.py`) and the semantic cache
(`src/cache/semantic_cache.py`) both use SQLite with path-based connection strings.
Replacing both with Postgres is a single-line change in each: swap `sqlite3.connect(path)`
for `psycopg2.connect(dsn)` (or use SQLAlchemy's `create_engine()` so either backend
works from a `DATABASE_URL` env var without code changes). The same `conversations.db`
tables can be created verbatim in Postgres; the cache's `semantic_cache` table is
identical. Neon serverless Postgres would be the natural host alongside Render (same
zero-infrastructure principle as the current deployment).

### Postgres for agent_turns history
There is no `agent_turns` table yet — a reasonable next addition for production monitoring.
The schema would mirror `operations-performance`'s `pipeline_runs` table: `turn_id`,
`question`, `tools_used`, `latency_ms`, `model`, `cost_usd`, `timestamp`. The per-request
logs already emit this information; persisting it in Postgres would enable "how has agent
latency trended over the last week" queries without log parsing.

### Arize Phoenix tracing
`src/observability/spans.py` already emits OTLP spans. Set `OTEL_EXPORTER=phoenix` and
install the optional extras to stream traces to a local Phoenix server:

```bash
pip install arize-phoenix openinference-instrumentation opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc
python -m phoenix.server.main serve   # UI at http://localhost:6006, OTLP on :4317
OTEL_EXPORTER=phoenix python -m src.api.main
```

What you see in Phoenix: one trace per agent turn, with child spans for
`retrieve`, `faithfulness_gate`, `tool:<name>`, and `llm_call`. The `agent` span
wraps them all with `question`, `provider`, and `model` attributes. Cost and token
counts are attached to each `llm_call` span, making per-question cost visible in the
UI without any additional instrumentation.

## What's NOT done
- No log aggregation/shipping configured (Cloud Logging, CloudWatch, etc.) -- these are structured
  JSON lines to stdout, ready to be picked up by whatever the deployment environment provides, not
  wired to a specific destination
- No persisted `agent_turns` table -- see "Postgres for agent_turns history" above for the
  design; deferred because the current per-request logs answer "what happened in this turn"
  well enough for a portfolio deployment that doesn't see production traffic volumes
