# Tool Design

## Core principle: controlled functions, never raw access
The agent never gets a database connection or the ability to construct arbitrary API calls. Every
capability is a named function in `src/tools/`, registered in `src/tools/registry.py`, with a
fixed JSON schema describing exactly what inputs it accepts. This is the same principle
`operations-performance` follows internally (its API never exposes raw SQL) applied one layer up:
the agent is a second, further-restricted consumer of that already-restricted API.

## Why a shared client (`src/tools/client.py`) instead of each tool rolling its own httpx call
One place to control timeout (`REQUEST_TIMEOUT_SECONDS`, matching `config/tools.yaml`), one place
to handle errors consistently, one exception type (`OpsPerformanceUnavailable`) the agent layer
(Phase 17) can catch uniformly rather than needing to know each tool's specific failure mode.

## Why `get()` and `get_text()` are separate functions
Most `operations-performance` endpoints return JSON; `GET /reports/management` returns markdown
text. Rather than overloading one function with a response-type flag, they're two functions with
genuinely different parsing and error-handling needs. This was a real decision made while writing
`analytics.py:get_management_report` -- the first draft awkwardly reached into `client.py`'s
internals with a local import rather than doing this properly; caught and fixed before shipping,
not after.

## Result-row caps
`config/tools.yaml`'s `max_result_rows` per tool exists so a single tool call can't flood the
agent's context with an unbounded result set -- `get_bottlenecks` defaults to `top_n=10`,
`get_supplier_performance` requires a minimum order volume before a supplier even appears. These
caps live in the tool functions themselves (as parameter defaults) as well as being documented in
config, so the behavior is enforced in code, not just described in a YAML file nobody reads at
runtime.

## Why `database.py` holds pipeline-health tools, not business data
The original stub layout named this file `database.py`, implying it should hold generic DB access
-- which would have duplicated `analytics.py`'s job. Repurposed to hold operational/pipeline-health
tools (`get_pipeline_status`, wrapping `operations-performance`'s `GET /observability/pipeline-runs`)
instead: a genuinely distinct concern (is the underlying data fresh, separate from what the data
says) that the agent can check before trusting a metric in an investigation. Named honestly in the
file's own docstring rather than silently repurposed without explanation.

## Why `process.py` is empty
It would wrap process-mining-specific tools (conformance rate, variant comparison, rework
statistics) the same way `analytics.py` wraps cycle time/bottlenecks/SLA -- but
`operations-performance` doesn't expose those as API endpoints yet (see that project's `PLAN.md`,
"What's left"). Wiring a tool to an endpoint that doesn't exist would produce a tool that always
fails, which is worse than an honestly empty file with a docstring explaining why. Fill this in
once the companion project adds those endpoints.

## The RAG tool is not like the others
`search_policy_documents` (`src/tools/documents.py`) calls `src/retrieval/search.py` directly, not
`operations-performance`'s API -- it's the one tool answering from this project's own document
corpus. Its schema description explicitly tells the agent when to use it vs. the analytics tools
("for questions about rules/procedures, not operational numbers") since tool *selection* is a real
part of what Phase 17 needs to get right, and a clear description is the cheapest lever for that.

## Testing approach
Every tool test mocks the underlying `httpx`/`semantic_search` call rather than hitting a real
`operations-performance` instance or the real vector store -- same principle as
`operations-performance`'s own `test_api.py`. This keeps the suite fast and runnable without both
projects' full stacks running simultaneously, at the cost of not testing the real network path
end-to-end (that's what actually running both services together, Phase 24, is for).
