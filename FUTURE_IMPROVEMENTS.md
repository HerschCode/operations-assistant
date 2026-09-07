# Future Improvements (Backlog)

## Post-v1.0 build session -- multi-provider agent
`src/agent/agent.py::run_agent` is now a thin dispatcher over `config["provider"]`
(anthropic/groq/gemini), so the portfolio deployment can run on Groq or Gemini's free
tiers instead of requiring a paid Anthropic key. `src/agent/providers.py` holds the
Groq (OpenAI-compatible `chat.completions`) and Gemini (`google-genai`, the current SDK
-- `google-generativeai` is end-of-life and warns on import, caught while wiring this up
and avoided) implementations, each converting the same `TOOL_SCHEMAS` (Anthropic's
`input_schema` shape) into their own tool-schema format rather than maintaining three
parallel tool definitions. `config/agent.yaml`'s committed default is deliberately left
as `provider: anthropic` -- changing it would have broken every existing Anthropic-shaped
fake-client test in `tests/test_agent.py`, since `load_agent_config` reads that same
file. Provider selection for a real deployment goes through `AGENT_PROVIDER`/
`AGENT_MODEL` env vars instead, checked at the end of `load_agent_config`, so the
checked-in test fixture never has to change. 14 new tests in `tests/test_providers.py`
(schema conversion, both providers' tool-loop/budget/error-handling paths, and dispatcher
routing), all passing alongside the existing 139. Anthropic's own integration is
untouched, not removed -- switching back is a one-line env var away.

## Done since v1.0 (moved out of this list)
Conversational context, citation-correctness/groundedness scoring, a real cost estimator,
`POST /documents`, process-conformance tooling, an in-process concurrency test, and API-key
authentication were all originally listed here and have since been built -- see `PLAN.md` for
details on each. A full CI workflow now exists (this project never had one at all until it was
built alongside `operations-performance`'s `pip-audit` addition -- a real gap, not just a missing
feature) -- test/lint/security jobs, matching the companion project's pattern.

## A real CI finding, tracked here until resolved
`pip-audit`'s first run found 4 real CVEs in `chromadb` (no fix version available yet) --
investigated and found to be in ChromaDB's HTTP server API, which this project never runs
(`vector_store.py` uses `PersistentClient` exclusively, no `HttpClient`, no standalone server).
Full writeup in `docs/security-notes.md`. Re-check this once a fix version ships, and definitely
re-check it if this project's Chroma usage ever changes to a networked server deployment.

## Still open
- Real load/concurrency testing against a live deployed server (the in-process `TestClient`
  version now exists and is genuinely useful, but it's not the same as measuring latency under
  real network load with a tool like locust/k6 against a running instance)
- Automated adversarial-answer scoring, not just tool-selection scoring, for the
  unanswerable/adversarial evaluation categories
- Hybrid retrieval (keyword + semantic) and reranking (Tier 3 in FEATURES.md)
- MCP-style tool exposure, if there's ever a genuine multi-client reason to need it
- LangGraph, only if a genuinely more complex multi-step workflow outgrows the current
  plain-loop agent -- not adopted speculatively
- Real API usage tracking (the cost estimator exists and works; it has no real spend numbers to
  report yet since no live model has been called)
- Persisted (not in-memory) conversation storage if this ever needs to survive a restart or run
  across multiple instances
- Per-client API keys, key rotation, and scopes/roles (single shared API key exists now)
