# Future Improvements (Backlog)

## Post-v1.0: hybrid retrieval (keyword + semantic) with RRF reranking
`src/retrieval/search.py::hybrid_search()`, real BM25 + semantic merged via
Reciprocal Rank Fusion, wired into the actual `search_policy_documents` tool
(not left unused). See PLAN.md.

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

## Post-v1.0: real load test against the live deployment
`scripts/load_test_live.py` closes this for real -- genuinely concurrent OS threads
against the actual Render URL, not TestClient. No race condition found in the rate
limiter under real concurrent load; a real different bug (provider rate-limit
surfaced as an indistinguishable 502) was found and fixed instead. See PLAN.md.

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
- MCP-style tool exposure, if there's ever a genuine multi-client reason to need it
- LangGraph, only if a genuinely more complex multi-step workflow outgrows the current
  plain-loop agent -- not adopted speculatively
- Real API usage tracking (the cost estimator exists and works; it has no real spend numbers to
  report yet since no live model has been called)

## Post-v1.0: persisted conversation storage
`src/agent/conversation_store.py` swapped from an in-memory dict to SQLite
(`data/conversations.db`) -- survives an ordinary process restart now, not just a
Render redeploy (that free tier's disk is still ephemeral across redeploys, same
limitation as the Chroma index). Public API unchanged, so all 5 existing tests pass
against the new backend with zero changes; added a 6th proving the data actually
lands on disk (read back through a separate raw sqlite3 connection, not this
module's own functions) rather than just living in a Python object a real restart
would drop.
