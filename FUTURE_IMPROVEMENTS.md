# Future Improvements (Backlog)

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
