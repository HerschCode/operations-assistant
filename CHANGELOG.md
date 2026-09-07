# Changelog

## v1.0.0 -- Freeze

All 24 originally-planned phases have been touched; 22 are genuinely complete (Phase 19 remains
an explicit first pass, documented as such in `docs/evaluation.md`). Phases 25-31 (audit,
correctness cross-checks, end-to-end scenarios, agent-report tooling, performance, and this
production-readiness review) are also complete. This is the frozen v1.0 baseline.

**To actually tag this once the repo is under real git version control:**
```
git init   # if not already
git add -A && git commit -m "v1.0.0 -- Operations Assistant"
git tag -a v1.0.0 -m "Frozen baseline after Phases 13-31"
```

### What's in v1.0.0
- Document ingestion: 4 real synthetic policy documents, section-aware chunking, local embeddings,
  Chroma vector store, safe re-indexing
- Retrieval with a real evaluation set and citation formatting
- 6 controlled, input-validated tools wrapping `operations-performance`'s API
- Agent orchestration with an enforced tool-call budget, full turn/round/tool-call logging
- Investigation mode with structured JSON-report compilation and defensive parsing (fenced JSON,
  invalid JSON, missing keys all handled without crashing)
- Structural (not just prompt-based) anti-injection defense, proven by regression test
- Dockerfile with a real health check, docker-compose bridging to the companion project
- 5 canonical end-to-end business scenarios traced through real code (2 with a real ephemeral
  vector store and real retrieval, not mocked)
- 103 tests, all passing

### Real bugs found and fixed during the build (kept for the record)
1. `load_all_documents()` was indexing `README.md` as if it were a policy document
2. A `get_management_report` draft awkwardly bypassed the client module instead of extending it
   properly -- caught and fixed before shipping, not after
3. `load_agent_config()` silently dropped the nested `investigation:` config section
4. A fragile monkeypatch-based config override was recognized as wrong and replaced with a
   proper `config_override` parameter before any test was written against the fragile version
5. `config/tools.yaml`'s `max_result_rows` was documentation-only for 4 phases -- never enforced
   in code until Phase 20
6. `client.py` never caught `json.JSONDecodeError` -- a malformed 200 response would crash
   instead of degrading to the same clean error every other failure mode produced
7. A `caplog`-vs-`capsys` testing lesson: `configure_logging()` clears handlers `caplog` relies on
8. Every documented script invocation was broken (`python scripts/index_documents.py` failed on
   import) -- the same class of bug the companion project's Phase 30 found, checked here and
   confirmed present, fixed the same way

### Known limitations, stated plainly
- No conversational context across `/chat` turns yet (`conversation_id` accepted, not yet used)
- No automated citation-correctness or groundedness scoring (Phase 19's explicit gap)
- No real load/concurrency testing (Phase 23's explicit gap -- this environment can't run a live
  server to test against)
- No cost/token tracking (needs a real API key to have real numbers to track)
- **Never run against a real `ANTHROPIC_API_KEY` or a live `operations-performance` instance** --
  every test mocks the LLM and/or the downstream API. This is the one thing every phase since 17
  has pointed at with increasing specificity, and it remains the single most important unknown
  in this entire project.
