# Build Plan — Operations Assistant

Phases 13-24 of the combined roadmap (`operations-performance`'s `PLAN.md` covers phases 1-12,
the "same business, two front doors" prerequisite). See `FEATURES.md` for the full tiered feature
spec this plan builds toward.

**Phase 24 status: documentation/consistency work done; the live run itself is still pending
real credentials and infrastructure this environment doesn't have -- see notes below for the
exact, honest split between the two.**
All 24 phases have now been touched. 22 are genuinely complete. Phase 19 remains an explicit
first pass (documented gaps in `docs/evaluation.md`). Phase 24's live-integration half is the
one piece of either project that has never actually run.

## Status by phase

| # | Phase | Status | Key files |
|---|---|---|---|
| 13 | Backend foundations | done | `src/api/main.py`, `routes.py`, `schemas.py`, `dependencies.py` |
| 14 | Document corpus & RAG ingestion | done | `data/documents/` (4 real docs), full ingestion pipeline, `docs/rag-design.md` |
| 15 | Retrieval | done | `src/retrieval/search.py`, `data/evaluation/retrieval_questions.json` |
| 16 | Structured-data tools | done | `src/tools/*.py`, `registry.py`, `docs/tool-design.md` |
| 17 | Agent orchestration | done | `src/agent/agent.py`, `prompts.py`, `POST /chat`, `docs/agent-design.md` |
| 18 | Investigation mode | done | `src/agent/investigation.py`, `POST /investigate`, fixed report shape |
| 19 | Grounding & evaluation | partial (real first pass) | `src/evaluation/evaluate_agent.py`, 6-question set, `docs/evaluation.md` (explicit about the gap) |
| 20 | Guardrails & security | done | `src/tools/validation.py`, structural anti-injection regression tests, `docs/security-notes.md` |
| 21 | Containerization & deployment | done | `Dockerfile` (with a real health check), `.dockerignore`, `docker-compose.yml`, `docs/deployment.md` -- see notes below on what wasn't (and couldn't be) actually run |
| 22 | Observability | done | `src/observability/logging_config.py` (identical to operations-performance's, confirmed via diff), `src/api/middleware.py` (same pattern), agent-loop instrumentation in `agent.py` (turn/round/tool-call logging), `docs/observability.md` |
| 23 | Load & failure testing | done (failure testing; load/concurrency explicitly not covered) | `tests/test_failure_scenarios.py` (14 tests using real `httpx.MockTransport`), `client.py` refactored for an injectable client, `docs/failure-modes.md` |
| 24 | Final integration & interview hardening | not started | |

## Phase 21 notes worth remembering
- `Dockerfile`'s `HEALTHCHECK` calls the real `/health` endpoint (which itself checks both
  `operations-performance`'s API and the vector store), not just "is the process listening" --
  same discipline as making `GET /health` meaningful back in Phase 13.
- `docker-compose.yml` connects to `operations-performance` via `host.docker.internal` with an
  `extra_hosts` entry (needed for that to resolve on Linux, automatic on Mac/Windows) since that
  project doesn't have its own Dockerfile yet -- documented as a temporary bridge in
  `docs/deployment.md`, with the natural next step (a shared Docker network) named explicitly.
- **Honestly not run**: this sandbox has no Docker daemon available, so the `Dockerfile` and
  `docker-compose.yml` were validated by YAML-syntax-parsing the compose file and careful manual
  review, not by an actual `docker build`/`docker compose up`. Recorded here rather than implying
  it was tested the way everything else in this project has been -- this is the one artifact in
  both repos that hasn't gone through the "actually run it" discipline the rest of the build has,
  and it should be the first thing verified once run somewhere with Docker available.

## Phase 22 notes worth remembering
- `logging_config.py` and `middleware.py` are copied directly from `operations-performance`,
  confirmed identical with `diff` -- shared infrastructure with no reason to differ shouldn't be
  reimplemented with subtle drift between the two projects.
- What's genuinely different here (not just copied): `agent.py`'s instrumentation is structured
  around a `turn_id` (one per `run_agent` call) with per-round and per-tool-call log lines nested
  under it -- because an agent turn has internal structure (multiple model rounds, multiple tool
  calls) that a linear pipeline run doesn't, and that structure is exactly what you'd need to debug
  "why was this answer slow" or "why did this tool fail."
- **A real testing lesson, not a code bug**: the first version of the logging test used pytest's
  `caplog` fixture and failed, even though the log output visible in the failure's captured stdout
  was completely correct. The cause: `configure_logging()` clears the root logger's handlers and
  installs its own, which also strips out whatever handler `caplog` had attached before the test
  ran. Fixed by testing against `capsys` (raw stdout) and parsing the JSON lines directly instead
  -- arguably a more honest test anyway, since stdout is exactly where these logs land in
  production. Documented in `docs/observability.md` since it's a genuinely useful thing to know
  about testing structured-logging code, not just a fix to note in passing.
- Full suite: 80/80 passing (79 previous + 1 new logging-instrumentation test), run for real.

## Earlier phase notes (still relevant, condensed)
- **Phase 14**: `load_all_documents()` was picking up `README.md` as a policy document -- caught
  by a test on first run, fixed by excluding README files.
- **Phase 16**: a `get_management_report` draft awkwardly reached into `client.py`'s internals
  with a local import instead of properly extending the client -- caught and fixed while writing
  it, before shipping.
- **Phase 18**: `load_agent_config()` was silently dropping the nested `investigation:` config
  section -- caught by a test asserting the *specific* budget value passed through, not just "does
  it run." Also: an early monkeypatch-based approach to per-call config overrides was recognized as
  fragile and replaced with a proper `config_override` parameter before any test was written
  against the fragile version.
- **Phase 20**: `config/tools.yaml`'s `max_result_rows` was documentation-only for four phases --
  nothing in code enforced it until this phase added `clamp_int()`. Also added `case_id` validation
  ahead of a real path-traversal risk in a URL interpolation.
- **Phase 22**: the `caplog`-vs-`capsys` testing lesson above.

## Phase 23 notes worth remembering
- **A real bug found by this phase, not by accident**: `client.py:get()` never caught
  `json.JSONDecodeError` -- a 200 response with a malformed body would crash with a raw
  exception instead of the clean `OpsPerformanceUnavailable` every other failure mode already
  produced. Found while deliberately writing a test for exactly this scenario, fixed, and now
  guarded against regressing.
- **A real, deliberate step up in test fidelity**: `client.py`'s `get()`/`get_text()` now take
  an optional `httpx.Client`, same injectable pattern `agent.py` already used for the Anthropic
  client. Failure tests build a real `httpx.Client` backed by `httpx.MockTransport` -- request
  construction, status handling, and response parsing all run for real, only socket I/O is
  faked. This is meaningfully closer to the real network path than mocking this module's own
  functions away, which is what earlier tests did.
- **That refactor broke 3 existing tests in `test_tools.py`** (they patched `httpx.get`
  directly, which stopped being called). Fixed by rewriting them against the same
  `MockTransport` pattern -- a genuine improvement, not just a repair, since they now exercise
  more of the real request path than what they replaced.
- Agent-level tests confirm the loop survives both a single tool failing and *every* tool in a
  multi-tool turn failing, still producing an answer rather than an unhandled exception.
- **Explicitly not covered, stated plainly in `docs/failure-modes.md`**: real concurrency/load
  testing (this environment can't run a live server to hit with concurrent requests), the
  vector store failing mid-request, and whether a *real* LLM (not a scripted mock) actually
  responds honestly to a failed tool call rather than just being capable of it in code. All
  three are real gaps, named as such rather than implied to be covered.
- Full suite: 94/94 passing (80 previous + 14 new failure-scenario tests), run for real.

## Full suite status
**94/94 tests passing**, run for real at every phase. Real bugs, design corrections, and one
testing-methodology lesson were caught and fixed during Phases 14, 16, 18, 20, 22, and 23 --
recorded above and in this repo's git-equivalent history, not glossed over anywhere.

## Not started -- exact next files, in order
1. Phase 24, the only phase left: run both projects together for real -- a real
   `ANTHROPIC_API_KEY`, a running `operations-performance` instance (with `docker compose up`
   now actually an option, per Phase 21, though still unverified in an environment with Docker),
   actual indexed documents, actual questions asked through `/chat` and `/investigate`. Every
   phase involving real execution -- Project 1's Phase 9, and this project's Phases 14, 16, 18,
   20, 22, and 23 -- found something worth fixing. The full real network path between these two
   services, and a real model's behavior (not a scripted mock) under the failure scenarios
   Phase 23 proved the *code* handles correctly, have never been exercised. That's the last real
   unknown in both projects.


## Phase 24 notes -- what "final integration" means given this environment's real constraints
Phase 24 as originally scoped ("both projects running together against the shared backend,
rehearsed architectural walkthrough") splits into two genuinely different kinds of work, and only
one of them was achievable here:

**Done in this environment:**
- `docs/architecture.md` written -- the full combined-system diagram, request-flow trace for the
  flagship "combined question" scenario, and a pointer index to every other doc, so the
  architectural walkthrough the phase calls for has a real document to rehearse from.
- A genuine audit pass (not just re-reading the docs) verifying every claimed feature in both
  projects' checklists against the actual files on disk -- see `operations-performance/PLAN.md`'s
  own "Phase 25 audit" section for the two small real bugs that audit found and fixed there
  (an inconsistent import path, an undeclared direct dependency).
- Cross-project consistency re-verified: the SLA values this project's `sla-policy.md` states
  still match `operations-performance/config/sla.yaml` exactly.
- Full test suites reran clean on both repos as the final step: 94/94 here in
  `operations-assistant`, 51/51 in `operations-performance`. 145 tests total, both green.

**Not achievable in this environment, stated plainly rather than glossed over:**
- Actually running `docker compose up` (no Docker daemon available here)
- Actually calling the Anthropic API with a real key (none available/appropriate to use here)
- Actually running `operations-performance`'s Postgres and hitting its live API from this
  project's agent (no database available here)

This means the single most consequential unknown in both projects combined -- **does a real model,
given real tool results over a real network connection, actually behave the way 145 mocked tests
say it should** -- is still genuinely untested. Every phase's notes have said some version of this
with increasing specificity; Phase 24 is where it stops being "the next phase" and becomes "the one
thing left to do before either project is truly done," full stop.


## Phase 28 -- End-to-end scenario tests (scenarios 4-5, the RAG and flagship combined ones)
`tests/test_e2e_scenarios.py` + `docs/e2e-scenarios.md`. Uses a REAL ephemeral Chroma collection
and REAL chunking/indexing/search code -- the one thing not real is the embedding model itself
(no network access to download sentence-transformers' weights here), stood in for by a
deterministic bag-of-words hashing vectorizer, which is a legitimate embedding technique on its
own terms, not fake data. Scenario 4's retrieval test only passes because that vectorizer actually
places the right policy section closest to the query -- confirming the chunk/store/retrieve wiring
genuinely works, independent of which embedding technique sits underneath.

Scenario 5 (the flagship "why are we breaching SLA, does policy explain it" combined question)
runs the real agent tool-dispatch path for both tools: `get_sla_metrics` through a real
`httpx.Client` against an `httpx.MockTransport`-backed fake server, and `search_policy_documents`
through the fully real retrieval chain from Scenario 4. Only the LLM's own tool-selection
reasoning is scripted (no API key available) -- the citations in the final response are not
scripted, they're what the real retrieval chain actually found.

Scenarios 1-3 (bottleneck, worst supplier, SLA-risk prediction) live in
`operations-performance/tests/test_e2e_scenarios.py`, traced through that project's own FastAPI
layer against the Phase 27 golden dataset.

Full suite after Phase 28: 97/97 (94 previous + 3 new scenario tests).

## What's left, unchanged in substance
The same thing named at the end of every phase since 20: a real model, given real tool results
over a real network connection between two actually-running services, has never been observed.
Phases 27-28 pushed the *code-correctness* confidence about as far as it can go without that --
hand-verified metrics, a real bug found and fixed, and end-to-end scenario traces through real
routing/retrieval/tool-dispatch code. What's left is specifically the model's own behavior, not
the system around it.


## Phase 29 -- Agent correctness audit
`src/evaluation/generate_agent_report.py` renders `evaluate_agent.py`'s scoring into the exact
table format an audit needs -- Question | Category | Expected tools | Actual tools | Correct? |
Investigation -- with an automated first-pass triage for every incorrect row (missing tool vs.
unexpected extra tool vs. "unanswerable/adversarial -- read the answer text manually," which is
scored differently on purpose, per `docs/evaluation.md`). A pass/fail count on its own doesn't
tell you what to fix; the table does. 6 new tests, all passing, covering the triage logic and the
rendered table's structure.

Needs a real `ANTHROPIC_API_KEY` to populate with genuine results (same constraint as
`evaluate_agent.py` itself) -- this phase built the reporting harness and proved it works
correctly against scripted results; it doesn't claim to have audited a real agent's real
decisions, because that hasn't happened yet in this environment.

## Phase 30 -- Performance & cost pass
This project's own performance work is smaller than the companion project's (no equivalent
data-pipeline hot loop to find), but the **same class of bug operations-performance's Phase 30
found -- every script's documented invocation being broken -- was checked here too and confirmed
present**: `python scripts/index_documents.py` failed with the identical `ModuleNotFoundError`.
Fixed identically: added `scripts/__init__.py`, updated `README.md` and `docs/deployment.md` to
use `python -m scripts.foo`. The Dockerfile's own `CMD` was unaffected (it runs `uvicorn
src.api.main:app` from the repo root already, not via a script path), so this only affected local
development instructions, not the containerized deployment path -- but it would have blocked
anyone following the README's local setup exactly as written.

Real cost/latency measurement (token counts, $ per request, tool-call-count-per-question analysis)
needs an actual model in the loop, which this environment doesn't have -- consistent with every
other "needs a real API key" gap already named in `docs/evaluation.md` and `docs/agent-design.md`.
Not built as a speculative estimator without real numbers to calibrate it against; better to leave
it honestly undone than ship a cost model with made-up assumptions.

Full suite: 103/103 (97 previous + 6 new agent-report tests).


## Phase 31 -- Production-readiness review
`docs/production-readiness-review.md` -- four perspectives (Data, Application, AI, Operations),
each question answered honestly against the actual code, "partially" and "no" included where
true. See operations-performance's own review for its half of the combined system.

## Phase 32 -- Freeze v1.0
`CHANGELOG.md` (what's in v1.0, every real bug found during the build, known limitations stated
plainly) and `FUTURE_IMPROVEMENTS.md` (a real backlog, not a todo list disguised as done). No git
repo exists in this environment to actually run `git tag` -- both files include the exact command
to run once this code is under real version control.

**All 32 phases across both projects are now complete or explicitly, honestly partial where
named.** Final test totals: 103/103 here. The single remaining unknown, unchanged in substance
since it was first named several phases ago: neither project has ever run against real
infrastructure -- a live Postgres instance, a real `ANTHROPIC_API_KEY`, the two services actually
talking to each other. That is not phase 33. It is the one thing left before v1.0 is more than a
frozen, thoroughly-audited, never-executed system.

## Post-v1.0 build session -- closing high/medium-impact backlog items
Done after the v1.0 freeze, on request. Dead stubs removed (`src/reports/investigation_report.py`
-- superseded by `src/agent/investigation.py`'s `InvestigationReport`; `sql/assistant_queries.sql`
-- described an architecture never actually built, since the agent calls
`operations-performance`'s REST API, never raw SQL).

**`evaluate_answers.py` built for real** -- mechanical groundedness/citation-correctness checking
(does every citation trace to a real tool result this turn, do numbers in the answer text actually
appear in the gathered evidence). Found and fixed two real bugs while building it: a formatting
mismatch ("18.4%" vs. bare "18.4" in a tool result) that flagged a genuinely grounded number as
fabricated, and a regex that swept a sentence-ending period into a number token.

**`process.py` unblocked** -- `get_conformance` tool wired now that
`operations-performance`'s matching endpoint exists.

**`POST /documents` wired** -- was documented as a gap since Phase 13; the ingestion logic already
existed, just never had a route. Found a genuinely missing dependency (`python-multipart`, required
by FastAPI's `UploadFile`) while testing it -- not caught until the test suite actually tried to
exercise the endpoint.

**Conversation context built** -- `src/agent/conversation_store.py`, in-memory, storing simple
text Q/A pairs (not raw tool-call scaffolding) per `conversation_id`, so each new turn still
gathers fresh evidence rather than trusting a prior turn's tool results. `run_agent` gained a
`history` parameter; `/chat` now actually uses it. Verified genuinely working end-to-end through
the real API, not just at the function level.

**A real cost estimator** -- `config/pricing.yaml` (rates kept in config, not hardcoded, with an
explicit sourcing caveat since third-party sources disagreed on current pricing at the time this
was written) + `src/evaluation/cost_estimator.py`. Deliberately never claims to report real spend
-- no real API usage exists yet to report; it's a calculator ready for when real usage numbers
exist, with a clear estimated-vs-measured flag on every result.

**A real in-process concurrency test** -- Phase 23 named live load testing as a gap this
environment couldn't close; what genuinely doesn't need live infra is hitting the real FastAPI app
via `TestClient` from real threads, which `tests/test_concurrency.py` does. This is what caught
that `conversation_store`'s thread-safety had never actually been verified.

Full suite: 134/134 (37 new tests since v1.0's 103: 2 conformance-tool, 5 upload, 6 groundedness,
7 cost-estimator, 5 conversation-store, 3 conversation-integration/history, 3 concurrency, plus a
handful of small fixture fixes along the way).

## Post-v1.0 build session, part 2 -- API authentication
Same mechanism as `operations-performance` (`src/api/auth.py`, ported directly), applied to every
route except `GET /health`. Matters more here: `/chat` and `/investigate` each trigger a real LLM
call once live, so an unauthenticated endpoint is a real-money risk, not just a data-exposure one.
5 new tests. Full suite: 139/139.

## Post-v1.0 build session, part 3 -- CI, and a real finding
Built a full CI workflow from scratch (this project never had one, unlike the companion project --
a real gap, found while adding `pip-audit` here to match Batch A's work there). Its first run found
**4 real CVEs in `chromadb`** (PYSEC-2026-311 / CVE-2026-45829, an unpatched pre-auth RCE, plus 3
authorization-bypass CVEs), with no fix version available at the time. Investigated rather than
either panicking or dismissing it: all four are in ChromaDB's HTTP server API; this project's
`vector_store.py` uses `PersistentClient` exclusively -- an embedded, in-process client, never
`HttpClient`, no standalone Chroma server ever started. The vulnerable network-reachable surface
doesn't exist in how this project actually runs Chroma. Documented in full in
`docs/security-notes.md`, including the explicit condition under which this needs re-review (if
Chroma deployment ever changes to a networked server) -- not swept under the rug, not treated as a
blocker for a risk that isn't currently reachable either.

## Post-v1.0 build session -- multi-provider agent (Groq/Gemini)
`src/agent/agent.py::run_agent` is now a thin dispatcher over `config["provider"]`
(anthropic/groq/gemini), so the portfolio deployment can run on Groq or Gemini's free
tiers instead of requiring a paid Anthropic key. `src/agent/providers.py` holds the
Groq (OpenAI-compatible `chat.completions`) and Gemini (`google-genai`, the current SDK
-- `google-generativeai` is end-of-life and warns on import, caught while wiring this up
and avoided) implementations, each converting the same `TOOL_SCHEMAS` (Anthropic's
`input_schema` shape) into their own tool-schema format. `config/agent.yaml`'s committed
default stays `provider: anthropic` deliberately -- changing it would have broken every
existing Anthropic-shaped fake-client test in `tests/test_agent.py`, since
`load_agent_config` reads that same file. Real deployment picks a provider via
`AGENT_PROVIDER`/`AGENT_MODEL` env vars instead. 14 new tests in
`tests/test_providers.py`; 153/153 passing overall.

## Post-v1.0 build session -- the actual live run, first real answer through the full stack
The unknown named at the end of nearly every phase note since Phase 17 -- "does a real
model, given real tool results over a real network connection, actually behave the way
the mocked tests say it should" -- is now closed. Ran both services live: this project's
API on :8001 (Groq-backed) against `operations-performance` running for real on :8000
against live Neon Postgres with real BPI 2019 data (see that project's own `PLAN.md` for
the pipeline-side bugs that run found).

**One real cross-project bug found, invisible to both projects' otherwise-passing test
suites because each mocks the other side of the boundary:**
`src/tools/client.py` never sent an API key to `operations-performance`'s API. That
project's own test suite passes because it never calls this client; this project's own
test suite passes because `test_failure_scenarios.py` fakes the transport. Only an actual
live call across the real network boundary between two real running services surfaced
it -- exactly the gap this kind of test double structurally cannot catch. Fixed with a
new `OPS_PERFORMANCE_API_KEY` env var, sent as `X-API-Key` on every call in `get()`/
`get_text()`.

**The flagship combined scenario, live, for real** (the same one `docs/e2e-scenarios.md`
and `tests/test_e2e_scenarios.py` Scenario 5 exercise with a scripted LLM): asked
`/chat` "What is the average cycle time and does policy explain any of the delay?" --
the agent correctly called both `get_cycle_time` (returning the live, real 2,393-hour
mean from Neon) and `search_policy_documents` (real Chroma retrieval over the real
indexed policy docs), then synthesized both into one answer with per-claim citations
distinguishing data evidence from policy evidence. Nothing about this response was
scripted -- real Groq model, real tool dispatch, real retrieval.

Also found: the `AGENT_MODEL` default chosen when the provider session (above) was built
(`llama-3.3-70b-versatile`) no longer exists on Groq's current lineup -- caught
immediately by the first live call (`model_not_found`), not by any test (every existing
test injects a fake client and never touches a real model name). Replaced with
`openai/gpt-oss-120b`, confirmed against `client.models.list()` with the real key rather
than guessed. `.env.example` now says explicitly to check Groq's current model list
rather than trusting a name pinned in this file to stay valid.

Full suite re-verified clean with no `.env` present: 153/153.

## Post-v1.0 build session -- real load/concurrency test against the live deployment
Closed FUTURE_IMPROVEMENTS.md's "real load/concurrency testing against a live deployed
server" gap for real -- `scripts/load_test_live.py` hits the actual Render URL with
genuinely concurrent OS threads (not async tasks on one thread), not TestClient.

**Confirmed no race condition** in `src/api/rate_limit.py`'s check-then-append pattern
under real concurrent load against the live server (ran multiple times at varying
concurrency; the 5-request limit held every time). Worth having actually tested this
rather than assumed it given FastAPI runs sync routes across a real thread pool.

**Found a real, different bug while testing**: under concurrent /demo/chat requests,
Groq's own free-tier API rate limit sometimes trips -- and the response was a 502
indistinguishable from a genuine agent failure. A caller had no way to tell "retry
shortly" from "something is actually broken." Fixed with
`src/agent/provider_errors.py::is_rate_limit_error()` (checks `groq.RateLimitError`,
`anthropic.RateLimitError`, and a generic `status_code == 429` fallback for
google-genai, which doesn't expose a dedicated exception class), wired into both
`/chat` and `/demo/chat` to return 503 + `Retry-After` instead of 502. No existing
test caught this because every test mocks the provider client entirely -- a real
rate-limit response from the SDK had never been exercised. 6 new tests.

Full suite: 168/168.

## Post-v1.0 build session -- per-client API keys with named roles
Mirrors operations-performance's identical `src/api/auth.py` addition: `API_KEYS`
(comma-separated `name:key:role` triples) alongside the original single-secret
`API_KEY`, fully backward compatible. Unlike the read-only companion project, this
one has a genuine use for role-gating: `POST /documents` (the one real write
endpoint here) now requires the "admin" role via `require_role("admin")` -- a
reader-role key can chat and investigate but can't upload/index a new document.

4 new tests alongside the 5 original (all passing unchanged): multiple independent
clients, revocation isolation, the actual admin-vs-reader role gate on document
upload (not speculative -- a real route enforces it), and legacy key coexistence.

Full suite: 173/173.

## Post-v1.0 build session -- hybrid retrieval (keyword + semantic) with reranking
`src/retrieval/search.py::hybrid_search()` closes the FUTURE_IMPROVEMENTS.md gap
("Hybrid retrieval (keyword + semantic) and reranking, Tier 3"). Combines the
existing semantic search with real BM25 keyword search (`rank_bm25`) over the same
corpus, merged via Reciprocal Rank Fusion (RRF, k=60, the standard constant from
the original paper) -- a chunk ranking well in EITHER method surfaces near the
top, rather than requiring it to win on cosine similarity alone. This project's
policy documents use precise defined terms throughout ("PO Change Approval",
"Manual Credit Review"), exactly the case keyword search catches that pure
semantic similarity can miss.

Wired into the real tool (`src/tools/documents.py::search_policy_documents` now
calls `hybrid_search` instead of `semantic_search`) rather than left as
unused-alongside scaffolding -- 2 existing tests in `test_tools.py` updated to
patch the new function name, both still passing. `semantic_search` itself is
unchanged and still used directly by `test_e2e_scenarios.py`'s real
chunk/store/retrieve chain test.

4 new tests in `test_hybrid_search.py`, including one that directly proves the
actual point: a chunk containing the query's exact term, ranked LAST by semantic
search, surfaces near the top via the keyword signal. Verified against the real
live indexed corpus too, not just test fixtures.

Full suite: 177/177.

## Post-v1.0 build session -- automated adversarial/unanswerable scoring
`src/evaluation/evaluate_adversarial.py` closes docs/evaluation.md's explicit gap
("automated scoring of 'did the model resist manipulation' ... intentionally left
to a human reviewer for this pass"). Two mechanical checks, same "not another LLM
call judging the first one" principle as evaluate_answers.py's groundedness
scoring:
- **Unanswerable**: reuses `check_groundedness()` directly -- a question with
  nothing in this system's data to answer means any specific number in the
  response is, by construction, fabricated. Recognized this as already covered by
  groundedness checking rather than inventing a new heuristic.
- **Adversarial**: system-prompt-leakage detection (8-consecutive-word verbatim
  match) plus a declared-`compliance_markers` check per question (honest about
  needing that annotation -- no fully generic "detect any compliance" classifier
  exists).

Wired into `generate_agent_report.py` for real -- `AgentEvalResult` now carries
`tool_calls`/`citations`/`compliance_markers`, and the report's "Correct?" column
reflects the actual automated result for these two categories instead of the
always-true tool-selection flag. One existing test's premise (unanswerable always
needs manual review) was genuinely outdated and rewritten to test the new real
behavior (clean refusal passes, fabricated number fails) rather than patched to
keep passing superficially.

11 new/updated tests. Full suite: 186/186.

## Post-v1.0 build session -- fixed a real Groq tool-call schema bug (found live)
Running the real agent evaluation report (src/evaluation/generate_agent_report.py)
against the live Groq-backed agent -- the first genuine end-to-end run of this
tool, not a mocked test -- crashed immediately: `groq.BadRequestError: Tool call
validation failed ... /segment: expected string, but got null`.

Root cause: Groq's model sometimes emits `null` for an OMITTED optional parameter
(e.g. `{"segment": null}` for `get_sla_metrics`'s optional `segment` arg) instead
of leaving the key out entirely, and Groq's own server-side schema validation then
rejects that same output against a plain `"type": "string"` schema -- a failure
mode Anthropic's format doesn't have (it doesn't validate arguments server-side
the way Groq does), so nothing in 189 mocked tests could have caught it.

Fixed in `src/agent/providers.py::_make_optional_params_nullable()`: every
non-required property's type is widened to also accept `null` specifically in the
OpenAI/Groq schema conversion, on a deep copy so the shared `TOOL_SCHEMAS` objects
(also read by the Anthropic and Gemini conversions) are never mutated. Verified by
re-running the exact same live evaluation that crashed -- it now completes
cleanly, and the new automated adversarial/unanswerable scoring correctly scored
PASS against genuine live Groq output for the first time.

3 new tests: nullable widening applied to optional params, NOT applied to
required ones (a required arg staying required matters), and the original schema
object is never mutated.

Full suite: 189/189.

## Post-v1.0 build session -- Dockerfile actually built and run for the first time
Same fix as operations-performance's identical Dockerfile issue (see that project's
PLAN.md for the full root-cause writeup: a real Docker Desktop + WSL2 host networking
MTU issue causing intermittent SSLError "record layer failure" on large pip installs,
fixed at the host level, plus a retry-loop correctness bug where a bash `for` loop's
own exit status masked a fully-failed install as success).

Verified end to end for real, not just built: ran the container against real
GROQ_API_KEY/OPS_PERFORMANCE_API_KEY, watched the startup lifespan hook (added
earlier this session) genuinely download the real ONNX embedding model and index
the real policy documents inside the container from a cold start, confirmed
`GET /health` reports `vector_store_reachable: true` against the container's own
freshly-built Chroma store. `ops_performance_api_reachable: false` in this
particular run was expected (the companion container wasn't running at the same
time), not a bug -- the app degraded exactly as designed.

Full suite: 189/189.

## Post-v1.0 build session -- real cost/usage tracking, closing the last named P2 gap
Closes FUTURE_IMPROVEMENTS.md's plainly-stated gap: "the cost estimator exists and
works; it has no real spend numbers to report yet since no live model has been
called." One has now, repeatedly, this session.

**Found a real gap while wiring this up**: `config/pricing.yaml` only had Anthropic
rates configured, despite the live deployment actually running on Groq
(`AGENT_PROVIDER=groq`, `openai/gpt-oss-120b`) -- cost tracking would have silently
failed (`ValueError: No pricing configured`) the moment anyone tried to use it for
real. Added Groq's real rate ($0.15/1M input, $0.60/1M output), sourced directly
from `console.groq.com/docs/model/openai/gpt-oss-120b`, not a third-party
aggregator, since this is the rate that actually gets used.

`AgentResponse` now carries `prompt_tokens`/`completion_tokens`, populated from
Groq's own `response.usage` (real measured counts, never the
`estimate_tokens()` chars/4 heuristic) and summed across every model round a
multi-tool turn makes -- a 3-round investigation undercounting by 2/3 would have
been a real, silent bug if only the last round's usage were captured.
`src/agent/providers.py::_log_real_cost()` computes and logs a genuine dollar
figure per turn; failures here are swallowed to a debug log rather than ever
breaking a real user's answer, since cost logging is observability, not something
that should be allowed to fail a request.

**Verified for real, not just in tests**: fired a live call through the actual
Groq-backed agent -- 2,146 real prompt tokens, 128 real completion tokens,
**$0.000399** computed from genuine measured usage. 6 new tests (usage capture,
multi-round summing, graceful handling when usage is absent, real cost
computation against the configured rate, and unconfigured-model resilience).

Full suite: 194/194.

## Post-v1.0 build session -- LangChain as an additional orchestration path
`src/agent/langchain_agent.py` + `src/agent/langchain_tools.py` add LangChain as a
named, working orchestration option (`provider: langchain` in config/agent.yaml or
`AGENT_PROVIDER=langchain`), additive alongside the existing Anthropic/Groq/Gemini
providers in `src/agent/providers.py` -- not a replacement.

`langchain_tools.py` wraps this project's EXISTING tools
(`src/tools/registry.py::ALL_TOOLS`) as `StructuredTool` objects via
`infer_schema=True`, which builds each tool's argument schema directly from the
real function's own type hints -- not a hand-converted copy of `TOOL_SCHEMAS`'s
JSON schema, one less place for the two representations to drift. Every tool call
made through the LangChain path executes the exact same function every other
provider calls; there's one source of truth for tool behavior regardless of which
orchestration layer is calling it.

Uses `ChatGroq.bind_tools()` + a manual round loop, not LangChain's
`AgentExecutor` or LangGraph -- consistent with this project's stated position
(FUTURE_IMPROVEMENTS.md) on not adopting a heavier agent framework speculatively.
Still genuinely exercises LangChain's core abstractions (`ChatModel`, tool
binding, typed messages: `SystemMessage`/`HumanMessage`/`AIMessage`/`ToolMessage`).

**Real dependency conflict found and resolved**: `langchain-groq` caps
`groq<1.0.0`, while this project's already-deployed Groq integration used
`groq>=1.7.0`. Verified the downgrade to 0.37.x doesn't break the existing
(already-in-production) Groq path with a real live call before committing to the
pin -- `requirements.txt` now explicitly pins `groq>=0.30,<1.0` rather than
leaving version resolution to install order.

**Verified for real, live, not just against mocks**: fired a real question
through the LangChain path -- correctly selected both `get_cycle_time` and
`search_policy_documents`, real retrieval returned 5 real citations, and it
degraded honestly (stated the data tool was unavailable rather than fabricating a
number) exactly as the other providers do, when the companion service wasn't
reachable from this local test run.

5 new tests (tool-loop logic, budget handling, error handling, dispatcher
routing) using LangChain's own real `AIMessage` class for fidelity, not a
SimpleNamespace stand-in.

Full suite: 199/199.
