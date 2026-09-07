# Architecture

## The whole system, both repos
```
                              NORTHSTAR MANUFACTURING
                                (fictional client)
                                       |
                  +--------------------+--------------------+
                  |                                         |
          operations-performance                  operations-assistant
        "understand the business"                "investigate the business"
                  |                                         |
        BPI 2019 event log (real)                4 synthetic policy docs
        + synthetic business rules                         |
                  |                                    chunk -> embed
        ETL -> Postgres -> analytics/ML                Chroma (local)
                  |                                         |
        FastAPI (src/api/) ------------- consumed as tools -+
                  |                                         |
        Dashboard + management report              Agent (src/agent/)
                  |                                         |
        BigQuery migration                    FastAPI (POST /chat, /investigate)
```

## The one rule that shapes everything else
`operations-performance` owns the data model and every analytics/ML calculation.
`operations-assistant` never re-derives a number -- it calls the other project's API
(`src/tools/client.py` -> `analytics.py`/`prediction.py`/`database.py`) and treats the response as
ground truth. This is why the SLA targets in this project's `sla-policy.md` document are
deliberately written to match `operations-performance/config/sla.yaml` exactly (verified again in
Phase 24) -- if an investigation question needs both the policy text and the live number, both
answers should agree, because they trace back to the same source of truth by construction, not
by coincidence.

## Request flow: a combined question
"Why are high-value orders breaching SLA, and does our procurement policy explain the extra step?"

```
POST /chat
    |
    v
run_agent() -- src/agent/agent.py
    |
    +-- model requests get_sla_metrics -- src/tools/analytics.py
    |         |
    |         v
    |   GET operations-performance:8000/metrics/sla  (real HTTP call)
    |         |
    |         v
    |   {"breach_rate_pct": 18.4, ...}
    |
    +-- model requests search_policy_documents -- src/tools/documents.py
    |         |
    |         v
    |   semantic_search() -- src/retrieval/search.py
    |         |
    |         v
    |   Chroma vector store (local) -- top-K matching chunks
    |
    +-- model synthesizes both tool_results into one grounded answer
    |
    v
ChatResponse { answer, tools_used: [...], citations: [{kind: "data", ...}, {kind: "document", ...}] }
```

Every step in that chain is real, tested code -- except the model's synthesis step, which is the
one part that needs a live `ANTHROPIC_API_KEY` to actually exercise (see `PLAN.md`'s Phase 24
notes).

## Investigation mode's extra step
`POST /investigate` runs the same loop above (with a larger tool-call budget), then makes a
**second** API call asking the model to compile the gathered evidence into the fixed
`InvestigationReport` JSON shape. Two calls, not one, because forcing every response into that
structure would be wasteful for a simple factual question -- see `docs/agent-design.md` for the
full reasoning.

## What's shared vs. what's project-specific
**Shared (copied directly, confirmed identical via `diff`)**: `src/observability/logging_config.py`,
`src/api/middleware.py` -- infrastructure with no reason to differ between the two projects.

**Project-specific**: everything else. `operations-performance` is a batch ETL + analytics + ML
system; `operations-assistant` is a request/response agent system. Their `src/` layouts reflect
that difference deliberately (`ingestion/cleaning/transformation/analytics/ml` vs.
`api/ingestion/retrieval/tools/agent/evaluation`) rather than forcing one folder structure onto two
different kinds of system.

## For the full picture
- `docs/rag-design.md` -- document/chunking/embedding/vector-store choices
- `docs/tool-design.md` -- how the agent gets controlled access to the other project's data
- `docs/agent-design.md` -- the tool-selection loop and investigation mode
- `docs/evaluation.md` -- what's measured and what isn't yet
- `docs/security-notes.md` -- guardrails, input validation, what's out of scope
- `docs/failure-modes.md` -- what happens when things break
- `docs/deployment.md` -- Docker/Cloud Run, and what's honestly untested
- `PLAN.md` -- the full phase-by-phase build log, including every real bug found along the way
