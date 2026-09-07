# Operations Assistant — Feature Specification

**Client scenario:** Northstar Manufacturing wants to ask plain-English operational questions
("why did SLA breaches increase this quarter?") and get answers grounded in real operational data
*and* real company policy — not a chatbot guessing, an assistant that investigates and cites its
evidence. It consumes the data model and analytical logic already built in `operations-performance`
rather than re-deriving it.

Tiering: **Tier 1 = required to call this project done. Tier 2 = what makes it a strong,
above-fresher project. Tier 3 = advanced, build only if time remains.**

---

## 1. Document knowledge base (RAG source material)
- [T1] 5–8 synthetic-but-realistic client documents: Procurement Policy, Approval Policy,
      Supplier Management SOP, PO Change Procedure, Invoice Processing SOP, SLA Policy,
      Exception Handling Procedure — each with title, version, effective date, section headings
- [T1] Explicitly labeled as synthetic (same honesty standard as Project 1's real/synthetic split)

## 2. Document ingestion pipeline
- [T1] Parse → clean → chunk → embed → index (PDF/TXT minimum)
- [T1] Metadata retained per chunk (source doc, section, version)
- [T2] Re-indexing support (a document gets updated, index reflects it)
- [T3] Document versioning (assistant knows which policy version is current)

## 3. RAG retrieval
- [T1] Semantic search over the vector DB, top-K retrieval
- [T1] Every document-grounded answer includes a citation (document + section)
- [T2] Metadata filtering (restrict retrieval to a specific document/department)
- [T3] Hybrid retrieval (keyword + semantic), reranking, query rewriting

## 4. Structured-data tools (consuming Project 1)
- [T1] The agent gets a small, fixed set of controlled functions — never raw/arbitrary SQL access:
      `get_cycle_time`, `get_sla_metrics`, `get_supplier_performance`, `get_bottlenecks`,
      `get_process_variants`, `get_rework_statistics`, `predict_sla_risk`
- [T1] Each tool wraps logic that already exists in `operations-performance/src/analytics` and
      `src/ml` — Project 2 does not reimplement the analysis, it calls it
- [T2] Input validation and row/result limits on every tool

## 5. Agent orchestration
- [T1] Single agent, 3–7 tools, explicit tool-selection logic (does this question need documents,
      data, both, or neither?)
- [T1] Structured system prompt defining scope, tone, and refusal behavior
- [T2] Multi-tool questions handled correctly (e.g. "why are high-value orders breaching SLA, and
      does policy explain the extra step?" → SQL/analytics tool + RAG tool, synthesized)
- [T3] LangGraph if it genuinely simplifies an explicit multi-step workflow — not adopted for its
      own sake

## 6. Investigation mode
- [T2] Beyond single Q&A: a structured multi-step investigation for open-ended prompts
      ("investigate why procurement delays increased this quarter")
- [T2] Fixed investigation output shape: Executive Summary → Problem → Evidence → Root Causes →
      Relevant Policy → Recommendations → Limitations
- [T2] Conversational follow-ups within a session ("what about suppliers?" after a prior answer)

## 7. Grounding & hallucination control
- [T1] Factual claims must trace to a tool result or a retrieved chunk — no invented numbers
- [T1] Explicit "insufficient data" response when the system genuinely doesn't have the answer
      (tested with an out-of-scope question, e.g. asking about revenue the system never ingested)
- [T2] Separate "data evidence" from "document evidence" in the response so a reader can trace
      each claim back to its source

## 8. Guardrails / security
- [T1] Uploaded/retrieved document text is never treated as an instruction (basic prompt-injection
      resistance — test with a document containing an embedded fake instruction)
- [T1] No destructive SQL possible from the agent; tool inputs validated; tool calls capped/timed out
- [T2] Logging of every tool call (question → tools invoked → latency → result) for later debugging

## 9. Evaluation
- [T1] A written evaluation set of 30–50 questions spanning: data-only, document-only, combined,
      multi-step, unanswerable, and at least a few adversarial/ambiguous ones
- [T1] Scored on: correct tool selection, numerical correctness (matches the DB), citation
      correctness, and appropriate refusal on unanswerable questions
- [T2] A short written evaluation report — not just a score, but where and why it fails

## 10. API
- [T1] FastAPI service, not a notebook: `POST /chat`, `POST /investigate`, `GET /health`
- [T2] `POST /documents`, `GET /documents`, `GET /metrics` (thin wrapper over Project 1 tools)
- [T3] Basic auth / role separation (analyst vs. manager access to certain tools/documents)

## 11. Deployment
- [T1] Dockerized (API + agent + retrieval + tools in one container, docker-compose for local deps)
- [T2] Deployed to Cloud Run, connected to the same BigQuery/Postgres backend as Project 1
- [T3] MCP — only as a genuine refactor of the existing tools into MCP-tool form, framed as "here's
      how this would plug into an MCP-based agent," not a dependency for the core project to work

## 12. Engineering hygiene
- [T1] `src/` organized by architectural role (api / ingestion / retrieval / tools / agent /
      evaluation), matching the actual data flow — not generic service/manager/handler layers
- [T1] Tests for retrieval, tools, agent tool-selection, and the API layer
- [T2] Config-driven agent/retrieval/tool settings (`config/*.yaml`) instead of hardcoded values

---

## Definition of done
**MVP (interview-ready minimum):** sections 1–5, 7 (basic), 9 (evaluation set exists and is scored),
10 (`/chat` working), 12 — all Tier 1.
**Strong version (what you should actually ship):** MVP + all Tier 2, especially Investigation
mode and grounding/evidence separation — this is what actually differentiates the project from a
"chat with PDF" tutorial.
**Stretch:** Tier 3, only once both projects clear their Tier 1 + Tier 2 bar.
