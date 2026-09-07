# Project Overview

Operations Assistant is an investigation assistant over Northstar Manufacturing's operational
data and policy documents -- the second of two connected projects. It doesn't reimplement any
analytics; it consumes [`operations-performance`](../operations-performance)'s API as agent tools.

## Start here, in this order
1. `docs/architecture.md` -- how this project fits together (once filled in, Phase 17)
2. `docs/rag-design.md` -- document corpus, chunking, embedding, vector store choices and why
3. `docs/tool-design.md` -- how the agent gets controlled access to operations-performance's data
   (Phase 16, not written yet)
4. `docs/agent-design.md` -- tool selection, investigation mode (Phase 17-18, not written yet)
5. `docs/evaluation.md` -- the benchmark question set and scoring (Phase 19, not written yet)

## Relationship to operations-performance
Same underlying business scenario (Northstar Manufacturing's Procure-to-Pay process), same
database. `operations-performance` owns the data model and all analytics/ML logic, exposed via its
own FastAPI service. This project's agent calls that API as tools rather than querying the
database directly or reimplementing any calculation -- one analytics engine, two front doors (a
human-facing dashboard in that project, an AI-agent interface here).

## Build status
See `PLAN.md` for the phase-by-phase build log (phases 13-24 of the combined roadmap;
`operations-performance`'s `PLAN.md` covers phases 1-12).

## Current state (Phase 13-14 complete)
- FastAPI service scaffolded (`src/api/`), `GET /health` and `GET /documents` working
- Document ingestion pipeline complete and tested: load (markdown+frontmatter, with PDF fallback)
  -> section-aware chunk -> local embed -> Chroma vector store
- 4 real synthetic policy documents written, cross-referenced against
  `operations-performance`'s actual `config/sla.yaml` values so an investigation using both
  projects together would find genuinely consistent numbers
- Not yet built: retrieval/search (Phase 15), tool wiring (Phase 16), the agent itself (Phase 17)
