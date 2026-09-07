# Production-Readiness Review (Phase 31)

Four perspectives, same discipline as `operations-performance`'s review -- honest answers, not
uniformly reassuring ones.

## Data
| Question | Answer |
|---|---|
| Can data be regenerated? | **Yes.** All 4 policy documents are version-controlled markdown; `scripts/index_documents.py` rebuilds the vector store from them deterministically, with safe re-indexing (Phase 14 -- old chunks deleted before new ones are added). |
| Can bad input be detected? | **Yes.** `validate_query`/`validate_case_id` (Phase 20) reject malformed tool input before it reaches a URL path or the retrieval layer. |
| Are the documents' claims reproducible? | **Yes, and cross-verified.** The SLA values in `sla-policy.md` are checked against `operations-performance/config/sla.yaml`'s actual values (re-verified again in Phase 24), not just written to sound plausible. |
| Is provenance documented? | **Yes.** `docs/rag-design.md` states plainly these are synthetic documents, not a real company's actual policies. |

## Application
| Question | Answer |
|---|---|
| Can APIs fail gracefully? | **Yes, and genuinely tested.** Phase 23's `test_failure_scenarios.py` exercises real `httpx.Client` behavior against 500s, malformed JSON, connection refused, and timeouts -- not just asserting an exception type exists. |
| Are inputs validated? | **Yes.** Every tool parameter an LLM could supply is validated (`src/tools/validation.py`) before use -- including a real path-traversal risk closed in `predict_sla_risk`'s `case_id` handling. |
| Are dependencies isolated? | **Yes for authentication now**, added in a later session (`src/api/auth.py`) -- matters more here than in the companion project since `/chat`/`/investigate` each cost real money per call once live. |

## AI
| Question | Answer |
|---|---|
| Can the model hallucinate? | **Yes, inherently** -- this is the core risk of an LLM-backed system. Mitigated, not eliminated: the system prompt instructs grounding in tool results only, tool results are structurally isolated from the system prompt (can't be mistaken for instructions), and a fixed "insufficient data" fallback exists for empty retrieval. **Not yet verified against a real model's actual behavior** -- every test here mocks the LLM's response. |
| Can retrieval fail? | **Yes, and the failure mode is explicit**, not silent: below `min_similarity_score`, results are dropped entirely and a fixed honest string is returned (Phase 15), rather than forcing a weak match. |
| Can tool selection fail? | **Untested against a real model.** The tool-call budget prevents an infinite loop if it does (Phase 17), and Phase 29's audit harness exists to measure it -- but has never been run against a real API key. |
| Can malicious documents influence behavior? | **Structurally, no** -- proven by regression test (Phase 20), not just asserted: tool results can never reach the system prompt, verified by a test that plants a fake instruction in a mocked document chunk and checks the actual `system` kwarg sent to the API stays unchanged. |

## Operations
| Question | Answer |
|---|---|
| Can I see what happened? | **Yes.** Per-turn, per-round, per-tool-call structured logging (Phase 22), with a `turn_id` tying it together. |
| Can I diagnose failure? | **Yes, more thoroughly tested here than in the companion project** -- Phase 23's real HTTP-layer failure tests, plus agent-level tests confirming the loop survives every tool in a turn failing at once. |
| Can someone reproduce deployment? | **Was also broken until this pass.** The same script-invocation bug `operations-performance`'s Phase 30 found was checked here too and confirmed present (`python scripts/index_documents.py` failed identically) -- fixed the same way. The Dockerfile itself was unaffected (it doesn't invoke scripts by path), but local development following the README exactly as written would have failed at the very first step. |

## What's genuinely still unknown, unchanged by this review
This review, like every phase before it, cannot answer: does a real model, given real tool
results, actually behave the way 103 mocked tests say it should. That's not a gap this document
can close -- it's the one thing that needs a real `ANTHROPIC_API_KEY` and a real running
`operations-performance` instance to find out.
