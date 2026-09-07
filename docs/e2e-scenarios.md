# End-to-End Scenarios (Phase 28)

Two of the five canonical business scenarios live here (`tests/test_e2e_scenarios.py`); the other
three (bottleneck identification, worst supplier, SLA-risk prediction) live in
`operations-performance/tests/test_e2e_scenarios.py`, traced through that project's FastAPI layer.

## What's genuinely real vs. necessarily stood-in
**Real**: document loading, section-aware chunking, Chroma storage and retrieval (an actual
ephemeral Chroma collection, not a mock), cosine-similarity search, the full agent tool-dispatch
path (`registry.call_tool` → the real tool function → the real `src/tools/client.py` HTTP layer
against an `httpx.MockTransport`-backed server).

**Stood in, and named as such**: the embedding *model* -- `sentence-transformers` needs a
download from a host this environment's network allowlist doesn't include. In its place, a
deterministic bag-of-words hashing vectorizer (`_hashing_embed` in the test file) -- a real,
legitimate embedding technique on its own terms (lexical-overlap-based rather than
learned-semantic), not fake data. It's what makes Scenario 4's retrieval test meaningful rather
than trivially mocked: the test only passes if the vectorizer actually places the right chunk
closest to the query, the same property a real embedding model needs to have.

**Also stood in**: the LLM's own reasoning in Scenario 5 -- no `ANTHROPIC_API_KEY` available here,
so the model's tool-selection decision is scripted. What's *not* scripted: the citations in the
final response, which come from genuinely running the retrieval chain against the real indexed
corpus, not from a canned string.

## Scenario 4 result
`semantic_search("What approval is required for purchase orders above $10,000?")` correctly
retrieves the Procurement Policy's Secondary Approval section as its top result, using only the
hashing vectorizer -- confirming the chunking/storage/retrieval wiring works correctly together,
independent of which embedding technique sits underneath it. A second test confirms an unrelated
question ("weather forecast") does *not* return a confident match, which is the retrieval-side
half of the grounding guarantee.

## Scenario 5 (flagship) result
The agent correctly calls both `get_sla_metrics` (the data tool) and `search_policy_documents`
(the RAG tool) for the combined question, and the final response's citations include both a
`"data"` and a `"document"` kind -- with the document citation's actual text traced back to a real
retrieval result, not asserted against a hardcoded string. This is the strongest evidence available
in this environment that the multi-tool synthesis path (`docs/agent-design.md`) is wired correctly
end-to-end, short of a real model actually making the tool-selection decision itself.

## What Phase 28 still doesn't prove
A real model's actual tool-selection judgment, and its actual synthesis quality given real
(not scripted) tool results. That remains the one thing that needs a real `ANTHROPIC_API_KEY` and
a real `operations-performance` instance to verify -- unchanged from every phase before this one.
