# Security & Guardrails Notes

Small and deliberately bounded, same spirit as `operations-performance`'s
`docs/security-notes.md` -- what's here demonstrates the practice, not a claim that this system
is hardened for real deployment.

## Prompt injection: structural defense, not a filter
The real defense against a policy document containing a fake instruction ("ignore your previous
instructions...") isn't a keyword blocklist -- it's that tool results are only ever placed inside
`tool_result` content blocks (`src/agent/agent.py:_execute_tool_calls`), never concatenated into
the `system` prompt or a plain user/assistant text message. The system prompt (`prompts.py`) also
explicitly tells the model that tool content is data, not instructions, but the prompt instruction
alone would be a weak defense on its own -- the structural separation is what actually matters.

**This is enforced by a regression test, not just documentation**:
`test_system_prompt_is_never_mutated_with_tool_content` plants a malicious-looking instruction
inside a mocked document search result and asserts the `system` kwarg sent to the Anthropic API on
every call remains byte-for-byte `SYSTEM_PROMPT`, with the injected text nowhere in it.
`test_tool_result_content_is_isolated_to_tool_result_blocks` confirms tool output always lands in
a `tool_result` block, never a plain text message. If a future change accidentally builds the
system prompt dynamically from conversation history, these tests fail immediately.

## Tool input validation (`src/tools/validation.py`)
Every parameter an LLM's tool call supplies is treated as untrusted input, same as a user-facing
form field would be, and validated before use:
- `validate_case_id` -- restricts to `[A-Za-z0-9_-]{1,50}`. This matters specifically because
  `case_id` is interpolated directly into a URL path (`predict_sla_risk`'s
  `f"/orders/{case_id}/risk"`) -- an unvalidated value here would be a path-traversal /
  request-smuggling risk against `operations-performance`'s API (e.g. `case_id="../../admin"`).
  Closed off here rather than trusting the downstream API to reject it gracefully.
- `validate_query` -- rejects empty input and caps length (500 chars) before it reaches
  `semantic_search` or gets embedded.
- `clamp_int` -- **`config/tools.yaml`'s `max_result_rows` was documentation-only until this
  phase** -- nothing in code actually enforced it. `get_bottlenecks`, `get_supplier_performance`,
  and `get_pipeline_status` now clamp their inputs to real bounds, so a model-supplied `top_n` of
  an arbitrary size can't flood the agent's context with an unbounded result set. Recording this
  gap explicitly: a config file describing a limit that the code doesn't enforce is a real and
  common class of bug, and it existed here for four phases before being caught while writing
  Phase 20's guardrails, not by a test failure.

## A real finding from CI's dependency audit (pip-audit), investigated, not just noted
The first run of the newly-added CI security job found **4 real CVEs in `chromadb`**
(`PYSEC-2026-311` / `CVE-2026-45829` -- pre-auth remote code execution via a malicious model
repository with `trust_remote_code`; `CVE-2026-45830`, `CVE-2026-45831`, `CVE-2026-45833` --
multi-tenant authorization bypass issues), with **no fix version available yet** even in the
latest release (1.5.9) at the time this was found.

Investigated rather than either panicking or ignoring it: all four CVEs are in ChromaDB's
**HTTP server API** (`/api/v2/tenants/{tenant}/databases/{db}/collections/...` endpoints,
tenant/database-scoped RBAC). This project's `src/retrieval/vector_store.py` uses
`chromadb.PersistentClient` exclusively -- an embedded, in-process client with no HTTP server
ever started, `chromadb.HttpClient` is never used anywhere in this codebase. The vulnerable
attack surface (a network-reachable Chroma server) doesn't exist in how this project actually
runs Chroma.

**This is not a reason to stop tracking it.** If this project's Chroma usage ever changes to a
standalone server (e.g. a shared multi-service deployment where the vector store runs as its own
process, reachable over the network), these CVEs become directly relevant and this decision needs
revisiting -- noted here specifically so that future change doesn't silently inherit an
unreviewed assumption. `pip-audit`'s CI job (`continue-on-error: true`) will keep surfacing this
on every run until a fix version ships; it's deliberately not blocking CI, since blocking on a
vulnerability with no available fix and no exploitable path in this project's actual usage would
just be noise, not a control.

## What's explicitly NOT done here, on purpose
- No automated scoring of whether the agent actually *resists* an injection attempt in its
  generated answer -- the structural tests above verify the injected text can't reach the system
  prompt, which is the stronger guarantee, but a live model could still choose to discuss or act
  on injected content within its normal answer. `data/evaluation/agent_questions.json`'s
  adversarial question is flagged for human review in `evaluate_agent.py`, not automatically
  scored (see `docs/evaluation.md` for why).
- No rate limiting on `/chat` or `/investigate` (Tier 3 in FEATURES.md)
- **API-key authentication is now real** (`src/api/auth.py`, ported from
  `operations-performance`'s identical mechanism) -- every endpoint except `GET /health` requires
  a matching `X-API-Key` header, constant-time compared. Fails open if `API_KEY` is unset (local
  dev only). This matters more here than in the read-only companion project: `/chat` and
  `/investigate` each trigger a real LLM call once live, so an unauthenticated endpoint isn't just
  a data-exposure risk, it's a real-money risk. Still just one shared secret -- no per-client keys,
  rotation, or scopes.
- No sandboxing of what a tool call could theoretically do beyond input validation -- every tool
  is read-only against `operations-performance`'s API by construction (`src/tools/client.py` only
  ever issues GET requests), which is the real containment here, not a runtime permission check

Being explicit about what's out of scope is the point, same principle as the companion project's
security doc -- a list that claims completeness is a worse signal than a bounded, honest one.
