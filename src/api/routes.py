from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
import time
import uuid
import tempfile
from pathlib import Path
from collections import Counter

from src.api.schemas import (
    HealthResponse, DocumentUploadResponse, DocumentListItem,
    ChatRequest, ChatResponse, SourceCitation,
    InvestigateRequest, InvestigateResponse, InvestigationReport,
    HITLChatResponse, InterventionProposal, InterventionStatusResponse,
    InterventionResumeResponse, RejectRequest,
)
from src.api.auth import require_role
from src.api.dependencies import check_ops_performance_reachable, check_vector_store_reachable
from src.api.rate_limit import is_allowed as rate_limit_is_allowed
from src.agent.provider_errors import is_rate_limit_error
from src.ingestion.document_loader import load_document
from src.ingestion.chunker import chunk_text
from src.ingestion.index_documents import index_document, list_indexed_documents
from src.agent.agent import run_agent
from src.agent.investigation import run_investigation
from src.agent.conversation_store import (
    get_history, append_turn,
    get_turns_to_summarize, get_summary, save_summary, delete_turns,
)
from src.observability.trace_log import log_trace, read_recent

router = APIRouter()

# /health and /demo/* are intentionally on an unauthenticated router -- /health for
# the conventional load-balancer/orchestrator liveness-probe reason (same as
# operations-performance's src/api/auth.py docstring), /demo/* because it's this
# project's public portfolio demo surface, guarded instead by src/api/rate_limit.py
# rather than an API key a site visitor obviously can't be expected to have.
health_router = APIRouter()


@health_router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        ops_performance_api_reachable=check_ops_performance_reachable(),
        vector_store_reachable=check_vector_store_reachable(),
    )


@health_router.get("/eval/recent")
def eval_recent(n: int = 100):
    """Online production eval stats computed from the last N demo traces logged
    to logs/agent_traces.jsonl. Returns latency percentiles, tool-call
    distribution, answered rate, and citation rate — a lightweight signal that
    the agent is working as expected in production without needing a test harness
    to run."""
    traces = read_recent(n)
    if not traces:
        return {"sample_size": 0, "message": "No traces logged yet"}

    latencies = [t["latency_ms"] for t in traces if t.get("latency_ms") is not None]
    latencies_sorted = sorted(latencies)

    def pct(data, p):
        if not data:
            return None
        idx = int(len(data) * p / 100)
        return round(data[min(idx, len(data) - 1)], 1)

    tool_counter: Counter = Counter()
    for t in traces:
        for tool in t.get("tools_used", []):
            tool_counter[tool] += 1

    answered = sum(1 for t in traces if t.get("answered"))
    cited = sum(1 for t in traces if t.get("num_citations", 0) > 0)
    total = len(traces)

    return {
        "sample_size": total,
        "window": f"last {n} requests",
        "latency_ms": {
            "p50": pct(latencies_sorted, 50),
            "p90": pct(latencies_sorted, 90),
            "p95": pct(latencies_sorted, 95),
            "mean": round(sum(latencies) / len(latencies), 1) if latencies else None,
        },
        "answered_rate_pct": round(100 * answered / total, 1),
        "citation_rate_pct": round(100 * cited / total, 1),
        "avg_tools_per_query": round(sum(t.get("num_tools", 0) for t in traces) / total, 2),
        "avg_citations_per_query": round(sum(t.get("num_citations", 0) for t in traces) / total, 2),
        "top_tools": [{"tool": t, "count": c} for t, c in tool_counter.most_common(10)],
        "cost_usd": {
            "total": round(sum(t["cost_usd"] for t in traces if t.get("cost_usd")), 6),
            "avg_per_query": round(
                sum(t["cost_usd"] for t in traces if t.get("cost_usd"))
                / max(1, sum(1 for t in traces if t.get("cost_usd"))),
                6,
            ) if any(t.get("cost_usd") for t in traces) else None,
            "queries_with_cost": sum(1 for t in traces if t.get("cost_usd")),
        },
    }


@health_router.get("/", response_class=HTMLResponse, include_in_schema=False)
def demo_page():
    return (Path(__file__).parent / "demo.html").read_text(encoding="utf-8")


@health_router.post("/demo/chat", response_model=ChatResponse)
def demo_chat(request: ChatRequest, http_request: Request):
    """Same underlying agent as the authenticated /chat, minus persisted
    conversation history (a public demo visitor's turns aren't worth storing) and
    rate-limited per client IP instead of requiring an API key -- see
    src/api/rate_limit.py for why a site visitor can't reasonably be asked to have
    one."""
    # Behind a reverse proxy (Render, this project's actual deployment target),
    # request.client.host is the proxy's own internal address for every request, not
    # the real visitor's IP -- which would rate-limit every demo visitor as a single
    # shared client. X-Forwarded-For's first entry is the original client; only trust
    # it because this project deliberately runs behind exactly one known proxy layer,
    # not directly exposed to the internet where a client could forge that header.
    forwarded_for = http_request.headers.get("x-forwarded-for")
    client_ip = forwarded_for.split(",")[0].strip() if forwarded_for else (
        http_request.client.host if http_request.client else "unknown"
    )
    if not rate_limit_is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Demo rate limit reached (5 questions per 10 minutes) -- please try again shortly.",
        )

    t0 = time.monotonic()
    try:
        result = run_agent(request.question)
    except Exception as exc:
        raise _agent_error_response(exc)
    latency_ms = round((time.monotonic() - t0) * 1000, 1)

    from src.agent.agent import load_agent_config
    from src.evaluation.cost_estimator import calculate_cost
    cfg = load_agent_config()
    raw_model = cfg.get("model", "")
    model_display = raw_model.split("/")[-1] if "/" in raw_model else raw_model

    cost_usd: float | None = None
    if result.prompt_tokens and result.completion_tokens:
        try:
            # Use raw_model (e.g. "openai/gpt-oss-120b") for pricing lookup so it
            # matches the key in config/pricing.yaml; model_display is stripped for
            # display only and would miss the "openai/" prefix the Groq key needs.
            est = calculate_cost(
                input_tokens=result.prompt_tokens,
                output_tokens=result.completion_tokens,
                model=raw_model or None,
                is_estimated_token_count=False,
            )
            cost_usd = est.total_cost_usd
        except Exception:
            pass

    response = ChatResponse(
        answer=result.answer,
        tools_used=result.tools_used,
        citations=[SourceCitation(kind=c["kind"], reference=c["reference"]) for c in result.citations],
        conversation_id="demo",
        latency_ms=latency_ms,
        model=model_display or None,
        input_tokens=result.prompt_tokens,
        output_tokens=result.completion_tokens,
        cost_usd=cost_usd,
    )
    log_trace(
        tools_used=result.tools_used,
        num_citations=len(result.citations),
        latency_ms=latency_ms,
        answered=bool(result.answer),
        model=model_display or None,
        cost_usd=cost_usd,
    )
    return response


@health_router.get("/demo/chat/stream")
def demo_chat_stream(question: str, http_request: Request):
    """SSE endpoint — streams real-time tool-call events then the final answer.
    Uses the same agent and rate-limiting as /demo/chat; clients should use an
    EventSource to consume the stream.

    Event shape:
      {"type": "thinking"}                           — agent has started
      {"type": "tool_start", "tool": "<name>"}       — tool call initiated
      {"type": "tool_done",  "tool": "<name>", "ok": bool}  — tool returned
      {"type": "answer",     "answer": "...", "tools_used": [...], "latency_ms": N}
      {"type": "error",      "detail": "..."}        — agent raised
    """
    import queue
    import threading
    import json as _json

    forwarded_for = http_request.headers.get("x-forwarded-for")
    client_ip = forwarded_for.split(",")[0].strip() if forwarded_for else (
        http_request.client.host if http_request.client else "unknown"
    )
    if not rate_limit_is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Demo rate limit reached (5 questions per 10 minutes) -- please try again shortly.",
        )

    event_queue: queue.Queue = queue.Queue()

    def on_event(event: dict) -> None:
        event_queue.put(event)

    def run() -> None:
        t0 = time.monotonic()
        try:
            on_event({"type": "thinking"})
            result = run_agent(question, event_cb=on_event)
            latency_ms = round((time.monotonic() - t0) * 1000, 1)

            from src.agent.agent import load_agent_config
            from src.evaluation.cost_estimator import calculate_cost
            cfg = load_agent_config()
            raw_m = cfg.get("model", "")
            disp_m = raw_m.split("/")[-1] if "/" in raw_m else raw_m
            cost_usd: float | None = None
            if result.prompt_tokens and result.completion_tokens:
                try:
                    est = calculate_cost(
                        input_tokens=result.prompt_tokens,
                        output_tokens=result.completion_tokens,
                        model=raw_m or None,
                        is_estimated_token_count=False,
                    )
                    cost_usd = est.total_cost_usd
                except Exception:
                    pass

            log_trace(
                tools_used=result.tools_used,
                num_citations=len(result.citations),
                latency_ms=latency_ms,
                answered=bool(result.answer),
                model=disp_m or None,
                cost_usd=cost_usd,
            )
            on_event({
                "type": "answer",
                "answer": result.answer,
                "tools_used": result.tools_used,
                "citations": [{"kind": c["kind"], "reference": c["reference"]} for c in result.citations],
                "latency_ms": latency_ms,
                "model": disp_m or None,
                "cost_usd": cost_usd,
            })
        except Exception as exc:
            on_event({"type": "error", "detail": str(exc)})
        finally:
            event_queue.put(None)

    threading.Thread(target=run, daemon=True).start()

    def generate():
        while True:
            event = event_queue.get()
            if event is None:
                return
            yield f"data: {_json.dumps(event)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _maybe_summarize(conversation_id: str) -> None:
    turns = get_turns_to_summarize(conversation_id)
    if not turns:
        return
    try:
        from src.agent.summarizer import summarize_turns
        batch = [{"role": t["role"], "content": t["content"]} for t in turns]
        new_text = summarize_turns(batch)
        existing = get_summary(conversation_id) or ""
        combined = f"{existing} {new_text}".strip() if existing else new_text
        save_summary(conversation_id, combined)
        delete_turns([t["id"] for t in turns])
    except Exception:
        pass  # never crash a conversation for a summarization failure


def _agent_error_response(exc: Exception) -> HTTPException:
    # A provider rate-limit (found by scripts/load_test_live.py hitting the real
    # deployed service with genuinely concurrent requests -- Groq's free-tier limit
    # tripped and surfaced as an indistinguishable-from-broken 502) is a distinct,
    # retryable condition, not "the agent failed." 503 + Retry-After tells a real
    # caller to back off and retry rather than treating this like a bug to report.
    if is_rate_limit_error(exc):
        return HTTPException(
            status_code=503,
            detail="The model provider is temporarily rate-limited -- please retry in a few seconds.",
            headers={"Retry-After": "10"},
        )
    return HTTPException(status_code=502, detail=f"Agent failed to produce a response: {exc}")


@router.get("/documents", response_model=list[DocumentListItem])
def list_documents():
    return list_indexed_documents()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """conversation_id, if provided, replays prior turns' text (not tool-call
    scaffolding -- see conversation_store.py) as context, then persists this turn
    back to the store. A fresh conversation_id is generated and returned if none was
    given, so a client can start a new conversation with nothing and continue it on
    subsequent calls by passing back what /chat returned.

    When SEMANTIC_CACHE=1 is set, a SemanticCache is consulted before calling the
    agent. A cache hit returns the stored answer immediately and still records the
    turn so conversation history stays consistent. Not applied to /chat/hitl (HITL
    turns have side-effects from propose_intervention and must not be cached).
    """
    from src.cache.semantic_cache import SemanticCache, is_enabled as cache_enabled

    conversation_id = request.conversation_id or str(uuid.uuid4())[:8]
    history = get_history(conversation_id)

    # ── semantic cache check ──────────────────────────────────────────────────
    _cache: SemanticCache | None = None
    if cache_enabled():
        _cache = SemanticCache()
        cached = _cache.get(request.question)
        if cached is not None:
            answer = cached.get("answer", "")
            tools_used = cached.get("tools_used", [])
            citations = [SourceCitation(**c) for c in cached.get("citations", [])]
            append_turn(conversation_id, request.question, answer)
            _maybe_summarize(conversation_id)
            return ChatResponse(
                answer=answer,
                tools_used=tools_used,
                citations=citations,
                conversation_id=conversation_id,
            )

    try:
        result = run_agent(request.question, history=history)
    except Exception as exc:
        raise _agent_error_response(exc)

    if _cache is not None:
        _cache.put(request.question, {
            "answer": result.answer,
            "tools_used": result.tools_used,
            "citations": result.citations,
        })

    append_turn(conversation_id, request.question, result.answer)

    # Summarize the oldest turns when the conversation grows past the threshold.
    # Runs synchronously but on the cheap Haiku model; never crashes the response
    # if the summarizer fails.
    _maybe_summarize(conversation_id)

    return ChatResponse(
        answer=result.answer,
        tools_used=result.tools_used,
        citations=[SourceCitation(kind=c["kind"], reference=c["reference"]) for c in result.citations],
        conversation_id=conversation_id,
    )


ALLOWED_UPLOAD_SUFFIXES = {".md", ".pdf", ".txt"}
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5MB -- generous for a policy document, small enough to reject an accidental wrong-file upload fast


@router.post("/documents", response_model=DocumentUploadResponse, dependencies=[Depends(require_role("admin"))])
async def upload_document(file: UploadFile = File(...)):
    """Accepts a document, indexes it through the real Phase 14 pipeline (load ->
    chunk -> embed -> upsert), and makes it immediately searchable via
    search_policy_documents. Uses a temp file rather than an in-memory path because
    load_document() is path-based (it also needs to re-derive frontmatter for .md
    files), and reusing that same function here -- rather than a second parsing path
    just for uploads -- is what keeps upload and the corpus-directory indexing script
    (scripts/index_documents.py) behaving identically."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}' -- allowed: {sorted(ALLOWED_UPLOAD_SUFFIXES)}",
        )

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024*1024)}MB upload limit")
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        doc = load_document(tmp_path)
        # the real document_id should reflect the uploaded filename, not the random
        # temp-file name load_document() would otherwise derive from tmp_path's stem
        doc.document_id = Path(file.filename).stem
        result = index_document(doc)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to process document: {exc}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return DocumentUploadResponse(
        document_id=result.document_id, title=result.title, chunk_count=result.chunk_count,
    )
# pipeline wired to an actual upload endpoint (chunking/indexing logic itself is
# already done, this route just calls it once file upload handling is added).


@router.post("/chat/hitl", response_model=HITLChatResponse)
def chat_hitl(http_req: Request, request: ChatRequest):
    """Agent turn with human-in-the-loop approval gate.

    If the model calls propose_intervention, the graph pauses and this endpoint
    returns {"status": "awaiting_approval", "intervention": {...}, "thread_id": "..."}.
    The caller must then POST /interventions/{intervention_id}/approve or /reject to
    resume the turn and receive the final answer.

    If the model does not call propose_intervention the response is identical to
    /chat with {"status": "done"}.
    """
    from src.agent.hitl_agent import start_hitl_turn

    client_obj = getattr(http_req.state, "api_client", None)
    initiated_by = client_obj.name if client_obj is not None else None

    try:
        result = start_hitl_turn(request.question, initiated_by=initiated_by)
    except Exception as exc:
        raise _agent_error_response(exc)

    if result["status"] == "awaiting_approval":
        iv = result["intervention"]
        return HITLChatResponse(
            status="awaiting_approval",
            thread_id=result["thread_id"],
            intervention=InterventionProposal(
                intervention_id=iv["intervention_id"],
                action=iv["action"],
                target=iv["target"],
                reason=iv["reason"],
                priority=iv["priority"],
            ),
        )

    resp = result["response"]
    return HITLChatResponse(
        status="done",
        thread_id=result["thread_id"],
        answer=resp.answer,
        tools_used=resp.tools_used,
        citations=[SourceCitation(kind=c["kind"], reference=c["reference"]) for c in resp.citations],
    )


@router.get("/interventions/{intervention_id}", response_model=InterventionStatusResponse)
def get_intervention_status(intervention_id: str):
    from src.tools.interventions import get_intervention
    record = get_intervention(intervention_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Intervention {intervention_id!r} not found")
    return InterventionStatusResponse(
        intervention_id=record.intervention_id,
        action=record.action,
        target=record.target,
        reason=record.reason,
        priority=record.priority,
        status=record.status,
        thread_id=record.thread_id,
        initiated_by=record.initiated_by,
        approved_by=record.approved_by,
        execution_note=record.execution_note,
    )


@router.post("/interventions/{intervention_id}/approve", response_model=InterventionResumeResponse)
def approve_intervention_endpoint(
    intervention_id: str,
    http_req: Request,
    _role: None = Depends(require_role("operator")),
):
    """Approve a pending intervention and resume the paused HITL agent turn.

    Requires the 'operator' role. The approver cannot be the same API client that
    initiated the intervention (separation of duties).
    The agent generates a final answer reflecting the approval outcome.
    """
    from src.tools.interventions import get_intervention, set_intervention_approved_by
    from src.agent.hitl_agent import resume_hitl_turn

    client_obj = getattr(http_req.state, "api_client", None)
    approved_by = client_obj.name if client_obj is not None else None

    record = get_intervention(intervention_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Intervention {intervention_id!r} not found")
    if record.status != "pending_approval":
        raise HTTPException(status_code=409, detail=f"Intervention is already {record.status!r}")
    if record.thread_id is None:
        raise HTTPException(status_code=409, detail="No active HITL session for this intervention")
    if approved_by and record.initiated_by and approved_by == record.initiated_by:
        raise HTTPException(status_code=403, detail="Cannot approve your own intervention")

    set_intervention_approved_by(intervention_id, approved_by)

    try:
        result = resume_hitl_turn(record.thread_id, decision="approved")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to resume agent turn: {exc}")

    resp = result["response"]
    return InterventionResumeResponse(
        intervention_id=intervention_id,
        decision="approved",
        answer=resp.answer,
        tools_used=resp.tools_used,
        citations=[SourceCitation(kind=c["kind"], reference=c["reference"]) for c in resp.citations],
    )


@router.post("/interventions/{intervention_id}/reject", response_model=InterventionResumeResponse)
def reject_intervention_endpoint(
    intervention_id: str,
    http_req: Request,
    body: RejectRequest = RejectRequest(),
    _role: None = Depends(require_role("operator")),
):
    """Reject a pending intervention and resume the paused HITL agent turn.

    Requires the 'operator' role.
    The agent generates a final answer reflecting the rejection.
    """
    from src.tools.interventions import get_intervention, set_intervention_approved_by
    from src.agent.hitl_agent import resume_hitl_turn

    client_obj = getattr(http_req.state, "api_client", None)
    rejected_by = client_obj.name if client_obj is not None else None

    record = get_intervention(intervention_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Intervention {intervention_id!r} not found")
    if record.status != "pending_approval":
        raise HTTPException(status_code=409, detail=f"Intervention is already {record.status!r}")
    if record.thread_id is None:
        raise HTTPException(status_code=409, detail="No active HITL session for this intervention")

    set_intervention_approved_by(intervention_id, rejected_by)

    decision = body.reason if body.reason else "rejected"
    try:
        result = resume_hitl_turn(record.thread_id, decision=decision)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to resume agent turn: {exc}")

    resp = result["response"]
    return InterventionResumeResponse(
        intervention_id=intervention_id,
        decision="rejected",
        answer=resp.answer,
        tools_used=resp.tools_used,
        citations=[SourceCitation(kind=c["kind"], reference=c["reference"]) for c in resp.citations],
    )


@router.post("/investigate", response_model=InvestigateResponse)
def investigate(request: InvestigateRequest):
    try:
        result = run_investigation(request.question)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Investigation failed: {exc}")

    r = result.report
    return InvestigateResponse(
        report=InvestigationReport(
            executive_summary=r.executive_summary,
            problem=r.problem,
            evidence=r.evidence,
            root_causes=r.root_causes,
            relevant_policy=[SourceCitation(kind="document", reference=p) for p in r.relevant_policy],
            recommendations=r.recommendations,
            limitations=r.limitations,
        ),
        tools_used=result.tools_used,
    )
