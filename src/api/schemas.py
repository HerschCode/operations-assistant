from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    ops_performance_api_reachable: bool
    vector_store_reachable: bool


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    conversation_id: str | None = Field(
        default=None,
        description="Reuse across turns for follow-up questions with context (see Phase 18).",
    )
    provider: str | None = Field(
        default=None,
        description="Override the server's default provider for this request: 'gemini', 'groq', or 'anthropic'. Demo endpoint only; ignored on authenticated /chat.",
    )


class SourceCitation(BaseModel):
    kind: str  # "document" | "data"
    reference: str  # e.g. "Procurement Policy, Section 4.2" or "GET /metrics/sla"


class ChatResponse(BaseModel):
    answer: str
    tools_used: list[str]
    citations: list[SourceCitation]
    conversation_id: str
    latency_ms: float | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class InvestigateRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class InvestigationReport(BaseModel):
    executive_summary: str
    problem: str
    evidence: list[str]
    root_causes: list[str]
    relevant_policy: list[SourceCitation]
    recommendations: list[str]
    limitations: str


class InvestigateResponse(BaseModel):
    report: InvestigationReport
    tools_used: list[str]


class DocumentUploadResponse(BaseModel):
    document_id: str
    title: str
    chunk_count: int


class DocumentListItem(BaseModel):
    document_id: str
    title: str
    version: str | None = None
    chunk_count: int


# ── HITL / intervention schemas ───────────────────────────────────────────────

class InterventionProposal(BaseModel):
    intervention_id: str
    action: str
    target: str
    reason: str
    priority: str


class HITLChatResponse(BaseModel):
    """Response from POST /chat/hitl.

    status="awaiting_approval" when the graph paused at propose_intervention;
    the caller must POST /interventions/{intervention_id}/approve|reject to continue.
    status="done" when the graph completed without interruption.
    """
    status: str   # "awaiting_approval" | "done"
    thread_id: str
    # Populated when status="awaiting_approval"
    intervention: InterventionProposal | None = None
    # Populated when status="done"
    answer: str | None = None
    tools_used: list[str] | None = None
    citations: list[SourceCitation] | None = None


class InterventionStatusResponse(BaseModel):
    intervention_id: str
    action: str
    target: str
    reason: str
    priority: str
    status: str
    thread_id: str | None = None
    initiated_by: str | None = None
    approved_by: str | None = None
    execution_note: str | None = None


class InterventionResumeResponse(BaseModel):
    """Response from POST /interventions/{id}/approve|reject.

    Returns the final agent answer after the graph resumes.
    """
    intervention_id: str
    decision: str   # "approved" | "rejected"
    answer: str
    tools_used: list[str]
    citations: list[SourceCitation]


class RejectRequest(BaseModel):
    reason: str = ""
