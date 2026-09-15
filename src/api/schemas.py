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
