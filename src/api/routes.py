from fastapi import APIRouter, HTTPException, UploadFile, File
import uuid
import tempfile
from pathlib import Path

from src.api.schemas import (
    HealthResponse, DocumentUploadResponse, DocumentListItem,
    ChatRequest, ChatResponse, SourceCitation,
    InvestigateRequest, InvestigateResponse, InvestigationReport,
)
from src.api.dependencies import check_ops_performance_reachable, check_vector_store_reachable
from src.ingestion.document_loader import load_document
from src.ingestion.chunker import chunk_text
from src.ingestion.index_documents import index_document, list_indexed_documents
from src.agent.agent import run_agent
from src.agent.investigation import run_investigation
from src.agent.conversation_store import get_history, append_turn

router = APIRouter()

# /health is intentionally on an unauthenticated router -- same reasoning as
# operations-performance's src/api/auth.py docstring (load balancer/orchestrator
# liveness probes conventionally don't carry a credential).
health_router = APIRouter()


@health_router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        ops_performance_api_reachable=check_ops_performance_reachable(),
        vector_store_reachable=check_vector_store_reachable(),
    )


@router.get("/documents", response_model=list[DocumentListItem])
def list_documents():
    return list_indexed_documents()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """conversation_id, if provided, replays prior turns' text (not tool-call
    scaffolding -- see conversation_store.py) as context, then persists this turn
    back to the store. A fresh conversation_id is generated and returned if none was
    given, so a client can start a new conversation with nothing and continue it on
    subsequent calls by passing back what /chat returned."""
    conversation_id = request.conversation_id or str(uuid.uuid4())[:8]
    history = get_history(conversation_id)

    try:
        result = run_agent(request.question, history=history)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Agent failed to produce a response: {exc}")

    append_turn(conversation_id, request.question, result.answer)

    return ChatResponse(
        answer=result.answer,
        tools_used=result.tools_used,
        citations=[SourceCitation(kind=c["kind"], reference=c["reference"]) for c in result.citations],
        conversation_id=conversation_id,
    )


ALLOWED_UPLOAD_SUFFIXES = {".md", ".pdf", ".txt"}
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5MB -- generous for a policy document, small enough to reject an accidental wrong-file upload fast


@router.post("/documents", response_model=DocumentUploadResponse)
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
