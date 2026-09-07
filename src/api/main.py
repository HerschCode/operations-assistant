import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router, health_router
from src.api.middleware import RequestLoggingMiddleware
from src.api.auth import require_api_key
from src.observability.logging_config import configure_logging, get_logger

load_dotenv()
configure_logging()
logger = get_logger("startup")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Self-heals the vector index on a fresh/ephemeral filesystem -- a PaaS host with
    # no persistent disk (Render's free tier, for instance) wipes data/chroma on every
    # restart or redeploy, so a one-time manual `python -m scripts.index_documents`
    # step doesn't survive past the first restart. Re-indexing 4 small documents costs
    # a couple seconds at startup; silently serving an empty vector store after every
    # restart would be a much worse failure mode than that small one-time cost.
    from src.ingestion.index_documents import index_all_documents, COLLECTION_NAME
    from src.retrieval.vector_store import get_collection
    if get_collection(COLLECTION_NAME).count() == 0:
        logger.info("Vector store empty at startup -- indexing documents")
        results = index_all_documents()
        logger.info(
            "Startup indexing complete",
            extra={"documents_indexed": len(results), "total_chunks": sum(r.chunk_count for r in results)},
        )
    yield


app = FastAPI(
    title="Operations Assistant API",
    description=(
        "Investigation assistant over Northstar Manufacturing's operational data and "
        "policy documents. Consumes operations-performance's API as agent tools rather "
        "than reimplementing any analytics -- one analytics engine, two front doors."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("API_ALLOWED_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(router, dependencies=[Depends(require_api_key)])


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("API_PORT", 8001))
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=port, reload=True)
