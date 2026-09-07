import os
from dotenv import load_dotenv
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router, health_router
from src.api.middleware import RequestLoggingMiddleware
from src.api.auth import require_api_key
from src.observability.logging_config import configure_logging

load_dotenv()
configure_logging()

app = FastAPI(
    title="Operations Assistant API",
    description=(
        "Investigation assistant over Northstar Manufacturing's operational data and "
        "policy documents. Consumes operations-performance's API as agent tools rather "
        "than reimplementing any analytics -- one analytics engine, two front doors."
    ),
    version="0.1.0",
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
