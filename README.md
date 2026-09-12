# Operations Assistant

![Tests](https://github.com/HerschCode/operations-assistant/actions/workflows/test.yml/badge.svg)

**🔗 [Live demo](https://operations-assistant.onrender.com)** — ask a real question, get a real
answer from a real Groq-backed agent over live data and real policy documents. Free-tier hosting,
so the first request after idling may take 30-60s to wake up.

An operations investigation assistant for **Northstar Manufacturing** that combines enterprise
documents (policies/SOPs), operational data, process analytics, and predictive SLA-risk scores to
answer business questions and conduct evidence-based investigations — grounded, cited, and
willing to say "insufficient data" rather than guess.

Consumes the data model and analytics built in [`operations-performance`](../operations-performance)
via a small set of controlled tools rather than re-implementing that logic.

## Stack
Python · FastAPI · PostgreSQL · RAG (ChromaDB + hybrid BM25/semantic retrieval) ·
LangChain · Groq / Anthropic / Gemini · Docker · Cloud Run · GCP

Containerised; deployable as a Kubernetes `Deployment` behind a `ClusterIP` `Service` — the
one stateful piece (`data/conversations.db`, SQLite) is the reason a real cluster deployment
would swap that for a shared store first, noted here rather than glossed over.

## Agent orchestration
Four interchangeable providers behind one interface (`config["provider"]`, switchable
via `AGENT_PROVIDER` without touching code): Anthropic, Groq, Gemini, and
**LangChain** (`src/agent/langchain_agent.py`) — the LangChain path wraps this
project's real tools as `StructuredTool` objects (schema inferred from the actual
function signatures, not a hand-copied duplicate) and drives them through
`ChatGroq.bind_tools()`. Every provider calls the exact same underlying tool
functions and returns the same response shape, so switching orchestration layers
never changes what a tool actually does.

## Architecture
![Architecture diagram](docs/architecture.svg)

## Status
See [`FEATURES.md`](FEATURES.md) for the full, tiered feature specification and definition of done.

## Running locally
```
cp .env.example .env
pip install -r requirements.txt
docker compose up -d          # local vector DB / deps
python -m scripts.index_documents
python -m scripts.run_server
```
