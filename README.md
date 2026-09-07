# Operations Assistant

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
Python · FastAPI · PostgreSQL · RAG (embeddings + vector DB) · LLM tool-calling / agent
orchestration · Docker · Cloud Run · GCP

## Architecture
```
                    User
                     │
                     ▼
                  FastAPI
                     │
                   Agent
                     │
       ┌─────────────┼──────────────┐
       ▼             ▼              ▼
     RAG tool     SQL/analytics   Prediction tool
       │            tools              │
       ▼             ▼                 ▼
   Documents    operations-performance data/logic
       │             │                 │
       └─────────────┼─────────────────┘
                     ▼
              Grounded, cited answer
              (or "insufficient data")
```

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
