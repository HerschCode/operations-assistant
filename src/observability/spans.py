"""
OpenTelemetry span helpers for the operations-assistant pipeline.

Architecture: each key operation (retrieve, rerank, LLM call, tool call, gate)
gets its own span with relevant attributes (token counts, cost, latency, model).
Spans nest naturally: the agent span contains tool-call spans; grounded_answer
spans contain retrieve + gate spans.

Backend selection (OTEL_EXPORTER env var, or defaults):
  "phoenix"  — send to a local Arize Phoenix instance
               (start with: python -m phoenix.server.main serve)
  "otlp"     — OTLP/gRPC to OTEL_EXPORTER_OTLP_ENDPOINT
  "console"  — print spans to stdout (development)
  "none"     — no-op (default when no exporter configured)

Install optional deps:
  pip install arize-phoenix openinference-instrumentation opentelemetry-sdk
              opentelemetry-exporter-otlp-proto-grpc

All helpers degrade gracefully when opentelemetry is not installed or no
exporter is configured — the wrapped code runs unchanged.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

_TRACER = None
_ENABLED = False
_SETUP_DONE = False


def _setup() -> None:
    global _TRACER, _ENABLED, _SETUP_DONE
    if _SETUP_DONE:
        return
    _SETUP_DONE = True

    exporter_name = os.environ.get("OTEL_EXPORTER", "none").lower()
    if exporter_name == "none":
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider()

        if exporter_name == "phoenix":
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:4317")
            exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        elif exporter_name == "otlp":
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
            exporter = OTLPSpanExporter(endpoint=endpoint)
        elif exporter_name == "console":
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            exporter = ConsoleSpanExporter()
        else:
            return

        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _TRACER = trace.get_tracer("operations-assistant", "1.0")
        _ENABLED = True
    except ImportError:
        pass  # opentelemetry or exporter not installed — run without tracing


@contextmanager
def retrieve_span(question: str, top_k: int, method: str = "hybrid") -> Generator:
    """Span for the retrieval step (BM25 + semantic + optional rerank)."""
    _setup()
    if not _ENABLED or _TRACER is None:
        yield None
        return
    with _TRACER.start_as_current_span("retrieve") as span:
        span.set_attribute("question", question[:500])
        span.set_attribute("top_k", top_k)
        span.set_attribute("method", method)
        yield span


@contextmanager
def rerank_span(question: str, n_candidates: int) -> Generator:
    """Span for the cross-encoder reranking step."""
    _setup()
    if not _ENABLED or _TRACER is None:
        yield None
        return
    with _TRACER.start_as_current_span("rerank") as span:
        span.set_attribute("question", question[:500])
        span.set_attribute("n_candidates", n_candidates)
        yield span


@contextmanager
def llm_call_span(model: str, prompt_tokens: int = 0, completion_tokens: int = 0, cost_usd: float = 0.0) -> Generator:
    """Span for a single LLM API call."""
    _setup()
    if not _ENABLED or _TRACER is None:
        yield None
        return
    with _TRACER.start_as_current_span("llm_call") as span:
        span.set_attribute("model", model)
        span.set_attribute("prompt_tokens", prompt_tokens)
        span.set_attribute("completion_tokens", completion_tokens)
        span.set_attribute("cost_usd", cost_usd)
        yield span


@contextmanager
def tool_call_span(name: str, args: dict | None = None) -> Generator:
    """Span for a single tool invocation."""
    _setup()
    if not _ENABLED or _TRACER is None:
        yield None
        return
    with _TRACER.start_as_current_span(f"tool:{name}") as span:
        span.set_attribute("tool.name", name)
        if args:
            for k, v in args.items():
                span.set_attribute(f"tool.arg.{k}", str(v)[:256])
        yield span


@contextmanager
def gate_span(faithfulness_score: float, gated: bool, threshold: float = 0.05) -> Generator:
    """Span for the NLI faithfulness gate decision."""
    _setup()
    if not _ENABLED or _TRACER is None:
        yield None
        return
    with _TRACER.start_as_current_span("faithfulness_gate") as span:
        span.set_attribute("faithfulness_score", faithfulness_score)
        span.set_attribute("gated", gated)
        span.set_attribute("threshold", threshold)
        yield span


@contextmanager
def agent_span(question: str, provider: str, model: str) -> Generator:
    """Root span for a full agent invocation."""
    _setup()
    if not _ENABLED or _TRACER is None:
        yield None
        return
    with _TRACER.start_as_current_span("agent") as span:
        span.set_attribute("question", question[:500])
        span.set_attribute("provider", provider)
        span.set_attribute("model", model)
        yield span
