"""
Write one JSON line per agent invocation to logs/agent_traces.jsonl.
Consumed by /eval/recent to report online production eval stats without
needing a database — the file stays small because the demo rate-limiter
(5 req / 10 min / IP) caps total write volume.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

_LOG_PATH = Path(os.getenv("TRACE_LOG_PATH", "logs/agent_traces.jsonl"))


def log_trace(
    *,
    tools_used: list[str],
    num_citations: int,
    latency_ms: float,
    answered: bool,
    model: str | None,
    cost_usd: float | None = None,
) -> None:
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "tools_used": tools_used,
        "num_tools": len(tools_used),
        "num_citations": num_citations,
        "latency_ms": latency_ms,
        "answered": answered,
        "model": model,
        "cost_usd": cost_usd,
    }
    try:
        with _LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except OSError:
        pass  # never crash a user request over a logging failure


def read_recent(n: int = 200) -> list[dict]:
    if not _LOG_PATH.exists():
        return []
    lines = _LOG_PATH.read_text(encoding="utf-8").splitlines()
    records = []
    for line in lines[-n:]:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records
