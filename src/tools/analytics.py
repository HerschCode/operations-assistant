"""
Wraps operations-performance's read-only analytics endpoints as agent tools. The agent
never sees raw SQL or a direct DB connection -- only these named, parameterized
functions, each with a fixed result-row cap (config/tools.yaml) so a single tool call
can't return an unbounded amount of data into the agent's context.
"""
from src.tools.client import get, get_text
from src.tools.validation import clamp_int

GET_CYCLE_TIME_SCHEMA = {
    "name": "get_cycle_time",
    "description": "Get procurement cycle time statistics (mean, median, p90). Omit segment for the overall rate; pass segment='category' for a per-procurement-category breakdown.",
    "input_schema": {
        "type": "object",
        "properties": {
            "segment": {
                "type": "string",
                "description": "Optional column to segment by. Use 'category' for a per-category breakdown. Omit for overall stats.",
            }
        },
        "required": [],
    },
}


def get_cycle_time(segment: str | None = None) -> dict | list:
    if segment:
        return get("/metrics/cycle-time", params={"segment": segment})
    return get("/metrics/cycle-time")


GET_BOTTLENECKS_SCHEMA = {
    "name": "get_bottlenecks",
    "description": "Get the slowest process stages (bottlenecks), ranked by average duration and percent of total delay.",
    "input_schema": {
        "type": "object",
        "properties": {
            "top_n": {"type": "integer", "description": "Number of stages to return, default 10, max 50.", "default": 10}
        },
        "required": [],
    },
}


def get_bottlenecks(top_n: int = 10) -> list:
    # config/tools.yaml documents max_result_rows: 10 for this tool -- previously that
    # was descriptive only and never actually enforced in code. Clamped here now, so a
    # model-supplied top_n of, say, 10000 can't flood the agent's context.
    top_n = clamp_int(top_n, max_value=50)
    return get("/metrics/bottlenecks", params={"top_n": top_n})


GET_SLA_METRICS_SCHEMA = {
    "name": "get_sla_metrics",
    "description": "Get SLA breach rate and case counts, optionally broken down by a segment (e.g. 'category').",
    "input_schema": {
        "type": "object",
        "properties": {
            "segment": {
                "type": "string",
                "description": "Optional column to segment by, e.g. 'category'. Omit for the overall rate.",
            }
        },
        "required": [],
    },
}


def get_sla_metrics(segment: str | None = None) -> list:
    params = {"segment": segment} if segment else None
    return get("/metrics/sla", params=params)


GET_SUPPLIER_PERFORMANCE_SCHEMA = {
    "name": "get_supplier_performance",
    "description": "Get the supplier performance scorecard (cycle time, SLA breach rate, order count) for suppliers with enough order volume to rank reliably. Returns the top suppliers by SLA breach rate.",
    "input_schema": {
        "type": "object",
        "properties": {
            "min_volume": {"type": "integer", "description": "Minimum order count to include a supplier, default 5.", "default": 5},
            "top_n": {"type": "integer", "description": "Return only the top N suppliers by SLA breach rate, default 15, max 50.", "default": 15},
        },
        "required": [],
    },
}


def get_supplier_performance(min_volume: int = 5, top_n: int = 15) -> list:
    min_volume = clamp_int(min_volume, max_value=1000, min_value=1)
    top_n = clamp_int(top_n, max_value=50, min_value=1)
    rows: list = get("/suppliers/performance", params={"min_volume": min_volume})
    rows.sort(key=lambda r: r.get("sla_breach_rate") or 0.0, reverse=True)
    return rows[:top_n]


GET_MANAGEMENT_REPORT_SCHEMA = {
    "name": "get_management_report",
    "description": "Get the current Finding -> Evidence -> Impact -> Recommendation management report, generated fresh from live data.",
    "input_schema": {"type": "object", "properties": {}, "required": []},
}


def get_management_report() -> str:
    return get_text("/reports/management")


ALL_ANALYTICS_TOOLS = [
    (GET_CYCLE_TIME_SCHEMA, get_cycle_time),
    (GET_BOTTLENECKS_SCHEMA, get_bottlenecks),
    (GET_SLA_METRICS_SCHEMA, get_sla_metrics),
    (GET_SUPPLIER_PERFORMANCE_SCHEMA, get_supplier_performance),
    (GET_MANAGEMENT_REPORT_SCHEMA, get_management_report),
]
