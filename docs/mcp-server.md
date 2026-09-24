# MCP Server

`src/mcp_server.py` exposes all 9 operations-assistant tools as a Model Context
Protocol server. Any MCP-compatible client (Claude Desktop, `mcp` CLI, MCP Inspector)
can call these tools directly without going through the built-in agent.

## Why this is not a duplicate of the agent

The agent in `src/agent/agent.py` wraps the same 9 tools in a conversational
question-answering loop driven by an LLM. The MCP server exposes the tools
directly — an MCP client decides when to call them and how to use the results,
rather than delegating that reasoning to this project's agent. Both paths share
the same underlying implementations in `src/tools/` and the same validation layer
(`src/tools/validation.py`), so the same input safety guarantees hold regardless
of which entry point is used.

## Tools exposed

| Tool | Description |
|---|---|
| `get_cycle_time` | Procurement cycle time stats (mean, median, p90), optionally by category |
| `get_bottlenecks` | Slowest process stages ranked by duration and delay share |
| `get_sla_metrics` | SLA breach rate and case counts, optionally segmented |
| `get_supplier_performance` | Supplier scorecard (cycle time, breach rate, order count) |
| `get_management_report` | Full Finding → Evidence → Impact → Recommendation report |
| `predict_sla_risk` | SLA-breach risk prediction for a specific case ID |
| `get_pipeline_status` | Recent data pipeline run status (freshness check) |
| `search_policy_documents` | Semantic search over the four procurement policy documents |
| `get_conformance` | Process conformance rate and deviation breakdown |

## Running the server

**stdio (default — for Claude Desktop and `mcp` CLI):**

```bash
python -m src.mcp_server
```

**HTTP/SSE (for remote or browser-based clients):**

```bash
python -m src.mcp_server --transport sse --port 8081
```

**Streamable HTTP (mcp 2.x preferred for remote):**

```bash
python -m src.mcp_server --transport streamable-http --port 8081
```

**MCP Inspector (stdio, interactive debugging):**

```bash
mcp dev src/mcp_server.py
```

## Claude Desktop configuration

Add this block to `~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "operations-assistant": {
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "cwd": "/path/to/operations-assistant",
      "env": {
        "OPS_PERFORMANCE_API_URL": "http://localhost:8000",
        "OPS_PERFORMANCE_API_KEY": "<your-key-if-set>"
      }
    }
  }
}
```

The `OPS_PERFORMANCE_API_URL` must point to a running instance of
`operations-performance`. The `OPS_PERFORMANCE_API_KEY` is only needed if that
project's `API_KEY` environment variable is set.

## Error handling

The MCP server converts two anticipated error types to `ToolError` (the MCP
typed error response) rather than letting them propagate as unexpected crashes:

| Exception | Source | What it means to a client |
|---|---|---|
| `ValueError` | `src/tools/validation.py` | Malformed input — invalid `case_id`, empty query, out-of-range integer |
| `OpsPerformanceUnavailable` | `src/tools/client.py` | Downstream API unreachable or returned an error |

Clients receive the error message in a typed MCP error response and can display
it or retry, rather than seeing an opaque internal crash.
