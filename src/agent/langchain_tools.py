"""
Wraps this project's EXISTING tools (src/tools/registry.py's ALL_TOOLS -- the same
functions src/agent/providers.py's hand-rolled loops call) as LangChain
StructuredTool objects, for src/agent/langchain_agent.py. Deliberately not a second
implementation of get_cycle_time/search_policy_documents/etc. -- every tool call
made through the LangChain path executes the exact same function as every other
provider, so there's one source of truth for what each tool does regardless of
which orchestration layer is calling it.

infer_schema=True (StructuredTool's default) builds each tool's argument schema
directly from the real function's own type hints (e.g. get_bottlenecks(top_n:
int = 10)), not from a hand-converted copy of TOOL_SCHEMAS's JSON schema -- one
less place for the two representations to drift apart.
"""
from langchain_core.tools import StructuredTool

from src.tools.registry import ALL_TOOLS


def build_langchain_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            func=fn, name=schema["name"], description=schema["description"],
        )
        for schema, fn in ALL_TOOLS
    ]
