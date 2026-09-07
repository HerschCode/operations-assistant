"""
Single place the agent (Phase 17) imports from to get every available tool's schema
and implementation. Adding a new tool means adding it to ALL_TOOLS here, not hunting
through the agent's code for where tools get wired in.
"""
from src.tools.analytics import ALL_ANALYTICS_TOOLS
from src.tools.prediction import ALL_PREDICTION_TOOLS
from src.tools.database import ALL_PIPELINE_TOOLS
from src.tools.documents import ALL_DOCUMENT_TOOLS
from src.tools.process import ALL_PROCESS_TOOLS

ALL_TOOLS = (
    ALL_ANALYTICS_TOOLS
    + ALL_PREDICTION_TOOLS
    + ALL_PIPELINE_TOOLS
    + ALL_DOCUMENT_TOOLS
    + ALL_PROCESS_TOOLS  # currently empty -- see process.py
)

TOOL_SCHEMAS = [schema for schema, _ in ALL_TOOLS]
TOOL_FUNCTIONS = {schema["name"]: fn for schema, fn in ALL_TOOLS}


def call_tool(name: str, **kwargs):
    if name not in TOOL_FUNCTIONS:
        raise ValueError(f"Unknown tool: {name}")
    return TOOL_FUNCTIONS[name](**kwargs)


if __name__ == "__main__":
    print(f"{len(ALL_TOOLS)} tools registered:")
    for schema in TOOL_SCHEMAS:
        print(f"  {schema['name']}: {schema['description'][:80]}")
