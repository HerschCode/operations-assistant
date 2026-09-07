"""
The RAG-backed tool -- calls straight into src/retrieval/search.py (Phase 15), not
operations-performance's API, since this is the one tool that answers from this
project's own document corpus rather than the companion project's data.
"""
from src.retrieval.search import semantic_search, no_relevant_results_response
from src.tools.validation import validate_query

SEARCH_POLICY_DOCUMENTS_SCHEMA = {
    "name": "search_policy_documents",
    "description": (
        "Search the company's policy and procedure documents (Procurement Policy, SLA Policy, "
        "Escalation Procedure, Exception Handling Procedure) for relevant sections. Use this for "
        "questions about rules, approval requirements, or procedures -- not for questions about "
        "actual operational numbers, which need get_cycle_time / get_sla_metrics / etc. instead."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for, in natural language."}
        },
        "required": ["query"],
    },
}


def search_policy_documents(query: str) -> dict:
    query = validate_query(query)
    results = semantic_search(query)
    if not results:
        return {"found": False, "message": no_relevant_results_response(), "results": []}

    return {
        "found": True,
        "results": [
            {"citation": r.citation, "text": r.text, "similarity_score": r.similarity_score}
            for r in results
        ],
    }


ALL_DOCUMENT_TOOLS = [(SEARCH_POLICY_DOCUMENTS_SCHEMA, search_policy_documents)]
