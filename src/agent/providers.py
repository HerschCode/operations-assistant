"""
Groq and Gemini implementations of the same tool-selection loop `_run_agent_anthropic`
(src/agent/agent.py) runs for Anthropic. Kept in a separate module rather than branching
inside agent.py's loop because each SDK returns a differently-shaped response (OpenAI-style
`choices[0].message.tool_calls` for Groq vs. Gemini's `candidates[0].content.parts` with
`function_call`/`function_response` parts) and expects a differently-shaped tool schema --
forcing all three into one function body would trade a little duplication for a lot of
provider-conditional branching throughout what is currently a clean, linear loop.

Both providers reuse agent.py's `TOOL_SCHEMAS` (Anthropic's `input_schema` shape) as the
single source of truth for what a tool accepts, converting it to each SDK's own schema
format here rather than maintaining three parallel tool definitions that could drift.

Both providers reuse agent.py's `AgentResponse`/`ToolCallRecord`/`_extract_citations`/
`call_tool` so tool bookkeeping, citation extraction, and the response shape returned to
the API layer are identical regardless of which model answered the question -- only the
API round-trip and message formatting differ.
"""
import json
import os
import time
import uuid

from src.tools.registry import TOOL_SCHEMAS, call_tool
from src.tools.client import OpsPerformanceUnavailable
from src.agent.prompts import SYSTEM_PROMPT
from src.observability.logging_config import get_logger

logger = get_logger("agent")


def _make_optional_params_nullable(input_schema: dict) -> dict:
    """Found by a real live agent evaluation run, not by inspection: Groq's model
    sometimes emits `null` for an OMITTED optional parameter (e.g.
    `{"segment": null}` for get_sla_metrics's optional segment argument) instead of
    leaving the key out entirely -- and Groq's own server-side schema validation
    then rejects that same output against a strict `"type": "string"` schema,
    producing a `tool_use_failed` 400 error the agent has no way to recover from.
    Anthropic's schema format (this project's source of truth, see TOOL_SCHEMAS)
    doesn't have this problem since it doesn't validate tool-call arguments against
    the schema server-side the way Groq does. Fixed here, in the OpenAI/Groq
    conversion specifically, by widening every non-required property's type to
    also accept null -- Anthropic and Gemini's conversions are untouched since
    neither exhibited this failure."""
    import copy
    schema = copy.deepcopy(input_schema)
    required = set(schema.get("required", []))
    for prop_name, prop_schema in schema.get("properties", {}).items():
        if prop_name in required:
            continue
        prop_type = prop_schema.get("type")
        if isinstance(prop_type, str) and prop_type != "null":
            prop_schema["type"] = [prop_type, "null"]
    return schema


def _tool_schemas_to_openai(tool_schemas: list[dict]) -> list[dict]:
    """Anthropic's {name, description, input_schema} -> OpenAI/Groq's
    {type: "function", function: {name, description, parameters}}."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": _make_optional_params_nullable(t["input_schema"]),
            },
        }
        for t in tool_schemas
    ]


def _tool_schemas_to_gemini(tool_schemas: list[dict]) -> list[dict]:
    """Anthropic's {name, description, input_schema} -> Gemini's function_declarations
    {name, description, parameters}. Gemini's schema is close enough to JSON Schema that
    input_schema can be passed through as-is, same as the OpenAI conversion above."""
    return [
        {
            "function_declarations": [
                {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}
                for t in tool_schemas
            ]
        }
    ]


def _log_real_cost(turn_id: str, model: str, prompt_tokens: int, completion_tokens: int) -> None:
    """Closes the FUTURE_IMPROVEMENTS.md gap named plainly: "the cost estimator
    exists and works; it has no real spend numbers to report yet since no live
    model has been called." Now one has -- this logs a real cost figure computed
    from real token counts (response.usage), not the estimate_tokens() heuristic.
    Failure here (e.g. model not in config/pricing.yaml) is swallowed to a debug
    log rather than breaking the actual turn -- cost logging is observability, not
    something that should ever fail a real user's question."""
    try:
        from src.evaluation.cost_estimator import calculate_cost
        cost = calculate_cost(
            input_tokens=prompt_tokens, output_tokens=completion_tokens,
            model=model, is_estimated_token_count=False,
        )
        logger.info(
            "real cost for this turn",
            extra={
                "turn_id": turn_id, "model": model,
                "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                "total_cost_usd": cost.total_cost_usd,
            },
        )
    except Exception as exc:
        logger.debug("cost logging skipped", extra={"turn_id": turn_id, "reason": str(exc)})


# --- Groq (OpenAI-compatible chat.completions API) -------------------------------------

def _default_groq_client():
    from groq import Groq
    return Groq(api_key=os.environ["GROQ_API_KEY"])


def run_agent_groq(question: str, config: dict, client=None, history: list[dict] | None = None, event_cb=None):
    from src.agent.agent import AgentResponse, ToolCallRecord, _extract_citations

    client = client or _default_groq_client()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": question})

    tools = _tool_schemas_to_openai(TOOL_SCHEMAS)
    tool_calls: list[ToolCallRecord] = []
    max_rounds = config["max_tool_calls_per_turn"]
    turn_id = str(uuid.uuid4())[:8]
    turn_start = time.monotonic()
    # A multi-tool turn calls the model more than once (one round per tool-call
    # cycle) -- real cost/usage tracking needs the SUM across every round this
    # turn made, not just the last one, or a 3-round investigation would silently
    # undercount its real token spend by 2/3.
    total_prompt_tokens = 0
    total_completion_tokens = 0

    logger.info("agent turn started", extra={"turn_id": turn_id, "provider": "groq", "question_length": len(question)})

    for round_num in range(max_rounds + 1):
        api_call_start = time.monotonic()
        response = client.chat.completions.create(
            model=config["model"],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
            messages=messages,
            tools=tools,
        )
        api_call_duration_ms = round((time.monotonic() - api_call_start) * 1000, 1)
        if response.usage is not None:
            total_prompt_tokens += response.usage.prompt_tokens
            total_completion_tokens += response.usage.completion_tokens
        message = response.choices[0].message
        requested_tool_calls = message.tool_calls or []

        # Groq's assistant message must be echoed back verbatim (including raw
        # tool_calls) for the follow-up round to be valid -- unlike Anthropic's content
        # blocks, OpenAI-shaped messages are dicts, so this is built explicitly rather
        # than re-using the SDK object.
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in requested_tool_calls
            ] or None,
        })

        logger.info(
            "model round complete",
            extra={
                "turn_id": turn_id, "round": round_num, "api_call_duration_ms": api_call_duration_ms,
                "tools_requested": [tc.function.name for tc in requested_tool_calls],
            },
        )

        if not requested_tool_calls:
            tools_used = list(dict.fromkeys(tc.name for tc in tool_calls))
            _log_real_cost(turn_id, config["model"], total_prompt_tokens, total_completion_tokens)
            logger.info(
                "agent turn complete",
                extra={
                    "turn_id": turn_id, "total_duration_ms": round((time.monotonic() - turn_start) * 1000, 1),
                    "rounds": round_num + 1, "tools_used": tools_used, "budget_exceeded": False,
                },
            )
            return AgentResponse(
                answer=message.content or "", tool_calls=tool_calls, tools_used=tools_used,
                citations=_extract_citations(tool_calls),
                prompt_tokens=total_prompt_tokens, completion_tokens=total_completion_tokens,
            )

        if round_num == max_rounds:
            tools_used = list(dict.fromkeys(tc.name for tc in tool_calls))
            _log_real_cost(turn_id, config["model"], total_prompt_tokens, total_completion_tokens)
            logger.warning(
                "agent turn hit tool-call budget",
                extra={
                    "turn_id": turn_id, "total_duration_ms": round((time.monotonic() - turn_start) * 1000, 1),
                    "rounds": round_num + 1, "tools_used": tools_used, "budget_exceeded": True,
                },
            )
            return AgentResponse(
                answer=(
                    "I gathered some information but reached the tool-call limit for this "
                    "question before I could fully answer it. Here's what I found so far, "
                    "though it may be incomplete."
                ),
                tool_calls=tool_calls, tools_used=tools_used,
                citations=_extract_citations(tool_calls), budget_exceeded=True,
                prompt_tokens=total_prompt_tokens, completion_tokens=total_completion_tokens,
            )

        for tc in requested_tool_calls:
            t0 = time.monotonic()
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            if event_cb:
                event_cb({"type": "tool_start", "tool": tc.function.name})
            try:
                result = call_tool(tc.function.name, **args)
                duration_ms = round((time.monotonic() - t0) * 1000, 1)
                tool_calls.append(ToolCallRecord(name=tc.function.name, input=args, result=result))
                messages.append({
                    "role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, default=str),
                })
                if event_cb:
                    event_cb({"type": "tool_done", "tool": tc.function.name, "ok": True})
                logger.info(
                    "tool call succeeded",
                    extra={"turn_id": turn_id, "tool_name": tc.function.name, "duration_ms": duration_ms},
                )
            except (OpsPerformanceUnavailable, Exception) as exc:
                duration_ms = round((time.monotonic() - t0) * 1000, 1)
                tool_calls.append(ToolCallRecord(name=tc.function.name, input=args, error=str(exc)))
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": f"Error: {exc}"})
                if event_cb:
                    event_cb({"type": "tool_done", "tool": tc.function.name, "ok": False, "error": str(exc)})
                logger.warning(
                    "tool call failed",
                    extra={"turn_id": turn_id, "tool_name": tc.function.name, "duration_ms": duration_ms, "error": str(exc)},
                )

    raise RuntimeError("Agent loop exited without producing a response")


# --- Gemini (google-generativeai function-calling API) ---------------------------------

def _default_gemini_client(model: str, history: list[dict] | None):
    # google-genai (the current, non-deprecated SDK -- google-generativeai was
    # end-of-life'd and warns on import) exposes a chat session the same shape this
    # loop already needs: `.send_message(content) -> response` with
    # `response.candidates[0].content.parts`, each part exposing `.text` or
    # `.function_call` (itself `.name`/`.args`). Building this here (rather than at
    # call time in run_agent_gemini) keeps that function's only real-vs-fake branch
    # to "was a client passed in", identical to the Anthropic and Groq paths.
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    config_obj = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[types.Tool(function_declarations=_tool_schemas_to_gemini(TOOL_SCHEMAS)[0]["function_declarations"])],
    )
    chat = client.chats.create(model=model, config=config_obj, history=history or [])
    # google-genai's Client owns the underlying httpx session and closes it in
    # __del__ -- the chat session returned here only holds a reference to the
    # client's internal _api_client, not to `client` itself, so without this the
    # outer Client object is unreferenced the moment this function returns and
    # gets garbage-collected before the chat's first real request completes,
    # raising "Cannot send a request, as the client has been closed." on live
    # calls (never caught by tests, since they always inject a fake client).
    # Keeping a strong reference on the chat object for its lifetime fixes this.
    chat._gateway_client_keepalive = client
    return chat


def _gemini_function_response_parts(response_parts: list[dict]):
    """Converts this module's plain-dict {"function_response": {...}} shape (used so
    tests don't need the google.genai package installed to exercise the loop logic)
    into real google.genai.types.Part objects for an actual API call."""
    from google.genai import types
    return [
        types.Part.from_function_response(
            name=p["function_response"]["name"], response=p["function_response"]["response"],
        )
        for p in response_parts
    ]


def run_agent_gemini(question: str, config: dict, client=None, history: list[dict] | None = None):
    """`client` here is a started chat session (an object with `.send_message(content)`
    returning a Gemini response) so tests can inject a fake one the same way
    tests/test_agent.py injects a fake Anthropic client -- constructing the real
    google-genai chat session is left to `_default_gemini_client` so a test never needs
    a real API key or the google-genai package's actual types to exercise the loop logic."""
    from src.agent.agent import AgentResponse, ToolCallRecord, _extract_citations

    real_client = client is None
    if real_client:
        client = _default_gemini_client(config["model"], history)

    tool_calls: list[ToolCallRecord] = []
    max_rounds = config["max_tool_calls_per_turn"]
    turn_id = str(uuid.uuid4())[:8]
    turn_start = time.monotonic()
    next_message = question

    logger.info("agent turn started", extra={"turn_id": turn_id, "provider": "gemini", "question_length": len(question)})

    for round_num in range(max_rounds + 1):
        api_call_start = time.monotonic()
        response = client.send_message(next_message)
        api_call_duration_ms = round((time.monotonic() - api_call_start) * 1000, 1)

        parts = response.candidates[0].content.parts
        function_calls = [p.function_call for p in parts if getattr(p, "function_call", None)]
        text = "".join(p.text for p in parts if getattr(p, "text", None))

        logger.info(
            "model round complete",
            extra={
                "turn_id": turn_id, "round": round_num, "api_call_duration_ms": api_call_duration_ms,
                "tools_requested": [fc.name for fc in function_calls],
            },
        )

        if not function_calls:
            tools_used = list(dict.fromkeys(tc.name for tc in tool_calls))
            logger.info(
                "agent turn complete",
                extra={
                    "turn_id": turn_id, "total_duration_ms": round((time.monotonic() - turn_start) * 1000, 1),
                    "rounds": round_num + 1, "tools_used": tools_used, "budget_exceeded": False,
                },
            )
            return AgentResponse(
                answer=text, tool_calls=tool_calls, tools_used=tools_used,
                citations=_extract_citations(tool_calls),
            )

        if round_num == max_rounds:
            tools_used = list(dict.fromkeys(tc.name for tc in tool_calls))
            logger.warning(
                "agent turn hit tool-call budget",
                extra={
                    "turn_id": turn_id, "total_duration_ms": round((time.monotonic() - turn_start) * 1000, 1),
                    "rounds": round_num + 1, "tools_used": tools_used, "budget_exceeded": True,
                },
            )
            return AgentResponse(
                answer=(
                    "I gathered some information but reached the tool-call limit for this "
                    "question before I could fully answer it. Here's what I found so far, "
                    "though it may be incomplete."
                ),
                tool_calls=tool_calls, tools_used=tools_used,
                citations=_extract_citations(tool_calls), budget_exceeded=True,
            )

        # Gemini expects every function_call in this turn answered with a matching
        # function_response part in a single follow-up message, not one message per call.
        response_parts = []
        for fc in function_calls:
            t0 = time.monotonic()
            args = dict(fc.args) if fc.args else {}
            try:
                result = call_tool(fc.name, **args)
                duration_ms = round((time.monotonic() - t0) * 1000, 1)
                tool_calls.append(ToolCallRecord(name=fc.name, input=args, result=result))
                response_parts.append({"function_response": {"name": fc.name, "response": {"result": result}}})
                logger.info(
                    "tool call succeeded",
                    extra={"turn_id": turn_id, "tool_name": fc.name, "duration_ms": duration_ms},
                )
            except (OpsPerformanceUnavailable, Exception) as exc:
                duration_ms = round((time.monotonic() - t0) * 1000, 1)
                tool_calls.append(ToolCallRecord(name=fc.name, input=args, error=str(exc)))
                response_parts.append({"function_response": {"name": fc.name, "response": {"error": str(exc)}}})
                logger.warning(
                    "tool call failed",
                    extra={"turn_id": turn_id, "tool_name": fc.name, "duration_ms": duration_ms, "error": str(exc)},
                )
        next_message = _gemini_function_response_parts(response_parts) if real_client else response_parts

    raise RuntimeError("Agent loop exited without producing a response")
