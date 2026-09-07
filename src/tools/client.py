"""
Shared HTTP client for calling operations-performance's API. Every tool goes through
this rather than each file rolling its own httpx call -- one place to control timeout,
error handling, and the base URL, matching config/tools.yaml's request_timeout_seconds.

Deliberately narrow: only GET is ever used (this project's tools never write to the
other project's data), and a failed call raises a specific exception type
(OpsPerformanceUnavailable) so the agent layer (Phase 17) can catch it and respond
gracefully rather than crashing the whole turn on a downstream API hiccup.

get()/get_text() accept an optional httpx.Client -- same injectable-client pattern as
src/agent/agent.py's Anthropic client. This is what makes Phase 23's failure-scenario
tests able to exercise this module's REAL request/response/error-handling logic against
an httpx.MockTransport, rather than mocking this module's own functions the way
tests/test_tools.py does. Both kinds of test earn their place: test_tools.py checks each
tool calls the right path with the right params; test_failure_scenarios.py checks this
module actually behaves correctly when the network/server misbehaves.
"""
import json
import os
import httpx

REQUEST_TIMEOUT_SECONDS = 15  # from config/tools.yaml


class OpsPerformanceUnavailable(Exception):
    """Raised when operations-performance's API can't be reached, returns an error, or
    returns a response this client can't parse. Caught at the agent layer -- a tool
    failure should produce a clear 'this data isn't available right now' response, not
    an unhandled crash."""


def _base_url() -> str:
    return os.environ.get("OPS_PERFORMANCE_API_URL", "http://localhost:8000")


def _get_client(client: httpx.Client | None) -> tuple[httpx.Client, bool]:
    """Returns (client, should_close). A caller-supplied client (e.g. a test's
    MockTransport-backed one) is never closed here -- that's the caller's
    responsibility. A client created internally for a single call is closed after."""
    if client is not None:
        return client, False
    return httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS), True


def get(path: str, params: dict | None = None, client: httpx.Client | None = None) -> dict | list:
    http_client, should_close = _get_client(client)
    try:
        response = http_client.get(f"{_base_url()}{path}", params=params)
        response.raise_for_status()
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            # Previously unhandled: a 200 response with a malformed/non-JSON body
            # would crash with a raw JSONDecodeError bubbling all the way up through
            # the tool and into the agent loop, rather than the clean
            # OpsPerformanceUnavailable every other failure mode produces. Found
            # while writing Phase 23's failure-scenario tests, fixed here.
            raise OpsPerformanceUnavailable(
                f"operations-performance API returned an unparseable response for {path}: {exc}"
            ) from exc
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            detail = "not found"
            try:
                detail = exc.response.json().get("detail", detail)
            except Exception:
                pass
            raise OpsPerformanceUnavailable(f"No data found for {path} -- {detail}") from exc
        raise OpsPerformanceUnavailable(
            f"operations-performance API returned {exc.response.status_code} for {path}"
        ) from exc
    except httpx.RequestError as exc:
        raise OpsPerformanceUnavailable(
            f"Could not reach operations-performance API at {_base_url()}{path}: {exc}"
        ) from exc
    finally:
        if should_close:
            http_client.close()


def get_text(path: str, client: httpx.Client | None = None) -> str:
    """Like get(), but for endpoints that return plain text (e.g. the markdown
    management report) rather than JSON -- kept separate rather than overloading
    get() with a response-type flag, since the two failure/parsing paths are
    genuinely different."""
    http_client, should_close = _get_client(client)
    try:
        response = http_client.get(f"{_base_url()}{path}")
        response.raise_for_status()
        return response.text
    except httpx.HTTPStatusError as exc:
        raise OpsPerformanceUnavailable(
            f"operations-performance API returned {exc.response.status_code} for {path}"
        ) from exc
    except httpx.RequestError as exc:
        raise OpsPerformanceUnavailable(
            f"Could not reach operations-performance API at {_base_url()}{path}: {exc}"
        ) from exc
    finally:
        if should_close:
            http_client.close()
