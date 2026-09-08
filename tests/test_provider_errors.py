import httpx
import groq
import anthropic

from src.agent.provider_errors import is_rate_limit_error


def _fake_groq_rate_limit_error():
    response = httpx.Response(status_code=429, request=httpx.Request("POST", "https://api.groq.com"))
    return groq.RateLimitError("rate limited", response=response, body=None)


def _fake_anthropic_rate_limit_error():
    response = httpx.Response(status_code=429, request=httpx.Request("POST", "https://api.anthropic.com"))
    return anthropic.RateLimitError("rate limited", response=response, body=None)


def test_detects_groq_rate_limit_error():
    assert is_rate_limit_error(_fake_groq_rate_limit_error()) is True


def test_detects_anthropic_rate_limit_error():
    assert is_rate_limit_error(_fake_anthropic_rate_limit_error()) is True


def test_detects_generic_429_status_code_attribute():
    class FakeGenaiError(Exception):
        status_code = 429

    assert is_rate_limit_error(FakeGenaiError("rate limited")) is True


def test_does_not_flag_unrelated_errors():
    assert is_rate_limit_error(RuntimeError("something else broke")) is False


def test_does_not_flag_other_http_status_errors():
    response = httpx.Response(status_code=500, request=httpx.Request("POST", "https://api.groq.com"))
    exc = groq.InternalServerError("server error", response=response, body=None)
    assert is_rate_limit_error(exc) is False
