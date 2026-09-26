"""Session-wide test fixtures.

Found while auditing this project: 32 of ~200 tests failed locally whenever a
real, populated `.env` file sat in the working directory (exactly the state
left behind after setting one up to run the live demo). Root cause: several
auth tests do `monkeypatch.delenv("API_KEY")` then `importlib.reload(main_module)`
to simulate an unconfigured key -- but `src/api/main.py` calls `load_dotenv()`
at module level, and that reload re-executes it. `load_dotenv()`'s default
`override=False` only skips variables already present in `os.environ`; since
the monkeypatch just deleted the variable, `load_dotenv()` happily refills it
straight back from the physical `.env` file, silently undoing the monkeypatch
and cascading into unrelated-looking failures (401s where 200/422 were
expected, empty tool-call histories, etc.) across test_auth.py, test_api.py,
test_concurrency.py, test_e2e_scenarios.py, and test_failure_scenarios.py --
all one root cause, not five different bugs.

This never showed up in CI, because `.env` is gitignored and never checked
out there -- purely a "works in CI, silently breaks locally" trap for anyone
running the suite from a checkout that also has a working `.env`.

That reload-refill was only half the story. `load_dotenv()` also runs once,
for real, the moment pytest's COLLECTION phase first imports `src.api.main`
(any test module doing `from src.api.main import app` at module level
triggers it) -- before any fixture, autouse or not, has a chance to run.
`API_KEY` (read live per-request by `src/api/auth.py`, not cached at import
time) stays set in `os.environ` for the rest of the whole test session after
that, so every test that never explicitly sets/clears it -- most of
test_api.py, test_concurrency.py, test_e2e_scenarios.py,
test_failure_scenarios.py, none of which have anything to do with auth --
silently starts requiring a key they don't send, and gets 401 instead of
whatever they actually expected (422, 200, isolated conversation state, ...).

The identical pattern hit a second, unrelated config: `src/agent/agent.py`'s
`load_agent_config()` reads `AGENT_PROVIDER`/`AGENT_MODEL` from `os.environ`
by design (its own comment: "config/agent.yaml's committed default, which
tests/test_agent.py's Anthropic-shaped fake clients rely on, never has to
change for a real deployment to run Groq or Gemini instead") -- a real `.env`
with `AGENT_PROVIDER=groq` (needed to actually run the live demo locally)
overrides that committed "anthropic" default for every test in the session,
so test_agent.py's Anthropic-shaped mocks get called through Groq's
differently-shaped response path instead and fail in ways that look like
unrelated agent-loop bugs.

Fix: neutralize `load_dotenv()` for the whole test session (covers the
reload-refill case), AND unconditionally strip every env var any test
implicitly assumes is absent (`API_KEY`, `API_KEYS`, `AGENT_PROVIDER`,
`AGENT_MODEL`) before every single test (covers the collection-time
poisoning) -- so the suite behaves identically whether or not a real,
populated `.env` happens to be sitting in the working directory, matching
what CI already gets for free by never having one.
"""
import pytest

_ENV_VARS_TESTS_ASSUME_ABSENT = ("API_KEY", "API_KEYS", "AGENT_PROVIDER", "AGENT_MODEL")


@pytest.fixture(autouse=True)
def _no_real_dotenv(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    for var in _ENV_VARS_TESTS_ASSUME_ABSENT:
        monkeypatch.delenv(var, raising=False)
    # Most tests exercise routes as local development does: no keys configured, auth opted
    # out explicitly. tests/test_auth_fail_closed.py removes this to test the default.
    monkeypatch.setenv("ALLOW_UNAUTHENTICATED", "1")
