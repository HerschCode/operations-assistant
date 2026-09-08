"""
Real load/concurrency testing against an actually-running deployed instance --
the gap named in FUTURE_IMPROVEMENTS.md ("the in-process TestClient version exists
and is genuinely useful, but it's not the same as measuring latency under real
network load"). Run against the live Render URL, not localhost, so this measures
real network latency (including cold starts) rather than in-process call overhead.

Two things are tested:
1. Plain concurrent load against /health -- does the server survive N simultaneous
   real HTTP connections without errors, and what's the latency distribution.
2. A targeted race-condition probe against /demo/chat's rate limiter
   (src/api/rate_limit.py): MAX_REQUESTS_PER_WINDOW=5 is enforced by a
   check-then-append on a plain dict/deque with no lock. FastAPI runs sync route
   handlers in a thread pool, so genuinely concurrent requests are real OS threads,
   not just async tasks on one thread -- if the check-then-act sequence isn't
   atomic, more than 5 requests could get allowed through when they arrive at
   nearly the same instant instead of sequentially.

Usage:
    python -m scripts.load_test_live --url https://operations-assistant.onrender.com
"""
import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx


def health_load_test(base_url: str, n_requests: int, concurrency: int) -> None:
    print(f"\n=== Health endpoint load test: {n_requests} requests, concurrency={concurrency} ===")
    latencies = []
    errors = 0

    def one_request(_):
        t0 = time.monotonic()
        try:
            r = httpx.get(f"{base_url}/health", timeout=30)
            ok = r.status_code == 200
        except Exception:
            ok = False
        return (time.monotonic() - t0) * 1000, ok

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(one_request, i) for i in range(n_requests)]
        for f in as_completed(futures):
            ms, ok = f.result()
            latencies.append(ms)
            if not ok:
                errors += 1

    latencies.sort()
    print(f"  Errors: {errors}/{n_requests}")
    print(f"  Latency (ms) -- p50: {statistics.median(latencies):.0f}  "
          f"p90: {latencies[int(len(latencies) * 0.9)]:.0f}  "
          f"max: {max(latencies):.0f}")


def rate_limit_race_probe(base_url: str, n_concurrent: int) -> None:
    print(f"\n=== Rate limiter race probe: {n_concurrent} truly concurrent /demo/chat requests ===")
    fake_ip = f"203.0.113.{int(time.time()) % 255}"  # TEST-NET-3, unique enough per run
    allowed = 0
    rate_limited = 0
    other = []

    def one_request(_):
        try:
            r = httpx.post(
                f"{base_url}/demo/chat",
                json={"question": "what is the average cycle time"},
                headers={"X-Forwarded-For": fake_ip},
                timeout=120,
            )
            return r.status_code, r.text[:200]
        except Exception as exc:
            return f"error:{type(exc).__name__}", str(exc)[:200]

    with ThreadPoolExecutor(max_workers=n_concurrent) as pool:
        futures = [pool.submit(one_request, i) for i in range(n_concurrent)]
        for f in as_completed(futures):
            status, detail = f.result()
            if status == 200:
                allowed += 1
            elif status == 429:
                rate_limited += 1
            else:
                other.append((status, detail))

    print(f"  Allowed (200): {allowed}  Rate-limited (429): {rate_limited}  Other: {len(other)}")
    for status, detail in other:
        print(f"    -> {status}: {detail}")
    if allowed > 5:
        print(f"  *** RACE CONDITION CONFIRMED: {allowed} requests allowed through, limit is 5 ***")
    else:
        print("  OK -- limit held under concurrent load")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="https://operations-assistant.onrender.com")
    parser.add_argument("--n-requests", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--race-probe-concurrency", type=int, default=8)
    args = parser.parse_args()

    print(f"Target: {args.url}")
    health_load_test(args.url, args.n_requests, args.concurrency)
    rate_limit_race_probe(args.url, args.race_probe_concurrency)


if __name__ == "__main__":
    main()
