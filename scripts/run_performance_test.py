#!/usr/bin/env python
"""
UC-14: API Performance Degradation — stress test with optional degradation mode.

Wraps locust to run a performance test against the API and print a
summary.  Falls back to a simpler urllib-based stress test if locust
is not installed.

Usage:
    python scripts/run_performance_test.py --degrade          # concurrent degradation test
    python scripts/run_performance_test.py --simple           # sequential baseline
    python scripts/run_performance_test.py --users 20 --spawn-rate 5 -t 30s
    python scripts/run_performance_test.py --api-url http://localhost:80
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

LOCUSTFILE = PROJECT_ROOT / "tests" / "performance" / "locustfile.py"

HEALTHY_PAYLOAD = b"""{
  "device_id": "00000000-0000-0000-0000-000000000099",
  "time_values": [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100],
  "amplitude_values": [0.025,-0.007,0.032,0.076,-0.012,-0.012,0.079,0.038,-0.023,0.028,-0.021,-0.018,0.025,-0.068,-0.029,0.082,0.148,0.354,0.495,0.741,1.218,1.505,1.89,2.135,2.396,2.506,2.366,2.225,1.857,1.502,1.114,0.904,0.54,0.285,0.24,0.049,0.067,-0.07,-0.054,0.015,0.039,0.009,-0.005,-0.015,-0.074,-0.036,-0.023,0.053,0.017,-0.088,0.016]
}"""


def _check_api(api_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{api_url}/health", timeout=5):  # noqa: S310
            return True
    except Exception:
        return False


def _simple_stress_test(api_url: str, n_requests: int, api_key: str) -> int:
    """Run a simple sequential stress test without locust."""
    print(f"\n[Simple stress test]  {n_requests} sequential requests → {api_url}/predict")
    url = f"{api_url.rstrip('/')}/predict"

    latencies = []
    errors = 0

    for i in range(1, n_requests + 1):
        req = urllib.request.Request(
            url,
            data=HEALTHY_PAYLOAD,
            headers={"Content-Type": "application/json", "X-API-Key": api_key},
            method="POST",
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                _ = resp.read()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed_ms)
        except Exception:
            errors += 1
        if i % 10 == 0:
            pct = i / n_requests * 100
            avg = sum(latencies[-10:]) / len(latencies[-10:]) if latencies else 0
            print(f"  [{pct:3.0f}%] {i}/{n_requests}  last-10 avg: {avg:.0f} ms")

    if latencies:
        import statistics

        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)
        print(f"\n  Count     : {len(latencies)}")
        print(f"  Errors    : {errors}")
        print(f"  Mean      : {statistics.mean(latencies):.1f} ms")
        print(f"  p50       : {latencies_sorted[int(n * 0.50)]:.1f} ms")
        print(f"  p95       : {latencies_sorted[int(n * 0.95)]:.1f} ms")
        print(f"  p99       : {latencies_sorted[int(n * 0.99)]:.1f} ms")
        print(f"  Max       : {max(latencies):.1f} ms")
        p95 = latencies_sorted[int(n * 0.95)]
        if p95 > 500:
            print(f"\n  ⚠️  p95 latency {p95:.0f} ms > 500 ms threshold — performance degraded!")
            return 1
        print(f"\n  ✅  p95 latency {p95:.0f} ms ≤ 500 ms — performance is acceptable.")

    return 0


def _degradation_stress_test(api_url: str, n_workers: int, n_requests: int, api_key: str) -> int:
    """Send concurrent requests with large payloads to simulate API performance degradation.

    Uses ThreadPoolExecutor to fire many requests in parallel, each with a
    large time-series payload (500 points).  Under concurrent load the API
    p95 latency typically exceeds the 500 ms alert threshold.
    """
    import math

    n_points = 500
    t_vals = list(range(0, n_points * 2, 2))  # [0, 2, 4, ..., 998]
    a_vals = [
        round(math.sin(t / 10.0) * 1.5 + 0.05 * (i % 7) - 0.02, 4) for i, t in enumerate(t_vals)
    ]
    large_payload = (
        '{"device_id":"00000000-0000-0000-0000-000000000099",'
        f'"time_values":{t_vals},'
        f'"amplitude_values":{a_vals}}}'
    ).encode()

    url = f"{api_url.rstrip('/')}/predict"
    headers = {"Content-Type": "application/json", "X-API-Key": api_key}

    print(f"\n[Degradation stress test]  {n_requests} requests × {n_workers} workers → {url}")
    print(f"  Payload size: {len(large_payload):,} bytes  ({n_points}-point signal)")

    latencies: list[float] = []
    errors = 0
    completed = 0

    def _send_one() -> float | None:
        req = urllib.request.Request(url, data=large_payload, headers=headers, method="POST")
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                _ = resp.read()
            return (time.perf_counter() - t0) * 1000
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_send_one) for _ in range(n_requests)]
        report_every = max(1, n_requests // 10)
        for fut in as_completed(futures):
            result = fut.result()
            completed += 1
            if result is None:
                errors += 1
            else:
                latencies.append(result)
            if completed % report_every == 0:
                pct = completed / n_requests * 100
                recent = latencies[-10:] if latencies else []
                avg = sum(recent) / len(recent) if recent else 0.0
                print(
                    f"  [{pct:3.0f}%] {completed}/{n_requests}  recent-10 avg: {avg:.0f} ms  errors: {errors}"
                )

    if latencies:
        import statistics

        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)
        p95 = latencies_sorted[int(n * 0.95)]
        print(f"\n  Count     : {len(latencies)}")
        print(f"  Errors    : {errors}")
        print(f"  Mean      : {statistics.mean(latencies):.1f} ms")
        print(f"  p50       : {latencies_sorted[int(n * 0.50)]:.1f} ms")
        print(f"  p95       : {p95:.1f} ms")
        print(f"  p99       : {latencies_sorted[int(n * 0.99)]:.1f} ms")
        print(f"  Max       : {max(latencies):.1f} ms")
        if p95 > 500:
            print(f"\n  ⚠️  p95 latency {p95:.0f} ms > 500 ms threshold — performance degraded!")
            print("     Check Grafana → System Health dashboard for the latency histogram.")
            return 1
        print(
            f"\n  ✅  p95 latency {p95:.0f} ms ≤ 500 ms — performance acceptable under this load."
        )
        print("     Increase --workers or --requests to push past the threshold.")
    return 0


def _run_locust(api_url: str, users: int, spawn_rate: int, duration: str) -> int:
    """Run locust in headless mode."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "locust",
            "-f",
            str(LOCUSTFILE),
            "--headless",
            "-u",
            str(users),
            "-r",
            str(spawn_rate),
            "-t",
            duration,
            "--host",
            api_url,
            "--only-summary",
        ],
        cwd=str(PROJECT_ROOT),
    )
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="UC-14: API Performance Degradation")
    parser.add_argument("--api-url", default="http://localhost:80", help="API base URL.")
    parser.add_argument("--api-key", default="dev-key-12345", help="API key.")
    parser.add_argument(
        "-u", "--users", type=int, default=50, help="Number of concurrent users (locust)."
    )
    parser.add_argument(
        "-r", "--spawn-rate", type=int, default=10, help="Users to spawn per second (locust)."
    )
    parser.add_argument("-t", "--duration", default="60s", help="Test duration (locust).")
    parser.add_argument("-n", "--requests", type=int, default=30, help="Requests for simple mode.")
    parser.add_argument(
        "--simple", action="store_true", help="Use simple urllib test instead of locust."
    )
    parser.add_argument(
        "--degrade",
        action="store_true",
        help="Run concurrent load test with large payloads to trigger the 500 ms alert threshold.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=30,
        help="Number of concurrent workers for --degrade mode.",
    )
    args = parser.parse_args()

    print("UC-14: API Performance Degradation")
    print("=" * 60)

    if not _check_api(args.api_url):
        print(f"\n[ERROR] API not reachable at {args.api_url}")
        print("        Start the stack: docker compose up -d")
        print("        Or locally: python scripts/start_api.py")
        return 1

    if args.degrade:
        return _degradation_stress_test(args.api_url, args.workers, args.requests, args.api_key)

    if args.simple:
        return _simple_stress_test(args.api_url, args.requests, args.api_key)

    # Try locust
    try:
        import locust  # noqa: F401

        print(f"\n[INFO] Running locust: {args.users} users, {args.spawn_rate}/s, {args.duration}")
        return _run_locust(args.api_url, args.users, args.spawn_rate, args.duration)
    except ImportError:
        print("[INFO] locust not installed — falling back to simple stress test.")
        print("       Install with: pip install locust")
        return _simple_stress_test(args.api_url, args.requests, args.api_key)


if __name__ == "__main__":
    sys.exit(main())
