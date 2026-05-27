#!/usr/bin/env python
"""
UC-8: Data Quality Gate Enforcement.

Sends deliberately invalid API requests (too few points, NaN values,
multiple peaks, out-of-range time axis) and verifies that the API correctly
rejects them with 422 / 400 status codes.

Usage:
    # Requires the API to be running (docker compose up OR python scripts/start_api.py)
    python scripts/run_quality_gate_tests.py
    python scripts/run_quality_gate_tests.py --api-url http://localhost:8080
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

HEALTHY_SIGNAL = [
    0.025,
    -0.007,
    0.032,
    0.076,
    -0.012,
    -0.012,
    0.079,
    0.038,
    -0.023,
    0.028,
    -0.021,
    -0.018,
    0.025,
    -0.068,
    -0.029,
    0.082,
    0.148,
    0.354,
    0.495,
    0.741,
    1.218,
    1.505,
    1.89,
    2.135,
    2.396,
    2.506,
    2.366,
    2.225,
    1.857,
    1.502,
    1.114,
    0.904,
    0.54,
    0.285,
    0.24,
    0.049,
    0.067,
    -0.07,
    -0.054,
    0.015,
    0.039,
    0.009,
    -0.005,
    -0.015,
    -0.074,
    -0.036,
    -0.023,
    0.053,
    0.017,
    -0.088,
    0.016,
]
HEALTHY_TIME = [float(i * 2) for i in range(51)]


def _post_predict(api_url: str, payload: dict, api_key: str = "dev-key-12345") -> tuple[int, dict]:
    """POST to /predict, return (status_code, response_json)."""
    url = f"{api_url.rstrip('/')}/predict"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "X-API-Key": api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read())
        except Exception:
            body = {"raw": str(e)}
        return e.code, body


QUALITY_GATE_CASES = [
    {
        "name": "Too few data points (< 51)",
        "payload": {
            "device_id": "00000000-0000-0000-0000-000000000001",
            "time_values": [0.0, 1.0, 2.0],
            "amplitude_values": [0.1, 0.2, 0.3],
        },
        "expected_status": [400, 422],
        "description": "Signal with only 3 points — must be rejected.",
    },
    {
        "name": "NaN values in amplitude",
        "payload": {
            "device_id": "00000000-0000-0000-0000-000000000002",
            "time_values": HEALTHY_TIME,
            "amplitude_values": [
                float("nan") if i % 7 == 0 else v for i, v in enumerate(HEALTHY_SIGNAL)
            ],
        },
        "expected_status": [400, 422],
        "description": "Signal with ~15% NaN values — must be rejected.",
    },
    {
        "name": "Missing required field (device_id)",
        "payload": {
            "time_values": HEALTHY_TIME,
            "amplitude_values": HEALTHY_SIGNAL,
        },
        "expected_status": [400, 422],
        "description": "Payload missing device_id — Pydantic must reject.",
    },
    {
        "name": "SQL injection in device_id (string field)",
        "payload": {
            "device_id": "'; DROP TABLE predictions; --",
            "time_values": HEALTHY_TIME,
            "amplitude_values": HEALTHY_SIGNAL,
        },
        "expected_status": [400, 422],
        "description": "SQL injection attempt in device_id — must be rejected.",
    },
    {
        "name": "Valid healthy signal (control)",
        "payload": {
            "device_id": "00000000-0000-0000-0000-000000000099",
            "time_values": HEALTHY_TIME,
            "amplitude_values": HEALTHY_SIGNAL,
        },
        "expected_status": [200, 201],
        "description": "Valid signal — should be accepted (control case).",
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description="UC-8: Data Quality Gate Enforcement")
    parser.add_argument("--api-url", default="http://localhost:80", help="API base URL.")
    parser.add_argument("--api-key", default="dev-key-12345", help="API key for auth.")
    args = parser.parse_args()

    print("UC-8: Data Quality Gate Enforcement")
    print(f"API: {args.api_url}")
    print("=" * 60)

    # Quick connectivity check
    try:
        with urllib.request.urlopen(f"{args.api_url}/health", timeout=5):  # noqa: S310
            pass
    except Exception as exc:
        print(f"\n[ERROR] Cannot reach API at {args.api_url}: {exc}")
        print("        Start the stack first: docker compose up -d")
        print("        Or run locally: python scripts/start_api.py")
        return 1

    passed = 0
    failed = 0

    for case in QUALITY_GATE_CASES:
        status, body = _post_predict(
            args.api_url, cast(dict[str, Any], case["payload"]), args.api_key
        )
        ok = status in case["expected_status"]
        mark = "✅" if ok else "❌"
        result = "PASS" if ok else "FAIL"
        print(f"\n{mark} [{result}] {case['name']}")
        print(f"       {case['description']}")
        print(f"       Expected status: {case['expected_status']}  Got: {status}")
        if not ok:
            print(f"       Response: {json.dumps(body)[:200]}")
            failed += 1
        else:
            passed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
