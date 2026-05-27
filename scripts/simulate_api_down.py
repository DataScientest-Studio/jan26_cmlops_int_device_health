#!/usr/bin/env python
"""
UC-18: API Down Simulation.

Pauses the mlops_api Docker container for 90 seconds so that Prometheus
scrapes fail and the 'MLOps API Down' alert fires in Grafana after
the 1-minute sustained-failure window.

Requirements:
  - Docker must be running and the MLOps stack must be up:
      docker compose up -d
  - The caller must have permission to run `docker pause / unpause`.

Usage:
    python scripts/simulate_api_down.py
    python scripts/simulate_api_down.py --container mlops_api --down-seconds 90
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def _docker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        ["docker", *args],
        capture_output=True,
        text=True,
    )


def _container_running(name: str) -> bool:
    result = _docker("inspect", "--format", "{{.State.Status}}", name)
    return result.returncode == 0 and result.stdout.strip() == "running"


def main() -> int:
    parser = argparse.ArgumentParser(description="UC-18: API Down Simulation")
    parser.add_argument(
        "--container",
        default="mlops_api",
        help="Docker container name to pause (default: mlops_api).",
    )
    parser.add_argument(
        "--down-seconds",
        type=int,
        default=150,
        help="Seconds to keep the container paused (default: 150).",
    )
    args = parser.parse_args()

    print("UC-18: API Down Simulation")
    print("=" * 60)

    # Verify Docker is available
    if _docker("info").returncode != 0:
        print("\n[ERROR] Docker is not running or not accessible.")
        print("        Start Docker Desktop and try again.")
        return 1

    # Verify the container is running
    if not _container_running(args.container):
        print(f"\n[ERROR] Container '{args.container}' is not in 'running' state.")
        print("        Start the stack first:  docker compose up -d")
        return 1

    print(f"\n[STEP 1] Pausing container '{args.container}'…")
    result = _docker("pause", args.container)
    if result.returncode != 0:
        print(f"[ERROR] docker pause failed:\n{result.stderr}")
        return 1
    print(f"         ✅  '{args.container}' is now paused — API is DOWN.")
    print()
    print("  Prometheus will detect scrape failures within ~15-30 s.")
    print("  The 'MLOps API Down' alert fires after the container has been")
    print("  unreachable for 1 minute (alert rule: for: 1m).")
    print()
    print(f"  Keeping the API down for {args.down_seconds} seconds…")
    print("  Navigate to Grafana now → Alerting → Alert rules → folder 'MLOps Alerts'")
    print("  to watch the 'MLOps API Down' alert transition from Normal → Pending → Firing.")
    print(f"  (Alert fires at ~t+75 s; container resumes at t+{args.down_seconds} s)")
    print()

    for remaining in range(args.down_seconds, 0, -10):
        print(f"  ⏱  {remaining:3d} s remaining …", flush=True)
        time.sleep(min(10, remaining))

    print()
    print(f"[STEP 2] Unpausing container '{args.container}'…")
    result = _docker("unpause", args.container)
    if result.returncode != 0:
        print(f"[ERROR] docker unpause failed:\n{result.stderr}")
        print(f"        Run manually:  docker unpause {args.container}")
        return 1

    print(f"         ✅  '{args.container}' is running again — API is UP.")
    print()
    print("  The alert will auto-resolve within ~1 minute once Prometheus")
    print("  confirms the API is healthy again.")
    print()
    print("  Where to verify:")
    print("    • Grafana → Alerting → Alert rules → 'MLOps API Down' → Firing / Resolved")
    print("    • Grafana → MLOps Alerts Overview dashboard → 'API Availability' panel")
    print("    • Prometheus → http://localhost:9090/alerts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
