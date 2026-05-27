#!/usr/bin/env python
"""
UC-19: Prediction Distribution Skew — Trigger PredictionDistributionSkew Alert.

Sends a large burst of highly drifted signals (shifted amplitude, increased
noise) to the API.  The model classifies the majority as "1" (unhealthy),
which drives the healthy prediction rate below 50% and triggers the
'PredictionDistributionSkew — Possible Concept Drift' Grafana/Prometheus
alert after the sustained-condition window (``for: 10m``).

Expected timeline
-----------------
  0 s   Script starts sending signals
  ~1 m  Prometheus scrapes updated ``model_predictions_total`` counters
  ~2 m  Grafana alert moves to "Pending" state (condition first met)
  ~12 m Alert fires (10-minute ``for:`` window elapsed)

Requirements:
  - MLOps stack running: ``docker compose up -d``

Usage:
    python scripts/simulate_prediction_skew.py
    python scripts/simulate_prediction_skew.py --n-samples 3000
    python scripts/simulate_prediction_skew.py --api-url http://localhost:80
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _check_api(api_url: str) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(f"{api_url}/health", timeout=5):  # noqa: S310
            return True
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="UC-19: Prediction Distribution Skew")
    parser.add_argument("--api-url", default="http://localhost:80", help="API base URL.")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=2000,
        help="Number of drifted signals to send (default: 2000).",
    )
    args = parser.parse_args()

    print("UC-19: Prediction Distribution Skew")
    print("=" * 60)

    if not _check_api(args.api_url):
        print(f"\n[ERROR] API not reachable at {args.api_url}")
        print("        Start the stack: docker compose up -d")
        return 1

    print()
    print(f"  Sending {args.n_samples:,} heavily drifted signals to {args.api_url}")
    print("  These signals have shifted amplitude centres and high noise,")
    print("  which the model classifies as 'unhealthy' (label 1).")
    print("  This drives the healthy prediction rate well below 50%.")
    print()
    print("  ⏳  Alert timeline:")
    print("       ~1 min  → Prometheus scrapes updated counters")
    print("       ~2 min  → Grafana alert enters 'Pending' state")
    print("      ~12 min  → 'PredictionDistributionSkew' alert FIRES")
    print("                 (10-minute 'for:' window in the alert rule)")
    print()

    simulate_script = str(PROJECT_ROOT / "scripts" / "simulate_drift.py")
    cmd = [
        sys.executable,
        simulate_script,
        "data-drift",
        "--n-samples",
        str(args.n_samples),
        "--send-to-api",
    ]

    print("[STEP 1] Running simulate_drift.py data-drift …")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))  # noqa: S603
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"\n[ERROR] Drift simulation failed (exit {result.returncode}).")
        print("        Ensure the MLOps stack is running: docker compose up -d")
        return result.returncode

    print(f"\n  ✅  {args.n_samples:,} signals sent in {elapsed:.1f} s")
    print()
    print("[STEP 2] What to watch now:")
    print("  • Grafana → Alerting → Alert rules → 'PredictionDistributionSkew'")
    print("    (http://localhost:3000/alerting/list)")
    print("  • Grafana → MLOps Alerts Overview dashboard")
    print("    (http://localhost:3000/d/mlops-alerts-overview)")
    print("  • Grafana → Model Performance dashboard → 'Prediction Distribution (1h)'")
    print("    (http://localhost:3000/d/model-performance)")
    print("  • Prometheus instant query:")
    print(
        '    sum(rate(model_predictions_total{predicted_label="0"}[1h]))'
        " / sum(rate(model_predictions_total[1h]))"
    )
    print()
    print("  The alert will auto-resolve once healthy predictions recover above 50%.")
    print("  To restore balance, run UC-01 or send healthy signals:")
    print("    python scripts/simulate_drift.py gradual --n-samples 500 --send-to-api")
    return 0


if __name__ == "__main__":
    sys.exit(main())
