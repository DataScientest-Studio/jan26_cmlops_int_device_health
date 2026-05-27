"""
K8s Master Test Runner
Runs all tier tests in sequence, collects results, and prints a summary.
"""

from __future__ import annotations

import importlib
import sys
import time

TIERS = [
    ("Tier 2 — Database", "tests.k8s.test_02_database"),
    ("Tier 3 — API E2E", "tests.k8s.test_03_api_e2e"),
    ("Tier 4 — MLflow", "tests.k8s.test_04_mlflow"),
    ("Tier 5 — Airflow DAGs", "tests.k8s.test_05_airflow_dags"),
    ("Tier 6 — Monitoring", "tests.k8s.test_06_monitoring"),
    ("Tier 7 — Drift & Retraining", "tests.k8s.test_07_drift_e2e"),
    ("Tier 8 — Resilience", "tests.k8s.test_08_resilience"),
]


def main() -> None:
    # Parse args: --skip-slow skips Tier 7 (5-min DAG wait)
    skip_slow = "--skip-slow" in sys.argv
    results: list[tuple[str, int]] = []

    for name, module_path in TIERS:
        if skip_slow and "drift_e2e" in module_path:
            print(f"\n{'=' * 60}")
            print(f"SKIPPING (--skip-slow): {name}")
            continue

        print(f"\n{'=' * 60}")
        print(f"RUNNING: {name}")
        print("=" * 60)
        t0 = time.time()
        try:
            mod = importlib.import_module(module_path)
            failures = mod.run_all()
        except Exception as e:
            print(f"  💥 Module import/run error: {e}")
            failures = 1
        elapsed = time.time() - t0
        results.append((name, failures))
        print(f"  (elapsed: {elapsed:.1f}s)")

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    total_failures = 0
    for name, failures in results:
        icon = "✅" if failures == 0 else "❌"
        print(f"  {icon} {name}: {failures} failure(s)")
        total_failures += failures

    print(f"\nTotal failures: {total_failures}")
    sys.exit(1 if total_failures else 0)


if __name__ == "__main__":
    main()
