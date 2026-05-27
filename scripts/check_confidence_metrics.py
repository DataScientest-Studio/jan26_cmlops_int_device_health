#!/usr/bin/env python
"""
UC-12: Confidence Calibration Monitoring.

Queries predictions from the database and reports confidence score
statistics (mean, std, min, max, percentiles).  Fires a warning if
mean confidence drops below 0.75 — matching the Grafana alert rule.

When Prometheus is available (Docker stack), also shows the raw metric.

Usage:
    python scripts/check_confidence_metrics.py
    python scripts/check_confidence_metrics.py --prometheus-url http://localhost:9090
    python scripts/check_confidence_metrics.py --threshold 0.80
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _query_database_confidence() -> dict:
    """Fetch confidence stats from the predictions table."""
    result: dict = {
        "count": 0,
        "mean": None,
        "std": None,
        "min": None,
        "max": None,
        "p25": None,
        "p50": None,
        "p75": None,
        "error": None,
    }
    try:
        from src.database.database import Database

        db_url = os.environ.get("DATABASE_URL", "")
        pg_host = os.environ.get("POSTGRES_HOST", "")
        if db_url and db_url.startswith("postgresql"):
            db = Database(db_url=db_url)
        elif pg_host:
            user = os.environ.get("POSTGRES_USER", "mlops_user")
            pw = os.environ.get("POSTGRES_PASSWORD", "changeme")
            port = os.environ.get("POSTGRES_PORT", "5432")
            dbname = os.environ.get("POSTGRES_DB", "mlops_db")
            db = Database(db_url=f"postgresql://{user}:{pw}@{pg_host}:{port}/{dbname}")
        else:
            db_path = PROJECT_ROOT / "data" / "database" / "mlops.db"
            db = Database(db_path=str(db_path))

        cursor = db.conn.cursor()
        cursor.execute(
            "SELECT prediction_confidence FROM predictions WHERE prediction_confidence IS NOT NULL"
        )
        rows = cursor.fetchall()
        db.close()

        scores = [
            float(r["prediction_confidence"])
            for r in rows
            if r["prediction_confidence"] is not None
        ]
        if not scores:
            result["error"] = "No predictions with confidence scores found."
            return result

        import statistics

        scores_sorted = sorted(scores)
        n = len(scores)
        result["count"] = n
        result["mean"] = statistics.mean(scores)
        result["std"] = statistics.stdev(scores) if n > 1 else 0.0
        result["min"] = min(scores)
        result["max"] = max(scores)
        result["p25"] = scores_sorted[int(n * 0.25)]
        result["p50"] = scores_sorted[int(n * 0.50)]
        result["p75"] = scores_sorted[int(n * 0.75)]

    except Exception as exc:
        result["error"] = str(exc)

    return result


def _query_prometheus(prometheus_url: str) -> str:
    """Fetch confidence histogram from Prometheus (best-effort)."""
    try:
        url = (
            f"{prometheus_url.rstrip('/')}/api/v1/query"
            "?query=histogram_quantile(0.50,rate(model_prediction_confidence_bucket[5m]))"
        )
        with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310
            import json

            data = json.loads(resp.read())
            results = data.get("data", {}).get("result", [])
            if results:
                val = results[0].get("value", [None, "—"])[1]
                return f"p50 from Prometheus: {val}"
            return "No histogram data in Prometheus."
    except Exception as exc:
        return f"Prometheus unavailable: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description="UC-12: Confidence Calibration Monitoring")
    parser.add_argument(
        "--threshold", type=float, default=0.75, help="Alert threshold (default: 0.75)."
    )
    parser.add_argument("--prometheus-url", default="http://localhost:9090", help="Prometheus URL.")
    args = parser.parse_args()

    print("UC-12: Confidence Calibration Monitoring")
    print("=" * 60)

    # ── Database stats ────────────────────────────────────────────────────────
    print("\n📊  Confidence scores from database:")
    stats = _query_database_confidence()

    if stats.get("error"):
        print(f"  ⚠️  {stats['error']}")
    else:
        print(f"  Count  : {stats['count']}")
        print(f"  Mean   : {stats['mean']:.4f}  (alert threshold: {args.threshold})")
        print(f"  Std    : {stats['std']:.4f}")
        print(f"  Min    : {stats['min']:.4f}")
        print(f"  p25    : {stats['p25']:.4f}")
        print(f"  p50    : {stats['p50']:.4f}")
        print(f"  p75    : {stats['p75']:.4f}")
        print(f"  Max    : {stats['max']:.4f}")

        mean = stats["mean"]
        if mean is not None and mean < args.threshold:
            print(
                f"\n  🚨  ALERT: Mean confidence {mean:.4f} < {args.threshold} — consider retraining!"
            )
            alert_code = 1
        else:
            print(f"\n  ✅  Mean confidence {mean:.4f} ≥ {args.threshold} — within normal range.")
            alert_code = 0

    # ── Prometheus ────────────────────────────────────────────────────────────
    print(f"\n📡  Prometheus ({args.prometheus_url}):")
    print(f"  {_query_prometheus(args.prometheus_url)}")

    print("\n📈  View full calibration dashboard:")
    print("     → Grafana → Model Performance → Confidence Score Distribution")

    return alert_code if not stats.get("error") else 0  # type: ignore[return-value]


if __name__ == "__main__":
    sys.exit(main())
