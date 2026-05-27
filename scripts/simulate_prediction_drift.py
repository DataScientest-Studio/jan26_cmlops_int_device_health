#!/usr/bin/env python
"""
UC-20: Prediction Drift Simulation.

Demonstrates **prediction drift**: the distribution of model outputs shifts
significantly compared to the baseline reference.  This is distinct from
data drift (input features shift) and concept drift (label relationships
change) — here the *model predictions themselves* are distributing
differently.

How it works
------------
1. Loads the production model (bootstrap or retrained) from disk.
2. Generates a **reference** batch of healthy baseline signals and runs the
   model on them locally → establishes the reference prediction distribution
   (mostly label 0, high confidence).
3. Sends a large batch of **drifted** signals to the live API, collecting the
   API's predictions → current prediction distribution (mostly label 1).
4. Runs EvidentlyAI's TargetDriftPreset comparing the two prediction columns.
5. Writes a ``drift_summary_<timestamp>.json`` to ``reports/drift/`` — the
   API ``/metrics`` endpoint reads these files on every Prometheus scrape and
   sets ``drift_detected_gauge{drift_type="prediction"}``.
6. Grafana's "Drift Gauge Over Time" panel and the
   "UC-20 Prediction Drift Detected" alert rule reflect the result within
   the next scrape interval (~15 s).

Expected observations
---------------------
* Prometheus: ``drift_detected_gauge{drift_type="prediction"} > 0``
* Grafana → Alerting → "UC-20 Prediction Drift Detected" → Firing
* Grafana → Alerts Overview → "Drift Gauge Over Time" → prediction series rises

Alert timeline
--------------
  0 s   Script finishes writing drift_summary JSON
  ~15 s API /metrics scrape picks up the JSON; gauge incremented
  ~30 s Grafana evaluates alert rule; condition first met
  ~90 s Alert fires (1-minute ``for:`` window elapses)

Requirements
------------
* MLOps stack running: ``docker compose up -d``
* Model file present: ``models/bootstrap_model.pkl``
  (or ``models/retrained_model.pkl`` — whichever exists)

Usage
-----
    python scripts/simulate_prediction_drift.py
    python scripts/simulate_prediction_drift.py --n-reference 200 --n-current 500
    python scripts/simulate_prediction_drift.py --api-url http://localhost:80
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

REPORTS_DIR = PROJECT_ROOT / "reports" / "drift"
BASELINE_JSON = PROJECT_ROOT / "data" / "raw" / "dataset_baseline_test.json"
MODEL_PATHS = [
    PROJECT_ROOT / "models" / "retrained_model.pkl",
    PROJECT_ROOT / "models" / "bootstrap_model.pkl",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_api(api_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{api_url}/health", timeout=5):  # noqa: S310
            return True
    except Exception:
        return False


def _load_model():
    """Load the best available model from disk."""
    import pickle

    for path in MODEL_PATHS:
        if path.exists():
            with open(path, "rb") as fh:
                model = pickle.load(fh)  # noqa: S301
            print(f"  ✅  Loaded model: {path.name}")
            return model

    raise FileNotFoundError(
        "No model found. Expected one of:\n" + "\n".join(f"  {p}" for p in MODEL_PATHS)
    )


def _build_reference_df(n_samples: int, model):
    """
    Generate baseline healthy signals, run the model on them, and return
    a DataFrame with features + predicted_label (the reference distribution).
    """
    import numpy as np
    import pandas as pd

    from src.signal_processing.feature_extractor import extract_features
    from src.signal_processing.signal_generator import generate_dataset

    # Unpack the model artifact dict (same format as predict.py / train.py)
    clf = model["model"]
    scaler = model["scaler"]
    feat_order = model.get(
        "feature_names",
        ["fwhm", "peak_height", "peak_area", "noise_level", "snr", "peak_center"],
    )

    print(f"  Generating {n_samples} baseline reference signals …")
    signals = generate_dataset(n_samples=n_samples, drift_scenario="baseline")

    rows = []
    for sig in signals:
        feats = extract_features(sig.signal)
        row = {k: (v if v is not None else float("nan")) for k, v in feats.items()}

        # Scale then predict — scaler is stored separately in the artifact
        x = np.array([[row.get(f) or 0.0 for f in feat_order]])
        x_scaled = scaler.transform(x)
        pred = int(clf.predict(x_scaled)[0])
        row["predicted_label"] = pred
        rows.append(row)

    df = pd.DataFrame(rows)
    n_healthy = (df["predicted_label"] == 0).sum()
    print(f"  Reference distribution: {n_healthy}/{len(df)} healthy ({n_healthy / len(df):.0%})")
    return df


def _build_current_df(n_samples: int, api_url: str, api_key: str) -> pd.DataFrame:
    """
    Send drifted signals to the live API and collect predictions.
    Returns DataFrame with features + predicted_label (the current distribution).
    """
    import pandas as pd

    from src.signal_processing.feature_extractor import extract_features
    from src.signal_processing.signal_generator import generate_dataset

    print(f"  Generating and sending {n_samples} drifted signals to {api_url} …")
    signals = generate_dataset(n_samples=n_samples, drift_scenario="data_drift")

    predict_url = f"{api_url.rstrip('/')}/predict"
    headers = {"Content-Type": "application/json", "X-API-Key": api_key}

    rows = []
    errors = 0
    for i, sig in enumerate(signals, 1):
        feats = extract_features(sig.signal)
        row = {k: (v if v is not None else float("nan")) for k, v in feats.items()}

        # Build API payload
        payload = json.dumps(
            {
                "device_id": "00000000-0000-0000-0000-pred-drift-uc20",
                "time_values": sig.signal.time,
                "amplitude_values": sig.signal.amplitude,
            }
        ).encode()

        try:
            req = urllib.request.Request(predict_url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
                result = json.loads(resp.read())
            pred_label = int(result.get("prediction", result.get("predicted_label", 1)))
        except Exception:
            errors += 1
            pred_label = 1  # assume unhealthy if API error during drift simulation

        row["predicted_label"] = pred_label
        rows.append(row)

        if i % max(1, n_samples // 10) == 0:
            pct = i / n_samples * 100
            print(f"    [{pct:3.0f}%] {i}/{n_samples}  errors: {errors}", flush=True)

    df = pd.DataFrame(rows)
    n_unhealthy = (df["predicted_label"] == 1).sum()
    print(
        f"  Current distribution: {n_unhealthy}/{len(df)} unhealthy "
        f"({n_unhealthy / len(df):.0%})  errors: {errors}"
    )
    return df


def _run_prediction_drift_detection(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> dict:
    """
    Use EvidentlyAI TargetDriftPreset to compare prediction distributions.
    Returns the drift result dict.
    """
    from src.monitoring.drift_detection import DriftDetector

    feature_cols = ["fwhm", "peak_height", "peak_area", "noise_level", "snr", "peak_center"]
    valid_features = [
        c
        for c in feature_cols
        if c in reference_df.columns
        and c in current_df.columns
        and not reference_df[c].isna().all()
        and not current_df[c].isna().all()
    ]

    detector = DriftDetector(
        feature_columns=valid_features,
        prediction_column="predicted_label",
    )
    result = detector.detect_prediction_drift(
        reference_data=reference_df,
        current_data=current_df,
    )
    return result


def _write_drift_summary(
    prediction_drift: dict,
    reference_n: int,
    current_n: int,
) -> Path:
    """
    Write a drift_summary JSON file to reports/drift/.
    The API /metrics endpoint reads these files to set drift_detected_gauge.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = REPORTS_DIR / f"drift_summary_{timestamp}.json"

    summary = {
        # Short keys — read by src/api/main.py /metrics endpoint:
        "data_drift": {"drift_detected": False},
        "target_drift": {"drift_detected": False},
        # prediction_drift_details is the key the API reads for drift_type="prediction":
        "prediction_drift_details": prediction_drift,
        # Metadata:
        "reference_samples": reference_n,
        "current_samples": current_n,
        "timestamp": timestamp,
        "uc": "UC-20",
        "drift_type": "prediction",
    }

    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description="UC-20: Prediction Drift Simulation")
    parser.add_argument("--api-url", default="http://localhost:80", help="API base URL.")
    parser.add_argument("--api-key", default="dev-key-12345", help="API key.")
    parser.add_argument(
        "--n-reference",
        type=int,
        default=200,
        help="Number of baseline signals for reference distribution (default: 200).",
    )
    parser.add_argument(
        "--n-current",
        type=int,
        default=400,
        help="Number of drifted signals for current distribution (default: 400).",
    )
    args = parser.parse_args()

    print("UC-20: Prediction Drift Simulation")
    print("=" * 60)

    if not _check_api(args.api_url):
        print(f"\n[ERROR] API not reachable at {args.api_url}")
        print("        Start the stack: docker compose up -d")
        return 1

    # ── Step 1: Build reference prediction distribution ───────────────────
    print("\n[STEP 1] Building reference prediction distribution …")
    try:
        model = _load_model()
        reference_df = _build_reference_df(args.n_reference, model)
    except Exception as exc:
        print(f"[ERROR] Could not build reference data: {exc}")
        return 1

    # ── Step 2: Build current prediction distribution ─────────────────────
    print("\n[STEP 2] Building current prediction distribution (drifted signals) …")
    t0 = time.perf_counter()
    try:
        current_df = _build_current_df(args.n_current, args.api_url, args.api_key)
    except Exception as exc:
        print(f"[ERROR] Could not build current data: {exc}")
        return 1
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f} s")

    # ── Step 3: Run EvidentlyAI prediction drift detection ────────────────
    print("\n[STEP 3] Running EvidentlyAI prediction drift detection …")
    try:
        prediction_drift = _run_prediction_drift_detection(reference_df, current_df)
    except Exception as exc:
        print(f"[ERROR] Drift detection failed: {exc}")
        return 1

    drift_detected = prediction_drift.get("drift_detected", False)
    drift_score = prediction_drift.get("drift_score", 0.0)

    if drift_detected:
        print(f"  ⚠️  Prediction drift DETECTED  (score={drift_score:.4f})")
    else:
        print(f"  ℹ️  No prediction drift detected (score={drift_score:.4f})")
        print("      The model predictions may not have shifted enough.")
        print("      Try --n-current 800 for a stronger signal.")

    # ── Step 4: Write drift summary JSON ─────────────────────────────────
    print("\n[STEP 4] Writing drift summary JSON for Prometheus …")
    summary_path = _write_drift_summary(
        prediction_drift=prediction_drift,
        reference_n=len(reference_df),
        current_n=len(current_df),
    )
    try:
        display_path = summary_path.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = summary_path
    print(f"  ✅  Written: {display_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    ref_healthy_pct = (reference_df["predicted_label"] == 0).mean() * 100
    cur_healthy_pct = (current_df["predicted_label"] == 0).mean() * 100
    print()
    print("=" * 60)
    print("PREDICTION DRIFT SUMMARY")
    print("=" * 60)
    print(f"  Reference:  {len(reference_df)} signals, {ref_healthy_pct:.0f}% healthy predictions")
    print(f"  Current:    {len(current_df)} signals, {cur_healthy_pct:.0f}% healthy predictions")
    print(
        f"  Drift detected:  {'YES ⚠️' if drift_detected else 'NO  ✅'}"
        f"   (EvidentlyAI score={drift_score:.4f})"
    )
    print()
    print("  What to watch now:")
    print("  • Grafana → Alerting → 'UC-20 Prediction Drift Detected'")
    print("    (http://localhost:3000/alerting/list)")
    print("  • Grafana → MLOps Alerts Overview → 'Drift Gauge Over Time' panel")
    print("    (http://localhost:3000/d/mlops-alerts-overview)")
    print("  • Prometheus instant query:")
    print("    drift_detected_gauge{drift_type='prediction'}")
    print()
    if drift_detected:
        print("  Alert timeline:")
        print("    ~15 s  → API /metrics scrape picks up the JSON, gauge incremented")
        print("    ~30 s  → Grafana evaluates alert rule, condition first met")
        print("    ~90 s  → Alert fires (1-minute 'for:' window elapses)")
    print()
    print("  To reset: delete the drift summary files in reports/drift/")
    print(
        "  or run: python scripts/detect_drift.py --reference-json "
        "data/raw/dataset_baseline_test.json --output-dir reports/drift "
        "--min-samples 1   (with fresh healthy data in DB)"
    )
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
