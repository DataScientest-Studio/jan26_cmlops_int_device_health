#!/usr/bin/env python3
"""
UC-06 — A/B Testing (Canary Deployment) demonstration.

Simulates Nginx-based canary routing by loading the Production champion and
the Staging challenger from the MLflow Model Registry, running identical test
signals through both, and producing a side-by-side comparison report.

How it works
------------
1. Load the Production (champion) model from MLflow.
2. Load the Staging (challenger) model from MLflow.
3. Generate N test signals from the baseline distribution (or load from a
   provided JSON dataset file).
4. Split the signals into a "champion batch" (1 - canary_fraction) and a
   "canary batch" (canary_fraction), simulating Nginx weighted upstream routing.
5. For each batch, run local predictions (no HTTP required).
6. Compare: accuracy, F1-score, mean confidence, prediction distribution.
7. Print a rich comparison table and a promotion recommendation.

Exit codes
----------
0  Normal execution (comparison printed, regardless of promotion outcome)
1  Fatal error (both models unavailable, or dataset missing)

Usage
-----
    python scripts/run_ab_test.py                        # default: 200 signals, 25% canary
    python scripts/run_ab_test.py --n-signals 500        # larger batch
    python scripts/run_ab_test.py --canary-fraction 0.20 --n-signals 400
    python scripts/run_ab_test.py --model-name device_health_classifier
    python scripts/run_ab_test.py --tracking-uri http://localhost:5001
    python scripts/run_ab_test.py --dry-run              # use dummy scores (no MLflow needed)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ── Helpers ────────────────────────────────────────────────────────────────────


def _load_model_from_registry(
    model_name: str,
    stage: str,
    tracking_uri: str,
) -> dict[str, Any] | None:
    """Load a model artifact dict from the MLflow registry.

    Returns the pickle-ready dict ``{"model": ..., "scaler": ...,
    "feature_names": [...], "model_version": str}`` or ``None`` if not found.
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()

        # Find the latest version via alias (MLflow 3.x)
        stage_to_alias = {"Production": "champion", "Staging": "challenger"}
        alias = stage_to_alias.get(stage)
        if not alias:
            return None

        try:
            mv = client.get_model_version_by_alias(model_name, alias)
        except Exception:
            # Fallback: scan all versions for matching alias
            all_versions = client.search_model_versions(f"name='{model_name}'")
            candidates = []
            for v in all_versions:
                v_aliases = getattr(v, "aliases", []) or []
                if alias in v_aliases:
                    candidates.append(v)
            versions = sorted(candidates, key=lambda v: int(v.version), reverse=True)[:1]
            if not versions:
                return None
            mv = versions[0]

        model_uri = f"models:/{model_name}/{mv.version}"
        artifact = mlflow.sklearn.load_model(model_uri)

        # The registry stores sklearn pipelines or raw models; wrap to standard dict
        if isinstance(artifact, dict):
            if "model_version" not in artifact:
                artifact["model_version"] = f"{model_name}_v{mv.version}_{stage}"
            return artifact

        # Bare sklearn pipeline — wrap it
        return {
            "model": artifact,
            "scaler": None,
            "feature_names": None,
            "model_version": f"{model_name}_v{mv.version}_{stage}",
        }

    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not load {stage} model from MLflow: {exc}")
        return None


def _load_model_from_filesystem(path: Path) -> dict[str, Any] | None:
    """Load a pickle model from the filesystem."""
    import pickle

    try:
        with open(path, "rb") as f:
            artifact = pickle.load(f)  # noqa: S301 — local files only
        if isinstance(artifact, dict):
            return artifact
        return {
            "model": artifact,
            "scaler": None,
            "feature_names": None,
            "model_version": path.stem,
        }
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not load model from {path}: {exc}")
        return None


def _run_predictions(
    artifact: dict[str, Any],
    signals: list[dict],
) -> dict[str, Any]:
    """Run predictions for a list of signal dicts; return metrics summary."""
    from src.training.predict import predict

    predictions = []
    confidences = []
    errors = 0

    for sig in signals:
        try:
            result = predict(
                time_values=sig["time"],
                amplitude_values=sig["amplitude"],
                model_path=artifact,
                return_probabilities=False,
            )
            predictions.append(
                {
                    "predicted": result["predicted_label"],
                    "ground_truth": sig.get("label"),
                    "confidence": result["confidence"],
                    "model_version": result.get("model_version", "unknown"),
                }
            )
            confidences.append(result["confidence"])
        except Exception:  # noqa: BLE001
            errors += 1

    if not predictions:
        return {
            "n_signals": len(signals),
            "n_errors": errors,
            "accuracy": None,
            "f1_score": None,
            "mean_confidence": None,
            "pred_healthy_pct": None,
            "pred_unhealthy_pct": None,
            "model_version": artifact.get("model_version", "unknown"),
        }

    predicted = [p["predicted"] for p in predictions]
    ground_truth = [p["ground_truth"] for p in predictions if p["ground_truth"] is not None]
    labeled_predicted = [p["predicted"] for p in predictions if p["ground_truth"] is not None]

    accuracy = None
    f1 = None
    if ground_truth:
        correct = sum(p == g for p, g in zip(labeled_predicted, ground_truth, strict=False))
        accuracy = correct / len(ground_truth)
        try:
            from sklearn.metrics import f1_score

            f1 = float(f1_score(ground_truth, labeled_predicted, average="binary", zero_division=0))
        except Exception:  # noqa: BLE001
            pass

    n = len(predicted)
    n_healthy = predicted.count(0)
    n_unhealthy = predicted.count(1)

    return {
        "n_signals": len(signals),
        "n_errors": errors,
        "accuracy": accuracy,
        "f1_score": f1,
        "mean_confidence": sum(confidences) / len(confidences) if confidences else None,
        "pred_healthy_pct": 100.0 * n_healthy / n if n else None,
        "pred_unhealthy_pct": 100.0 * n_unhealthy / n if n else None,
        "model_version": artifact.get("model_version", "unknown"),
    }


def _load_signals(dataset_path: Path, n: int, seed: int = 42) -> list[dict]:
    """Load up to n signals from a JSON dataset file."""
    with open(dataset_path) as f:
        data = json.load(f)
    signals = data.get("signals", [])
    rng = random.Random(seed)
    rng.shuffle(signals)
    return signals[:n]


def _generate_baseline_signals(n: int, seed: int = 42) -> list[dict]:
    """Generate fresh baseline signals when no dataset file is available."""
    from src.signal_processing.signal_generator import (
        add_gaussian_noise,
        create_time_array,
        generate_gaussian_peak,
        generate_lorentzian_peak,
    )

    rng = random.Random(seed)
    signals = []
    time = create_time_array(n_points=200)

    for i in range(n):
        label = i % 2  # alternate healthy/unhealthy
        noise = rng.uniform(0.01, 0.04)
        if label == 0:  # healthy: Gaussian
            clean = generate_gaussian_peak(time, mu=50.0, sigma=2.0, height=1.0)
        else:  # unhealthy: Lorentzian
            clean = generate_lorentzian_peak(time, mu=50.0, gamma=2.0, height=1.0)
        amp: list[float] = [
            float(v)
            for v in add_gaussian_noise(clean, noise_level=noise, seed=rng.randint(0, 99999))
        ]
        signals.append({"time": list(time), "amplitude": amp, "label": label})

    return signals


def _fmt(value: float | None, fmt: str = ".4f") -> str:
    return f"{value:{fmt}}" if value is not None else "N/A"


def _dummy_metrics(model_version: str, champion: bool) -> dict[str, Any]:
    """Return plausible dummy metrics for dry-run mode."""
    if champion:
        return {
            "n_signals": 150,
            "n_errors": 0,
            "accuracy": 0.667,
            "f1_score": 0.667,
            "mean_confidence": 0.71,
            "pred_healthy_pct": 55.3,
            "pred_unhealthy_pct": 44.7,
            "model_version": model_version,
        }
    return {
        "n_signals": 50,
        "n_errors": 0,
        "accuracy": 0.98,
        "f1_score": 0.98,
        "mean_confidence": 0.97,
        "pred_healthy_pct": 49.0,
        "pred_unhealthy_pct": 51.0,
        "model_version": model_version,
    }


# ── Main ───────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="UC-06 A/B Testing (Canary Deployment) demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get("MODEL_REGISTRY_NAME", "device_health_classifier"),
        help="Registered model name in MLflow (default: $MODEL_REGISTRY_NAME or device_health_classifier)",
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"),
        help="MLflow tracking URI (default: $MLFLOW_TRACKING_URI or http://localhost:5001)",
    )
    parser.add_argument(
        "--n-signals",
        type=int,
        default=200,
        help="Total number of test signals to generate (default: 200)",
    )
    parser.add_argument(
        "--canary-fraction",
        type=float,
        default=0.25,
        help="Fraction of traffic routed to the Staging (challenger) model (default: 0.25)",
    )
    parser.add_argument(
        "--dataset",
        default=str(PROJECT_ROOT / "data" / "raw" / "dataset_baseline_full.json"),
        help="Path to labeled JSON dataset (default: data/raw/dataset_baseline_full.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use dummy model metrics without loading MLflow; useful for UI demos",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.02,
        help="Minimum F1 improvement required to recommend promotion (default: 0.02)",
    )
    args = parser.parse_args(argv)

    if args.canary_fraction <= 0 or args.canary_fraction >= 1:
        print("[ERROR] --canary-fraction must be between 0 and 1 (exclusive).")
        return 1
    if args.n_signals < 10:
        print("[ERROR] --n-signals must be at least 10.")
        return 1

    n_canary = max(1, int(args.n_signals * args.canary_fraction))
    n_champion = args.n_signals - n_canary

    print("═" * 65)
    print("  UC-06 — A/B Testing (Canary Deployment)")
    print("═" * 65)
    print(f"  Model registry : {args.model_name}")
    print(f"  MLflow URI     : {args.tracking_uri}")
    print(f"  Total signals  : {args.n_signals}")
    print(
        f"  Traffic split  : {100 * (1 - args.canary_fraction):.0f}% champion / "
        f"{100 * args.canary_fraction:.0f}% canary (challenger)"
    )
    print(f"  Champion batch : {n_champion} signals")
    print(f"  Canary batch   : {n_canary} signals")
    print("═" * 65)

    if args.dry_run:
        print("\n[INFO] Dry-run mode — using synthetic model metrics.\n")
        champ_metrics = _dummy_metrics(f"{args.model_name}_Production", champion=True)
        chal_metrics = _dummy_metrics(f"{args.model_name}_Staging", champion=False)
        champ_artifact: dict[str, Any] | None = {"model_version": f"{args.model_name}_Production"}
        chal_artifact: dict[str, Any] | None = {"model_version": f"{args.model_name}_Staging"}
    else:
        # ── Step 1: Load models ─────────────────────────────────────────
        print("\n[1/3] Loading models from MLflow registry…")
        champ_artifact = _load_model_from_registry(args.model_name, "Production", args.tracking_uri)
        if champ_artifact is None:
            # Try filesystem fallback
            fallback = PROJECT_ROOT / "models" / "bootstrap_model.pkl"
            if fallback.exists():
                print(f"      → Production model not in registry; using {fallback.name}")
                champ_artifact = _load_model_from_filesystem(fallback)

        chal_artifact = _load_model_from_registry(args.model_name, "Staging", args.tracking_uri)
        if chal_artifact is None:
            fallback_chal = PROJECT_ROOT / "models" / "retrained_model.pkl"
            if fallback_chal.exists():
                print(f"      → Staging model not in registry; using {fallback_chal.name}")
                chal_artifact = _load_model_from_filesystem(fallback_chal)

        if champ_artifact is None:
            print("[ERROR] No Production/champion model found in registry or filesystem.")
            print("        Run: python scripts/bootstrap_model.py  to create one, or")
            print("             python scripts/create_degrading_champion.py  to set up the demo.")
            return 1

        if chal_artifact is None:
            print("[WARN] No Staging/challenger model found.")
            print(
                "       Run: python scripts/trigger_retraining.py --force  to train a challenger."
            )
            print("       Canary comparison skipped — showing champion metrics only.\n")

        print(f"      ✓ Champion : {champ_artifact.get('model_version', 'unknown')}")
        if chal_artifact:
            print(f"      ✓ Challenger: {chal_artifact.get('model_version', 'unknown')}")
        else:
            print("      ✗ Challenger: not available")

        # ── Step 2: Load/generate signals ──────────────────────────────
        print("\n[2/3] Preparing test signals…")
        dataset_path = Path(args.dataset)
        if dataset_path.exists():
            all_signals = _load_signals(dataset_path, args.n_signals, seed=args.seed)
            print(f"      Loaded {len(all_signals)} signals from {dataset_path.name}")
        else:
            print(f"      Dataset not found at {dataset_path.name} — generating baseline signals.")
            all_signals = _generate_baseline_signals(args.n_signals, seed=args.seed)
            print(f"      Generated {len(all_signals)} synthetic baseline signals")

        # Ensure we have enough
        if len(all_signals) < 10:
            print("[ERROR] Not enough signals to run A/B test (need at least 10).")
            return 1

        # Duplicate/cycle if dataset smaller than requested
        while len(all_signals) < args.n_signals:
            all_signals = all_signals + all_signals
        all_signals = all_signals[: args.n_signals]

        rng = random.Random(args.seed)
        rng.shuffle(all_signals)
        champion_signals = all_signals[:n_champion]
        canary_signals = all_signals[n_champion : n_champion + n_canary]

        # ── Step 3: Run predictions ────────────────────────────────────
        print("\n[3/3] Running predictions…")
        print(f"      Running champion on {len(champion_signals)} signals…")
        champ_metrics = _run_predictions(champ_artifact, champion_signals)

        if chal_artifact:
            print(f"      Running challenger on {len(canary_signals)} signals…")
            chal_metrics = _run_predictions(chal_artifact, canary_signals)
        else:
            chal_metrics = None

    # ── Report ─────────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  RESULTS: Champion vs Challenger (Canary)")
    print("═" * 65)

    col_w = 20
    header = f"{'Metric':<22}  {'Champion (Prod)':>{col_w}}  {'Challenger (Stage)':>{col_w}}"
    print(header)
    print("-" * len(header))

    def row(
        label: str, champ_val: float | None, chal_val: float | None, higher_is_better: bool = True
    ) -> None:
        cv = _fmt(champ_val)
        cav = _fmt(chal_val) if chal_val is not None else "N/A"
        delta_str = ""
        if champ_val is not None and chal_val is not None:
            delta = chal_val - champ_val
            sign = "+" if delta >= 0 else ""
            symbol = (
                "▲"
                if (delta > 0 and higher_is_better) or (delta < 0 and not higher_is_better)
                else "▼"
            )
            delta_str = f"  Δ={sign}{delta:.4f} {symbol}"
        print(f"  {label:<20}  {cv:>{col_w}}  {cav:>{col_w}}{delta_str}")

    # Model version names
    print(f"  {'Model version':<20}  {champ_metrics['model_version']!r:>{col_w}}")
    if chal_metrics:
        print(f"  {'':22}  {'':>{col_w}}  {chal_metrics['model_version']!r:>{col_w}}")
    print()

    row(
        "Accuracy",
        champ_metrics.get("accuracy"),
        chal_metrics.get("accuracy") if chal_metrics else None,
    )
    row(
        "F1 Score",
        champ_metrics.get("f1_score"),
        chal_metrics.get("f1_score") if chal_metrics else None,
    )
    row(
        "Mean Confidence",
        champ_metrics.get("mean_confidence"),
        chal_metrics.get("mean_confidence") if chal_metrics else None,
    )
    row(
        "Pred Healthy %",
        champ_metrics.get("pred_healthy_pct"),
        chal_metrics.get("pred_healthy_pct") if chal_metrics else None,
        higher_is_better=False,
    )
    row(
        "Pred Unhealthy %",
        champ_metrics.get("pred_unhealthy_pct"),
        chal_metrics.get("pred_unhealthy_pct") if chal_metrics else None,
        higher_is_better=False,
    )
    row(
        "N signals tested",
        float(champ_metrics["n_signals"]),
        float(chal_metrics["n_signals"]) if chal_metrics else None,
        higher_is_better=True,
    )
    row(
        "Errors",
        float(champ_metrics["n_errors"]),
        float(chal_metrics["n_errors"]) if chal_metrics else None,
        higher_is_better=False,
    )
    print("═" * 65)

    # Recommendation
    print("\n  RECOMMENDATION")
    print("  " + "-" * 40)
    if chal_metrics is None:
        print("  No challenger available — keep current champion in production.")
        print("  Action: Run 'python scripts/trigger_retraining.py --force' to train a challenger.")
    else:
        champ_f1 = champ_metrics.get("f1_score") or 0.0
        chal_f1 = chal_metrics.get("f1_score") or 0.0
        delta_f1 = chal_f1 - champ_f1

        if delta_f1 >= args.min_improvement:
            print("  ✅ PROMOTE challenger to Production!")
            print(f"     F1 improvement: +{delta_f1:.4f} (threshold: {args.min_improvement:.4f})")
            print(f"     Run: python scripts/promote_model.py --model-name {args.model_name}")
            print(f"              --metric test_f1_score --min-improvement {args.min_improvement}")
        elif delta_f1 < 0:
            print(f"  ❌ KEEP champion — challenger is WORSE by {-delta_f1:.4f} F1 points.")
            print("     Action: Investigate challenger training or increase dataset quality.")
        else:
            print(
                f"  ↔  No significant improvement (ΔF1={delta_f1:+.4f}, threshold={args.min_improvement:.4f})."
            )
            print("     Action: Continue canary period or increase sample size.")

    print()
    print("  Prometheus metrics visible in Grafana:")
    print("    model_predictions_total{model_version='production'}")
    print("    model_predictions_total{model_version='staging'}")
    print()
    print("  Nginx canary routing configured in docker/nginx/conf.d/default.conf")
    print("  upstream api_backend { server api:8000 weight=75; }")
    print("  upstream api_canary  { server api:8000 weight=25; }  # Staging weights")
    print("═" * 65)

    return 0


if __name__ == "__main__":
    sys.exit(main())
