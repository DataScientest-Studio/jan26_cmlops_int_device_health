#!/usr/bin/env python3
"""
Create an intentionally weak initial champion model for champion/challenger demos.

This script trains an initial Production model on deliberately degraded data
(label noise injected into the training split) so that the model has a
measurably lower ``test_f1_score`` than a model trained on clean data.

Workflow:
  1.  Load (or generate) the balanced baseline dataset.
  2.  Split 80 % train / 20 % test (stratified, random_state=42).
  3.  Inject ``--label-noise`` fraction of label flips into ONLY the train split.
      The test split remains clean so evaluation is fair.
  4.  Serialize the noisy train + clean test back into a temporary JSON file.
  5.  Call ``train_model()`` on the noisy dataset → lower test_f1_score.
  6.  Register the resulting model in MLflow as *Production* (the champion).

UC-05 happy path:
  After running this script, trigger retraining (``trigger_retraining.py --force``)
  which trains a *clean* model from the real baseline data and puts it in
  *Staging* (challenger).  Running ``promote_model.py`` then compares champion
  (low f1) vs challenger (high f1) → challenger wins and is promoted.

Usage:
    # Create weak champion (default: 30 % label noise)
    python scripts/create_degrading_champion.py

    # Custom noise level
    python scripts/create_degrading_champion.py --label-noise 0.40

    # Dry run (train without registering in MLflow)
    python scripts/create_degrading_champion.py --dry-run

    # Custom MLflow URI (auto-detected from .current_mode by default)
    python scripts/create_degrading_champion.py --mlflow-uri http://localhost:5001
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _build_noisy_dataset(
    source_path: Path,
    label_noise: float,
    seed: int = 42,
    *,
    test_fraction: float = 0.20,
) -> tuple[dict, dict]:
    """Return (noisy_train_json, clean_test_json) dicts ready for ``train_model()``.

    Args:
        source_path:    Path to a clean labeled JSON with ``signals`` array.
        label_noise:    Fraction of training-set labels to flip (0.0 – 1.0).
        seed:           Random seed for reproducibility.
        test_fraction:  Fraction to hold out as a clean test set.

    Returns:
        A pair of dicts, both with ``{"n_samples": N, "signals": [...]}``.
    """
    with open(source_path) as f:
        data = json.load(f)

    signals = data["signals"]
    rng = random.Random(seed)
    rng.shuffle(signals)

    n_test = max(2, int(len(signals) * test_fraction))

    # Stratified split: preserve approximate 50/50 ratio
    class0 = [s for s in signals if s.get("label") == 0]
    class1 = [s for s in signals if s.get("label") == 1]

    n_test_0 = max(1, int(n_test * len(class0) / len(signals)))
    n_test_1 = n_test - n_test_0

    test_signals = class0[:n_test_0] + class1[:n_test_1]
    train_signals = class0[n_test_0:] + class1[n_test_1:]
    rng.shuffle(train_signals)

    # Inject label noise into the TRAINING split only
    n_flip = max(0, int(len(train_signals) * label_noise))
    flip_indices = rng.sample(range(len(train_signals)), n_flip)
    for idx in flip_indices:
        orig = train_signals[idx]["label"]
        train_signals[idx] = {**train_signals[idx], "label": 1 - orig}

    n_flipped_0_to_1 = sum(1 for i in flip_indices if train_signals[i]["label"] == 1)
    print(
        f"[INFO] Label noise: flipped {n_flip}/{len(train_signals)} training labels "
        f"({label_noise:.0%}); {n_flipped_0_to_1} → unhealthy, "
        f"{n_flip - n_flipped_0_to_1} → healthy"
    )

    # Combine noisy train + clean test into a single dataset for train_model()
    combined_signals = train_signals + test_signals
    combined = {"n_samples": len(combined_signals), "signals": combined_signals}

    clean_test = {"n_samples": len(test_signals), "signals": test_signals}
    return combined, clean_test


def _resolve_mlflow_uri(override: str | None) -> str:
    """Resolve the correct MLflow tracking URI for the current deployment mode."""
    if override:
        return override

    mode_file = PROJECT_ROOT / ".current_mode"
    mode = mode_file.read_text().strip() if mode_file.exists() else "local"

    raw_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if mode == "cloud" and raw_uri and raw_uri.startswith("https://"):
        dagshub_user = os.environ.get("MLFLOW_TRACKING_USERNAME") or os.environ.get(
            "DAGSHUB_USER", ""
        )
        dagshub_token = os.environ.get("MLFLOW_TRACKING_PASSWORD") or os.environ.get(
            "DAGSHUB_TOKEN", ""
        )
        if dagshub_user and dagshub_token:
            os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_user
            os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        return raw_uri

    return "http://localhost:5001"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a weak initial champion model for champion/challenger demos."
    )
    parser.add_argument(
        "--label-noise",
        type=float,
        default=0.35,
        metavar="FRAC",
        help="Fraction of *training* labels to flip (default: 0.35 = 35%% noise).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        metavar="N",
        help="Number of baseline signals to generate if the file is missing.",
    )
    parser.add_argument(
        "--mlflow-uri",
        default=None,
        metavar="URI",
        help="MLflow tracking URI (auto-detected from .current_mode if not set).",
    )
    parser.add_argument(
        "--model-name",
        default="DeviceHealthModel",
        metavar="NAME",
        help="Registered model name in MLflow (default: DeviceHealthModel).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Train without registering in MLflow or changing the registry.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Create Degrading Champion Model")
    print("=" * 60)
    print(f"  Label noise  : {args.label_noise:.0%}")
    print(f"  Model name   : {args.model_name}")
    print(f"  Dry run      : {args.dry_run}")
    print()

    # ── 1. Ensure baseline dataset exists ─────────────────────────────────
    baseline_path = PROJECT_ROOT / "data" / "raw" / "dataset_baseline_full.json"
    if not baseline_path.exists():
        print(f"[INFO] Generating baseline data → {baseline_path}")
        import subprocess

        gen_proc = subprocess.run(
            [
                sys.executable,
                "scripts/generate_data.py",
                "generate",
                "--n-samples",
                str(args.n_samples),
                "--drift-scenario",
                "baseline",
                "--output-dir",
                str(baseline_path.parent),
                "--seed",
                str(args.seed),
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        if gen_proc.returncode != 0:
            print(f"[ERROR] Data generation failed: {gen_proc.stderr[:500]}")
            return 1
        print("[OK]   Baseline data generated.")

    # ── 2. Build noisy training dataset ───────────────────────────────────
    print(f"[INFO] Injecting {args.label_noise:.0%} label noise for champion training…")
    noisy_dataset, clean_test = _build_noisy_dataset(
        baseline_path, label_noise=args.label_noise, seed=args.seed
    )

    print(f"       Dataset size : {noisy_dataset['n_samples']} signals")
    n_c0 = sum(1 for s in noisy_dataset["signals"] if s.get("label") == 0)
    n_c1 = len(noisy_dataset["signals"]) - n_c0
    print(f"       Class dist.  : {n_c0} healthy, {n_c1} unhealthy (after noise injection)")

    # ── 3. Configure MLflow ────────────────────────────────────────────────
    mlflow_uri = _resolve_mlflow_uri(args.mlflow_uri)
    print(f"[INFO] MLflow tracking URI: {mlflow_uri}")

    import mlflow

    mlflow.set_tracking_uri(mlflow_uri)

    # ── 4. Train on noisy data ─────────────────────────────────────────────
    from src.training.train import train_model

    with tempfile.TemporaryDirectory() as tmp:
        noisy_path = Path(tmp) / "noisy_champion_data.json"
        with open(noisy_path, "w") as f:
            json.dump(noisy_dataset, f)

        model_output_path = PROJECT_ROOT / "models" / "degrading_champion.pkl"
        model_output_path.parent.mkdir(parents=True, exist_ok=True)

        print("\n[INFO] Training degrading champion model…")
        try:
            result = train_model(
                train_data_path=str(noisy_path),
                model_output_path=str(model_output_path),
                use_mlflow=not args.dry_run,
                model_version="degrading-champion-v1",
                test_size=0.2,
            )
        except Exception as exc:
            print(f"[ERROR] Training failed: {exc}")
            return 1

    print("\n[OK]   Champion training complete")
    print(f"       Train F1     : {result.get('train_f1_score', 'N/A'):.4f}")
    print(f"       Test  F1     : {result.get('test_f1_score', 'N/A'):.4f}")
    print(f"       Test  Acc    : {result.get('test_accuracy', 'N/A'):.4f}")
    if result.get("gold_standard_path"):
        print(f"       Validation set: {result['gold_standard_path']}")

    if not result.get("test_f1_score", 1.0) < 0.95:
        print(
            "\n[WARN] Champion test F1 is not significantly lower than 0.95 "
            f"({result.get('test_f1_score', 'N/A'):.4f}). "
            "Try --label-noise 0.50 for a more degraded model."
        )

    if args.dry_run:
        print("\n[INFO] Dry run — skipping MLflow registration.")
        return 0

    # ── 5. Register in MLflow and promote to Production ───────────────────
    run_id = result.get("mlflow_run_id")
    if not run_id:
        print("[WARN] No MLflow run_id — skipping model registration.")
        return 0

    from mlflow.tracking import MlflowClient

    from src.training.registry import get_production_models, promote_model

    client = MlflowClient()
    model_name = args.model_name

    # Create registered model if it doesn't exist
    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(
            model_name,
            description="Device health classifier — created by create_degrading_champion.py",
        )

    try:
        mv = client.create_model_version(
            name=model_name,
            source=f"runs:/{run_id}/model",
            run_id=run_id,
            description=(
                f"Degrading champion (label_noise={args.label_noise:.0%}) — "
                f"test_f1={result.get('test_f1_score', 0):.4f}"
            ),
        )
        version = int(mv.version)
        print(f"\n[OK]   Registered {model_name} v{version}")
    except Exception as exc:
        print(f"[ERROR] Could not create model version: {exc}")
        return 1

    # Archive any existing Production models, promote this one
    existing_prod = get_production_models(model_name)
    if existing_prod:
        print(
            f"[INFO] Archiving existing Production model(s): {[m['version'] for m in existing_prod]}"
        )
        for prod in existing_prod:
            with contextlib.suppress(Exception):
                promote_model(model_name, version=prod["version"], stage="Archived")

    promote_model(model_name, version=version, stage="Production")
    print(f"[OK]   Promoted {model_name} v{version} → Production (champion)")

    print(
        f"\n[INFO] Next steps for champion/challenger demo:"
        f"\n       1. Run: python scripts/trigger_retraining.py --force"
        f"\n          → Trains a clean challenger (Staging) on baseline data"
        f"\n       2. Run: python scripts/promote_model.py --model-name {model_name}"
        f"  --min-improvement 0.02 --tracking-uri {mlflow_uri}"
        f"\n          → Compares champion (F1≈{result.get('test_f1_score', 0):.2f})"
        f" vs challenger (F1≈high)  → challenger wins!"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
