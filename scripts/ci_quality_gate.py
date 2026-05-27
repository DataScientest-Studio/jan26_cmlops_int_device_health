"""CI model quality gate — train on golden reference signals and verify thresholds.

This script is called by ``.github/workflows/model-quality-gate.yml``.
It proves that the **current training code** (not a pre-trained artifact) still
produces a model that meets minimum quality thresholds when trained on the fixed
golden reference dataset.

What it tests
-------------
Not the infrastructure (no PostgreSQL, no MLflow, no DagsHub, no Docker).
It tests: does src/features/ + src/training/ still produce a model with
acceptable accuracy and F1 when run against a fixed synthetic dataset?

This catches regressions in:
  - Feature extraction code (src/signal_processing/feature_extractor.py)
  - Training code and hyperparameters (params.yaml train.*)
  - Model selection / classifier type

Pipeline (identical to greenfield bootstrap)
---------------------------------------------
1. Load ``data/ci/quality_gate_signals.csv`` (raw amplitudes + labels)
2. Reconstruct SignalData objects (time = np.linspace(0, 100, 101))
3. Extract features using src/signal_processing/feature_extractor.py
4. Split 80% train / 20% test (random_state from params.yaml)
5. Scale with StandardScaler
6. Train LogisticRegression (params from params.yaml train.*)
7. Evaluate accuracy + F1 on test split
8. Assert accuracy ≥ params.yaml quality_gate.min_accuracy
9. Assert F1 ≥ params.yaml quality_gate.min_f1
10. Write JSON report to reports/ci_quality_gate_results.json

Exit codes
----------
0 — all thresholds passed
1 — one or more thresholds failed (also printed to stderr for CI log)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_SIGNALS_CSV = PROJECT_ROOT / "data" / "ci" / "quality_gate_signals.csv"
_PARAMS_YAML = PROJECT_ROOT / "params.yaml"
_REPORTS_DIR = PROJECT_ROOT / "reports"
_RESULTS_JSON = _REPORTS_DIR / "ci_quality_gate_results.json"

_N_POINTS = 101  # must match generate_ci_quality_gate_data.py


# ── Feature names — MUST MATCH FEATURE_NAMES in ci_generate_bootstrap_model.py ─

FEATURE_NAMES = [
    "fwhm",
    "peak_height",
    "peak_area",
    "noise_level",
    "snr",
    "peak_center",
]


def _load_params() -> dict:
    with _PARAMS_YAML.open() as f:
        return yaml.safe_load(f)


def _load_signals() -> tuple[list[list[float]], list[int], list[str]]:
    """Load quality gate signals from CSV.

    Returns:
        amplitudes: list of 101-element amplitude arrays
        labels: list of integer labels (0=healthy, 1=unhealthy)
        shape_types: list of shape_type strings
    """
    if not _SIGNALS_CSV.exists():
        print(
            f"ERROR: Quality gate signals not found: {_SIGNALS_CSV}\n"
            "  Run: python scripts/generate_ci_quality_gate_data.py\n"
            "  Then commit data/ci/quality_gate_signals.csv to git.",
            file=sys.stderr,
        )
        sys.exit(1)

    amplitudes: list[list[float]] = []
    labels: list[int] = []
    shape_types: list[str] = []

    with _SIGNALS_CSV.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(int(row["label"]))
            shape_types.append(row["shape_type"])
            amps = [float(row[f"a_{i}"]) for i in range(_N_POINTS)]
            amplitudes.append(amps)

    return amplitudes, labels, shape_types


def _extract_all_features(
    amplitudes: list[list[float]],
    shape_types: list[str],
) -> np.ndarray:
    """Extract the 6-feature vector from every signal.

    Returns feature matrix of shape (N, 6).
    """
    from src.signal_processing.feature_extractor import extract_features
    from src.signal_processing.signal_models import SignalData

    time = np.linspace(0, 100, _N_POINTS)
    rows: list[list[float]] = []
    for amps, st in zip(amplitudes, shape_types, strict=False):
        sd = SignalData(time=list(time), amplitude=amps, shape_type=st)
        feats = extract_features(sd)
        rows.append([feats.get(name) or 0.0 for name in FEATURE_NAMES])
    return np.array(rows, dtype=float)


def run_quality_gate() -> int:
    """Run the quality gate.

    Returns:
        0 if all thresholds pass, 1 if any fail.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    params = _load_params()
    train_cfg = params.get("train", {})
    qg_cfg = params.get("quality_gate", {})

    # Thresholds
    min_accuracy = float(qg_cfg.get("min_accuracy", 0.80))
    min_f1 = float(qg_cfg.get("min_f1", 0.75))

    # Training hyperparameters — MUST match greenfield defaults
    random_state = int(train_cfg.get("random_state", 42))
    test_size = float(train_cfg.get("test_size", 0.2))
    C = float(train_cfg.get("C", 1.0))
    penalty = str(train_cfg.get("penalty", "l2"))
    solver = str(train_cfg.get("solver", "lbfgs"))
    max_iter = int(train_cfg.get("max_iter", 1000))

    print("=" * 60)
    print("  CI Model Quality Gate")
    print("=" * 60)
    print(f"\nThresholds:  accuracy ≥ {min_accuracy:.2f}  |  F1 ≥ {min_f1:.2f}")
    print(
        f"Classifier:  LogisticRegression(C={C}, penalty={penalty!r}, "
        f"solver={solver!r}, max_iter={max_iter})"
    )
    print(
        f"Split:       {int((1 - test_size) * 100)}% train / {int(test_size * 100)}% test "
        f"(random_state={random_state})"
    )
    print()

    # ── Load signals ──────────────────────────────────────────────
    print(f"Loading signals from {_SIGNALS_CSV.relative_to(PROJECT_ROOT)}…")
    amplitudes, labels, shape_types = _load_signals()
    print(f"  {len(labels)} signals ({labels.count(0)} healthy, {labels.count(1)} unhealthy)")

    # ── Extract features ──────────────────────────────────────────
    print("Extracting features…")
    X = _extract_all_features(amplitudes, shape_types)  # noqa: N806
    y = np.array(labels, dtype=int)
    print(f"  Feature matrix: {X.shape}")

    # ── Split ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(  # noqa: N806
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"  Train: {len(y_train)} | Test: {len(y_test)}")

    # ── Scale ─────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # noqa: N806
    X_test_scaled = scaler.transform(X_test)  # noqa: N806

    # ── Train ─────────────────────────────────────────────────────
    print("Training classifier…")
    clf = LogisticRegression(
        C=C,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
    )
    clf.fit(X_train_scaled, y_train)

    # ── Evaluate ──────────────────────────────────────────────────
    y_pred = clf.predict(X_test_scaled)
    accuracy = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    print("\nResults:")
    acc_status = "✅ PASS" if accuracy >= min_accuracy else "❌ FAIL"
    f1_status = "✅ PASS" if f1 >= min_f1 else "❌ FAIL"
    print(f"  Accuracy: {accuracy:.4f}  (threshold: {min_accuracy:.2f})  {acc_status}")
    print(f"  F1 score: {f1:.4f}  (threshold: {min_f1:.2f})  {f1_status}")
    print()

    # ── Write report ──────────────────────────────────────────────
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "accuracy": accuracy,
        "f1_score": f1,
        "min_accuracy": min_accuracy,
        "min_f1": min_f1,
        "accuracy_pass": accuracy >= min_accuracy,
        "f1_pass": f1 >= min_f1,
        "overall_pass": accuracy >= min_accuracy and f1 >= min_f1,
        "n_test_samples": int(len(y_test)),
        "n_healthy": int(labels.count(0)),
        "n_unhealthy": int(labels.count(1)),
        "classifier": f"LogisticRegression(C={C}, penalty={penalty!r})",
        "feature_names": FEATURE_NAMES,
    }
    with _RESULTS_JSON.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"Report written to {_RESULTS_JSON.relative_to(PROJECT_ROOT)}")

    # ── Assert ────────────────────────────────────────────────────
    failed: list[str] = []
    if not report["accuracy_pass"]:
        failed.append(
            f"Accuracy {accuracy:.4f} < threshold {min_accuracy:.2f} "
            f"— training pipeline regression detected!"
        )
    if not report["f1_pass"]:
        failed.append(
            f"F1 score {f1:.4f} < threshold {min_f1:.2f} — training pipeline regression detected!"
        )

    if failed:
        print("=" * 60, file=sys.stderr)
        print("  ❌  QUALITY GATE FAILED", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        for msg in failed:
            print(f"  • {msg}", file=sys.stderr)
        print(
            "\n  Investigate: did a recent change to src/features/, src/training/,\n"
            "  or params.yaml cause a performance regression?\n"
            "  Run locally: python scripts/ci_quality_gate.py",
            file=sys.stderr,
        )
        return 1

    print("=" * 60)
    print("  ✅  QUALITY GATE PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(run_quality_gate())
