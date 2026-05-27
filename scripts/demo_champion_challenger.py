#!/usr/bin/env python3
"""
UC-05 Self-Contained Champion/Challenger Demo.

Creates a full champion vs challenger scenario entirely within this script,
without requiring manual setup steps.  Two scenarios are supported:

  promotion:
      Registers a WEAK champion (under-regularised logistic regression, very
      few iterations) then creates a STRONG challenger (properly tuned random
      forest).  The challenger beats the champion → promotion is executed and
      the challenger becomes Production.

  no-promotion:
      Registers a STRONG champion (random forest) then creates a WEAK
      challenger (under-regularised logistic regression).  The challenger
      cannot beat the champion → no promotion; champion stays in Production.

Works with:
  - Local MLflow 2.x container: uses ``artifact_path=`` (NOT ``name=``) to
    avoid the 3.x-only LoggedModel API that returns 404 on v2.x servers.
  - DagsHub MLflow: uses ``client.create_model_version(source=artifact_uri)``
    to bypass ``search_logged_models`` which DagsHub does not support.

Usage:
    # Challenger wins (promotion path)
    python scripts/demo_uc05_challenger.py --scenario promotion

    # Champion wins (no-promotion path)
    python scripts/demo_uc05_challenger.py --scenario no-promotion

    # Dry run (train + evaluate, skip MLflow writes)
    python scripts/demo_uc05_challenger.py --scenario promotion --dry-run
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

FEATURE_NAMES = [
    "fwhm",
    "peak_height",
    "peak_area",
    "noise_level",
    "snr",
    "peak_center",
]
_DEFAULT_MODEL_NAME = os.environ.get("MODEL_REGISTRY_NAME", "device_health_classifier")
_BASELINE_PATH = PROJECT_ROOT / "data" / "raw" / "dataset_baseline_full.json"
_MIN_IMPROVEMENT = 0.02
_EXPERIMENT_NAME = "mlops_device_health"


# ── Data helpers ──────────────────────────────────────────────────────────────


def _ensure_baseline_data(n_samples: int = 200) -> Path:
    """Generate ``dataset_baseline_full.json`` if absent; return its path."""
    if _BASELINE_PATH.exists():
        return _BASELINE_PATH
    print(f"[demo_uc05] Generating baseline data → {_BASELINE_PATH.name}")
    import subprocess

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/generate_data.py",
            "generate",
            "--n-samples",
            str(n_samples),
            "--drift-scenario",
            "baseline",
            "--output-dir",
            str(_BASELINE_PATH.parent),
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Data generation failed:\n{proc.stderr[:600]}")
    if not _BASELINE_PATH.exists():
        raise FileNotFoundError(f"Generation ran but {_BASELINE_PATH} not found.")
    return _BASELINE_PATH


def _load_feature_matrix(json_path: Path) -> tuple:
    """Return (X, y) numpy arrays from a raw signal JSON file."""
    import numpy as np

    from src.signal_processing.feature_extractor import extract_features
    from src.signal_processing.signal_models import SignalData

    with json_path.open() as fh:
        data = json.load(fh)

    rows: list[list[float]] = []
    labels: list[int] = []
    for sig in data["signals"]:
        label = sig.get("label")
        amp = sig.get("amplitude", [])
        if label is None or not amp:
            continue
        sd = SignalData(
            time=sig.get("time", list(range(len(amp)))),
            amplitude=amp,
            shape_type=sig.get("shape_type", "gaussian"),  # type: ignore[arg-type]
        )
        feats = extract_features(sd)
        rows.append([feats.get(name) or 0.0 for name in FEATURE_NAMES])
        labels.append(int(label))

    return np.array(rows, dtype=float), np.array(labels, dtype=int)


def _train_test_split(X, y, test_fraction: float = 0.20, seed: int = 42) -> tuple:
    """Simple stratified train/test split without sklearn dependency."""
    import numpy as np

    rng = np.random.default_rng(seed)
    class0 = np.where(y == 0)[0]
    class1 = np.where(y == 1)[0]
    rng.shuffle(class0)
    rng.shuffle(class1)

    n_test0 = max(2, int(len(class0) * test_fraction))
    n_test1 = max(2, int(len(class1) * test_fraction))

    test_idx = np.concatenate([class0[:n_test0], class1[:n_test1]])
    train_idx = np.concatenate([class0[n_test0:], class1[n_test1:]])

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ── Model factories ────────────────────────────────────────────────────────────


def _make_strong_model(seed: int = 42):
    """Return a properly tuned RandomForestClassifier (high F1)."""
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(n_estimators=100, max_depth=None, random_state=seed)


def _make_weak_model(seed: int = 42):
    """Return a heavily regularised, severely under-trained LogisticRegression.

    ``max_iter=3`` forces the solver to stop long before convergence.
    ``C=1e-6``  (very high regularisation) pushes weights toward zero —
    the model effectively ignores the features and predicts based on the
    class prior, yielding F1 ≈ 0.50-0.67 on balanced data.
    """
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=3, C=1e-6, random_state=seed, solver="lbfgs")


def _fit_and_eval(model, X_train, y_train, X_test, y_test) -> tuple[dict[str, float], object]:
    """Fit *model* on train/test splits; return (metrics, fitted_pipeline).

    The pipeline wraps ``StandardScaler → model`` so that the MLflow artifact
    can perform inference on raw (unscaled) features.
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", model)])

    with warnings.catch_warnings():
        # Suppress ConvergenceWarning for the deliberately weak model
        warnings.simplefilter("ignore")
        pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_train_pred = pipeline.predict(X_train)

    # Full metrics matching train.py output
    metrics: dict[str, float] = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "train_accuracy": float(accuracy_score(y_train, y_train_pred)),
        "train_f1_score": float(f1_score(y_train, y_train_pred, zero_division=0)),
        "n_test_samples": float(len(y_test)),
        "gold_standard_test_size": float(len(y_test)),
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        metrics["true_negatives"] = float(cm[0][0])
        metrics["false_positives"] = float(cm[0][1])
        metrics["false_negatives"] = float(cm[1][0])
        metrics["true_positives"] = float(cm[1][1])

    metrics["primary_metric"] = metrics["test_f1_score"]

    return metrics, pipeline


# ── MLflow helpers ──────────────────────────────────────────────────────────


def _resolve_uri() -> str:
    """Pick the correct MLflow tracking URI from env / .current_mode."""
    mode_file = PROJECT_ROOT / ".current_mode"
    mode = mode_file.read_text().strip() if mode_file.exists() else "local"
    raw = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if mode == "cloud" and raw.startswith("https://"):
        for key, fallback_env in [
            ("MLFLOW_TRACKING_USERNAME", "DAGSHUB_USER"),
            ("MLFLOW_TRACKING_PASSWORD", "DAGSHUB_TOKEN"),
        ]:
            if not os.environ.get(key):
                os.environ[key] = os.environ.get(fallback_env, "")
        return raw
    return "http://localhost:5001"


def _archive_all(client, model_name: str) -> None:
    """Clear champion/challenger aliases for a clean slate (MLflow v3 — no legacy stages)."""
    for alias in ("champion", "challenger"):
        try:
            client.delete_registered_model_alias(model_name, alias)
            print(f"[demo_uc05]   Removed alias '{alias}' from {model_name}.")
        except Exception:
            pass  # alias not set — nothing to remove


def _log_and_register(
    pipeline,
    metrics: dict[str, float],
    *,
    model_name: str,
    run_name: str,
    target_stage: str,
    model_label: str,
    dry_run: bool,
) -> str:
    """Log sklearn pipeline to MLflow, register it, and set its stage.

    Uses ``artifact_path=`` (not ``name=``) so the call is compatible with both
    MLflow 2.x and 3.x tracking servers.  ``name=`` triggers the LoggedModel
    API (``/api/2.0/mlflow/logged-models``) which returns 404 on v2.x servers.

    Uses ``client.create_model_version(source=artifact_uri)`` with the
    resolved storage URI so that MLflow 3.x does **not** call
    ``search_logged_models()`` — an API endpoint that DagsHub does not support.

    Returns:
        The registered version string, or ``run_id[:8]`` for dry runs.
    """
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient

    clf_step = pipeline.named_steps.get("clf", pipeline)
    mlflow.set_experiment(_EXPERIMENT_NAME)
    client = MlflowClient()

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "algorithm": type(clf_step).__name__,
                "features": ",".join(FEATURE_NAMES),
                "demo_scenario": run_name,
                "model_label": model_label,
            }
        )
        mlflow.log_metrics(metrics)

        # ── Lineage tags (git, DVC, Airflow) ─────────────────────────────
        # These tags are read by _enrich_artifact_lineage() at prediction
        # time so that every prediction row carries full traceability IDs.
        from src.training.mlflow_utils import get_dvc_data_version, get_git_commit_hash

        _git_sha = get_git_commit_hash()
        _dvc_hash = get_dvc_data_version(_BASELINE_PATH)
        if _git_sha:
            mlflow.set_tag("git_commit", _git_sha)
        if _dvc_hash:
            mlflow.set_tag("dvc_data_version", _dvc_hash)

        # Capture artifact URI inside the run context BEFORE attempting upload.
        # The URI is stored in the run record as metadata; create_model_version
        # references it but does NOT validate the path on the tracking server
        # during creation — only during model load.
        run_id = run.info.run_id
        artifact_uri = mlflow.get_artifact_uri("model")

        # Attempt to upload the model artifact.  This step:
        #  - Succeeds on cloud (DagsHub) which uses HTTP-based artifact storage.
        #  - Fails silently on the local Docker MLflow 2.x server because the
        #    artifact store is inside the container (`file:///mlflow/artifacts/`)
        #    and is not writable from the macOS host.  The promotion demo does
        #    not load the model; it only compares logged metrics — so missing
        #    artifact files do not affect the demo outcome.
        try:
            import tempfile

            with tempfile.TemporaryDirectory() as _tmp:
                _model_local = os.path.join(_tmp, "model")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mlflow.sklearn.save_model(pipeline, _model_local)
                mlflow.log_artifacts(_model_local, artifact_path="model")
        except Exception as _art_err:
            print(
                f"[demo_uc05]   NOTE: artifact upload skipped "
                f"({type(_art_err).__name__} — normal for local Docker server). "
                f"Metrics are logged; promotion comparison is unaffected."
            )

    if dry_run:
        print(
            f"[demo_uc05]   [DRY RUN] Logged run {run_id[:8]} "
            f"(f1={metrics['test_f1_score']:.4f}) — registry skip."
        )
        return run_id[:8]

    with contextlib.suppress(Exception):
        client.create_registered_model(
            model_name,
            description="MLOps Device Health Classifier — demo_uc05_challenger",
        )

    version_obj = client.create_model_version(
        name=model_name,
        source=artifact_uri,
        run_id=run_id,
        tags={"demo": "uc05", "model_label": model_label, "target_stage": target_stage},
    )
    version = str(version_obj.version)
    print(
        f"[demo_uc05]   Registered v{version} "
        f"(f1={metrics['test_f1_score']:.4f}) → setting alias for {target_stage}…"
    )
    stage_to_alias = {"Production": "champion", "Staging": "challenger"}
    alias = stage_to_alias.get(target_stage)
    if alias:
        client.set_registered_model_alias(model_name, alias, version)
    return version


# ── Scenario runners ──────────────────────────────────────────────────────────


def run_promotion_scenario(model_name: str, *, dry_run: bool = False, seed: int = 42) -> dict:
    """Champion is weak; challenger is strong → challenger is promoted.

    Champion model:  LogisticRegression(max_iter=3, C=1e-6) — deliberately
                     under-trained and over-regularised.  F1 ≈ 0.50–0.70.

    Challenger model: RandomForestClassifier(n_estimators=100) — properly
                      configured ensemble.  F1 ≈ 0.92–1.00.

    Expected outcome: challenger f1 > champion f1 + 0.02 → PROMOTED.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    from src.training.promotion import auto_promote_model

    uri = _resolve_uri()
    mlflow.set_tracking_uri(uri)
    print(f"[demo_uc05] MLflow tracking URI : {uri}")
    print("[demo_uc05] Scenario            : promotion")

    baseline = _ensure_baseline_data()
    X, y = _load_feature_matrix(baseline)
    X_train, X_test, y_train, y_test = _train_test_split(X, y, seed=seed)
    print(f"[demo_uc05] Dataset: {len(y)} signals  (train={len(y_train)}, test={len(y_test)})")

    # ── Step 1: train models ─────────────────────────────────────────────────
    print("\n[demo_uc05] ── Step 1 / 4  Train WEAK champion ──────────────────────")
    weak_model = _make_weak_model(seed=seed)
    champ_metrics, champ_pipeline = _fit_and_eval(weak_model, X_train, y_train, X_test, y_test)
    print(
        f"[demo_uc05]   Champion  "
        f"accuracy={champ_metrics['test_accuracy']:.4f}  "
        f"f1={champ_metrics['test_f1_score']:.4f}  "
        f"(LogReg max_iter=3, C=1e-6 — deliberately weak)"
    )

    print("\n[demo_uc05] ── Step 2 / 4  Train STRONG challenger ──────────────────")
    strong_model = _make_strong_model(seed=seed)
    chall_metrics, chall_pipeline = _fit_and_eval(strong_model, X_train, y_train, X_test, y_test)
    print(
        f"[demo_uc05]   Challenger "
        f"accuracy={chall_metrics['test_accuracy']:.4f}  "
        f"f1={chall_metrics['test_f1_score']:.4f}  "
        f"(RandomForest n_estimators=100 — properly tuned)"
    )

    delta = chall_metrics["test_f1_score"] - champ_metrics["test_f1_score"]
    print(f"\n[demo_uc05]   Δ F1 = {delta:+.4f}  (threshold: >{_MIN_IMPROVEMENT:.2f})")

    # ── Step 3: register in MLflow ───────────────────────────────────────────
    print("\n[demo_uc05] ── Step 3 / 4  Register models in MLflow ─────────────────")
    client = MlflowClient()
    if not dry_run:
        _archive_all(client, model_name)

    champ_ver = _log_and_register(
        champ_pipeline,
        champ_metrics,
        model_name=model_name,
        run_name="uc05_weak_champion",
        target_stage="Production",
        model_label="weak_champion",
        dry_run=dry_run,
    )
    print(f"[demo_uc05]   Champion  → Production  v{champ_ver}")

    chall_ver = _log_and_register(
        chall_pipeline,
        chall_metrics,
        model_name=model_name,
        run_name="uc05_strong_challenger",
        target_stage="Staging",
        model_label="strong_challenger",
        dry_run=dry_run,
    )
    print(f"[demo_uc05]   Challenger → Staging    v{chall_ver}")

    # ── Step 4: evaluate promotion ───────────────────────────────────────────
    print("\n[demo_uc05] ── Step 4 / 4  Evaluate promotion ───────────────────────")
    result = auto_promote_model(
        model_name=model_name,
        metric_name="test_f1_score",
        min_improvement=_MIN_IMPROVEMENT,
        archive_old_champion=True,
        dry_run=dry_run,
    )

    _print_summary(
        scenario="promotion",
        champ_ver=champ_ver,
        chall_ver=chall_ver,
        champ_label="WEAK champion (LogReg max_iter=3, C=1e-6)",
        chall_label="STRONG challenger (RandomForest n_estimators=100)",
        champ_metrics=champ_metrics,
        chall_metrics=chall_metrics,
        result=result,
    )
    return {
        "scenario": "promotion",
        "champion_metrics": champ_metrics,
        "challenger_metrics": chall_metrics,
        "delta_f1": delta,
        "promotion_result": result,
        "promoted": result["promoted"],
    }


def run_no_promotion_scenario(model_name: str, *, dry_run: bool = False, seed: int = 42) -> dict:
    """Champion is strong; challenger is weak → champion is retained.

    Champion model:  RandomForestClassifier(n_estimators=100).  F1 ≈ 0.92–1.00.

    Challenger model: LogisticRegression(max_iter=3, C=1e-6) — deliberately
                      under-trained and over-regularised.  F1 ≈ 0.50–0.70.

    Expected outcome: challenger f1 ≤ champion f1 + 0.02 → NOT PROMOTED.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    from src.training.promotion import auto_promote_model

    uri = _resolve_uri()
    mlflow.set_tracking_uri(uri)
    print(f"[demo_uc05] MLflow tracking URI : {uri}")
    print("[demo_uc05] Scenario            : no-promotion")

    baseline = _ensure_baseline_data()
    X, y = _load_feature_matrix(baseline)
    X_train, X_test, y_train, y_test = _train_test_split(X, y, seed=seed)
    print(f"[demo_uc05] Dataset: {len(y)} signals  (train={len(y_train)}, test={len(y_test)})")

    # ── Step 1: train models ─────────────────────────────────────────────────
    print("\n[demo_uc05] ── Step 1 / 4  Train STRONG champion ────────────────────")
    strong_model = _make_strong_model(seed=seed)
    champ_metrics, champ_pipeline = _fit_and_eval(strong_model, X_train, y_train, X_test, y_test)
    print(
        f"[demo_uc05]   Champion  "
        f"accuracy={champ_metrics['test_accuracy']:.4f}  "
        f"f1={champ_metrics['test_f1_score']:.4f}  "
        f"(RandomForest n_estimators=100 — properly tuned)"
    )

    print("\n[demo_uc05] ── Step 2 / 4  Train WEAK challenger ───────────────────")
    weak_model = _make_weak_model(seed=seed)
    chall_metrics, chall_pipeline = _fit_and_eval(weak_model, X_train, y_train, X_test, y_test)
    print(
        f"[demo_uc05]   Challenger "
        f"accuracy={chall_metrics['test_accuracy']:.4f}  "
        f"f1={chall_metrics['test_f1_score']:.4f}  "
        f"(LogReg max_iter=3, C=1e-6 — deliberately weak)"
    )

    delta = chall_metrics["test_f1_score"] - champ_metrics["test_f1_score"]
    print(f"\n[demo_uc05]   Δ F1 = {delta:+.4f}  (threshold: >{_MIN_IMPROVEMENT:.2f})")

    # ── Step 3: register in MLflow ───────────────────────────────────────────
    print("\n[demo_uc05] ── Step 3 / 4  Register models in MLflow ─────────────────")
    client = MlflowClient()
    if not dry_run:
        _archive_all(client, model_name)

    champ_ver = _log_and_register(
        champ_pipeline,
        champ_metrics,
        model_name=model_name,
        run_name="uc05_strong_champion",
        target_stage="Production",
        model_label="strong_champion",
        dry_run=dry_run,
    )
    print(f"[demo_uc05]   Champion  → Production  v{champ_ver}")

    chall_ver = _log_and_register(
        chall_pipeline,
        chall_metrics,
        model_name=model_name,
        run_name="uc05_weak_challenger",
        target_stage="Staging",
        model_label="weak_challenger",
        dry_run=dry_run,
    )
    print(f"[demo_uc05]   Challenger → Staging    v{chall_ver}")

    # ── Step 4: evaluate promotion ───────────────────────────────────────────
    print("\n[demo_uc05] ── Step 4 / 4  Evaluate promotion ───────────────────────")
    result = auto_promote_model(
        model_name=model_name,
        metric_name="test_f1_score",
        min_improvement=_MIN_IMPROVEMENT,
        archive_old_champion=False,
        dry_run=dry_run,
    )

    _print_summary(
        scenario="no-promotion",
        champ_ver=champ_ver,
        chall_ver=chall_ver,
        champ_label="STRONG champion (RandomForest n_estimators=100)",
        chall_label="WEAK challenger (LogReg max_iter=3, C=1e-6)",
        champ_metrics=champ_metrics,
        chall_metrics=chall_metrics,
        result=result,
    )
    return {
        "scenario": "no-promotion",
        "champion_metrics": champ_metrics,
        "challenger_metrics": chall_metrics,
        "delta_f1": delta,
        "promotion_result": result,
        "promoted": result["promoted"],
    }


# ── Summary printer ───────────────────────────────────────────────────────────


def _print_summary(
    *,
    scenario: str,
    champ_ver: str,
    chall_ver: str,
    champ_label: str,
    chall_label: str,
    champ_metrics: dict,
    chall_metrics: dict,
    result: dict,
) -> None:
    width = 64
    print("\n" + "═" * width)
    title = "UC-05 Promotion" if scenario == "promotion" else "UC-05 No-Promotion"
    print(f"  {title} Scenario — Summary")
    print("═" * width)
    print(
        f"  Champion  v{champ_ver:>3}  F1={champ_metrics['test_f1_score']:.4f}"
        f"  acc={champ_metrics['test_accuracy']:.4f}"
    )
    print(f"              → {champ_label}")
    print(
        f"  Challenger v{chall_ver:>3}  F1={chall_metrics['test_f1_score']:.4f}"
        f"  acc={chall_metrics['test_accuracy']:.4f}"
    )
    print(f"              → {chall_label}")
    print()

    if result.get("promoted"):
        new_v = result.get("new_champion_version", "?")
        delta = result.get("decision", {}).get("improvement", 0.0)
        print(f"  ✅  PROMOTED — Challenger v{new_v} is now Production")
        print(f"     Δ F1 = {delta:+.4f}  (threshold: >{_MIN_IMPROVEMENT:.2f})")
    else:
        old_v = result.get("old_champion_version", "?")
        print(f"  🛡️  CHAMPION RETAINED — v{old_v} stays in Production")
        print(f"     Reason: {result.get('reason', 'unknown')}")
    print("═" * width)


# ── CLI entry point ───────────────────────────────────────────────────────────


def main() -> int:  # noqa: PLR0912
    parser = argparse.ArgumentParser(
        description="UC-05 self-contained champion/challenger demo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--scenario",
        choices=["promotion", "no-promotion"],
        default="promotion",
        help="Which scenario to run (default: promotion)",
    )
    parser.add_argument(
        "--model-name",
        default=_DEFAULT_MODEL_NAME,
        metavar="NAME",
        help=f"Registered model name (default: {_DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Train and evaluate without writing to the MLflow registry",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    print(
        f"[demo_uc05] UC-05 demo  scenario={args.scenario}  "
        f"model={args.model_name}  dry_run={args.dry_run}"
    )

    try:
        if args.scenario == "promotion":
            result = run_promotion_scenario(args.model_name, dry_run=args.dry_run, seed=args.seed)
        else:
            result = run_no_promotion_scenario(
                args.model_name, dry_run=args.dry_run, seed=args.seed
            )
    except Exception as exc:
        print(f"\n[demo_uc05] ERROR: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1

    # Exit 2 for unexpected outcomes (demo config issue, not a code bug)
    if not args.dry_run:
        if args.scenario == "promotion" and not result["promoted"]:
            print(
                "\n[demo_uc05] WARNING: Expected promotion but outcome was "
                "no-promotion.  Delta F1 may be too small.",
                file=sys.stderr,
            )
            return 2
        if args.scenario == "no-promotion" and result["promoted"]:
            print(
                "\n[demo_uc05] WARNING: Expected no-promotion but outcome was "
                "promotion.  Weak model may have fitted better than expected.",
                file=sys.stderr,
            )
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
