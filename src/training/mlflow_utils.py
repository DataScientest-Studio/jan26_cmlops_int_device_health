"""
MLflow utilities for experiment tracking and model registry.

Provides:
- MLflow configuration and setup
- Experiment context management
- Auto-tagging with Git metadata
- DVC data version tracking
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient


def get_git_commit_hash() -> str | None:
    """
    Get current Git commit hash.

    Checks GIT_SHA / GIT_COMMIT_SHA environment variables first (set when
    running inside Docker where the .git directory is not mounted) before
    falling back to the ``git`` subprocess.

    Returns:
        Git commit hash (short SHA) or None if not in git repo
    """
    # Docker containers typically have no .git directory.  Inject the SHA via
    # an environment variable (set by ``make cloud`` / ``make ghcr``).
    for env_key in ("GIT_SHA", "GIT_COMMIT_SHA", "GIT_COMMIT"):
        val = os.environ.get(env_key, "").strip()
        if val and val not in ("unknown", "HEAD"):
            return val
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def get_git_branch() -> str | None:
    """
    Get current Git branch name.

    Returns:
        Git branch name or None if not in git repo
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def get_dvc_data_version(data_path: Path | str) -> str | None:
    """
    Get DVC version hash for a data file or directory.

    Checks both standalone ``.dvc`` files and ``dvc.lock`` (for pipeline
    outputs defined in ``dvc.yaml``).

    Args:
        data_path: Path to DVC-tracked file or directory

    Returns:
        DVC hash (MD5) or None if not DVC-tracked
    """
    data_path = Path(data_path)

    # 1. Check standalone .dvc file
    dvc_file = Path(str(data_path) + ".dvc")
    if dvc_file.exists():
        try:
            with open(dvc_file) as f:
                for line in f:
                    if "md5:" in line:
                        return line.split("md5:")[-1].strip()
        except Exception:
            pass

    # 2. Fallback: check dvc.lock for pipeline outputs
    lock_file = data_path.parent / "dvc.lock"
    if not lock_file.exists():
        # Try project root (walk up to find dvc.lock)
        for parent in data_path.resolve().parents:
            candidate = parent / "dvc.lock"
            if candidate.exists():
                lock_file = candidate
                break

    if lock_file.exists():
        try:
            import yaml

            lock_data = yaml.safe_load(lock_file.read_text())
            # Compute relative path from dvc.lock's directory
            lock_dir = lock_file.parent
            try:
                rel = str(data_path.resolve().relative_to(lock_dir.resolve()))
            except ValueError:
                rel = str(data_path)

            for _stage, stage_data in (lock_data or {}).get("stages", {}).items():
                for out in stage_data.get("outs", []):
                    if out.get("path") == rel:
                        return out.get("md5")
        except Exception:
            pass

    return None


def setup_mlflow(
    tracking_uri: str | None = None,
    experiment_name: str = "device_health_classifier",
) -> str:
    """
    Configure MLflow tracking server and experiment.

    Args:
        tracking_uri: MLflow tracking server URI (None = local ./mlruns)
        experiment_name: Name of MLflow experiment

    Returns:
        Experiment ID

    Example:
        >>> setup_mlflow(tracking_uri="http://localhost:5000")
        >>> # Now mlflow.start_run() will use this configuration
    """
    # Set tracking URI (default: local ./mlruns directory)
    if tracking_uri is None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")

    mlflow.set_tracking_uri(tracking_uri)

    # Create or get experiment
    # set_experiment handles creation and lookup, but in MLflow 3.x it still
    # raises on soft-deleted experiments.  Detect that case and restore first.
    try:
        experiment = mlflow.set_experiment(experiment_name)
    except Exception as exc:
        if "deleted" in str(exc).lower():
            # Experiment was soft-deleted — restore it and retry
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            deleted_exp = client.get_experiment_by_name(experiment_name)
            if deleted_exp is not None:
                client.restore_experiment(deleted_exp.experiment_id)
            experiment = mlflow.set_experiment(experiment_name)
        else:
            raise
    experiment_id = experiment.experiment_id

    return experiment_id


def log_training_metadata(
    train_data_path: Path | str,
    model_version: str,
    algorithm: str,
    git_commit: str | None = None,
    git_branch: str | None = None,
    airflow_run_id: str | None = None,
    trained_by: str | None = None,
    deployment_mode: str | None = None,
) -> None:
    """
    Log standard metadata tags to current MLflow run.

    Args:
        train_data_path: Path to training data
        model_version: Model version identifier
        algorithm: Algorithm name
        git_commit: Git commit hash (auto-detected if None)
        git_branch: Git branch name (auto-detected if None)
        airflow_run_id: Airflow DAG run ID (if triggered by Airflow)
        trained_by: Source that triggered training (e.g. greenfield_init, airflow)
        deployment_mode: 'local' or 'cloud'
    """
    # Auto-detect Git metadata
    if git_commit is None:
        git_commit = get_git_commit_hash()
    if git_branch is None:
        git_branch = get_git_branch()

    # Get DVC data version (or compute MD5 fingerprint as fallback)
    dvc_version = get_dvc_data_version(train_data_path)
    if not dvc_version:
        # No DVC tracking for this file (e.g. temp files from automated_retraining).
        # Compute a content MD5 fingerprint so at least some data lineage is captured.
        try:
            import hashlib

            _p = Path(train_data_path)
            if _p.exists():
                h = hashlib.md5()  # noqa: S324
                with _p.open("rb") as _f:
                    for _chunk in iter(lambda: _f.read(8192), b""):
                        h.update(_chunk)
                dvc_version = f"md5:{h.hexdigest()}"
        except Exception:
            pass

    # Log tags
    mlflow.set_tag("model_version", model_version)
    mlflow.set_tag("algorithm", algorithm)
    mlflow.set_tag("timestamp", datetime.utcnow().isoformat())

    if git_commit:
        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("git_sha", git_commit)  # alias for lineage audit
    if git_branch:
        mlflow.set_tag("git_branch", git_branch)
    if dvc_version:
        mlflow.set_tag("dvc_data_version", dvc_version)
        mlflow.set_tag("dvc_data_hash", dvc_version)  # alias for lineage audit
    if airflow_run_id:
        mlflow.set_tag("airflow_run_id", airflow_run_id)
    if trained_by:
        mlflow.set_tag("trained_by", trained_by)
    if deployment_mode:
        mlflow.set_tag("deployment_mode", deployment_mode)


def log_dataset_info(
    train_samples: int,
    test_samples: int | None = None,
    class_distribution: dict[int, int] | None = None,
) -> None:
    """
    Log dataset statistics to current MLflow run.

    Args:
        train_samples: Number of training samples
        test_samples: Number of test samples (optional)
        class_distribution: Dict of {label: count} (optional)
    """
    mlflow.log_param("train_samples", train_samples)

    if test_samples is not None:
        mlflow.log_param("test_samples", test_samples)

    if class_distribution is not None:
        for label, count in class_distribution.items():
            mlflow.log_param(f"class_{label}_count", count)


def log_feature_importance(
    feature_names: list[str],
    importances: list[float],
) -> None:
    """
    Log feature importance as metrics.

    Args:
        feature_names: List of feature names
        importances: Corresponding importance values
    """
    for name, importance in zip(feature_names, importances):  # noqa: B905
        mlflow.log_metric(f"feature_importance_{name}", float(importance))


def get_best_run(
    experiment_name: str,
    metric_name: str = "test_accuracy",
    ascending: bool = False,
) -> dict[str, Any] | None:
    """
    Get best run from experiment based on metric.

    Args:
        experiment_name: MLflow experiment name
        metric_name: Metric to optimize (e.g., "test_accuracy")
        ascending: If True, lower is better; if False, higher is better

    Returns:
        Dict with run info and metrics, or None if no runs found
    """
    client = MlflowClient()

    # Get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    # Search runs sorted by metric
    order_by = f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[order_by],
        max_results=1,
    )

    if not runs:
        return None

    run = runs[0]

    return {
        "run_id": run.info.run_id,
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "metrics": run.data.metrics,
        "params": run.data.params,
        "tags": run.data.tags,
    }


def compare_runs(run_id_1: str, run_id_2: str) -> dict[str, Any]:
    """
    Compare two MLflow runs.

    Args:
        run_id_1: First run ID (e.g., Champion)
        run_id_2: Second run ID (e.g., Challenger)

    Returns:
        Dict with comparison data:
        {
            "run_1": {...},
            "run_2": {...},
            "metric_diff": {metric_name: diff, ...},
            "param_diff": {param_name: (val1, val2), ...},
        }
    """
    client = MlflowClient()

    run_1 = client.get_run(run_id_1)
    run_2 = client.get_run(run_id_2)

    # Compare metrics
    metric_diff = {}
    all_metrics = set(run_1.data.metrics.keys()) | set(run_2.data.metrics.keys())

    for metric in all_metrics:
        val_1 = run_1.data.metrics.get(metric, 0.0)
        val_2 = run_2.data.metrics.get(metric, 0.0)
        metric_diff[metric] = val_2 - val_1

    # Compare parameters
    param_diff = {}
    all_params = set(run_1.data.params.keys()) | set(run_2.data.params.keys())

    for param in all_params:
        val_1 = run_1.data.params.get(param)
        val_2 = run_2.data.params.get(param)
        if val_1 != val_2:
            param_diff[param] = (val_1, val_2)

    return {
        "run_1": {
            "run_id": run_1.info.run_id,
            "metrics": run_1.data.metrics,
            "params": run_1.data.params,
            "tags": run_1.data.tags,
        },
        "run_2": {
            "run_id": run_2.info.run_id,
            "metrics": run_2.data.metrics,
            "params": run_2.data.params,
            "tags": run_2.data.tags,
        },
        "metric_diff": metric_diff,
        "param_diff": param_diff,
    }
