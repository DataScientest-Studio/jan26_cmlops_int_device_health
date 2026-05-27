"""
Training pipeline for MLOps device health monitoring.

Provides:
- train_model(): Train LogisticRegression classifier using semi-supervised K-means
  clustering with label propagation.  Feature extraction is performed internally —
  the function accepts raw signal JSON and produces a ready-to-deploy model pickle.
- retrain_model(): Retrain from database sparse labels
- evaluate_model(): Compute metrics on test set

Integrates with:
- Database layer for training data retrieval
- Feature extractor for signal processing
- Model serialization (pickle)
- MLflow for experiment tracking and model registry
- Semi-supervised learning for scarce label scenarios
- DVC for data versioning and production logs synchronization
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
from datetime import datetime, timezone
from hashlib import md5
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

CLASSIFIER_MAP: dict[str, type] = {
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "svc": SVC,
}

_CLASSIFIER_DEFAULTS: dict[str, dict] = {
    "logistic_regression": {"max_iter": 1000, "random_state": 42},
    "decision_tree": {"random_state": 42},
    "random_forest": {"n_estimators": 100, "random_state": 42},
    "svc": {"kernel": "rbf", "probability": True, "max_iter": 1000, "random_state": 42},
}


def _is_mlflow_rate_limited(exc: Exception) -> bool:
    """Return True if *exc* indicates a DagsHub / MLflow server 429 rate-limit."""
    msg = str(exc).lower()
    return "429" in msg or "too many" in msg or "rate" in msg


def _local_mlflow_fallback_uri() -> str:
    """Return a safe local file-based MLflow tracking URI.

    Prefers the env var ``MLFLOW_LOCAL_TRACKING_URI`` when set, then falls
    back to ``<repo_root>/mlruns`` resolved from this file's location so it
    works both on the host and inside the Airflow container.
    """
    if os.getenv("MLFLOW_LOCAL_TRACKING_URI"):
        return os.environ["MLFLOW_LOCAL_TRACKING_URI"]
    # Walk up from src/training/ to repo root
    repo_root = Path(__file__).resolve().parents[2]
    return f"file:///{repo_root / 'mlruns'}".replace("\\", "/")


from src.signal_processing.feature_extractor import extract_features
from src.signal_processing.signal_models import LabeledSignal, SignalData
from src.training.mlflow_utils import (
    log_dataset_info,
    log_training_metadata,
    setup_mlflow,
)
from src.training.semi_supervised import (
    SemiSupervisedTrainer,
    create_gold_standard_split,
    select_sliding_window_data,
)


def load_signals_from_json(file_path: Path, allow_unlabeled: bool = False) -> list[LabeledSignal]:
    """
    Load labeled signals from JSON file.

    Args:
        file_path: Path to JSON dataset file
        allow_unlabeled: If True, allow missing labels (defaults to -1 for semi-supervised)

    Returns:
        List of LabeledSignal instances

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON format is invalid
    """
    with open(file_path) as f:
        data = json.load(f)

    labeled_signals = []

    for entry in data["signals"]:
        signal = SignalData(
            time=entry["time"],
            amplitude=entry["amplitude"],
            shape_type=entry.get("shape_type", "unknown"),
        )

        # Label is required for training (unless allow_unlabeled=True)
        if "label" not in entry:
            if not allow_unlabeled:
                raise ValueError(f"Signal {entry.get('id', 'unknown')} missing label")
            label = -1  # Use -1 for unlabeled samples in semi-supervised learning
        else:
            label = entry["label"]
            if label is None and not allow_unlabeled:
                raise ValueError(f"Signal {entry.get('id', 'unknown')} has None label")
            if label is None:
                label = -1  # Convert None to -1 for consistency

        metadata = entry.get("metadata", {}) or {}
        # Store the signal's id field so train_model() can track split membership per signal_id
        if "id" in entry and "signal_id" not in metadata:
            metadata["signal_id"] = entry["id"]

        labeled_signal = LabeledSignal(signal=signal, label=label, metadata=metadata)
        labeled_signals.append(labeled_signal)

    return labeled_signals


def dvc_pull_data(data_path: Path | str, verify: bool = True) -> dict[str, Any]:
    """
    Pull data from DVC remote storage.

    Args:
        data_path: Path to file or directory to pull
        verify: Whether to verify DVC hash after pull (default: True)

    Returns:
        Dict with:
        {
            "success": bool,
            "path": str,
            "dvc_hash": str | None,  # MD5 hash from .dvc file
            "message": str,
        }

    Raises:
        RuntimeError: If DVC pull fails
    """
    data_path = Path(data_path)

    print(f"Pulling data from DVC: {data_path}...")

    # Run dvc pull
    result = subprocess.run(
        ["dvc", "pull", str(data_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        error_msg = f"DVC pull failed: {result.stderr}"
        raise RuntimeError(error_msg)

    # Extract DVC hash if verify=True
    dvc_hash = None
    if verify:
        dvc_file = Path(str(data_path) + ".dvc")
        if dvc_file.exists():
            dvc_hash = get_dvc_file_hash(dvc_file)

    return {
        "success": True,
        "path": str(data_path),
        "dvc_hash": dvc_hash,
        "message": f"Successfully pulled {data_path}",
    }


def get_dvc_file_hash(dvc_file_path: Path | str) -> str | None:
    """
    Extract MD5 hash from .dvc file.

    Args:
        dvc_file_path: Path to .dvc file

    Returns:
        MD5 hash string or None if not found
    """
    dvc_file_path = Path(dvc_file_path)

    if not dvc_file_path.exists():
        return None

    try:
        import yaml
    except ImportError:
        print("Warning: PyYAML not installed, cannot extract DVC hash")
        return None

    try:
        with open(dvc_file_path) as f:
            dvc_content = yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Failed to parse .dvc file: {e}")
        return None

    # Hash can be at top level or in outs[0]
    if "md5" in dvc_content:
        return dvc_content["md5"]
    elif "outs" in dvc_content and len(dvc_content["outs"]) > 0:
        return dvc_content["outs"][0].get("md5")

    return None


def load_signals_from_production_logs(
    metadata_csv_path: Path | str,
    signals_dir: Path | str,
    labeled_only: bool = True,
) -> list[LabeledSignal]:
    """
    Load signals from production logs synchronized from SQL database.

    Expected structure:
    - metadata_csv_path: CSV with columns [prediction_id, device_id, predicted_label, ground_truth_label, timestamp, ...]
    - signals_dir: Directory with JSON files at {prefix1}/{prefix2}/{device_id}/{prediction_id}.json

    Args:
        metadata_csv_path: Path to predictions metadata CSV
        signals_dir: Root directory containing signal JSON files
        labeled_only: If True, load only signals with ground_truth_label (default: True)

    Returns:
        List of LabeledSignal instances

    Raises:
        FileNotFoundError: If metadata CSV doesn't exist
        ValueError: If required columns are missing
    """
    metadata_csv_path = Path(metadata_csv_path)
    signals_dir = Path(signals_dir)

    if not metadata_csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv_path}")

    # Load metadata
    df_metadata = pd.read_csv(metadata_csv_path)

    # Validate required columns
    required_cols = ["prediction_id", "device_id", "ground_truth_label"]
    missing_cols = [col for col in required_cols if col not in df_metadata.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Filter for labeled samples if requested
    if labeled_only:
        df_metadata = df_metadata[df_metadata["ground_truth_label"].notna()]

    print(f"Loading {len(df_metadata)} signals from production logs...")

    labeled_signals = []

    for _, row in df_metadata.iterrows():
        prediction_id = row["prediction_id"]
        device_id = row["device_id"]
        ground_truth_label = row["ground_truth_label"]

        # Construct path using UUID sharding
        uuid_hex = device_id.replace("-", "")
        prefix1 = uuid_hex[:2]
        prefix2 = uuid_hex[2:4]
        signal_path = signals_dir / prefix1 / prefix2 / device_id / f"{prediction_id}.json"

        if not signal_path.exists():
            print(f"Warning: Signal file not found: {signal_path}, skipping...")
            continue

        # Load signal JSON
        with open(signal_path) as f:
            signal_data = json.load(f)

        # Create SignalData
        signal = SignalData(
            time=signal_data["time_values"],
            amplitude=signal_data["amplitude_values"],
            shape_type=signal_data.get("shape_type", "unknown"),
        )

        # Create metadata
        metadata = {
            "prediction_id": prediction_id,
            "device_id": device_id,
            "timestamp": signal_data.get("prediction_timestamp"),
            "predicted_label": signal_data.get("predicted_label"),
        }

        # For unlabeled signals, use predicted_label as fallback
        # (LabeledSignal model requires a label, even for "unlabeled" signals)
        if pd.isna(ground_truth_label):
            # Use predicted label for unlabeled signals
            label_value = int(signal_data.get("predicted_label", 0))
        else:
            label_value = int(ground_truth_label)

        # Create LabeledSignal
        labeled_signal = LabeledSignal(
            signal=signal,
            label=label_value,
            metadata=metadata,
        )
        labeled_signals.append(labeled_signal)

    print(f"Successfully loaded {len(labeled_signals)} signals")

    return labeled_signals


def load_model(model_path: Path | str) -> dict[str, Any]:
    """
    Load trained model from disk.

    Args:
        model_path: Path to model pickle file

    Returns:
        Dict with:
        {
            "model": trained classifier,
            "scaler": StandardScaler,
            "feature_names": list[str],
            "model_version": str,
            "algorithm": str,
            "trained_at": str,
        }

    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(model_path, "rb") as f:
        model_artifact = pickle.load(f)

    return model_artifact


def evaluate_model(
    model_path: Path | str,
    test_data_path: Path | str,
) -> dict[str, Any]:
    """
    Evaluate trained model on test dataset.

    Args:
        model_path: Path to trained model (pickle)
        test_data_path: Path to test dataset (JSON)

    Returns:
        Dict with evaluation metrics:
        {
            "test_samples": int,
            "test_accuracy": float,
            "confusion_matrix": list[list[int]],
            "classification_report": str,
        }

    Raises:
        FileNotFoundError: If files don't exist
    """
    # Load model
    model_artifact = load_model(model_path)
    model = model_artifact["model"]
    scaler = model_artifact["scaler"]
    feature_names = model_artifact["feature_names"]

    # Load test data
    test_data_path = Path(test_data_path)
    if not test_data_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_data_path}")

    test_signals = load_signals_from_json(test_data_path)

    # Extract features
    test_features_list = [extract_features(signal.signal) for signal in test_signals]
    test_labels = [signal.label for signal in test_signals]

    X_test = np.array([[f.get(name) or 0.0 for name in feature_names] for f in test_features_list])
    y_test = np.array(test_labels)

    # Scale and predict
    X_test_scaled = scaler.transform(X_test)
    y_test_pred = model.predict(X_test_scaled)

    # Compute metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    conf_matrix = confusion_matrix(y_test, y_test_pred).tolist()
    class_report = classification_report(
        y_test,
        y_test_pred,
        target_names=["Healthy", "Unhealthy"],
        zero_division=0,
    )

    return {
        "test_samples": len(test_signals),
        "test_accuracy": float(test_accuracy),
        "confusion_matrix": conf_matrix,
        "classification_report": class_report,
    }


def train_model(
    train_data_path: Path | str = "__from_db__",
    model_output_path: Path | str = "models/model.pkl",
    model_version: str = "v1.0_semi",
    use_mlflow: bool = True,
    mlflow_experiment_name: str = "device_health_classifier",
    # DVC integration
    dvc_pull: bool = False,
    # Production logs integration
    from_production_logs: bool = False,
    production_logs_metadata_csv: Path | str | None = None,
    production_logs_signals_dir: Path | str | None = None,
    # DB-backed training (Phase 5 — bypass JSON file entirely)
    from_db: bool = False,
    # Sliding window parameters
    window_size: int | None = None,
    window_days: int | None = 90,
    # Semi-supervised parameters
    k_range: tuple[int, int] = (2, 10),
    k_method: str = "silhouette",
    distance_threshold: float = 2.0,
    knn_neighbors: int = 5,
    use_domain_heuristics: bool = True,
    # Unlabeled data support
    allow_unlabeled: bool = False,
    filter_unlabeled: bool = True,
    # Gold standard split
    test_size: float = 0.2,
    stratify: bool = True,
    # Evaluation metric
    primary_metric: str = "f1_score",
    # Airflow lineage
    airflow_run_id: str | None = None,
    # Training split tracking (Phase 2/3)
    db: Any = None,
    # Bootstrap/champion-challenger: restrict DB query to specific signal IDs
    signal_ids_filter: list[int] | None = None,
    # Random seed — passed from signal generator UI so MLflow logs the actual seed used
    random_state: int = 42,
    **model_kwargs,
) -> dict[str, Any]:
    """
    Train model using semi-supervised learning with K-means clustering.

    This approach is designed for scenarios where ground truth labels are scarce (5-10%).
    It uses:
    1. DVC integration for production data synchronization
    2. Support for SQL production logs (CSV metadata + JSON signals)
    3. Sliding window data selection (last N samples or M days)
    4. 80/20 split with fully-labeled gold standard test set
    5. K-means clustering with optimal K selection
    6. Label propagation via majority voting
    7. Handling of unlabeled clusters (distance/proximity/heuristics)
    8. F1 score as primary metric

    Args:
        train_data_path: Path to training dataset (JSON with 'signals' array)
        model_output_path: Path to save trained model (pickle)
        model_version: Model version identifier (default: "v1.0_semi")
        use_mlflow: Enable MLflow experiment tracking (default: True)
        mlflow_experiment_name: MLflow experiment name
        dvc_pull: Pull data from DVC remote before training (default: False)
        from_production_logs: Load from production logs format (default: False)
        production_logs_metadata_csv: Path to metadata CSV (required if from_production_logs=True)
        production_logs_signals_dir: Path to signals directory (required if from_production_logs=True)
        window_size: Number of most recent samples for sliding window (None = use window_days)
        window_days: Number of most recent days for sliding window (default: 90)
        k_range: Tuple of (min_k, max_k) for K optimization (default: (2, 10))
        k_method: Method for K selection - "silhouette", "elbow", or "calinski" (default: "silhouette")
        distance_threshold: Std deviations from healthy centroid for distance heuristic (default: 2.0)
        knn_neighbors: Number of neighbors for KNN-based labeling (default: 5)
        use_domain_heuristics: Whether to apply domain-specific rules (default: True)
        allow_unlabeled: Allow loading samples without labels (for semi-supervised) (default: False)
        filter_unlabeled: Filter out unlabeled samples before training (default: True)
        test_size: Fraction for gold standard test set (default: 0.2)
        stratify: Whether to stratify split by labels (default: True)
        primary_metric: Primary evaluation metric - "f1_score" or "accuracy" (default: "f1_score")
        **model_kwargs: Additional arguments for LogisticRegression (e.g., max_iter, C)

    Returns:
        Dict with training metrics:
        {
            "model_version": str,
            "model_path": str,
            "train_samples": int,
            "test_samples": int,
            "optimal_k": int,
            "pseudo_labeled_clusters": int,
            "train_accuracy": float,
            "train_f1_score": float,
            "test_accuracy": float,
            "test_f1_score": float,
            "primary_metric": str,
            "primary_metric_value": float,
            "features_used": list[str],
            "confusion_matrix": list[list[int]],
            "classification_report": str,
            "trained_at": str,
            "mlflow_run_id": str,  # If use_mlflow=True
        }

    Raises:
        ValueError: If training data has < 2 labeled samples
        FileNotFoundError: If train_data_path doesn't exist
    """
    train_data_path = Path(train_data_path)
    model_output_path = Path(model_output_path)
    # Always ensure the output has a .pkl extension so load_production_model_artifact
    # can find it by extension when scanning MLflow artifacts. Callers that pass
    # "models/challenger" (no extension) get "models/challenger.pkl".
    if model_output_path.suffix != ".pkl":
        model_output_path = model_output_path.with_suffix(".pkl")

    # Setup MLflow if enabled — with automatic fallback to local tracking when
    # DagsHub returns HTTP 429 (rate-limited).
    _mlflow_using_local = False
    if use_mlflow:
        try:
            setup_mlflow(experiment_name=mlflow_experiment_name)
        except Exception as _setup_exc:
            if _is_mlflow_rate_limited(_setup_exc):
                _local_uri = _local_mlflow_fallback_uri()
                print(
                    f"⚠️  DagsHub rate-limited during experiment setup — "
                    f"falling back to local MLflow: {_local_uri}"
                )
                setup_mlflow(tracking_uri=_local_uri, experiment_name=mlflow_experiment_name)
                _mlflow_using_local = True
            else:
                raise

    # Start MLflow run — retry with local fallback if DagsHub returns 429.
    mlflow_run = None
    if use_mlflow:
        try:
            mlflow_run = mlflow.start_run()
        except Exception as _run_exc:
            if _is_mlflow_rate_limited(_run_exc):
                if not _mlflow_using_local:
                    _local_uri = _local_mlflow_fallback_uri()
                    print(
                        f"⚠️  DagsHub rate-limited on run create — "
                        f"falling back to local MLflow: {_local_uri}"
                    )
                    setup_mlflow(tracking_uri=_local_uri, experiment_name=mlflow_experiment_name)
                    _mlflow_using_local = True
                mlflow_run = mlflow.start_run()
            else:
                raise
    if _mlflow_using_local:
        print("ℹ️  Training metrics will be tracked locally (DagsHub unavailable).")

    try:
        # DVC pull if requested
        dvc_hash = None
        if dvc_pull:
            if from_production_logs:
                # Pull both CSV and signals directory
                if production_logs_metadata_csv:
                    dvc_result = dvc_pull_data(production_logs_metadata_csv)
                    dvc_hash = dvc_result["dvc_hash"]
                    print(f"DVC hash (metadata): {dvc_hash}")
                if production_logs_signals_dir:
                    dvc_pull_data(production_logs_signals_dir)
            else:
                # Pull single training data file
                dvc_result = dvc_pull_data(train_data_path)
                dvc_hash = dvc_result["dvc_hash"]
                print(f"DVC hash: {dvc_hash}")

            # Log DVC hash to MLflow
            if use_mlflow and dvc_hash:
                mlflow.log_param("dvc_data_hash", dvc_hash)

        # Load training data based on source
        if from_db:
            # Phase 5 — DB-backed training: fetch signals directly from the database.
            # The window_days filter is already applied at SQL level via
            # get_labeled_signal_ids() / get_unlabeled_signal_ids() so the
            # dead pandas filter block below is bypassed entirely.
            # signal_ids_filter (optional) restricts the query to a specific set of
            # signal_ids — used by bootstrap/champion-challenger to train only on the
            # signals that were just inserted, not all historical signals.
            if db is None:
                raise ValueError("from_db=True requires db parameter to be provided")
            print("Loading training data from database (DB-backed training)...")
            _labeled_ids = db.get_labeled_signal_ids(
                window_days=window_days,
                signal_ids_filter=signal_ids_filter,
            )
            _unlabeled_ids: list = (
                db.get_unlabeled_signal_ids(
                    window_days=window_days,
                    signal_ids_filter=signal_ids_filter,
                )
                if allow_unlabeled
                else []
            )
            train_signals = []
            for _sid in _labeled_ids:
                _raw = db.get_signal_data_by_id(_sid)
                _lbl = db.get_label_by_signal_id(_sid)
                if _raw is None or _lbl is None:
                    continue
                _eff_lbl = int(_lbl) if int(_lbl) in (0, 1) else -1
                train_signals.append(
                    LabeledSignal(
                        signal=SignalData(
                            time=_raw["time_values"],
                            amplitude=_raw["amplitude_values"],
                            shape_type="gaussian",
                        ),
                        label=_eff_lbl,
                        metadata={"signal_id": _sid},
                    )
                )
            for _sid in _unlabeled_ids:
                _raw = db.get_signal_data_by_id(_sid)
                if _raw is None:
                    continue
                train_signals.append(
                    LabeledSignal(
                        signal=SignalData(
                            time=_raw["time_values"],
                            amplitude=_raw["amplitude_values"],
                            shape_type="gaussian",
                        ),
                        label=-1,
                        metadata={"signal_id": _sid},
                    )
                )
            print(
                f"Loaded {len(_labeled_ids)} labeled + {len(_unlabeled_ids)} unlabeled "
                f"signals from database."
            )
            if use_mlflow:
                mlflow.log_param("data_source", "database")
                mlflow.log_param("train_data_path", "__from_database__")
                if window_days:
                    mlflow.log_param("window_days", window_days)

        elif from_production_logs:
            # Load from production logs (CSV + JSON hybrid format)
            if not production_logs_metadata_csv or not production_logs_signals_dir:
                raise ValueError(
                    "production_logs_metadata_csv and production_logs_signals_dir "
                    "are required when from_production_logs=True"
                )

            print("Loading training data from production logs...")
            print(f"  Metadata CSV: {production_logs_metadata_csv}")
            print(f"  Signals directory: {production_logs_signals_dir}")

            train_signals = load_signals_from_production_logs(
                metadata_csv_path=production_logs_metadata_csv,
                signals_dir=production_logs_signals_dir,
                labeled_only=True,  # Only use labeled samples for training
            )

            if use_mlflow:
                mlflow.log_param("data_source", "production_logs")
                mlflow.log_param("metadata_csv", str(production_logs_metadata_csv))
                mlflow.log_param("signals_dir", str(production_logs_signals_dir))
        else:
            # Load from standard JSON format
            if not train_data_path.exists():
                raise FileNotFoundError(f"Training data not found: {train_data_path}")

            print(f"Loading training data from {train_data_path}...")
            train_signals = load_signals_from_json(train_data_path, allow_unlabeled=allow_unlabeled)

            if use_mlflow:
                mlflow.log_param("data_source", "json_file")
                mlflow.log_param("train_data_path", str(train_data_path))

        if len(train_signals) < 2:
            raise ValueError(f"Insufficient training samples: {len(train_signals)} < 2")

        # Extract features and create DataFrame
        print(f"Extracting features from {len(train_signals)} signals...")
        train_features_list = [extract_features(signal.signal) for signal in train_signals]
        train_labels = [signal.label for signal in train_signals]

        # Extract signal IDs from metadata (set by load_signals_from_json when "id" field present)
        _raw_signal_ids: list[int | None] = [
            s.metadata.get("signal_id") if s.metadata else None for s in train_signals
        ]

        # Extract timestamps from metadata (if available)
        timestamps = []
        for signal in train_signals:
            if signal.metadata and "timestamp" in signal.metadata:
                timestamps.append(pd.to_datetime(signal.metadata["timestamp"]))
            else:
                # Use current time as fallback
                timestamps.append(pd.Timestamp.now())

        # Convert to DataFrame
        feature_names = ["fwhm", "peak_height", "peak_area", "noise_level", "snr", "peak_center"]
        feature_data = []
        for f_dict in train_features_list:
            feature_data.append([f_dict.get(name) or 0.0 for name in feature_names])

        df_all = pd.DataFrame(feature_data, columns=feature_names)
        df_all["ground_truth_label"] = train_labels
        df_all["timestamp"] = timestamps
        df_all["_signal_id"] = _raw_signal_ids  # tracks original DB signal ID through windowing

        # ── Data contract validation (Task 6) ─────────────────────────────
        try:
            from src.data.schemas import validate_features

            labeled_mask = df_all["ground_truth_label"] != -1
            if labeled_mask.any():
                _df_check = df_all.loc[labeled_mask, feature_names].copy()
                _df_check["label"] = df_all.loc[labeled_mask, "ground_truth_label"].astype(int)
                validate_features(_df_check, require_label=True)
                print(f"✓ Data contract validated ({labeled_mask.sum()} labeled rows)")
        except ImportError:
            pass  # pandera not installed — skip silently
        except Exception as _contract_exc:
            print(f"⚠️ Data contract warning: {_contract_exc} — continuing with training")

        n_total = len(df_all)
        n_labeled_before_window = (df_all["ground_truth_label"] != -1).sum()

        print(f"Total samples loaded: {n_total}")
        print(
            f"Labeled samples before windowing: {n_labeled_before_window} "
            f"({100 * n_labeled_before_window / n_total:.1f}%)"
        )

        # Apply sliding window if specified
        # DEPRECATED: window_size/window_days filtering here is a no-op when signals
        # are loaded from a static JSON file (the 'timestamp' column is set to
        # pd.Timestamp.now() as a fallback, so every signal appears brand-new).
        # The real window filter is now applied in SQL by get_labeled_signal_ids()
        # and get_unlabeled_signal_ids() before signals are fetched.  This block is
        # kept for backward compatibility when loading from JSON with real timestamps.
        # When from_db=True the filter was already applied at SQL level — skip here.
        if not from_db and (window_size is not None or window_days is not None):
            df_windowed = select_sliding_window_data(
                df_all,
                window_size=window_size,
                window_days=window_days,
                timestamp_col="timestamp",
                label_col="ground_truth_label",
            )

            if use_mlflow:
                if window_size:
                    mlflow.log_param("window_size", window_size)
                if window_days:
                    mlflow.log_param("window_days", window_days)
        else:
            if filter_unlabeled:
                print("\nNo sliding window specified, using all labeled data...")
                # Filter for labeled samples only
                df_windowed = df_all[df_all["ground_truth_label"] != -1].copy()
            else:
                print("\nNo sliding window specified, using all data (labeled + unlabeled)...")
                # Keep all samples (labeled + unlabeled) for semi-supervised learning
                df_windowed = df_all.copy()

        # Extract features and labels from windowed data
        X_all_data = df_windowed[feature_names].values
        y_all_data = df_windowed["ground_truth_label"].values
        # Carry signal IDs through windowing so we can map split indices → signal IDs
        _windowed_signal_ids: list[int | None] = df_windowed["_signal_id"].tolist()

        # Count labeled vs unlabeled
        n_labeled = (y_all_data != -1).sum()  # type: ignore[union-attr]
        n_unlabeled = (y_all_data == -1).sum()  # type: ignore[union-attr]
        n_total_windowed = len(y_all_data)

        print(
            f"Samples after windowing: {n_total_windowed} total "
            f"(labeled: {n_labeled}, unlabeled: {n_unlabeled})"
        )

        if n_labeled < 2:
            raise ValueError(f"Need at least 2 labeled samples, got {n_labeled}")

        # Gold standard split — proportional to total samples, not just labeled.
        # When unlabeled data is available, we use a larger portion of labeled
        # samples for testing (gold standard) since the semi-supervised trainer
        # can learn from unlabeled data.  This gives more meaningful test sets.
        print(f"\nCreating gold standard split (test_size={test_size:.0%})...")

        labeled_mask = y_all_data != -1  # type: ignore[assignment]

        if n_unlabeled > 0 and not filter_unlabeled:
            # Semi-supervised mode: ALL labeled go to test, unlabeled to train.
            # We keep a small labeled seed in train for cluster quality.
            X_labeled = X_all_data[labeled_mask]  # type: ignore[call-overload]
            y_labeled = y_all_data[labeled_mask]  # type: ignore[call-overload]
            X_unlabeled = X_all_data[~labeled_mask]

            # Reserve a small fraction of labeled for training seed
            min_train_labeled = max(2, int(n_labeled * 0.2))  # at least 2
            if n_labeled <= min_train_labeled + 2:
                # Too few labeled — fall back to standard split
                X_train, X_test, y_train, y_test = create_gold_standard_split(
                    X_labeled,
                    y_labeled,
                    test_size=test_size,
                    stratify=stratify,
                    random_state=random_state,
                )
            else:
                # Split labeled: most go to test (gold standard)
                test_fraction = 1.0 - (min_train_labeled / n_labeled)
                X_train_labeled, X_test, y_train_labeled, y_test = create_gold_standard_split(
                    X_labeled,
                    y_labeled,
                    test_size=test_fraction,
                    stratify=stratify,
                    random_state=random_state,
                )
                X_train = X_train_labeled
                y_train = y_train_labeled

            # Add unlabeled to training for semi-supervised learning
            y_unlabeled = np.full(n_unlabeled, -1)
            X_train_with_unlabeled = np.vstack([X_train, X_unlabeled])
            y_train_with_unlabeled = np.concatenate([y_train, y_unlabeled])

            print(
                f"Training set: {len(y_train)} labeled + {n_unlabeled} unlabeled "
                f"= {len(y_train_with_unlabeled)} total"
            )
            print(f"Test set (gold standard): {len(y_test)} labeled samples")
        else:
            # Fully supervised: standard split on labeled data only
            X_labeled = X_all_data[labeled_mask]  # type: ignore[call-overload]
            y_labeled = y_all_data[labeled_mask]  # type: ignore[call-overload]

            X_train, X_test, y_train, y_test = create_gold_standard_split(
                X_labeled,
                y_labeled,
                test_size=test_size,
                stratify=stratify,
                random_state=random_state,
            )
            X_train_with_unlabeled = X_train
            y_train_with_unlabeled = y_train

        # Save and version gold standard test set for reproducibility
        gold_standard_dir = Path("data/gold_standard")
        gold_standard_dir.mkdir(parents=True, exist_ok=True)

        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        gold_standard_filename = f"test_set_{model_version}_{timestamp_str}.csv"
        gold_standard_path = gold_standard_dir / gold_standard_filename

        # ── Phase 2: record training split in DB; Phase 3: compute MD5 hash ─
        # Recover which signal IDs ended up in train vs test by replaying the
        # same train_test_split call on a parallel index array.  This is
        # deterministic because train_test_split uses random_state=42 in both calls.
        _split_hash: str | None = None
        _train_signal_ids: list[int] = []
        _test_signal_ids: list[int] = []
        try:
            from sklearn.model_selection import train_test_split as _tts

            _labeled_ids_windowed = [
                sid
                for sid, lbl in zip(_windowed_signal_ids, y_all_data)  # noqa: B905
                if lbl != -1 and sid is not None
            ]
            _n_lbl = len(_labeled_ids_windowed)
            if _n_lbl >= 2:
                _eff_test_size = test_size
                if (
                    n_unlabeled > 0
                    and not filter_unlabeled
                    and _n_lbl > max(2, int(_n_lbl * 0.2)) + 2
                ):
                    _eff_test_size = 1.0 - (max(2, int(_n_lbl * 0.2)) / _n_lbl)
                _y_lbl = [lbl for lbl in y_all_data if lbl != -1]
                _strat = _y_lbl if stratify else None
                _ids_train, _ids_test = _tts(
                    _labeled_ids_windowed,
                    test_size=_eff_test_size,
                    stratify=_strat,
                    random_state=random_state,
                )
                _train_signal_ids = [sid for sid in _ids_train if sid is not None]
                _test_signal_ids = [sid for sid in _ids_test if sid is not None]

                # Add unlabeled signal IDs to train set (they were included in training)
                _unlabeled_ids = [
                    sid
                    for sid, lbl in zip(_windowed_signal_ids, y_all_data)  # noqa: B905
                    if lbl == -1 and sid is not None
                ]
                _train_signal_ids = list(_train_signal_ids) + _unlabeled_ids

            # Phase 3: export split signals to JSON and compute MD5 hash
            if _train_signal_ids or _test_signal_ids:
                _mlflow_run_id_for_hash = (
                    mlflow_run.info.run_id if (use_mlflow and mlflow_run) else None
                )
                _repo_root = Path(__file__).resolve().parents[2]
                _split_dir = (
                    _repo_root
                    / "data"
                    / "processed"
                    / "training_splits"
                    / (_mlflow_run_id_for_hash or model_version)
                )
                _split_dir.mkdir(parents=True, exist_ok=True)

                # Build lookup: signal_id → (time_values, amplitude_values, label)
                _sid_to_signal: dict[int, dict] = {}
                for _sig, _sid in zip(train_signals, _raw_signal_ids):  # noqa: B905
                    if _sid is not None:
                        _sid_to_signal[_sid] = {
                            "id": _sid,
                            "time": _sig.signal.time,
                            "amplitude": _sig.signal.amplitude,
                            "label": _sig.label,
                            "shape_type": getattr(_sig.signal, "shape_type", "gaussian"),
                        }

                def _write_split_json(ids: list[int], path: Path) -> None:
                    payload = {
                        "signals": [_sid_to_signal[sid] for sid in ids if sid in _sid_to_signal]
                    }
                    with open(path, "w") as _f:
                        json.dump(payload, _f)

                _train_json = _split_dir / "train.json"
                _test_json = _split_dir / "test.json"
                _write_split_json(_train_signal_ids, _train_json)
                _write_split_json(_test_signal_ids, _test_json)

                # Compute MD5 hash over both files concatenated
                _hasher = md5()
                for _jf in (_train_json, _test_json):
                    with open(_jf, "rb") as _f:
                        _hasher.update(_f.read())
                _split_hash = _hasher.hexdigest()
                print(f"✓ Training split exported to {_split_dir}; MD5={_split_hash}")

                # Record in DB if provided
                if db is not None and _mlflow_run_id_for_hash:
                    try:
                        db.record_training_split(
                            mlflow_run_id=_mlflow_run_id_for_hash,
                            train_signal_ids=[
                                sid for sid in _train_signal_ids if isinstance(sid, int)
                            ],
                            test_signal_ids=[
                                sid for sid in _test_signal_ids if isinstance(sid, int)
                            ],
                            model_version=model_version,
                        )
                        print(
                            f"✓ Recorded split: {len(_train_signal_ids)} train, {len(_test_signal_ids)} test signal IDs"
                        )
                    except Exception as _db_err:
                        print(f"⚠️  Could not record training split in DB: {_db_err}")

                # Log hash to MLflow (replaces the static DVC file hash)
                if use_mlflow and _split_hash:
                    try:
                        mlflow.log_param("dvc_data_hash", f"md5:{_split_hash}")
                        mlflow.log_param("train_split_size", len(_train_signal_ids))
                        mlflow.log_param("test_split_size", len(_test_signal_ids))
                        mlflow.log_param("split_export_path", str(_split_dir))
                    except Exception as _ml_err:
                        print(f"⚠️  Could not log split hash to MLflow: {_ml_err}")
        except Exception as _split_track_err:
            print(f"⚠️  Training split tracking failed (non-fatal): {_split_track_err}")
        # ──────────────────────────────────────────────────────────────────────

        # Create DataFrame with gold standard test set
        gold_standard_df = pd.DataFrame(X_test, columns=feature_names)
        gold_standard_df["ground_truth_label"] = y_test
        gold_standard_df["model_version"] = model_version
        gold_standard_df["created_at"] = timestamp_str

        # Save to CSV
        gold_standard_df.to_csv(gold_standard_path, index=False)
        print(f"✓ Saved gold standard test set to: {gold_standard_path}")

        # Track with DVC (if available and in cloud mode)
        # In local sandbox mode, DVC is intentionally disabled — no data should
        # leave the sandbox, and the DagsHub remote is not configured.
        _project_root = Path(__file__).resolve().parents[2]
        _mode_file = _project_root / ".current_mode"
        _deployment_mode = (
            _mode_file.read_text().strip() if _mode_file.exists() else ""
        ) or os.environ.get("DEPLOYMENT_MODE", "local")

        dvc_hash = None
        if _deployment_mode != "local":
            try:
                import subprocess

                _dvc_exe = _project_root / ".venv" / "Scripts" / "dvc.exe"
                _dvc_cmd = str(_dvc_exe) if _dvc_exe.exists() else "dvc"
                result = subprocess.run(
                    [_dvc_cmd, "add", str(gold_standard_path)],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=90,
                )
                if result.returncode == 0:
                    print("✓ Added gold standard to DVC tracking")
                    # Extract DVC hash from .dvc file
                    dvc_file = gold_standard_path.with_suffix(".csv.dvc")
                    if dvc_file.exists():
                        import yaml

                        with open(dvc_file) as _dvc_f:
                            dvc_data = yaml.safe_load(_dvc_f)
                            if "outs" in dvc_data and len(dvc_data["outs"]) > 0:
                                dvc_hash = dvc_data["outs"][0].get("md5")
                                print(f"✓ DVC hash: {dvc_hash}")
                else:
                    print(f"Warning: DVC add failed: {result.stderr}")
            except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
                print(f"Warning: Could not add to DVC: {e}")
        else:
            print("ℹ️  DVC tracking skipped (local sandbox mode — no remote configured)")

        # Log to MLflow
        if use_mlflow:
            # Log gold standard metadata
            mlflow.log_param("gold_standard_test_size", len(y_test))
            mlflow.log_param("gold_standard_path", str(gold_standard_path))
            if dvc_hash:
                mlflow.log_param("gold_standard_dvc_hash", dvc_hash)

            # Log gold standard as artifact
            try:
                mlflow.log_artifact(str(gold_standard_path), artifact_path="gold_standard")
            except Exception as _art_err:
                print(f"[WARN] Could not log gold standard artifact: {_art_err}")

            # Log test set statistics
            test_class_dist = np.bincount(y_test.astype(int))
            for idx, count in enumerate(test_class_dist):
                mlflow.log_metric(f"gold_standard_class_{idx}_count", int(count))

        # Standardize features (fit on labeled training set only, then transform all)
        print("Standardizing features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)  # Fit on labeled only
        X_test_scaled = scaler.transform(X_test)

        # Transform training set (with unlabeled if present)
        if n_unlabeled > 0 and not filter_unlabeled:
            X_train_with_unlabeled_scaled = scaler.transform(X_train_with_unlabeled)
        else:
            X_train_with_unlabeled_scaled = X_train_scaled

        # Initialize semi-supervised trainer
        print(f"\nInitializing semi-supervised trainer (K range={k_range}, method={k_method})...")
        semi_trainer = SemiSupervisedTrainer(
            k_range=k_range,
            k_method=k_method,
            distance_threshold=distance_threshold,
            knn_neighbors=knn_neighbors,
            use_domain_heuristics=use_domain_heuristics,
            random_state=random_state,
        )

        # Perform clustering and label propagation
        # Use X_train_with_unlabeled_scaled for clustering (includes unlabeled if present)
        print("\nPerforming K-means clustering with label propagation...")
        cluster_assignments, propagated_labels, cluster_info = semi_trainer.cluster_and_label(
            X_train_with_unlabeled_scaled,
            y_train_with_unlabeled,
            X_features_df=None,  # Can add if needed for domain heuristics
            k=None,  # Auto-optimize
        )

        optimal_k = semi_trainer.optimal_k_
        n_pseudo_clusters = sum(1 for info in cluster_info.values() if info["is_pseudo_label"])

        print(f"✓ Optimal K: {optimal_k}")
        print(f"✓ Pseudo-labeled clusters: {n_pseudo_clusters}/{optimal_k}")

        # Log clustering info to MLflow
        if use_mlflow:
            mlflow.log_param("optimal_k", optimal_k)
            mlflow.log_param("k_method", k_method)
            mlflow.log_param("k_range_min", k_range[0])
            mlflow.log_param("k_range_max", k_range[1])
            mlflow.log_metric("pseudo_labeled_clusters", n_pseudo_clusters)
            mlflow.log_metric("pseudo_labeled_ratio", n_pseudo_clusters / optimal_k)  # type: ignore[operator]

            # Log cluster info as artifact
            try:
                mlflow.log_dict(cluster_info, "cluster_info.json")  # type: ignore[arg-type]
            except Exception as _art_err:
                print(f"[WARN] Could not log cluster info artifact: {_art_err}")

        # Train classifier on propagated labels
        classifier_type = model_kwargs.pop("classifier_type", "logistic_regression")
        cls = CLASSIFIER_MAP.get(classifier_type, LogisticRegression)
        default_kwargs = dict(_CLASSIFIER_DEFAULTS.get(classifier_type, {"random_state": 42}))
        if "random_state" in default_kwargs:
            default_kwargs["random_state"] = random_state
        default_kwargs.update(model_kwargs)
        model = cls(**default_kwargs)

        algo_label = f"{classifier_type}_semi_supervised"
        print(f"\nTraining {classifier_type} on cluster-labeled data...")

        if use_mlflow:
            mlflow.log_params(default_kwargs)
            mlflow.log_param("algorithm", algo_label)
            mlflow.log_param("classifier_type", classifier_type)
            mlflow.log_param("random_state", random_state)

        # Train on all samples (labeled + unlabeled with propagated labels)
        model.fit(X_train_with_unlabeled_scaled, propagated_labels)

        # Evaluate on training set (using propagated labels)
        y_train_pred = model.predict(X_train_with_unlabeled_scaled)
        train_accuracy = accuracy_score(propagated_labels, y_train_pred)
        train_f1 = f1_score(propagated_labels, y_train_pred, average="binary", zero_division=0)

        print(f"Training accuracy: {train_accuracy:.2%}")
        print(f"Training F1 score: {train_f1:.4f}")

        if use_mlflow:
            mlflow.log_metric("train_accuracy", float(train_accuracy))
            mlflow.log_metric("train_f1_score", float(train_f1))
            mlflow.log_metric("n_train_samples", len(propagated_labels))

        # Evaluate on gold standard test set (GROUND TRUTH labels)
        print("\nEvaluating on gold standard test set...")
        y_test_pred = model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average="binary", zero_division=0)
        test_precision = precision_score(y_test, y_test_pred, average="binary", zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, average="binary", zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_test_pred).tolist()
        class_report = classification_report(
            y_test,
            y_test_pred,
            target_names=["Healthy", "Unhealthy"],
            zero_division=0,
        )

        print(f"Test accuracy: {test_accuracy:.2%}")
        print(f"Test F1 score: {test_f1:.4f}")
        print("\nConfusion Matrix (Gold Standard):")
        print(conf_matrix)
        print("\nClassification Report (Gold Standard):")
        print(class_report)

        # Log test metrics to MLflow
        if use_mlflow:
            mlflow.log_metric("test_accuracy", float(test_accuracy))
            mlflow.log_metric("test_f1_score", float(test_f1))
            mlflow.log_metric("test_precision", float(test_precision))
            mlflow.log_metric("test_recall", float(test_recall))
            mlflow.log_metric("n_test_samples", len(y_test))

            # Log confusion matrix values
            if len(conf_matrix) == 2:
                mlflow.log_metric("true_negatives", conf_matrix[0][0])
                mlflow.log_metric("false_positives", conf_matrix[0][1])
                mlflow.log_metric("false_negatives", conf_matrix[1][0])
                mlflow.log_metric("true_positives", conf_matrix[1][1])

            mlflow.log_param("primary_metric_name", primary_metric)

        # Prepare results
        primary_value = test_f1 if primary_metric == "f1_score" else test_accuracy

        results = {
            "model_version": model_version,
            "model_path": str(model_output_path),
            "algorithm": algo_label,
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "optimal_k": optimal_k,
            "pseudo_labeled_clusters": n_pseudo_clusters,
            "train_accuracy": float(train_accuracy),
            "train_f1_score": float(train_f1),
            "test_accuracy": float(test_accuracy),
            "test_f1_score": float(test_f1),
            "primary_metric": primary_metric,
            "primary_metric_value": float(primary_value),
            "features_used": feature_names,
            "confusion_matrix": conf_matrix,
            "classification_report": class_report,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            # Labeled validation set (gold standard test split) path — included
            # so callers can display / further log the held-out dataset path.
            "gold_standard_path": str(gold_standard_path),
        }

        # Save model (with scaler and semi-supervised trainer for reproducibility)
        model_output_path.parent.mkdir(parents=True, exist_ok=True)
        model_artifact = {
            "model": model,
            "scaler": scaler,
            "semi_trainer": semi_trainer,  # Include for inference
            "feature_names": feature_names,
            "model_version": model_version,
            "algorithm": algo_label,
            "optimal_k": optimal_k,
            "cluster_info": cluster_info,
            "trained_at": results["trained_at"],
        }

        with open(model_output_path, "wb") as f:
            pickle.dump(model_artifact, f)

        print(f"\n✓ Model saved to: {model_output_path}")

        # Log model to MLflow
        if use_mlflow:
            # Log dataset info
            class_distribution = {
                int(label): int(count)
                for label, count in zip(*np.unique(y_train, return_counts=True))  # noqa: B905
            }
            log_dataset_info(
                train_samples=len(y_train),
                test_samples=len(y_test),
                class_distribution=class_distribution,
            )

            # Log model artifact — may fail when running locally against a
            # Docker MLflow server whose artifact store is inside the container.
            # We tolerate the failure so metrics/params and the run_id are still
            # returned; model registration uses the local pickle file instead.
            #
            # MLflow v3 compatibility note: log_model(name=...) triggers the
            # /api/2.0/mlflow/logged-models endpoint. To avoid 404 errors when
            # the server does not yet support that endpoint we use save_model +
            # log_artifacts instead, which only touches the standard artifact
            # store — compatible with MLflow v2.x and v3.x servers.
            try:
                import tempfile as _tempfile

                with _tempfile.TemporaryDirectory() as _model_tmpdir:
                    mlflow.sklearn.save_model(model, os.path.join(_model_tmpdir, "model"))
                    mlflow.log_artifacts(
                        os.path.join(_model_tmpdir, "model"), artifact_path="model"
                    )
            except Exception as _art_err:
                print(f"[WARN] Could not log model artifact to MLflow: {_art_err}")

            # Log scaler and metadata separately
            try:
                mlflow.log_dict(
                    {
                        "feature_names": feature_names,
                        "model_version": model_version,
                        "optimal_k": optimal_k,
                        "primary_metric": primary_metric,
                    },
                    "model_metadata.json",
                )
            except Exception as _art_err:
                print(f"[WARN] Could not log metadata dict to MLflow: {_art_err}")

            # Log the full pickle so load_production_model_artifact() can
            # restore scaler + feature_names without re-deriving them.
            try:
                mlflow.log_artifact(str(model_output_path))
                # Validate the pickle was actually stored — log_artifact can fail
                # silently on network timeouts (DagsHub, corporate proxies).  If the
                # artifact is missing we do one immediate retry so the registry loader
                # always finds the scaler and never falls back to scaler-less inference.
                try:
                    if mlflow_run:
                        import mlflow as _mlflow_val

                        _run_id = mlflow_run.info.run_id
                        _pkl_name = os.path.basename(str(model_output_path))
                        _arts = _mlflow_val.MlflowClient().list_artifacts(_run_id, path="")
                        _found = any(a.path == _pkl_name for a in _arts)
                        if not _found:
                            print(f"[WARN] pkl '{_pkl_name}' not found after upload — retrying...")
                            mlflow.log_artifact(str(model_output_path))
                            _arts2 = _mlflow_val.MlflowClient().list_artifacts(_run_id, path="")
                            _found2 = any(a.path == _pkl_name for a in _arts2)
                            if _found2:
                                print("[OK]  pkl artifact re-uploaded successfully.")
                            else:
                                print(
                                    "[WARN] pkl artifact still missing after retry — "
                                    "registry load will fall back to scaler-less model."
                                )
                        else:
                            print(f"[OK]  pkl artifact '{_pkl_name}' verified in MLflow.")
                except Exception as _val_err:
                    print(f"[WARN] Could not validate pkl artifact upload: {_val_err}")
            except Exception as _art_err:
                print(f"[WARN] Could not log model pickle to MLflow: {_art_err}")

            # Log training metadata tags
            log_training_metadata(
                train_data_path=train_data_path,
                model_version=model_version,
                algorithm=algo_label,
                airflow_run_id=airflow_run_id,
                trained_by=os.environ.get("TRAINED_BY", "train_model"),
                deployment_mode=os.environ.get("DEPLOYMENT_MODE", "local"),
            )

            # Store run ID in results
            if mlflow_run:
                results["mlflow_run_id"] = mlflow_run.info.run_id
                print(f"✓ MLflow run ID: {mlflow_run.info.run_id}")

        return results

    finally:
        # End MLflow run — swallow 429 so the caller is not penalised for a
        # transient DagsHub rate-limit that occurs only during finalisation.
        if use_mlflow and mlflow_run:
            try:
                mlflow.end_run()
            except Exception as _end_exc:
                if _is_mlflow_rate_limited(_end_exc):
                    print(
                        "⚠️  DagsHub rate-limited on run end — "
                        "run data was logged; finalisation call skipped."
                    )
                else:
                    raise


def cleanup_old_training_splits(
    keep_n: int = 10,
    repo_root: Path | None = None,
    champion_run_id: str | None = None,
) -> dict[str, int]:
    """Remove stale training-split artifacts from ``data/processed/training_splits/``.

    Keeps the *keep_n* most-recent run directories plus the current champion's
    directory unconditionally.  Older directories are deleted.

    Args:
        keep_n:           Number of most-recent run directories to keep.
        repo_root:        Repository root.  Defaults to the project root derived
                          from this file's location.
        champion_run_id:  MLflow run-id of the current champion.  Directories
                          matching this ID are never deleted.  When *None*, the
                          function skips champion-protection.

    Returns:
        Dict with keys ``kept`` and ``deleted``.
    """
    import shutil

    _root = repo_root or Path(__file__).resolve().parents[2]
    splits_dir = _root / "data" / "processed" / "training_splits"
    if not splits_dir.exists():
        return {"kept": 0, "deleted": 0}

    run_dirs = sorted(
        [d for d in splits_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,  # newest first
    )

    deleted = 0
    kept = 0
    for i, d in enumerate(run_dirs):
        is_champion = d.name == champion_run_id
        if is_champion or i < keep_n:
            kept += 1
            continue
        try:
            shutil.rmtree(d)
            deleted += 1
        except Exception as _err:
            print(f"⚠️  Could not delete {d}: {_err}")

    return {"kept": kept, "deleted": deleted}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train device health classifier")
    parser.add_argument(
        "--train-data",
        type=Path,
        required=True,
        help="Path to training dataset (JSON)",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        help="Path to test dataset (JSON, optional)",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default="models/trained_model.pkl",
        help="Path to save trained model (default: models/trained_model.pkl)",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="v1.0",
        help="Model version identifier (default: v1.0)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum iterations for logistic regression (default: 1000)",
    )

    args = parser.parse_args()

    # Train model
    results = train_model(
        train_data_path=args.train_data,
        test_data_path=args.test_data,
        model_output_path=args.model_output,
        model_version=args.model_version,
        max_iter=args.max_iter,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model version: {results['model_version']}")
    print(f"Model saved to: {results['model_path']}")
    print(f"Training samples: {results['train_samples']}")
    print(f"Training accuracy: {results['train_accuracy']:.2%}")
    if "test_accuracy" in results:
        print(f"Test samples: {results['test_samples']}")
        print(f"Test accuracy: {results['test_accuracy']:.2%}")
