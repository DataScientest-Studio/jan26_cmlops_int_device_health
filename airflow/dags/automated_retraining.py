"""
Airflow DAG for automated model retraining.

This DAG reads directly from PostgreSQL — no DVC pull required.
PostgreSQL is always the authoritative, most up-to-date source of
labeled signals.  DVC/DagsHub is a versioning/backup mechanism that
runs independently via the sync_production_data DAG.

Workflow:
1. Validate data quality (labeled signal count in PostgreSQL)
2. Extract features from labeled signals
3. Train a challenger model with sliding window (90 days)
4. Compare challenger vs. champion in MLflow
5. Auto-promote challenger if improvement > threshold
6. Send notification

Schedule: Weekly (Sunday at 2 AM UTC)
"""

import contextlib
import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup

from airflow import DAG

# ──────────────────────────────────────────────────────────────────────────
# MLflow tracking URI strategy (local-first, Section 27):
#   MLFLOW_TRACKING_URI env var points to mlops_mlflow_buffer container
#   (http://mlflow_buffer:5000) which is always running in cloud mode.
#   No DagsHub calls during training.  DagsHub sync is done by the
#   separate sync_mlflow_to_dagshub DAG on a schedule.
# ──────────────────────────────────────────────────────────────────────────

try:
    from src.config import MIN_LABELED_SIGNALS, MIN_LABELED_SIGNALS_RECOMMENDED
except ImportError:
    MIN_LABELED_SIGNALS = 20
    MIN_LABELED_SIGNALS_RECOMMENDED = 100

_DAG_ID = "automated_retraining"


def _notify_retraining_failure(context: dict) -> None:
    """
    DAG-level on_failure_callback: increment retraining_failures_total via the API.

    Airflow task processes cannot write to the API's in-process Prometheus
    registry directly.  We POST to the API's internal endpoint so the counter
    surfaces on the /metrics scrape.  The call is best-effort: failures are
    printed but never re-raised (we don't want a metric call to mask the
    original failure).
    """
    import urllib.error
    import urllib.request

    api_url = os.environ.get("MLOPS_API_URL", "http://api:8000")
    dag_run = context.get("dag_run")
    reason = "dag_failure"
    if dag_run:
        # Narrow the reason if a specific task caused the failure
        failed_tasks = [
            ti.task_id
            for ti in (
                dag_run.get_task_instances() if hasattr(dag_run, "get_task_instances") else []
            )
            if ti.state == "failed"
        ]
        if "train_challenger" in " ".join(failed_tasks):
            reason = "training_error"
        elif "validate_data" in " ".join(failed_tasks):
            reason = "data_quality"
        elif "compare_models" in " ".join(failed_tasks) or "promote_if_better" in " ".join(
            failed_tasks
        ):
            reason = "promotion_error"

    try:
        req = urllib.request.Request(
            f"{api_url}/internal/metrics/retraining-failure?reason={reason}",
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5):
            pass
        print(f"✅ retraining_failures_total incremented (reason={reason})")
    except (urllib.error.URLError, OSError) as exc:
        print(f"⚠️  Could not increment retraining failure counter: {exc}")


def _notify_retraining_trigger(context: dict) -> None:
    """
    DAG-level on_success_callback: increment retraining_triggers_total via the API.

    Called when the full DAG completes successfully so NoRetrainingInWeek has
    data to evaluate.  Best-effort: exceptions are printed, never re-raised.
    """
    import urllib.error
    import urllib.request

    api_url = os.environ.get("MLOPS_API_URL", "http://api:8000")
    try:
        req = urllib.request.Request(
            f"{api_url}/internal/metrics/retraining-trigger?reason=scheduled",
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5):
            pass
        print("✅ retraining_triggers_total incremented (reason=scheduled)")
    except (urllib.error.URLError, OSError) as exc:
        print(f"⚠️  Could not increment retraining trigger counter: {exc}")


def _open_db():
    """Return a Database connected to PostgreSQL (Docker) or SQLite (fallback)."""
    from src.database.database import Database

    db_url = os.environ.get("DATABASE_URL", "")
    return (
        Database(db_url=db_url)
        if db_url.startswith("postgresql")
        else Database("/opt/airflow/data/database/mlops.db")
    )


def validate_data_quality() -> dict:
    """
    Validate data quality before training.

    Returns:
        Dict with validation metrics
    """

    db = _open_db()

    # Count labeled and unlabeled data
    labeled_count = db.count_labeled_signals()
    total_count = db.count_all_signals()

    if labeled_count == 0:
        raise ValueError("No labeled signals available for training")

    if labeled_count < MIN_LABELED_SIGNALS_RECOMMENDED:
        print(
            f"⚠️ Warning: Only {labeled_count} labeled signals (minimum {MIN_LABELED_SIGNALS_RECOMMENDED} recommended)"
        )

    metrics = {
        "labeled_signals": labeled_count,
        "total_signals": total_count,
        "label_ratio": labeled_count / total_count if total_count > 0 else 0,
        "validation_passed": labeled_count >= MIN_LABELED_SIGNALS,
    }

    print(f"✅ Data quality validation: {metrics}")
    return metrics


def extract_features_from_signals() -> dict:
    """
    Extract features from all labeled signals.

    Returns:
        Dict with feature extraction statistics
    """
    from src.signal_processing.feature_extractor import extract_features
    from src.signal_processing.signal_models import SignalData

    db = _open_db()

    # Get all labeled signal IDs
    signal_ids = db.get_labeled_signal_ids()

    if not signal_ids:
        raise ValueError("No labeled signals found for feature extraction")

    print(f"🔄 Extracting features from {len(signal_ids)} signals...")

    extracted_count = 0
    failed_count = 0

    for signal_id in signal_ids:
        try:
            # Load signal data
            signal_data = db.get_signal_data_by_id(signal_id)

            if signal_data is None:
                failed_count += 1
                continue

            # Build SignalData model from raw arrays
            sd = SignalData(
                time=signal_data["time_values"],
                amplitude=signal_data["amplitude_values"],
                shape_type="gaussian",  # Default; refined downstream if needed
            )

            # Extract features
            features = extract_features(sd)

            # Store in database
            db.store_features(signal_id, features)
            extracted_count += 1

        except Exception as e:
            print(f"⚠️ Feature extraction failed for signal {signal_id}: {e}")
            failed_count += 1

    stats = {
        "total_signals": len(signal_ids),
        "extracted": extracted_count,
        "failed": failed_count,
        "success_rate": extracted_count / len(signal_ids),
    }

    print(f"✅ Feature extraction complete: {stats}")
    return stats


def train_challenger_model(**context) -> dict:
    """
    Train new model using features already extracted to the DB.

    Instead of reading from a file path, this function passes the database
    connection directly to ``train_model(from_db=True)`` which fetches signals
    from PostgreSQL ``raw_signals`` directly — no temp JSON file required.
    The window filter is applied at SQL level via get_labeled_signal_ids().

    Returns:
        Dict with training metrics and model info
    """
    import mlflow

    from src.training.train import train_model

    print("🔄 Training Challenger model — loading signals from database...")

    # ── Configuration ──────────────────────────────────────────────────────
    sliding_window_days = int(os.getenv("SLIDING_WINDOW_DAYS", "90"))
    k_range_min = int(os.getenv("K_RANGE_MIN", "2"))
    k_range_max = int(os.getenv("K_RANGE_MAX", "5"))

    dag_run = context.get("dag_run")
    airflow_run_id = dag_run.run_id if dag_run else None

    # Determine the trigger reason to tag the model appropriately in lineage.
    # drift_triggered_retraining.py passes trigger_reason="evidently_drift_detection"
    # when it triggers this DAG; scheduled runs have no trigger_reason in conf.
    dag_conf = (dag_run.conf or {}) if dag_run else {}
    trigger_reason = dag_conf.get("trigger_reason", "scheduled")
    if trigger_reason == "evidently_drift_detection":
        _trained_by_tag = "drift_triggered_retraining"
    else:
        _trained_by_tag = "airflow_automated_retraining"

    # ── Load signal data from PostgreSQL via DB-backed training (Phase 5) ──
    # train_model(from_db=True) fetches signals directly from the DB,
    # eliminating the intermediate temp JSON file. The window filter is
    # applied at SQL level inside get_labeled_signal_ids().
    db = _open_db()
    try:
        # Verify there are signals before calling train_model (fail-fast)
        _n_labeled = len(db.get_labeled_signal_ids(window_days=sliding_window_days))
        if _n_labeled == 0:
            raise ValueError("No labeled signals found in database — cannot train challenger model")
        _n_unlabeled = len(db.get_unlabeled_signal_ids(window_days=sliding_window_days))
        print(
            f"📊 DB has {_n_labeled} labeled + {_n_unlabeled} unlabeled signals (window={sliding_window_days}d)"
        )

        # ── MLflow tracking URI ────────────────────────────────────────────
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_buffer:5000")
        mlflow.set_tracking_uri(tracking_uri)
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "automated_retraining_cloud")
        print(f"ℹ️  Using MLflow buffer: {tracking_uri}")

        results = train_model(
            from_db=True,
            db=db,
            model_output_path="models/challenger",
            model_version="retrained_pending",
            window_days=sliding_window_days,
            k_range=(k_range_min, k_range_max),
            mlflow_experiment_name=experiment_name,
            use_mlflow=True,
            allow_unlabeled=True,
            filter_unlabeled=False,
            airflow_run_id=airflow_run_id,
        )

    finally:
        db.close()

    run_id = results.get("mlflow_run_id", "")
    model_path = results.get("model_path", "")
    f1 = results.get("test_f1_score", 0.0)
    acc = results.get("test_accuracy", 0.0)

    # ── Register challenger in MLflow buffer registry ─────────────────────
    model_name: str = (
        os.getenv("MODEL_REGISTRY_NAME") or os.getenv("MODEL_NAME") or "device_health_classifier"
    )
    model_version = None
    if run_id:
        try:
            client = mlflow.tracking.MlflowClient()
            with contextlib.suppress(Exception):
                client.create_registered_model(model_name)
            mv = client.create_model_version(
                name=model_name,
                source=f"runs:/{run_id}/model",
                run_id=run_id,
                tags={"role": "Challenger", "trained_by": _trained_by_tag},
            )
            model_version = mv.version
            with contextlib.suppress(Exception):
                client.set_registered_model_alias(model_name, "challenger", str(mv.version))
            print(f"✅ Challenger v{mv.version} registered in MLflow buffer registry")
            # Backfill the canonical version in model_training_data now that we
            # know the real version number.
            if run_id:
                try:
                    _db_ver = _open_db()
                    _db_ver.update_model_version_by_run_id(run_id, f"v{mv.version}")
                    _db_ver.close()
                    print(f"✅ model_training_data.model_version updated to v{mv.version}")
                except Exception as _ver_exc:
                    print(f"⚠️ model_version backfill failed (non-fatal): {_ver_exc}")
        except Exception as exc:
            print(f"⚠️ Buffer registry registration failed (non-fatal): {exc}")

    return {
        "run_id": run_id,
        "model_version": model_version or "unknown",
        "model_path": str(model_path),
        "f1_score": f1,
        "accuracy": acc,
        "sliding_window_days": sliding_window_days,
        "tracking_uri": tracking_uri,
    }


def compare_with_champion(**context) -> dict:
    """
    Compare Challenger with current Champion model.

    Uses local MLflow registry for the challenger (always available).
    Attempts DagsHub for champion info; falls back to file-based comparison
    using ``models/champion_model.pkl`` when DagsHub is unavailable.

    Returns:
        Dict with comparison results
    """

    from src.training.promotion import auto_promote_model

    # Read training results from XCom for potential file-based fallback
    ti = context.get("task_instance")
    training_info = ti.xcom_pull(task_ids="training_group.train_challenger") if ti else {}
    challenger_f1 = float((training_info or {}).get("f1_score", 0.0))
    challenger_acc = float((training_info or {}).get("accuracy", 0.0))
    challenger_run_id = (training_info or {}).get("run_id", "")

    model_name: str = (
        os.getenv("MODEL_REGISTRY_NAME") or os.getenv("MODEL_NAME") or "device_health_classifier"
    )
    primary_metric = os.getenv("PRIMARY_METRIC", "f1_score")

    print(f"🔄 Comparing models on metric: {primary_metric}")
    print(f"   Challenger F1={challenger_f1:.4f}, Acc={challenger_acc:.4f}")

    # ── Phase 4: Re-evaluate champion on challenger's test split ──────────
    # This is the correct apples-to-apples comparison: both models evaluated
    # on the same (most recent) test signals.
    champion_f1_on_challenger_test: float | None = None
    if challenger_run_id:
        try:
            import json as _json
            import pickle as _pickle

            from sklearn.metrics import f1_score as _f1_score

            from src.signal_processing.feature_extractor import extract_features as _extract
            from src.signal_processing.signal_models import SignalData as _SignalData

            _db = _open_db()
            try:
                _test_ids = _db.get_training_signal_ids(challenger_run_id, split="test")
            finally:
                _db.close()

            if _test_ids:
                # Load champion model artifact
                _champion_pkl = Path("/opt/airflow/models/champion_model.pkl")
                if _champion_pkl.exists():
                    with open(_champion_pkl, "rb") as _f:
                        _champion_bundle = _pickle.load(_f)

                    # Load test signals from the exported JSON if available
                    _repo_root = Path("/opt/airflow")
                    _split_json = (
                        _repo_root
                        / "data"
                        / "processed"
                        / "training_splits"
                        / challenger_run_id
                        / "test.json"
                    )
                    if _split_json.exists():
                        with open(_split_json) as _split_fh:
                            _test_data = _json.load(_split_fh)
                        _test_signals = _test_data.get("signals", [])
                    else:
                        # Fall back to loading from DB
                        _db2 = _open_db()
                        try:
                            _test_signals = []
                            for _sid in _test_ids:
                                _raw = _db2.get_signal_data_by_id(_sid)
                                _lbl = _db2.get_label_by_signal_id(_sid)
                                if _raw and _lbl in (0, 1):
                                    _test_signals.append(
                                        {
                                            "time": _raw["time_values"],
                                            "amplitude": _raw["amplitude_values"],
                                            "label": _lbl,
                                        }
                                    )
                        finally:
                            _db2.close()

                    if _test_signals:
                        _feats = []
                        _labels_true = []
                        for _s in _test_signals:
                            _sd = _SignalData(
                                time=_s["time"],
                                amplitude=_s["amplitude"],
                                shape_type="gaussian",
                            )
                            _feat_dict = _extract(_sd)
                            _feat_names = [
                                "fwhm",
                                "peak_height",
                                "peak_area",
                                "noise_level",
                                "snr",
                                "peak_center",
                            ]
                            _feats.append([_feat_dict.get(n) or 0.0 for n in _feat_names])
                            _labels_true.append(_s.get("label", -1))

                        import numpy as _np

                        _X_test = _np.array(_feats)
                        _y_true = _np.array(_labels_true)
                        _valid = _y_true != -1
                        if _valid.sum() >= 2:
                            _scaler = _champion_bundle.get("scaler")
                            _clf = _champion_bundle.get("model")
                            if _scaler is not None and _clf is not None:
                                _X_scaled = _scaler.transform(_X_test[_valid])
                                _y_pred = _clf.predict(_X_scaled)
                                champion_f1_on_challenger_test = float(
                                    _f1_score(
                                        _y_true[_valid], _y_pred, average="binary", zero_division=0
                                    )
                                )
                                print(
                                    f"✅ Champion re-evaluated on {_valid.sum()} challenger test signals: "
                                    f"F1={champion_f1_on_challenger_test:.4f}"
                                )
        except Exception as _reeval_exc:
            print(f"⚠️  Champion re-evaluation failed (non-fatal): {_reeval_exc}")
    # ─────────────────────────────────────────────────────────────────────

    # ── Try MLflow buffer registry ────────────────────────────────────────
    try:
        comparison = auto_promote_model(
            model_name=model_name,
            metric_name=f"test_{primary_metric}",
            dry_run=True,
        )
        if (
            comparison.get("reason")
            != "No production model exists. Register and promote a baseline model first."
        ):
            # Inject challenger_f1 / champion_f1 so request_human_approval can read them
            decision = comparison.get("decision", {}) or {}
            if "challenger_f1" not in comparison:
                comparison["challenger_f1"] = float(
                    decision.get("challenger_metric", 0.0) or challenger_f1
                )
            if "champion_f1" not in comparison:
                comparison["champion_f1"] = float(decision.get("champion_metric", 0.0) or 0.0)
            # Phase 4: override champion_f1 with re-evaluated value if available
            if champion_f1_on_challenger_test is not None:
                comparison["champion_f1"] = champion_f1_on_challenger_test
                comparison["champion_f1_on_challenger_test"] = champion_f1_on_challenger_test
            print(f"✅ Local comparison complete: {comparison}")
            return comparison
    except Exception as exc:
        print(f"⚠️ Local MLflow comparison failed: {exc}")

    # ── Fallback: file-based champion comparison ───────────────────────────
    champion_pkl = Path("/opt/airflow/models/champion_model.pkl")
    if champion_pkl.exists():
        print("ℹ️  Comparing via file-based champion model (no registry champion found)")
        # We don't have stored champion metrics; assume challenger is an improvement
        # if it passes a minimum quality threshold (F1 > 0.5).
        float(os.getenv("CHAMPION_PROMOTION_THRESHOLD", "0.02"))
        should_promote = challenger_f1 >= 0.5  # basic quality gate
        comparison = {
            "promoted": False,  # dry_run=True equivalent
            "should_promote": should_promote,
            "challenger_f1": challenger_f1,
            "champion_f1": champion_f1_on_challenger_test,
            "champion_f1_on_challenger_test": champion_f1_on_challenger_test,
            "reason": "File-based comparison (no champion in local registry)",
        }
        print(f"✅ File-based comparison: should_promote={should_promote}")
        return comparison

    # ── No champion at all — first run ────────────────────────────────────
    comparison = {
        "promoted": False,
        "should_promote": True,  # Always promote when no champion exists
        "challenger_f1": challenger_f1,
        "champion_f1": None,
        "champion_f1_on_challenger_test": None,
        "reason": "No champion model found — challenger will be promoted on next step",
    }
    print(f"✅ No champion found — new model will be promoted: {comparison}")
    return comparison


def auto_promote_if_better(**context):
    """
    Promote Challenger to Champion if performance improves.

    Uses local MLflow registry; if no registry champion exists, falls back to
    direct file copy (challenger PKL → champion PKL) when the comparison step
    determined that promotion is warranted.

    When human approval was required and was granted, the F1 improvement check
    is bypassed — human approval means unconditional promotion.
    """
    import shutil

    from src.training.promotion import auto_promote_model

    model_name: str = (
        os.getenv("MODEL_REGISTRY_NAME") or os.getenv("MODEL_NAME") or "device_health_classifier"
    )
    promotion_threshold = float(os.getenv("CHAMPION_PROMOTION_THRESHOLD", "0.02"))
    primary_metric = os.getenv("PRIMARY_METRIC", "f1_score")

    # Read results from upstream XCom tasks
    ti = context.get("task_instance")
    training_info = ti.xcom_pull(task_ids="training_group.train_challenger") if ti else {}
    comparison = ti.xcom_pull(task_ids="evaluation_group.compare_models") if ti else {}

    challenger_model_path = (training_info or {}).get("model_path", "")
    should_promote = (comparison or {}).get("should_promote", False)

    # Check if human approval gate was used and whether it was approved.
    # If approved, bypass the min_improvement threshold entirely.
    dag_run = context.get("dag_run")
    conf = (dag_run.conf or {}) if dag_run else {}
    _raw_approval = conf.get("require_human_approval", False)
    human_approval_requested = (
        _raw_approval
        if isinstance(_raw_approval, bool)
        else str(_raw_approval).lower() in ("true", "1", "yes")
    )
    # If the human gate was active AND we reached this task (i.e., the
    # wait_for_human_approval task did NOT raise AirflowSkipException),
    # that means the approval was GRANTED → force-promote regardless of F1 delta.
    effective_threshold = 0.0 if human_approval_requested else promotion_threshold
    force_promote = human_approval_requested  # bypass decision logic if approved

    if human_approval_requested:
        print("🔐 Human approval gate was active and approved — bypassing F1 threshold check")
        should_promote = True

    print(f"🔄 Checking promotion criteria: improvement > {effective_threshold}")
    print(f"   should_promote={should_promote}, challenger_path={challenger_model_path}")

    # ── When human approval was granted, promote challenger directly ────────
    # Use promote_model() directly to bypass F1 comparison entirely.
    # This covers the case where challenger_f1 == champion_f1 (e.g. both 1.0),
    # which would cause auto_promote_model to skip promotion (0.0 > 0.0 = False).
    if force_promote:
        result = {"promoted": False, "reason": "force-promote path not yet completed"}
        try:
            from src.training.registry import get_staging_models, promote_model

            staging = get_staging_models(model_name)
            if staging:
                best = max(staging, key=lambda m: m["version"])
                challenger_version = best["version"]
                promote_model(
                    model_name=model_name,
                    version=challenger_version,
                    stage="Production",
                    archive_existing_production=True,
                )
                result = {
                    "promoted": True,
                    "new_champion_version": challenger_version,
                    "reason": f"Human-approved promotion of v{challenger_version}",
                }
                print(f"✅ Human-approved: challenger v{challenger_version} promoted to champion!")
            else:
                print("⚠️ No challenger in Staging — falling through to file-based promotion")
        except Exception as exc:
            print(f"⚠️ Human-approval direct promotion failed: {exc}")
            result = {"promoted": False, "reason": str(exc)}

    # ── Try MLflow buffer registry promotion (non-human-approval path) ───
    elif not force_promote:
        try:
            result = auto_promote_model(
                model_name=model_name,
                metric_name=f"test_{primary_metric}",
                min_improvement=effective_threshold,
            )
            if result.get("promoted", False):
                print("✅ Challenger promoted via local MLflow registry!")
            else:
                print(f"ℹ️  Local registry promotion skipped: {result.get('reason')}")
        except Exception as exc:
            print(f"⚠️ Local MLflow promotion failed: {exc}")
            result = {"promoted": False, "reason": str(exc)}

    # ── File-based promotion fallback ─────────────────────────────────────
    # When local MLflow couldn't promote (no champion in registry), do a
    # direct file copy if the comparison step said to promote.
    if not result.get("promoted") and should_promote and challenger_model_path:
        challenger_pkl = Path(challenger_model_path)
        if not challenger_pkl.exists():
            # Try common paths
            for candidate in [
                Path("/opt/airflow/models/challenger/model.pkl"),
                Path("/opt/airflow/models/challenger.pkl"),
            ]:
                if candidate.exists():
                    challenger_pkl = candidate
                    break

        champion_pkl = Path("/opt/airflow/models/champion_model.pkl")
        if challenger_pkl.exists():
            # Backup existing champion
            if champion_pkl.exists():
                backup = champion_pkl.with_suffix(".pkl.backup")
                shutil.copy2(champion_pkl, backup)
                print(f"ℹ️  Champion backed up → {backup}")
            # Promote challenger
            champion_pkl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(challenger_pkl, champion_pkl)
            print(f"✅ Challenger promoted via file copy: {challenger_pkl} → {champion_pkl}")
            result = {
                "promoted": True,
                "reason": "file-based promotion",
                "champion_path": str(champion_pkl),
            }
        else:
            print(f"⚠️ Cannot promote: challenger model file not found at {challenger_model_path}")

    if result.get("promoted", False):
        print("✅ Challenger promoted to Champion!")
    else:
        print(f"⏸️ Challenger not promoted: {result.get('reason', 'insufficient improvement')}")

    # ── Update KPI gauges via the API ─────────────────────────────────────
    # model_deploy_time_seconds: seconds from DAG trigger to promotion
    # automation_rate_gauge: 1.0 if triggered by schedule/drift, 0.0 if manual
    try:
        import urllib.error
        import urllib.request

        api_url = os.environ.get("MLOPS_API_URL", "http://api:8000")
        dag_run = context.get("dag_run")

        # Deploy time = now minus logical_date of this dag run
        deploy_time_s: float | None = None
        logical_date = getattr(dag_run, "logical_date", None) or getattr(
            dag_run, "execution_date", None
        )
        if logical_date is not None:
            import datetime as _dt

            now_utc = _dt.datetime.now(_dt.timezone.utc)
            if logical_date.tzinfo is None:
                logical_date = logical_date.replace(tzinfo=_dt.timezone.utc)
            deploy_time_s = (now_utc - logical_date).total_seconds()

        # Automation rate: 0.0 if manually triggered (run_type == "manual")
        run_type = getattr(dag_run, "run_type", "scheduled")
        automation_rate = 0.0 if str(run_type) == "manual" else 1.0

        payload_bytes = (
            __import__("json")
            .dumps(
                {
                    "deploy_time_seconds": deploy_time_s,
                    "automation_rate": automation_rate,
                }
            )
            .encode()
        )
        req = urllib.request.Request(
            f"{api_url}/internal/kpi-metrics",
            data=payload_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as _resp:
            print(
                f"✓ KPI gauges updated (deploy_time={deploy_time_s:.0f}s, automation_rate={automation_rate})"
            )
    except Exception as exc:
        print(f"⚠️ Could not update KPI gauges: {exc} (non-fatal)")


def cleanup_old_training_splits(**context) -> dict:
    """
    Phase 6 — Retention policy: delete training split artifacts older than the
    last KEEP_N model runs, retaining the current champion's splits unconditionally.

    This prevents ``data/processed/training_splits/`` from growing indefinitely.
    Configuration via env vars:
      TRAINING_SPLITS_KEEP_N  — number of most-recent runs to keep (default: 10)
    """
    import shutil

    import mlflow

    keep_n = int(os.getenv("TRAINING_SPLITS_KEEP_N", "10"))

    # Resolve repo root (mounted into Airflow container as /opt/airflow)
    _repo_root = Path("/opt/airflow")
    splits_dir = _repo_root / "data" / "processed" / "training_splits"
    if not splits_dir.exists():
        print("ℹ️  No training splits directory found — nothing to clean up")
        return {"deleted": 0, "kept": 0}

    # Discover all run-id subdirectories sorted by mtime descending
    run_dirs = sorted(
        [d for d in splits_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,  # newest first
    )

    # Identify the current champion's run_id so we never delete its splits
    champion_run_id: str | None = None
    try:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_buffer:5000")
        mlflow.set_tracking_uri(tracking_uri)
        model_name = (
            os.getenv("MODEL_REGISTRY_NAME")
            or os.getenv("MODEL_NAME")
            or "device_health_classifier"
        )
        client = mlflow.tracking.MlflowClient()
        mv_list = client.get_model_version_by_alias(model_name, "champion")
        champion_run_id = mv_list.run_id if mv_list else None
    except Exception as _exc:
        print(f"⚠️  Could not determine champion run_id: {_exc}")

    deleted = 0
    kept = 0
    for i, d in enumerate(run_dirs):
        run_id = d.name
        is_champion = run_id == champion_run_id
        if is_champion or i < keep_n:
            kept += 1
            continue
        try:
            shutil.rmtree(d)
            deleted += 1
            print(f"🗑️  Deleted stale split artifacts: {d.name}")
        except Exception as _del_err:
            print(f"⚠️  Could not delete {d}: {_del_err}")

    print(f"✅ Cleanup complete: kept={kept}, deleted={deleted} (keep_n={keep_n})")
    return {"deleted": deleted, "kept": kept}


def send_retraining_notification(**context):
    """
    Send notification on retraining completion.

    Args:
        context: Airflow context with task instance info
    """
    task_instance = context["task_instance"]
    dag_run = context["dag_run"]

    # Get XCom results from previous tasks
    training_info = task_instance.xcom_pull(task_ids="training_group.train_challenger") or {}
    comparison = task_instance.xcom_pull(task_ids="evaluation_group.compare_models") or {}

    message = f"""
    🤖 Automated Retraining Complete

    DAG Run: {dag_run.run_id}
    Start: {dag_run.start_date}
    End: {datetime.now()}

    Training Results:
    - Model Version: {training_info.get("model_version", "N/A")}
    - F1 Score: {training_info.get("f1_score", "N/A"):.4f}
    - Accuracy: {training_info.get("accuracy", "N/A"):.4f}

    Comparison:
    - Champion F1: {comparison.get("champion_f1", "N/A")}
    - Challenger F1: {comparison.get("challenger_f1", "N/A")}
    - Promoted: {comparison.get("promoted", False)}
    """

    print(message)
    # In production, integrate with Slack/email/PagerDuty


def _request_human_approval(**context):
    """
    Task 5 (Option B) — Human Review Gate: write a pending approval record to the DB.

    Creates a row in model_approvals with status='pending'.  The Streamlit UI
    reads this table and lets a human approve or reject before promotion proceeds.
    The approval_id is pushed to XCom so the wait task can poll it.

    This gate is DISABLED by default.  Enable it by triggering the DAG with
    Conf: {"require_human_approval": true}.  When disabled, the wait task is
    skipped automatically and promotion proceeds without a review.
    """
    import contextlib

    ti = context.get("task_instance")

    # Check whether human approval is required for this run
    dag_run = context.get("dag_run")
    conf = dag_run.conf or {} if dag_run else {}
    _raw_approval = conf.get("require_human_approval", False)
    # Handle both bool and string (Jinja rendering in TriggerDagRunOperator yields strings)
    require_approval = (
        _raw_approval
        if isinstance(_raw_approval, bool)
        else str(_raw_approval).lower() in ("true", "1", "yes")
    )
    if not require_approval:
        print("ℹ️  Human approval gate disabled (require_human_approval=False) — skipping")
        if ti:
            ti.xcom_push(key="approval_id", value=None)
        return {"approval_id": None, "gate_enabled": False}
    training_info = ti.xcom_pull(task_ids="training_group.train_challenger") or {} if ti else {}
    comparison = ti.xcom_pull(task_ids="evaluation_group.compare_models") or {} if ti else {}

    model_version = training_info.get("model_version", "unknown")
    mlflow_run_id = training_info.get("mlflow_run_id", "")
    challenger_f1 = float(comparison.get("challenger_f1", 0.0) or 0.0)
    # Prefer champion_f1_on_challenger_test for fair comparison when available
    champion_f1 = float(
        comparison.get("champion_f1_on_challenger_test")
        or comparison.get("champion_f1", 0.0)
        or 0.0
    )
    champion_f1_on_challenger_test = comparison.get("champion_f1_on_challenger_test")

    # Try PostgreSQL first, fall back to SQLite
    approval_id: int | None = None

    pg_url = os.environ.get("DATABASE_URL", "")
    if pg_url:
        try:
            import psycopg2

            conn = psycopg2.connect(pg_url)
            with conn, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO model_approvals
                        (model_version, mlflow_run_id, challenger_f1, champion_f1,
                         champion_f1_on_challenger_test, status)
                    VALUES (%s, %s, %s, %s, %s, 'pending')
                    RETURNING id
                    """,
                    (
                        model_version,
                        mlflow_run_id,
                        challenger_f1,
                        champion_f1,
                        champion_f1_on_challenger_test,
                    ),
                )
                approval_id = cur.fetchone()[0]
            conn.close()
            print(
                f"✓ Approval request created (PostgreSQL) id={approval_id}, model={model_version}"
            )
        except Exception as exc:
            print(f"⚠️ PostgreSQL insert failed: {exc} — trying SQLite fallback")

    if approval_id is None:
        import sqlite3

        _project_root = Path(__file__).resolve().parents[1]
        db_path = _project_root / "data" / "mlops.db"
        with contextlib.closing(sqlite3.connect(str(db_path))) as con:
            cur2 = con.cursor()
            cur2.execute(
                """
                INSERT INTO model_approvals
                    (model_version, mlflow_run_id, challenger_f1, champion_f1,
                     champion_f1_on_challenger_test, status)
                VALUES (?, ?, ?, ?, ?, 'pending')
                """,
                (
                    model_version,
                    mlflow_run_id,
                    challenger_f1,
                    champion_f1,
                    champion_f1_on_challenger_test,
                ),
            )
            approval_id = cur2.lastrowid
            con.commit()
        print(f"✓ Approval request created (SQLite) id={approval_id}, model={model_version}")

    if ti:
        ti.xcom_push(key="approval_id", value=approval_id)

    return {"approval_id": approval_id, "model_version": model_version}


def _wait_for_human_approval(**context):
    """
    Task 5 (Option B) — poll model_approvals until status != 'pending'.

    Polls every 60 seconds.  If execution_timeout is reached (48 h), Airflow
    marks the task as failed and promotion is skipped.  A status of 'rejected'
    also skips promotion by raising AirflowSkipException.
    """
    import time

    try:
        from airflow.exceptions import AirflowSkipException
    except ImportError:
        AirflowSkipException = RuntimeError

    ti = context.get("task_instance")
    approval_id = (
        ti.xcom_pull(task_ids="evaluation_group.request_human_approval", key="approval_id")
        if ti
        else None
    )

    if approval_id is None:
        print("⚠️ No approval_id found — skipping gate (approval record not created)")
        return

    pg_url = os.environ.get("DATABASE_URL", "")
    _project_root = Path(__file__).resolve().parents[1]
    db_path = _project_root / "data" / "mlops.db"

    def _get_status() -> str:
        if pg_url:
            try:
                import psycopg2

                conn = psycopg2.connect(pg_url)
                with conn, conn.cursor() as cur:
                    cur.execute("SELECT status FROM model_approvals WHERE id = %s", (approval_id,))
                    row = cur.fetchone()
                conn.close()
                return row[0] if row else "pending"
            except Exception as exc:
                print(f"⚠️ PostgreSQL poll failed: {exc}")
        import contextlib
        import sqlite3

        with contextlib.closing(sqlite3.connect(str(db_path))) as con:
            row = con.execute(
                "SELECT status FROM model_approvals WHERE id = ?", (approval_id,)
            ).fetchone()
        return row[0] if row else "pending"

    # Check whether gate was enabled for this run
    dag_run = context.get("dag_run")
    conf = dag_run.conf or {} if dag_run else {}
    _raw_approval = conf.get("require_human_approval", False)
    # Handle both bool and string (Jinja rendering in TriggerDagRunOperator yields strings)
    require_approval = (
        _raw_approval
        if isinstance(_raw_approval, bool)
        else str(_raw_approval).lower() in ("true", "1", "yes")
    )
    if not require_approval or approval_id is None:
        print("ℹ️  Human approval gate disabled or not requested — auto-proceeding to promotion")
        return

    print(f"⏳ Waiting for human approval (id={approval_id}) — check the Streamlit UI…")
    while True:
        status = _get_status()
        if status == "approved":
            print(f"✅ Approval id={approval_id} approved — proceeding to promotion")
            return
        if status == "rejected":
            print(f"🚫 Approval id={approval_id} rejected — skipping promotion")
            raise AirflowSkipException(f"Model approval id={approval_id} rejected by reviewer")
        print(f"   Still pending… sleeping 60 s (status={status})")
        time.sleep(60)


# ========================================
# DAG Definition
# ========================================

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "on_failure_callback": _notify_retraining_failure,
    "on_success_callback": _notify_retraining_trigger,
}

with DAG(
    dag_id="automated_retraining",
    description="Weekly automated model retraining — reads from PostgreSQL directly",
    default_args=default_args,
    schedule_interval="0 2 * * 0",  # Sunday at 2 AM UTC
    start_date=datetime(2026, 4, 6),  # First Sunday
    catchup=False,
    is_paused_upon_creation=False,
    tags=["training", "mlflow", "champion-challenger"],
    max_active_runs=1,
) as dag:
    # Task 1: Validate data quality
    validate_data_task = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data_quality,
        doc_md="""
        ### Validate Data Quality
        Checks minimum labeled signal count and data completeness in PostgreSQL.
        """,
    )

    # Task Group: Feature Engineering
    with TaskGroup("feature_group", tooltip="Feature extraction pipeline") as feature_group:
        extract_features_task = PythonOperator(
            task_id="extract_features",
            python_callable=extract_features_from_signals,
        )

    # Task Group: Model Training
    with TaskGroup("training_group", tooltip="Model training pipeline") as training_group:
        train_task = PythonOperator(
            task_id="train_challenger",
            python_callable=train_challenger_model,
            provide_context=True,
        )

    # Task Group: Model Evaluation
    with TaskGroup(
        "evaluation_group", tooltip="Model evaluation and promotion"
    ) as evaluation_group:
        compare_task = PythonOperator(
            task_id="compare_models",
            python_callable=compare_with_champion,
            provide_context=True,
        )

        request_approval_task = PythonOperator(
            task_id="request_human_approval",
            python_callable=_request_human_approval,
            provide_context=True,
        )

        wait_for_approval_task = PythonOperator(
            task_id="wait_for_human_approval",
            python_callable=_wait_for_human_approval,
            provide_context=True,
            execution_timeout=timedelta(hours=4),  # fail if no response within 4 h
            retries=0,
        )

        promote_task = PythonOperator(
            task_id="promote_if_better",
            python_callable=auto_promote_if_better,
            provide_context=True,
        )

        compare_task >> request_approval_task >> wait_for_approval_task >> promote_task

    # Task: Send notification
    notify_task = PythonOperator(
        task_id="send_notification",
        python_callable=send_retraining_notification,
        provide_context=True,
    )

    # Task: Clean up old training split artifacts (Phase 6 — retention policy)
    cleanup_task = PythonOperator(
        task_id="cleanup_training_splits",
        python_callable=cleanup_old_training_splits,
        provide_context=True,
        doc_md="""
        ### Cleanup Old Training Splits
        Deletes ``data/processed/training_splits/`` subdirectories older than
        ``TRAINING_SPLITS_KEEP_N`` (default 10) most-recent runs.
        The current champion's split artifacts are always retained.
        """,
    )

    # Task dependencies
    (
        validate_data_task
        >> feature_group
        >> training_group
        >> evaluation_group
        >> notify_task
        >> cleanup_task
    )
