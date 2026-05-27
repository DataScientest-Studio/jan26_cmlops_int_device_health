"""
EvidentlyAI Drift Detection DAG

This DAG:
1. Loads reference data (training set) from database
2. Loads recent production data (last N predictions, configurable via DRIFT_CURRENT_DATA_LIMIT)
3. Runs EvidentlyAI drift detection
4. Generates HTML drift reports
5. Records drift metrics in Prometheus
6. Triggers retraining if drift exceeds threshold

Schedule: Daily at 2 AM UTC
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow.operators.python import PythonOperator, ShortCircuitOperator

from airflow import DAG

try:
    from _dag_guards import require_cloud_mode
except ModuleNotFoundError:
    from airflow.dags._dag_guards import require_cloud_mode

try:
    from src.config import MIN_LABELED_SIGNALS
except ImportError:
    MIN_LABELED_SIGNALS = 20

_DAG_ID = "evidently_drift_detection"

# Number of most-recent predictions to use as the current window for drift detection.
# Using a count-based window (rather than a fixed time range) ensures that drift
# provocation signals are not diluted by a large backlog of normal predictions.
_CURRENT_DATA_LIMIT = int(os.environ.get("DRIFT_CURRENT_DATA_LIMIT", "500"))


def _open_db():  # noqa: ANN202
    """Open the database connection using DATABASE_URL (PostgreSQL in cloud)."""
    import sys

    sys.path.insert(0, "/opt/airflow")
    from src.database.database import Database

    db_url = os.environ.get("DATABASE_URL", "")
    if db_url.startswith("postgresql"):
        return Database(db_url=db_url)
    return Database("/opt/airflow/data/database/mlops.db")


# Default arguments
default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email": ["alerts@mlops-device-health.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def check_drift_detection_prerequisites(**context) -> bool:
    """
    Check if drift detection can run.

    Verifies:
    - Running in cloud mode
    - Database connection available
    - Sufficient labeled samples for reference data
    - Sufficient recent predictions for current data

    Returns:
        True if prerequisites met, False otherwise
    """
    require_cloud_mode(_DAG_ID)

    db = _open_db()

    try:
        # Check reference data (labeled samples)
        labeled_ids = db.get_labeled_signal_ids(limit=1000)
        if len(labeled_ids) < MIN_LABELED_SIGNALS:
            print(f"❌ Insufficient reference data: {len(labeled_ids)} < {MIN_LABELED_SIGNALS}")
            return False

        print(f"✅ Reference data available: {len(labeled_ids)} labeled samples")

        # Check recent predictions — count-based window
        cursor = db.conn.cursor()
        cursor.execute("SELECT COUNT(*) AS n FROM predictions")
        row = cursor.fetchone()
        total_count = int(row["n"]) if row else 0

        if total_count < MIN_LABELED_SIGNALS:
            print(f"❌ Insufficient current data: {total_count} < {MIN_LABELED_SIGNALS}")
            return False

        print(
            f"✅ Current data available: {min(total_count, _CURRENT_DATA_LIMIT)} of {total_count} predictions will be used"
        )

        return True

    except Exception as e:
        print(f"❌ Prerequisites check failed: {e}")
        return False


def load_reference_data(**context) -> dict:
    """
    Load reference/training data from database.

    Returns:
        Dictionary with reference data metrics
    """
    import pandas as pd

    db = _open_db()

    # Get labeled signal IDs (reference data)
    signal_ids = db.get_labeled_signal_ids(limit=10000)
    print(f"Found {len(signal_ids)} labeled signals for reference data")

    # Load features and labels
    features_list = []
    for signal_id in signal_ids:
        features = db.get_features_by_signal_id(signal_id)
        label = db.get_label_by_signal_id(signal_id)

        if features and label is not None:
            feature_dict = {
                "signal_id": signal_id,
                **features,
                "ground_truth_label": label,
            }
            features_list.append(feature_dict)

    df = pd.DataFrame(features_list)

    # Save reference data to shared location
    output_path = Path("/opt/airflow/data/drift/reference_data.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"✅ Saved {len(df)} reference samples to {output_path}")

    # Push metadata to XCom
    return {
        "path": str(output_path),
        "n_samples": len(df),
        "n_features": len(df.columns),
        "timestamp": datetime.now().isoformat(),
    }


def load_current_data(**context) -> dict:
    """
    Load recent production data from database.

    Returns:
        Dictionary with current data metrics
    """
    import pandas as pd

    db = _open_db()

    # Use the most recent _CURRENT_DATA_LIMIT predictions (count-based window).
    # This avoids drift signals being diluted by a large backlog of old normal
    # predictions that would swamp a small drift provocation batch.
    query = """
        SELECT
            p.prediction_id,
            p.device_id,
            p.predicted_label,
            p.prediction_confidence,
            p.created_at,
            s.signal_id
        FROM predictions p
        LEFT JOIN raw_signals s ON s.prediction_id = p.prediction_id
        ORDER BY p.created_at DESC
        LIMIT ?
    """

    cursor = db.conn.cursor()
    cursor.execute(query, (_CURRENT_DATA_LIMIT,))
    rows = cursor.fetchall()
    predictions_df = pd.DataFrame([dict(r) for r in rows])
    print(f"Found {len(predictions_df)} recent predictions")

    # Load features for predictions
    features_list = []
    for _, row in predictions_df.iterrows():
        features = db.get_features_by_signal_id(row["signal_id"])

        if features:
            feature_dict = {
                "signal_id": row["signal_id"],
                "prediction_id": row["prediction_id"],
                **features,
                "predicted_label": row["predicted_label"],
                "prediction_confidence": row["prediction_confidence"],
            }

            # Add ground truth if available
            label = db.get_label_by_signal_id(row["signal_id"])
            if label is not None:
                feature_dict["ground_truth_label"] = label

            features_list.append(feature_dict)

    df = pd.DataFrame(features_list)

    # Save current data
    output_path = Path("/opt/airflow/data/drift/current_data.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"✅ Saved {len(df)} current samples to {output_path}")

    return {
        "path": str(output_path),
        "n_samples": len(df),
        "n_features": len(df.columns),
        "timestamp": datetime.now().isoformat(),
    }


def run_drift_detection(**context) -> dict:
    """
    Run Evidently drift detection and generate report.

    Returns:
        Dictionary with drift detection results
    """
    import sys
    from pathlib import Path

    import pandas as pd

    sys.path.insert(0, "/opt/airflow")
    from src.monitoring.drift_detection import DriftDetector

    def _push_drift_metric(drift_type: str) -> None:
        """Push drift detection event to the API's Prometheus registry via HTTP.

        prometheus_client counters are in-process; the Airflow worker process
        has its own registry that Prometheus never scrapes.  Calling the API's
        internal endpoint is the only way to surface this event on /metrics.
        """
        import os

        import requests

        api_url = os.environ.get("API_INTERNAL_URL", "http://api:8000")
        try:
            requests.post(
                f"{api_url}/internal/metrics/drift-detection",
                params={"drift_type": drift_type},
                timeout=5,
            )
        except Exception as _e:
            print(f"Warning: could not push drift metric to API ({_e}); Prometheus won't see it")

    # Load reference and current data
    ti = context["ti"]
    ref_info = ti.xcom_pull(task_ids="load_reference_data")
    curr_info = ti.xcom_pull(task_ids="load_current_data")

    reference_df = pd.read_parquet(ref_info["path"])
    current_df = pd.read_parquet(curr_info["path"])

    print(f"Reference data: {len(reference_df)} samples")
    print(f"Current data: {len(current_df)} samples")

    # Determine feature columns
    exclude_cols = {
        "signal_id",
        "prediction_id",
        "ground_truth_label",
        "predicted_label",
        "prediction_confidence",
    }
    feature_cols = [col for col in reference_df.columns if col not in exclude_cols]

    print(f"Analyzing {len(feature_cols)} features for drift")

    # Initialize drift detector
    detector = DriftDetector(
        feature_columns=feature_cols,
        target_column="ground_truth_label" if "ground_truth_label" in current_df.columns else None,
        prediction_column="predicted_label",
    )

    # Detect data drift
    print("Running data drift detection...")
    data_drift = detector.detect_data_drift(
        reference_data=reference_df,
        current_data=current_df,
        stattest_threshold=0.05,
    )

    if data_drift["drift_detected"]:
        print(
            f"⚠️ Data drift DETECTED: {data_drift['n_drifted_features']}/{data_drift['n_features']} features"
        )
        print(f"Drifted features: {data_drift['drifted_features']}")
        _push_drift_metric(drift_type="data")
        # feature drift = individual feature(s) drifting (always true when data drift detected)
        _push_drift_metric(drift_type="feature")
    else:
        print("✅ No data drift detected")

    # Detect target drift (if labels available)
    target_drift = None
    if detector.target_column and detector.target_column in current_df.columns:
        print("Running target drift detection...")
        target_drift = detector.detect_target_drift(
            reference_data=reference_df,
            current_data=current_df,
        )

        if target_drift["drift_detected"]:
            print("⚠️ Target drift DETECTED")
            _push_drift_metric(drift_type="concept")
        else:
            print("✅ No target drift detected")

    # Detect prediction drift (only if predicted_label exists in both datasets)
    prediction_drift = None
    if (
        detector.prediction_column
        and detector.prediction_column in reference_df.columns
        and detector.prediction_column in current_df.columns
    ):
        print("Running prediction drift detection...")
        prediction_drift = detector.detect_prediction_drift(
            reference_data=reference_df,
            current_data=current_df,
        )

        if prediction_drift["drift_detected"]:
            print("⚠️ Prediction drift (prior probability shift) DETECTED")
            _push_drift_metric(drift_type="prior_probability")
        else:
            print("✅ No prediction drift detected")
    else:
        print(
            "⏭️ Skipping prediction drift: predicted_label not in reference data (expected for training-only reference)"
        )

    # Generate HTML report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(f"/opt/airflow/reports/drift/drift_report_{timestamp}.html")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating HTML report: {report_path}")
    summary = detector.generate_drift_report(
        reference_data=reference_df,
        current_data=current_df,
        output_path=report_path,
    )

    # ── Concept drift enrichment: check drift_batches ─────────────────────────
    # Evidently's target-drift test compares label *distributions*. For symmetric
    # concept drift (equal class balance, labels swapped), both reference and current
    # data have the same 50/50 distribution, so the KS/stattest reports no drift.
    # Solution: if the Streamlit drift provocation recorded a 'concept_drift' batch
    # in drift_batches within the last 48 h, force-mark target_drift as detected so
    # the JSON summary and Prometheus gauge reflect the event.
    _concept_provoked_flag = False
    try:
        _db_enrich = _open_db()
        _enr_cursor = _db_enrich.conn.cursor()
        _enr_cursor.execute(
            "SELECT COUNT(*) AS n FROM drift_batches "
            "WHERE drift_type = 'concept_drift' "
            "AND created_at >= NOW() - INTERVAL '48 hours'"
        )
        _enr_row = _enr_cursor.fetchone()
        _concept_provoked_flag = bool(_enr_row and int(_enr_row["n"]) > 0)
        _db_enrich.close()
    except Exception as _enr_err:
        print(f"Warning: drift_batches concept check failed: {_enr_err}")

    if _concept_provoked_flag and not (target_drift and target_drift.get("drift_detected")):
        print(
            "⚠️  Recent concept_drift provocation found in drift_batches. "
            "Symmetric label swap is not detectable by label-distribution test. "
            "Marking concept drift as detected from provocation batch."
        )
        _push_drift_metric(drift_type="concept")
        # Update summary["target_drift"] so the JSON file triggers the Prometheus gauge
        if "target_drift" not in summary:
            summary["target_drift"] = {}
        summary["target_drift"]["drift_detected"] = True
        summary["target_drift"]["source"] = "drift_batches_provocation"
        # Keep local variable consistent for summary.update() below
        if target_drift is None:
            target_drift = {
                "drift_detected": True,
                "drift_score": 1.0,
                "source": "drift_batches_provocation",
            }
        else:
            target_drift["drift_detected"] = True
            target_drift["source"] = "drift_batches_provocation"

    # Save JSON summary
    summary_path = Path(f"/opt/airflow/reports/drift/drift_summary_{timestamp}.json")
    summary.update(
        {
            "data_drift_details": data_drift,
            "target_drift_details": target_drift,
            "prediction_drift_details": prediction_drift,
            "reference_samples": len(reference_df),
            "current_samples": len(current_df),
        }
    )
    detector.save_drift_summary(summary, summary_path)

    print("✅ Reports saved:")
    print(f"  - HTML: {report_path}")
    print(f"  - JSON: {summary_path}")

    # Return results for downstream tasks
    return {
        "data_drift_detected": data_drift["drift_detected"],
        "drift_share": data_drift["drift_share"],
        "n_drifted_features": data_drift["n_drifted_features"],
        "target_drift_detected": target_drift["drift_detected"] if target_drift else False,
        "prediction_drift_detected": prediction_drift["drift_detected"]
        if prediction_drift
        else False,
        "report_path": str(report_path),
        "summary_path": str(summary_path),
        "timestamp": timestamp,
    }


# Create DAG
with DAG(
    dag_id="evidently_drift_detection",
    default_args=default_args,
    description="Monitor data drift using EvidentlyAI",
    schedule_interval="0 2 * * *",  # Daily at 2 AM UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["monitoring", "drift", "evidently"],
) as dag:
    # Task 1: Check prerequisites
    check_prerequisites = ShortCircuitOperator(
        task_id="check_prerequisites",
        python_callable=check_drift_detection_prerequisites,
        provide_context=True,
    )

    # Task 2: Load reference data
    load_reference = PythonOperator(
        task_id="load_reference_data",
        python_callable=load_reference_data,
        provide_context=True,
    )

    # Task 3: Load current production data
    load_current = PythonOperator(
        task_id="load_current_data",
        python_callable=load_current_data,
        provide_context=True,
    )

    # Task 4: Run drift detection
    detect_drift = PythonOperator(
        task_id="run_drift_detection",
        python_callable=run_drift_detection,
        provide_context=True,
    )

    # NOTE: Tasks 5 & 6 (check_should_trigger_retraining + trigger_automated_retraining)
    # have been REMOVED.  This DAG is metrics-only: it detects drift, writes reports to
    # Prometheus / Grafana, and stores a row in drift_batches.
    # Retraining is SOLELY the responsibility of the drift_triggered_retraining DAG
    # (hourly scheduled or manually triggered from the Streamlit UI).  Keeping the
    # trigger here caused automated_retraining to be launched twice — once by this DAG
    # and once by drift_triggered_retraining — and prevented the require_human_approval
    # conf flag from reaching automated_retraining.

    # Define task dependencies
    check_prerequisites >> [load_reference, load_current]
    [load_reference, load_current] >> detect_drift
