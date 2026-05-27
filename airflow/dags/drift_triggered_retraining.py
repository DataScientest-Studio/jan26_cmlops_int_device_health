"""
Airflow DAG for drift-triggered model retraining.

This DAG runs hourly to:
1. Query the PostgreSQL drift_batches table for recent drift events
2. Check if drift_batch count exceeds threshold in last 24h
3. Verify minimum labeled signals exist for retraining
4. Trigger retraining workflow if drift detected

The workflow reuses tasks from automated_retraining.py but triggers
on drift conditions rather than schedule.

Schedule: Hourly
Triggers: When drift batches in last 24h >= DRIFT_THRESHOLD (default: 1)
"""

import os
from datetime import datetime, timedelta

from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from airflow import DAG

try:
    from _dag_guards import require_cloud_mode
except ModuleNotFoundError:
    from airflow.dags._dag_guards import require_cloud_mode

_DAG_ID = "drift_triggered_retraining"

# Configuration — 1 drift batch in 24h is enough to trigger retraining
DRIFT_THRESHOLD = int(os.getenv("DRIFT_THRESHOLD", "1"))

try:
    from src.config import MIN_LABELED_SIGNALS
except Exception:
    MIN_LABELED_SIGNALS = 20


def _open_db():
    """Open a database connection using DATABASE_URL env var (PostgreSQL in cloud)."""
    import sys

    sys.path.insert(0, "/opt/airflow")
    from src.database.database import Database

    db_url = os.environ.get("DATABASE_URL", "")
    return (
        Database(db_url=db_url)
        if db_url.startswith("postgresql")
        else Database("/opt/airflow/data/database/mlops.db")
    )


def _count_recent_drift_batches(db, hours: int = 24) -> int:
    """Count drift batches created in the last N hours."""
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    cursor = db.conn.cursor()
    db_url = os.environ.get("DATABASE_URL", "")
    ph = "%s" if db_url.startswith("postgresql") else "?"
    cursor.execute(
        f"SELECT COUNT(*) AS n FROM drift_batches WHERE created_at >= {ph}",
        (cutoff,),
    )
    row = cursor.fetchone()
    if row is None:
        return 0
    try:
        return int(row["n"])
    except (TypeError, KeyError, IndexError):
        return int(row[0])


def check_drift_condition(**context) -> bool:
    """
    Check if drift conditions warrant retraining using the drift_batches DB table.

    Checks:
        1. Drift batches created in last 24h >= DRIFT_THRESHOLD (default: 1)
        2. Labeled signals in DB >= MIN_LABELED_SIGNALS (default: 20)

    Returns:
        True if retraining should be triggered, False otherwise
    """
    require_cloud_mode(_DAG_ID)
    print("🔍 Checking drift conditions via database...")

    db = _open_db()

    # Check 1: Drift batches in last 24 hours
    drift_count = _count_recent_drift_batches(db, hours=24)
    print(f"📊 Drift batches (24h): {drift_count} (threshold: {DRIFT_THRESHOLD})")

    # Check 2: Labeled signals available
    labeled_count = db.count_labeled_signals()
    print(f"📊 Labeled signals: {labeled_count} (minimum: {MIN_LABELED_SIGNALS})")

    # Decision logic
    drift_detected = drift_count >= DRIFT_THRESHOLD
    sufficient_labels = labeled_count >= MIN_LABELED_SIGNALS

    print(f"🎯 Drift detected: {drift_detected}")
    print(f"🎯 Sufficient labels: {sufficient_labels}")

    # Push metrics to XCom for downstream tasks
    context["task_instance"].xcom_push(key="drift_count", value=drift_count)
    context["task_instance"].xcom_push(key="label_count", value=labeled_count)

    should_trigger = drift_detected and sufficient_labels

    if should_trigger:
        print("✅ Drift threshold exceeded with sufficient labels — triggering retraining")
    elif drift_detected and not sufficient_labels:
        print(
            f"⚠️ Drift detected but only {labeled_count} labeled signals "
            f"(need {MIN_LABELED_SIGNALS}) — waiting for more labels"
        )
    else:
        print("✅ No retraining needed — no recent drift batches found")

    return should_trigger


def record_drift_trigger(**context) -> dict:
    """
    Record drift-triggered retraining event.

    Returns:
        Dict with trigger metadata
    """
    ti = context["task_instance"]
    drift_count = ti.xcom_pull(key="drift_count", task_ids="check_drift")
    label_count = ti.xcom_pull(key="label_count", task_ids="check_drift")

    trigger_info = {
        "trigger_time": datetime.now().isoformat(),
        "trigger_type": "drift",
        "drift_batches_24h": drift_count,
        "labeled_signals": label_count,
        "drift_threshold": DRIFT_THRESHOLD,
    }

    print(f"📝 Drift-triggered retraining initiated: {trigger_info}")
    return trigger_info


def notify_drift_retraining(**context) -> None:
    """Send notification that drift-triggered retraining was initiated."""
    ti = context["task_instance"]
    drift_count = ti.xcom_pull(key="drift_count", task_ids="check_drift")
    label_count = ti.xcom_pull(key="label_count", task_ids="check_drift")

    message = f"""
    🚨 Drift-Triggered Retraining Initiated

    Drift Batches (24h): {drift_count}
    Labeled Signals: {label_count}
    Drift Threshold: {DRIFT_THRESHOLD}

    The automated_retraining DAG has been triggered to retrain the model
    with the latest data.

    Monitor progress: http://localhost:8080/dags/automated_retraining
    """

    print(message)


def _notify_retraining_failure(context: dict) -> None:
    """
    DAG-level on_failure_callback: increment retraining_failures_total via the API.

    See automated_retraining.py for full rationale.  Failure reason is set to
    'drift_dag_failure' to distinguish from scheduled retraining failures.
    """
    import urllib.error
    import urllib.request

    api_url = os.environ.get("MLOPS_API_URL", "http://api:8000")
    reason = "drift_dag_failure"
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

    Called when the drift-triggered retraining DAG completes successfully.
    Uses reason='drift' to distinguish from scheduled retraining triggers.
    Best-effort: exceptions are printed, never re-raised.
    """
    import urllib.error
    import urllib.request

    api_url = os.environ.get("MLOPS_API_URL", "http://api:8000")
    # Increment retraining trigger counter
    try:
        req = urllib.request.Request(
            f"{api_url}/internal/metrics/retraining-trigger?reason=drift",
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5):
            pass
        print("✅ retraining_triggers_total incremented (reason=drift)")
    except (urllib.error.URLError, OSError) as exc:
        print(f"⚠️  Could not increment retraining trigger counter: {exc}")
    # Also increment drift_detections_total so Grafana drift panels reflect
    # manually triggered drift-retraining runs (DAG bypasses evidently_drift_detection).
    try:
        req = urllib.request.Request(
            f"{api_url}/internal/metrics/drift-detection?drift_type=data",
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5):
            pass
        print("✅ drift_detections_total incremented (drift_type=data)")
    except (urllib.error.URLError, OSError) as exc:
        print(f"⚠️  Could not increment drift detections counter: {exc}")


# DAG default arguments
default_args = {
    "owner": "mlops_team",
    "depends_on_past": False,
    "email": ["mlops-alerts@example.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": _notify_retraining_failure,
    "on_success_callback": _notify_retraining_trigger,
    "start_date": datetime(2026, 2, 18),
}

# DAG definition
with DAG(
    dag_id="drift_triggered_retraining",
    default_args=default_args,
    description="Trigger retraining when data drift exceeds threshold",
    schedule_interval="@hourly",  # Check every hour
    start_date=datetime(2026, 2, 18),
    catchup=False,
    tags=["mlops", "monitoring", "drift", "retraining"],
    doc_md=__doc__,
) as dag:
    # Task 1: Check drift conditions via DB
    check_drift = ShortCircuitOperator(
        task_id="check_drift",
        python_callable=check_drift_condition,
        provide_context=True,
        doc_md="""
        ### Check Drift Conditions

        Queries the drift_batches database table to determine if retraining is needed:
        - Drift batches created in last 24h >= DRIFT_THRESHOLD (default: 1)
        - Labeled signals in DB >= MIN_LABELED_SIGNALS (default: 20)

        Returns True to continue if both conditions are met.
        """,
    )

    # Task 2: Record trigger event
    record_trigger = PythonOperator(
        task_id="record_trigger",
        python_callable=record_drift_trigger,
        provide_context=True,
        doc_md="""
        ### Record Retraining Trigger

        Records the drift-triggered retraining event metadata for audit trail.
        """,
    )

    # Task 3: Trigger retraining DAG
    trigger_retraining = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="automated_retraining",
        wait_for_completion=False,  # Don't block - retraining takes time
        poke_interval=60,
        conf={
            "trigger_type": "drift",
            "triggered_by": "drift_triggered_retraining",
            "trigger_reason": "evidently_drift_detection",
            "require_human_approval": "{{ dag_run.conf.get('require_human_approval', False) }}",
        },
        doc_md="""
        ### Trigger Retraining Workflow

        Triggers the automated_retraining DAG which performs:
        1. Data validation
        2. Feature extraction
        3. Model training
        4. Evaluation and promotion
        """,
    )

    # Task 4: Send notification
    notify = PythonOperator(
        task_id="notify_team",
        python_callable=notify_drift_retraining,
        provide_context=True,
        doc_md="""
        ### Send Notification

        Notifies the ML team that drift-triggered retraining has been initiated.
        """,
    )

    # Task dependencies
    # If check_drift returns False, pipeline stops (ShortCircuitOperator behavior)
    check_drift >> record_trigger >> trigger_retraining >> notify
