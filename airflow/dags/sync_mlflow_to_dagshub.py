"""
Airflow DAG: sync_mlflow_to_dagshub

Pushes new experiments and runs from the local MLflow buffer container
(mlops_mlflow_buffer) to DagsHub MLflow incrementally.

Architecture context (Section 27 — Local-First MLflow):
  All live MLflow operations (training, promotion, Streamlit, API) use the
  local buffer as the primary server.  DagsHub is only written to by this
  DAG on a schedule (or on-demand manual trigger).

Sync direction:
  mlops_mlflow_buffer (local, http://mlflow_buffer:5000)
        → DagsHub (https://dagshub.com/<user>/<repo>.mlflow)

Incremental strategy:
  A state file (data/.mlflow_sync_state.json) records the timestamp of the
  last successful push.  Only runs completed after that timestamp are synced.
  Runs already present on DagsHub (identified by the mlflow_sync.source_run_id
  tag) are skipped to prevent duplicates.

Manual trigger:
  Use the Airflow UI or the Streamlit "Sync Buffer → DagsHub" button.

Scheduled trigger:
  Nightly at 02:30 UTC (30 minutes after the retraining DAG at 02:00).
  Schedule can be changed via the MLFLOW_SYNC_SCHEDULE env var.
"""

import os
from datetime import datetime, timedelta

from airflow.operators.python import PythonOperator

from airflow import DAG

_DAG_ID = "sync_mlflow_to_dagshub"


def _get_buffer_uri() -> str:
    """Return the MLflow buffer URI (container-internal)."""
    return os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_buffer:5000")


def _get_dagshub_uri() -> str:
    """Return the DagsHub MLflow URI from env."""
    uri = os.getenv("MLFLOW_DAGSHUB_URI", "")
    if uri:
        return uri
    user = os.getenv("DAGSHUB_USER", "")
    repo = os.getenv("DAGSHUB_REPO", "")
    if user and repo:
        return f"https://dagshub.com/{user}/{repo}.mlflow"
    raise ValueError("MLFLOW_DAGSHUB_URI or DAGSHUB_USER+DAGSHUB_REPO must be set in environment")


def check_dagshub_reachable() -> dict:
    """
    Pre-flight: verify DagsHub is reachable and not rate-limiting.

    Raises RuntimeError if DagsHub is unreachable so the DAG fails fast
    with a clear message rather than timing out during actual sync.

    Returns:
        Dict with connectivity status info.
    """
    import requests

    dagshub_uri = _get_dagshub_uri()
    user = os.getenv("MLFLOW_DAGSHUB_USERNAME") or os.getenv("DAGSHUB_USER", "")
    token = os.getenv("MLFLOW_DAGSHUB_PASSWORD") or os.getenv("DAGSHUB_TOKEN", "")

    probe_url = dagshub_uri.rstrip("/") + "/api/2.0/mlflow/experiments/search"
    print(f"🔍 Probing DagsHub at {probe_url}")

    try:
        resp = requests.post(
            probe_url,
            json={"max_results": 1},
            auth=(user, token) if user else None,
            timeout=15,
            verify=False,
        )
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(f"DagsHub unreachable: {exc}") from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("DagsHub probe timed out after 15 s") from exc

    if resp.status_code == 429:
        raise RuntimeError(
            "DagsHub is rate-limiting (HTTP 429). Wait and retry later, or reduce sync frequency."
        )
    if resp.status_code not in (200, 400):
        raise RuntimeError(
            f"DagsHub probe returned unexpected HTTP {resp.status_code}: {resp.text[:200]}"
        )

    result = {
        "dagshub_uri": dagshub_uri,
        "http_status": resp.status_code,
        "reachable": True,
    }
    print(f"✅ DagsHub reachable: {result}")
    return result


def sync_experiments_and_runs(sync_artifacts: bool = True, **context) -> dict:
    """
    Core sync task: push buffer runs to DagsHub (incremental).

    Reads all non-Default experiments from the buffer, finds runs not yet
    present on DagsHub, and uploads metrics, params, tags, and optionally
    artifacts (model .pkl files, evaluation plots).

    Returns:
        Dict with sync summary (experiments_synced, runs_synced, etc.)
    """
    import sys
    from pathlib import Path

    # Ensure src/ is on the path (Airflow worker may not have it)
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.training.mlflow_sync import push_to_dagshub

    buffer_uri = _get_buffer_uri()
    dagshub_uri = _get_dagshub_uri()
    user = os.getenv("MLFLOW_DAGSHUB_USERNAME") or os.getenv("DAGSHUB_USER", "")
    token = os.getenv("MLFLOW_DAGSHUB_PASSWORD") or os.getenv("DAGSHUB_TOKEN", "")

    # params injected via DAG trigger conf (if any)
    conf = (context.get("dag_run") or {}).conf or {}
    experiment_filter = conf.get("experiment_names")  # list[str] | None
    do_artifacts = conf.get("sync_artifacts", sync_artifacts)

    print(
        f"🔄 Syncing buffer ({buffer_uri}) → DagsHub ({dagshub_uri})\n"
        f"   Experiments filter: {experiment_filter or 'all'}\n"
        f"   Sync artifacts: {do_artifacts}"
    )

    summary = push_to_dagshub(
        local_mlflow_uri=buffer_uri,
        dagshub_uri=dagshub_uri,
        dagshub_user=user,
        dagshub_token=token,
        experiment_names=experiment_filter,
        sync_artifacts=bool(do_artifacts),
    )

    print(
        f"✅ Sync complete — {summary['runs_synced']} run(s) across "
        f"{summary['experiments_synced']} experiment(s)"
    )
    return summary


def notify_sync_result(**context) -> None:
    """Log sync result summary (extend to Slack/email as needed)."""
    ti = context["task_instance"]
    summary = ti.xcom_pull(task_ids="sync_runs") or {}
    dag_run = context["dag_run"]

    message = (
        f"\n📤 MLflow Buffer → DagsHub Sync Complete\n"
        f"   DAG run:         {dag_run.run_id}\n"
        f"   Experiments:     {summary.get('experiments_synced', '?')}\n"
        f"   Runs synced:     {summary.get('runs_synced', '?')}\n"
        f"   Source (buffer): {summary.get('source', '?')}\n"
        f"   Target (DagsHub):{summary.get('target', '?')}\n"
        f"   Timestamp:       {summary.get('timestamp_iso', '?')}\n"
    )
    print(message)


# ── DAG definition ────────────────────────────────────────────────────────────

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id=_DAG_ID,
    description=(
        "Incremental push: MLflow buffer container → DagsHub. "
        "Part of the local-first MLflow architecture (Section 27)."
    ),
    default_args=default_args,
    schedule_interval=os.getenv("MLFLOW_SYNC_SCHEDULE", "30 2 * * *"),  # 02:30 UTC nightly
    start_date=datetime(2026, 4, 29),
    catchup=False,
    # FU-2: schedule activation is controlled by MLFLOW_SYNC_PAUSED env var.
    # Default: paused=True (manual trigger only). Set MLFLOW_SYNC_PAUSED=false in
    # .env.cloud and restart the Airflow container to enable automatic nightly sync.
    is_paused_upon_creation=os.getenv("MLFLOW_SYNC_PAUSED", "true").lower() != "false",
    tags=["mlflow", "sync", "dagshub", "local-first"],
    max_active_runs=1,
    params={
        "sync_artifacts": True,
        "experiment_names": None,  # null = all non-Default experiments
    },
) as dag:
    check_task = PythonOperator(
        task_id="check_dagshub_reachable",
        python_callable=check_dagshub_reachable,
        doc_md="""
        ### Check DagsHub Reachability
        Probes the DagsHub MLflow API before sync starts.
        Fails fast with a clear error if DagsHub is down or rate-limiting.
        """,
    )

    sync_task = PythonOperator(
        task_id="sync_runs",
        python_callable=sync_experiments_and_runs,
        provide_context=True,
        doc_md="""
        ### Sync Buffer Runs to DagsHub
        Finds all runs in the buffer that are not yet on DagsHub
        (using ``mlflow_sync.source_run_id`` tag as the deduplication key)
        and uploads them incrementally.
        """,
    )

    notify_task = PythonOperator(
        task_id="notify_result",
        python_callable=notify_sync_result,
        provide_context=True,
        doc_md="""
        ### Notify Sync Result
        Logs sync summary.  Extend to send Slack/email notifications.
        """,
    )

    check_task >> sync_task >> notify_task
