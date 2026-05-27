"""
Airflow DAG for syncing production SQL database to DagsHub.

This DAG runs daily to:
1. Export metadata tables (predictions, features, devices, sparse_labels) to CSV
2. Export labeled signals (100%) to JSON files
3. Sample unlabeled signals (10%) for drift monitoring
4. Version and push to DagsHub via DVC

Directory structure (UUID prefix sharding for scalability):
    data/sync/
        predictions.csv
        features.csv
        devices.csv
        sparse_labels.csv
    data/raw_signals/
        {prefix1}/          # First 2 hex chars of device UUID (00-ff)
            {prefix2}/      # Next 2 hex chars of device UUID (00-ff)
                {device_id}/
                    {prediction_id}.json

Sharding Rationale:
    - 2-level UUID prefix sharding creates 256 × 256 = 65,536 possible shards
    - Prevents file system performance degradation (>10,000 files per directory)
    - Enables DVC parallel hash computation and efficient caching
    - Optimizes S3/DagsHub list operations (pagination limits)
    - Uniform distribution: UUIDs are random, ensuring balanced shards
    - Supports 10M+ devices without performance issues

Example:
    device_id = "550e8400-e29b-41d4-a716-446655440000"
    Output: data/raw_signals/55/0e/550e8400-.../123.json
"""

from __future__ import annotations

import os
import random
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from airflow.operators.python import PythonOperator

from airflow import DAG

try:
    from _dag_guards import require_cloud_mode
except ModuleNotFoundError:
    from airflow.dags._dag_guards import require_cloud_mode

_DAG_ID = "sync_production_data"


def _open_db():
    """Return a Database connected to PostgreSQL (Docker) or SQLite (fallback)."""
    from src.database.database import Database

    db_url = os.environ.get("DATABASE_URL", "")
    return (
        Database(db_url=db_url)
        if db_url.startswith("postgresql")
        else Database("/opt/airflow/data/database/mlops.db")
    )


def export_metadata_csvs() -> dict[str, int]:
    """
    Export predictions, features, devices, sparse_labels to CSV.

    Returns:
        Dict with counts of exported rows per table
    """
    require_cloud_mode(_DAG_ID)

    db = _open_db()

    counts = {
        "predictions": db.export_predictions_to_csv("data/sync/predictions.csv"),
        "features": db.export_features_to_csv("data/sync/features.csv"),
        "devices": db.export_devices_to_csv("data/sync/devices.csv"),
        "sparse_labels": db.export_sparse_labels_to_csv("data/sync/sparse_labels.csv"),
    }

    print(f"✅ Exported metadata: {counts}")
    return counts


def export_labeled_signals() -> int:
    """
    Export all signals with ground_truth_label IS NOT NULL.

    Returns:
        Number of labeled signals exported
    """
    require_cloud_mode(_DAG_ID)

    db = _open_db()
    signal_ids = db.get_labeled_signal_ids()

    if not signal_ids:
        print("⚠️ No labeled signals found")
        return 0

    exported = db.export_signals_to_json("data/raw_signals/", signal_ids)
    print(f"✅ Exported {exported} labeled signals")
    return exported


def sample_unlabeled_signals() -> int:
    """
    Export random 10% of unlabeled signals for drift monitoring.

    Returns:
        Number of unlabeled signals exported
    """
    require_cloud_mode(_DAG_ID)

    db = _open_db()
    unlabeled_ids = db.get_unlabeled_signal_ids()

    if not unlabeled_ids:
        print("⚠️ No unlabeled signals found")
        return 0

    # Sample 10% (minimum 1 signal if any exist)
    sample_size = max(1, len(unlabeled_ids) // 10)
    sample_ids = random.sample(unlabeled_ids, k=sample_size)

    exported = db.export_signals_to_json("data/raw_signals/", sample_ids)
    print(f"✅ Sampled {exported}/{len(unlabeled_ids)} unlabeled signals (10%)")
    return exported


def dvc_push_to_dagshub():
    """
    Version and push data to DagsHub via DVC (cloud mode only).

    Workflow:
      1. ``dvc add`` to track the exported directories (creates/updates .dvc pointers)
      2. ``dvc push`` to upload the actual data to DagsHub S3 storage
      3. ``git add`` + ``git commit`` to record the updated .dvc pointer files locally

    Note on git push:
      The container does NOT push to the git remote (GitHub).  Pushing from
      inside a container requires credentials that are intentionally not stored
      in the container for security reasons.  The .dvc pointer-file commit is
      kept local; changes can be pushed to GitHub via the normal development
      workflow (``git push`` from the host machine).

    Git identity is taken from environment variables ``GIT_USER_NAME`` and
    ``GIT_USER_EMAIL`` (set in .env.cloud and passed via docker-compose).
    Defaults to "Airflow Bot <airflow@mlops.local>" if not set.

    Git is available in the custom Airflow image
    (``docker/airflow_mlops.Dockerfile`` installs it via apt-get).
    """
    require_cloud_mode(_DAG_ID)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # All DVC/git operations must run from the DVC repo root inside the container.
    # The volume mounts in docker-compose.yml place .dvc/, dvc.yaml, dvc.lock, and
    # params.yaml under /opt/airflow, making it the effective DVC working directory.
    DVC_ROOT = Path("/opt/airflow")

    # Resolve data dirs relative to DVC root
    sync_dir = DVC_ROOT / "data/sync"
    signals_dir = DVC_ROOT / "data/raw_signals"

    if not sync_dir.exists():
        print("⚠️ data/sync/ not found, skipping DVC add")
        return

    def _run(cmd):
        """Run a subprocess command from DVC_ROOT; log warning on failure."""
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(DVC_ROOT))
        if result.returncode != 0:
            print(
                f"⚠️ Command {cmd} exited with code {result.returncode}. "
                f"stdout={result.stdout.strip()!r}  stderr={result.stderr.strip()!r}. "
                "Continuing (non-fatal: data exports are preserved even if DVC/git step fails)."
            )
            return False
        return True

    # Configure git identity (required for git commit inside the container)
    git_email = os.environ.get("GIT_USER_EMAIL", "airflow@mlops.local")
    git_name = os.environ.get("GIT_USER_NAME", "Airflow Bot")
    _run(["git", "config", "--global", "user.email", git_email])
    _run(["git", "config", "--global", "user.name", git_name])
    print(f"🔧 Git identity: {git_name} <{git_email}>")

    # DVC add (track directories — paths are relative to DVC_ROOT)
    print("🔄 Adding data to DVC...")
    if not _run(["dvc", "add", "data/sync/"]):
        print("⚠️ DVC add failed — skipping remaining DVC/git steps.")
        return

    if signals_dir.exists():
        _run(["dvc", "add", "data/raw_signals/"])

    # DVC push to DagsHub (the important part — uploads actual data files)
    print("🔄 Pushing to DagsHub...")
    if not _run(["dvc", "push"]):
        print("⚠️ DVC push failed — skipping git commit.")
        return

    # Git commit .dvc pointer files locally (does NOT push to GitHub remote)
    print("🔄 Committing DVC metadata locally...")
    _run(["git", "add", "data/sync.dvc", ".gitignore"])

    if signals_dir.exists():
        _run(["git", "add", "data/raw_signals.dvc"])

    _run(["git", "commit", "-m", f"[airflow] Data sync: {timestamp}"])
    # Note: intentionally no git push — container has no GitHub credentials.
    # Push .dvc pointer changes to GitHub from the host machine after review.

    print(f"✅ Synced data to DagsHub at {timestamp}")


# ========================================
# DAG Definition
# ========================================

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="sync_production_data",
    description="Export SQL database to DagsHub for versioning and drift analysis",
    default_args=default_args,
    schedule_interval="@daily",  # Run at midnight UTC
    start_date=datetime(2026, 4, 1),
    catchup=False,  # Don't backfill historical runs
    tags=["data-sync", "dagshub", "production"],
    max_active_runs=1,  # Prevent concurrent runs
) as dag:
    # Task 1: Export metadata tables to CSV
    export_metadata_task = PythonOperator(
        task_id="export_metadata",
        python_callable=export_metadata_csvs,
        doc_md="""
        ### Export Metadata to CSV
        Exports predictions, features, devices, and sparse_labels tables.
        Output: `data/sync/*.csv`
        """,
    )

    # Task 2: Export labeled signals to JSON
    export_labeled_task = PythonOperator(
        task_id="export_labeled",
        python_callable=export_labeled_signals,
        doc_md="""
        ### Export Labeled Signals
        Exports all signals with ground_truth_label IS NOT NULL.
        Output: `data/raw_signals/{device_id}/{prediction_id}.json`
        """,
    )

    # Task 3: Sample unlabeled signals (10%)
    sample_unlabeled_task = PythonOperator(
        task_id="sample_unlabeled",
        python_callable=sample_unlabeled_signals,
        doc_md="""
        ### Sample Unlabeled Signals
        Exports random 10% of unlabeled signals for drift analysis.
        Output: `data/raw_signals/{device_id}/{prediction_id}.json`
        """,
    )

    # Task 4: DVC push to DagsHub
    dvc_push_task = PythonOperator(
        task_id="dvc_push",
        python_callable=dvc_push_to_dagshub,
        doc_md="""
        ### Push to DagsHub
        Versions data with DVC and pushes to DagsHub remote.
        Commits .dvc files to Git.
        """,
    )

    # Task dependencies: Parallel exports, then push
    [export_metadata_task, export_labeled_task, sample_unlabeled_task] >> dvc_push_task
