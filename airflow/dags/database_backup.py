"""
Airflow DAG: Scheduled PostgreSQL database backup for MLOps device-health.

Schedule  : Daily at 02:00 UTC (configurable via Airflow Variable ``backup_schedule``)
Owner     : mlops-infra
Retention : Keep the 7 newest backups (configurable via Variable ``backup_keep_n``)

Tasks
-----
backup_database      – runs pg_dump via backup_postgres()
cleanup_old_backups  – removes stale backups, keeping the last N files
log_backup_result    – prints stats to task log for observability

Environment variables required in the Airflow worker:
  DATABASE_URL       – PostgreSQL connection URL
  BACKUP_DIR         – (optional) override default backup directory
                        default: data/backups/  (relative to AIRFLOW_HOME)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow.models import Variable
from airflow.operators.python import PythonOperator

from airflow import DAG

try:
    from _dag_guards import require_cloud_mode
except ModuleNotFoundError:
    from airflow.dags._dag_guards import require_cloud_mode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DAG configuration
# ---------------------------------------------------------------------------

_DEFAULT_ARGS = {
    "owner": "mlops-infra",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

_SCHEDULE = Variable.get("backup_schedule", default_var="0 2 * * *")
_KEEP_N = int(Variable.get("backup_keep_n", default_var="7"))

# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------


def _get_backup_dir() -> Path:
    base = os.environ.get("BACKUP_DIR", "data/backups")
    return Path(base).resolve()


def task_backup_database(**context: object) -> str:
    """
    Run pg_dump and push the backup file path to XCom.
    """
    require_cloud_mode("database_backup")
    from src.database.backup import backup_postgres, get_backup_filename

    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url.startswith("postgresql"):
        logger.warning("DATABASE_URL is not a PostgreSQL URL (%r); skipping backup.", db_url)
        return "skipped"

    backup_dir = _get_backup_dir()
    filename = get_backup_filename()
    output_path = backup_dir / filename

    backup_path = backup_postgres(db_url, output_path)
    logger.info("Backup created: %s", backup_path)

    # push path string to XCom so downstream tasks can read it
    context["ti"].xcom_push(key="backup_path", value=str(backup_path))  # type: ignore[index]
    return str(backup_path)


def task_cleanup_old_backups(**context: object) -> int:
    """
    Remove stale backups, keeping the _KEEP_N most recent files.
    """
    from src.database.backup import cleanup_old_backups

    backup_dir = _get_backup_dir()
    deleted = cleanup_old_backups(backup_dir, keep_n=_KEEP_N)
    logger.info("Cleaned up %d old backup(s); kept last %d.", deleted, _KEEP_N)
    return deleted


def task_log_backup_result(**context: object) -> None:
    """
    Pull backup path from XCom and log file size to the task log.
    """
    backup_path_str: str | None = context["ti"].xcom_pull(  # type: ignore[index]
        task_ids="backup_database", key="backup_path"
    )
    if backup_path_str and backup_path_str != "skipped":
        p = Path(backup_path_str)
        size_mb = p.stat().st_size / 1_048_576 if p.exists() else 0
        logger.info("Backup stats: file=%s  size=%.2f MB", p.name, size_mb)
    else:
        logger.info("No backup was created (DATABASE_URL not set or not PostgreSQL).")


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="database_backup",
    description="Daily PostgreSQL database backup with rotation",
    default_args=_DEFAULT_ARGS,
    schedule_interval=_SCHEDULE,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["database", "backup", "mlops"],
) as dag:
    backup_task = PythonOperator(
        task_id="backup_database",
        python_callable=task_backup_database,
        provide_context=True,
    )

    cleanup_task = PythonOperator(
        task_id="cleanup_old_backups",
        python_callable=task_cleanup_old_backups,
        provide_context=True,
    )

    log_task = PythonOperator(
        task_id="log_backup_result",
        python_callable=task_log_backup_result,
        provide_context=True,
    )

    backup_task >> cleanup_task >> log_task
