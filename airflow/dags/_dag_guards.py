"""Shared guard helpers for Airflow DAGs.

Import with the try/except pattern to support both Airflow's DAG loader
(which adds the dags folder to sys.path) and regular Python package imports
(used by tests):

    try:
        from _dag_guards import require_cloud_mode
    except ModuleNotFoundError:
        from airflow.dags._dag_guards import require_cloud_mode
"""

from __future__ import annotations

import os
from pathlib import Path


def _deployment_mode() -> str:
    """Return 'local', 'cloud', or 'k8s' based on environment / .current_mode."""
    mode = os.environ.get("DEPLOYMENT_MODE", "").lower()
    if mode in ("local", "cloud", "k8s"):
        return mode
    mode_file = Path(".current_mode")
    if mode_file.exists():
        return mode_file.read_text().strip().lower()
    return "local"


def require_cloud_mode(dag_id: str) -> None:
    """Raise RuntimeError if not running in cloud or k8s mode.

    Call this at the top of every task callable that must never execute
    inside the local sandbox.
    """
    mode = _deployment_mode()
    if mode not in ("cloud", "k8s"):
        raise RuntimeError(
            f"DAG '{dag_id}' is cloud/k8s-mode only (current mode: {mode}). "
            "Airflow DAGs are disabled in local sandbox mode."
        )
