"""Use Cases — backward-compatible shim.

All logic now lives in ``src.ui.views.use_cases_pkg``.
This file re-exports the public ``render()`` function and private symbols
that existing tests and other code reference using their **original** names
(single underscore prefix).

Previously, USE_CASES / _detect_deployment_mode / _mlflow_tracking_uri /
_run_command lived in the now-deleted use_cases_legacy.py module.
They are defined here so any remaining references keep working.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from src.ui.components.docker_utils import get_host
from src.ui.views.use_cases_pkg import render  # noqa: F401
from src.ui.views.use_cases_pkg._common import MODEL_NAME as _MODEL_NAME  # noqa: F401
from src.ui.views.use_cases_pkg._common import SECTION_CSS as _SECTION_CSS  # noqa: F401
from src.ui.views.use_cases_pkg._common import detect_mode as _detect_mode  # noqa: F401
from src.ui.views.use_cases_pkg._common import (
    fetch_champion_info as _fetch_champion_info,  # noqa: F401
)
from src.ui.views.use_cases_pkg.drift_provocation import DRIFT_TYPES as _DRIFT_TYPES  # noqa: F401
from src.ui.views.use_cases_pkg.drift_provocation import (
    generate_batch as _generate_batch,  # noqa: F401
)
from src.ui.views.use_cases_pkg.drift_provocation import ks_tests as _ks_tests  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# USE_CASES list (kept here for backward-compat with tests_archive)
# ---------------------------------------------------------------------------

USE_CASES: list[dict] = [
    {
        "id": 1,
        "title": "Data Drift Detection",
        "icon": "📉",
        "tools": "EvidentlyAI, Grafana",
        "description": "Sends 300 drifted signals to the API, runs EvidentlyAI drift detection.",
        "command": "python scripts/simulate_drift.py data-drift --n-samples 300 --send-to-api",
    },
    {
        "id": 2,
        "title": "Concept Drift Simulation",
        "icon": "🔀",
        "tools": "Prometheus, Grafana",
        "description": "Intermediate parameter ranges blur the healthy/unhealthy decision boundary.",
        "command": "python scripts/simulate_drift.py concept-drift --n-samples 300 --send-to-api",
    },
    {
        "id": 3,
        "title": "Sparse Label Audit",
        "icon": "🏷️",
        "tools": "SQL, Airflow",
        "description": "Inject 50 ground truth labels into the predictions database.",
        "command": "python scripts/inject_sparse_labels.py inject --source data/raw/dataset_baseline_test.json --n-labels 50",
    },
    {
        "id": 4,
        "title": "Automated Retraining Trigger",
        "icon": "🔄",
        "tools": "Airflow, DVC, MLflow",
        "description": "Drift + sparse label audit trigger retraining DAG.",
        "command": "python scripts/trigger_retraining.py --force",
    },
    {
        "id": 5,
        "title": "Champion/Challenger Promotion",
        "icon": "🏆",
        "tools": "MLflow Registry",
        "description": "Self-contained champion/challenger demo.",
        "command": "python scripts/demo_champion_challenger.py --scenario promotion --model-name device_health_classifier",
    },
    {
        "id": 6,
        "title": "A/B Testing (Canary Deployment)",
        "icon": "🔬",
        "tools": "FastAPI, Nginx, Prometheus",
        "description": "Simulates Nginx canary routing with side-by-side model comparison.",
        "command": "python scripts/run_ab_test.py --n-signals 200 --canary-fraction 0.25",
    },
    {
        "id": 7,
        "title": "Automated Rollback (Smoke Test Failure)",
        "icon": "⏪",
        "tools": "GitHub Actions",
        "description": "Deploy model that fails smoke test → previous model retained.",
        "command": "python scripts/smoke_test_model.py --model models/bootstrap_model.pkl",
    },
    {
        "id": 8,
        "title": "Data Quality Gate Enforcement",
        "icon": "🛡️",
        "tools": "FastAPI Pydantic",
        "description": "Submit invalid signals — API returns 422/400 with descriptive messages.",
        "command": "python scripts/run_quality_gate_tests.py",
    },
    {
        "id": 9,
        "title": "Prediction Distribution Monitoring",
        "icon": "📊",
        "tools": "Prometheus, Grafana",
        "description": "Tracks healthy/unhealthy prediction distribution over time.",
        "command": "python scripts/simulate_drift.py gradual --n-samples 1000 --send-to-api",
    },
    {
        "id": 10,
        "title": "Feature Distribution Drift",
        "icon": "📈",
        "tools": "EvidentlyAI, Grafana",
        "description": "EvidentlyAI DriftDetector computes statistical distance per feature.",
        "command": "python scripts/detect_drift.py --reference-json data/raw/dataset_baseline_test.json --output-dir reports/drift --min-samples 1",
    },
    {
        "id": 11,
        "title": "Model Lineage & Reproducibility",
        "icon": "🔗",
        "tools": "DVC, MLflow, Git",
        "description": "Reproduce a model from any point using DVC checkout + MLflow run ID.",
        "command": "python scripts/show_model_lineage.py",
    },
    {
        "id": 12,
        "title": "Confidence Calibration Monitoring",
        "icon": "🎯",
        "tools": "Prometheus, Grafana",
        "description": "Queries Prometheus for prediction_confidence histogram (P50/P95).",
        "command": "python scripts/check_confidence_metrics.py",
    },
    {
        "id": 13,
        "title": "Batch Prediction Pipeline",
        "icon": "📦",
        "tools": "Airflow, DVC, SQL",
        "description": "Airflow DAG exports signals, versions with DVC, pushes to DagsHub.",
        "command": "python scripts/sync_production_data.py",
    },
    {
        "id": 14,
        "title": "API Performance Degradation",
        "icon": "⚡",
        "tools": "Prometheus, Grafana, Nginx",
        "description": "60 concurrent requests push p95 latency above alert threshold.",
        "command": "python scripts/run_performance_test.py --degrade --workers 30 --requests 60",
    },
    {
        "id": 15,
        "title": "Security – Malicious Input Injection",
        "icon": "🔒",
        "tools": "Pydantic, Nginx",
        "description": "Submit SQL injection, oversized payloads — Pydantic/Nginx reject them.",
        "command": "python scripts/test_api_manual.py",
    },
    {
        "id": 16,
        "title": "PostgreSQL Backup & Recovery",
        "icon": "🗄️",
        "tools": "PostgreSQL, Airflow",
        "description": "Airflow database_backup DAG runs daily pg_dump; recovery via pg_restore.",
        "command": "python scripts/run_pg_backup.py",
    },
    {
        "id": 17,
        "title": "Database Schema Migration (SQLite → PostgreSQL)",
        "icon": "🛠️",
        "tools": "PostgreSQL, SQLite",
        "description": "Dual-mode Database class runs identically on SQLite and PostgreSQL.",
        "command": "python scripts/demo_schema_migration.py",
    },
    {
        "id": 18,
        "title": "API Down Simulation",
        "icon": "🔴",
        "tools": "Prometheus, Grafana, Docker",
        "description": "Pauses mlops_api container for 150 s — MLOps API Down alert fires.",
        "command": "python scripts/simulate_api_down.py --down-seconds 150",
    },
    {
        "id": 19,
        "title": "Prediction Distribution Skew",
        "icon": "📊",
        "tools": "Prometheus, Grafana",
        "description": "2000 heavily drifted signals → PredictionDistributionSkew alert fires.",
        "command": "python scripts/simulate_prediction_skew.py --n-samples 2000",
    },
    {
        "id": 20,
        "title": "Prediction Drift Detection",
        "icon": "🔮",
        "tools": "EvidentlyAI, Prometheus",
        "description": "TargetDriftPreset compares prediction distributions → gauge set.",
        "command": "python scripts/simulate_prediction_drift.py --n-reference 200 --n-current 400",
    },
]


# ---------------------------------------------------------------------------
# Helpers (previously in use_cases_legacy.py)
# ---------------------------------------------------------------------------


def _detect_deployment_mode() -> str:
    """Return the current deployment mode ('local', 'cloud', 'k8s', or 'unknown')."""
    mode_file = PROJECT_ROOT / ".current_mode"
    if mode_file.exists():
        mode = mode_file.read_text().strip()
        if mode in ("local", "cloud", "k8s"):
            return mode
    env_mode = os.environ.get("DEPLOYMENT_MODE", "").strip()
    if env_mode in ("local", "cloud", "k8s"):
        return env_mode
    return "local"


def _mlflow_tracking_uri(mode: str) -> str:
    """Return the correct MLflow tracking URI for ``mode``."""
    if mode == "cloud":
        user = os.environ.get("DAGSHUB_USER", "")
        repo = os.environ.get("DAGSHUB_REPO", "")
        if user and repo:
            return f"https://dagshub.com/{user}/{repo}.mlflow"
        return os.environ.get("MLFLOW_TRACKING_URI") or f"http://{get_host()}:5001"
    return f"http://{get_host()}:5001"


def _run_command(cmd: str) -> tuple[str, int]:
    """Run a shell command from project root with the correct deployment env."""
    try:
        mode = _detect_deployment_mode()
        _py = f'"{sys.executable}"'
        resolved_cmd = re.sub(r"\bpython3?\b", _py, cmd)
        env: dict[str, str] = {
            **{k: v for k, v in os.environ.items() if isinstance(v, str)},
            "MLFLOW_HTTP_REQUEST_TIMEOUT": "10",
            "MLFLOW_TRACKING_URI": _mlflow_tracking_uri(mode),
        }
        if mode == "cloud":
            if not env.get("MLFLOW_TRACKING_USERNAME"):
                env["MLFLOW_TRACKING_USERNAME"] = env.get("DAGSHUB_USER", "")
            if not env.get("MLFLOW_TRACKING_PASSWORD"):
                env["MLFLOW_TRACKING_PASSWORD"] = env.get("DAGSHUB_TOKEN", "")
        if "DATABASE_URL" not in env:
            _pg_host = env.get("POSTGRES_HOST") or env.get("DB_HOST") or "localhost"
            if _pg_host in ("postgres", "db"):
                _pg_host = "localhost"
            _pg_port = env.get("POSTGRES_PORT") or env.get("DB_PORT") or "5432"
            _pg_user = env.get("POSTGRES_USER") or env.get("DB_USER") or "mlops_user"
            _pg_pass = (
                env.get("POSTGRES_PASSWORD") or env.get("DB_PASSWORD") or "local_dev_password"
            )
            _pg_db = env.get("POSTGRES_DB") or env.get("DB_NAME") or "mlops_db"
            env["DATABASE_URL"] = (
                f"postgresql://{_pg_user}:{_pg_pass}@{_pg_host}:{_pg_port}/{_pg_db}"
            )
        result = subprocess.run(
            resolved_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(PROJECT_ROOT),
            env={**env, "PYTHONUNBUFFERED": "1"},
        )
        output = result.stdout
        if result.stderr:
            output += "\n--- stderr ---\n" + result.stderr
        return output.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "Command timed out after 300 seconds.", 1
    except Exception as e:
        return str(e), 1
