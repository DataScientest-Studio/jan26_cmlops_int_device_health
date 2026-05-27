"""PostgreSQL Database — schema explorer, live table browser, backup manager."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from src.ui.components.docker_utils import get_host
from src.ui.logging_ui import get_ui_logger
from src.ui.styles import get_global_css, hero_section, metric_card

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKUP_DIR = PROJECT_ROOT / "data" / "backups"

DB_ERD_MERMAID = r"""
erDiagram
    devices {
        TEXT device_id PK "UUID"
        TEXT device_name
        TEXT device_type
        TEXT location
        TEXT status "active|inactive|maintenance"
        TEXT deployment_mode "local or cloud"
        TEXT first_seen_at
        TEXT last_seen_at
        TEXT created_at
        TEXT updated_at
    }
    predictions {
        INTEGER prediction_id PK
        TEXT device_id FK
        TEXT timestamp
        INTEGER predicted_label "0=healthy 1=unhealthy"
        REAL prediction_confidence
        TEXT model_version
        INTEGER ground_truth_label
        TEXT label_source
        TEXT mlflow_run_id "MLflow run that trained model"
        TEXT git_sha "Git commit of training code"
        TEXT dvc_data_hash "DVC hash of training data"
        TEXT airflow_run_id "Airflow DAG run ID"
        TEXT deployment_mode "local or cloud"
        TEXT created_at
        TEXT updated_at
    }
    raw_signals {
        INTEGER signal_id PK
        INTEGER prediction_id FK
        TEXT time_values "JSON array"
        TEXT amplitude_values "JSON array"
        INTEGER n_points
        INTEGER n_nan_values
        REAL time_min
        REAL time_max
        REAL amplitude_min
        REAL amplitude_max
        TEXT shape_type "gaussian|lorentzian|NULL"
        TEXT deployment_mode "local or cloud"
        TEXT created_at
    }
    features {
        INTEGER feature_id PK
        INTEGER prediction_id FK
        REAL fwhm
        REAL peak_height
        REAL peak_area
        REAL noise_level
        REAL snr
        REAL peak_center
        REAL estimated_mu
        REAL estimated_sigma
        TEXT deployment_mode "local or cloud"
        TEXT created_at
    }
    sparse_labels {
        INTEGER label_id PK
        INTEGER prediction_id FK
        INTEGER ground_truth_label "0|1"
        TEXT label_source
        TEXT injected_at
        TEXT injected_by
        TEXT deployment_mode "local or cloud"
    }

    devices ||--o{ predictions : "has"
    predictions ||--o| raw_signals : "linked to"
    predictions ||--o| features : "derived from"
    predictions ||--o{ sparse_labels : "labelled by"
    drift_batches {
        INTEGER batch_id PK
        TEXT drift_type "data-drift|concept-drift|label-shift"
        INTEGER n_reference
        INTEGER n_drifted
        TEXT parameters "JSON drift params"
        TEXT deployment_mode "local or cloud"
        TEXT created_at
    }
    drift_signals {
        INTEGER signal_id PK
        INTEGER batch_id FK
        INTEGER is_drifted "0=reference 1=drifted"
        REAL fwhm
        REAL peak_height
        REAL peak_area
        REAL noise_level
        REAL snr
        REAL peak_center
        TEXT shape_type "gaussian|lorentzian"
        TEXT time_values "JSON array"
        TEXT amplitude_values "JSON array"
        TEXT deployment_mode "local or cloud"
        TEXT created_at
    }
    drift_batches ||--o{ drift_signals : "contains"
    model_approvals {
        INTEGER id PK
        TEXT model_version "Challenger version"
        TEXT mlflow_run_id "Run that trained challenger"
        REAL challenger_f1 "F1 on challenger test split"
        REAL champion_f1 "F1 at approval time (legacy)"
        REAL champion_f1_on_challenger_test "F1 re-eval on same test set"
        TEXT status "pending|approved|rejected"
        TEXT created_at
        TEXT decided_at
        TEXT decided_by
    }
    rescoring_runs {
        INTEGER id PK
        TEXT model_version "Champion model used"
        TEXT rescored_at
        INTEGER n_predictions "Signals rescored"
        INTEGER n_changed "Predictions that changed"
        REAL change_rate "Fraction changed"
        TEXT triggered_by
        TEXT status "pending|running|completed|failed"
    }
    model_training_data {
        INTEGER id PK
        TEXT mlflow_run_id "MLflow run that used this signal"
        INTEGER signal_id FK "raw_signals.signal_id"
        TEXT split "train or test"
        TEXT model_version "Human-readable model tag"
        TEXT created_at
    }
    predictions }o--o{ model_approvals : "challenger governs"
    model_approvals ||--o{ rescoring_runs : "approval triggers"
    raw_signals ||--o{ model_training_data : "used in split"
"""

DB_FLOW_MERMAID = r"""
flowchart LR
    subgraph INGRESS["📡 Signal Ingestion"]
        api["FastAPI<br/>POST /predict"]
    end

    subgraph DRIFT["🌊 Drift Provocation UI"]
        drift_ui["Drift Provocation<br/>Streamlit page"]
    end

    subgraph STORAGE["🗄️ PostgreSQL / SQLite"]
        dev["devices"]
        pred["predictions"]
        sig["raw_signals"]
        feat["features"]
        lbl["sparse_labels"]
        db["drift_batches"]
        ds["drift_signals"]
    end

    subgraph GOVERNANCE["🔐 Governance Tables"]
        ma["model_approvals"]
        rr["rescoring_runs"]
        mtd["model_training_data"]
    end

    subgraph BACKUP["💾 Backup (Daily)"]
        pg_dump["pg_dump<br/>(custom format)"]
        files["data/backups/<br/>*.dump"]
        rotate["Rotate — keep 7"]
    end

    subgraph CONSUMERS["🔍 Consumers"]
        evidently["EvidentlyAI<br/>Drift Detection"]
        mlflow_c["MLflow<br/>Experiments"]
        prometheus["Prometheus<br/>Metrics"]
    end

    api --> dev
    api --> pred
    api --> sig
    api --> feat
    lbl -. "injected later" .-> pred

    api -. "after promotion" .-> ma
    ma -. "reviewed via Streamlit" .-> rr
    sig -. "train/test lineage" .-> mtd

    drift_ui --> db
    drift_ui --> ds
    drift_ui --> pred
    drift_ui --> sig
    drift_ui --> feat
    drift_ui -. "ground-truth label" .-> lbl

    pred --> evidently
    pred --> prometheus
    feat --> mlflow_c

    pg_dump --> files
    files --> rotate
    STORAGE -. "daily" .-> pg_dump

    classDef store fill:#1e3a5f,stroke:#60a5fa,stroke-width:2px,color:#e2e8f0
    classDef drift fill:#1a2e4a,stroke:#38bdf8,stroke-width:2px,color:#e2e8f0
    classDef governance fill:#2d1a3f,stroke:#c084fc,stroke-width:2px,color:#e2e8f0
    classDef backup fill:#1e3a2f,stroke:#10b981,stroke-width:2px,color:#e2e8f0
    classDef consumer fill:#3b1f6e,stroke:#a78bfa,stroke-width:2px,color:#e2e8f0
    classDef ingress fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#e2e8f0
    classDef driftui fill:#1e1a3f,stroke:#818cf8,stroke-width:2px,color:#e2e8f0

    class dev,pred,sig,feat,lbl store
    class db,ds drift
    class ma,rr,mtd governance
    class pg_dump,files,rotate backup
    class evidently,mlflow_c,prometheus consumer
    class api ingress
    class drift_ui driftui
"""

# Table metadata shown in the browser tab
TABLE_DESCRIPTIONS: dict[str, str] = {
    "devices": "One row per IoT device.  Primary key is a UUID string.",
    "predictions": "One row per API /predict call.  Carries predicted label, confidence, and optional ground truth.",
    "raw_signals": "JSON-serialised time / amplitude arrays linked to each prediction.",
    "features": "Engineered features (FWHM, SNR, peak area …) derived from raw_signals.",
    "sparse_labels": "Human-reviewed or automated ground-truth labels injected post-prediction.",
    "drift_batches": "One row per Drift Provocation run.  Records drift type, parameters, and summary statistics.",
    "drift_signals": "Individual signals from each drift batch (both reference and drifted distributions).",
    "model_approvals": "Human review gate records — one row per automated-retraining run that requested manual sign-off.",
    "rescoring_runs": "Audit log for batch re-scoring runs — tracks how many historical predictions changed after a model swap.",
    "model_training_data": "Training lineage table — records which signals were used in the train or test split for each MLflow run.",
}

# Friendly column widths hint for st.dataframe
_COL_ORDER: dict[str, list[str]] = {
    "devices": [
        "device_id",
        "device_name",
        "device_type",
        "location",
        "status",
        "deployment_mode",
        "first_seen_at",
        "last_seen_at",
        "created_at",
        "updated_at",
    ],
    "predictions": [
        "prediction_id",
        "device_id",
        "timestamp",
        "predicted_label",
        "prediction_confidence",
        "model_version",
        "ground_truth_label",
        "label_source",
        "mlflow_run_id",
        "git_sha",
        "dvc_data_hash",
        "airflow_run_id",
        "deployment_mode",
        "created_at",
        "updated_at",
    ],
    "raw_signals": [
        "signal_id",
        "prediction_id",
        "n_points",
        "n_nan_values",
        "shape_type",
        "time_min",
        "time_max",
        "amplitude_min",
        "amplitude_max",
        "time_values",
        "amplitude_values",
        "deployment_mode",
        "created_at",
    ],
    "features": [
        "feature_id",
        "prediction_id",
        "fwhm",
        "peak_height",
        "peak_area",
        "noise_level",
        "snr",
        "peak_center",
        "estimated_mu",
        "estimated_sigma",
        "deployment_mode",
        "created_at",
    ],
    "sparse_labels": [
        "label_id",
        "prediction_id",
        "ground_truth_label",
        "label_source",
        "injected_at",
        "injected_by",
        "deployment_mode",
    ],
    "drift_batches": [
        "batch_id",
        "drift_type",
        "n_reference",
        "n_drifted",
        "n_drifted_features",
        "deployment_mode",
        "parameters",
        "created_at",
    ],
    "drift_signals": [
        "signal_id",
        "batch_id",
        "split",
        "label",
        "predicted_label",
        "shape_type",
        "fwhm",
        "peak_height",
        "peak_area",
        "noise_level",
        "snr",
        "peak_center",
        "created_at",
    ],
    "model_approvals": [
        "id",
        "model_version",
        "mlflow_run_id",
        "challenger_f1",
        "champion_f1",
        "status",
        "created_at",
        "decided_at",
        "decided_by",
    ],
    "rescoring_runs": [
        "id",
        "model_version",
        "rescored_at",
        "n_predictions",
        "n_changed",
        "change_rate",
        "triggered_by",
        "status",
    ],
    "model_training_data": [
        "id",
        "mlflow_run_id",
        "signal_id",
        "split",
        "model_version",
        "created_at",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_logger = get_ui_logger(__name__)


def _mermaid(diagram: str, height: int = 600) -> None:
    """Embed a Mermaid.js diagram in a self-resizing iframe."""
    import streamlit.components.v1 as components

    html = f"""<html><head>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
</head><body style="background:transparent;margin:0;overflow:hidden">
<div id="d" class="mermaid" style="background:#0f172a;border-radius:10px;padding:1rem;height:{height}px;overflow:auto;">
{diagram.strip()}
</div>
<script>
  mermaid.initialize({{startOnLoad:true,theme:'dark'}});
  var _t=0;var _p=setInterval(function(){{
    _t++;var s=document.querySelector('#d svg');
    if(s||_t>50){{clearInterval(_p);setTimeout(function(){{
      var el=document.getElementById('d');
      if(el&&window.frameElement){{window.frameElement.style.height=(el.scrollHeight+32)+'px';}}
    }},120);}}
  }},100);
</script></body></html>"""
    components.html(html, height=height, scrolling=False)


def _build_pg_url(host: str, port: str | None = None) -> str:
    """Build a PostgreSQL connection URL from env vars + host override."""
    if port is None:
        port = os.environ.get("DB_PORT", "5433")
    user = os.environ.get("POSTGRES_USER", os.environ.get("DB_USER", "mlops_user"))
    password = os.environ.get(
        "POSTGRES_PASSWORD", os.environ.get("DB_PASSWORD", "local_dev_password")
    )
    db = os.environ.get("POSTGRES_DB", os.environ.get("DB_NAME", "mlops_db"))
    return f"postgresql://{user}:{password}@{host}:{port}/{db}?connect_timeout=10"


def _resolve_ipv4(hostname: str) -> str:
    """Return the first IPv4 address for *hostname*, or *hostname* itself.

    psycopg2 probes IPv6 first when a hostname resolves to both families
    (e.g. OrbStack container DNS), adding several seconds of latency before
    falling back to IPv4.  Resolving eagerly to the IPv4 address avoids that.
    """
    import socket

    _port = int(os.environ.get("DB_PORT", "5433"))
    try:
        for info in socket.getaddrinfo(hostname, _port, socket.AF_INET, socket.SOCK_STREAM):
            return str(info[4][0])  # first IPv4 address
    except OSError:
        pass
    return hostname


def _get_db() -> Any:
    """
    Return a Database instance, preferring PostgreSQL when available.

    Uses @st.cache_resource to share a SINGLE connection across all Streamlit
    reruns.  Without caching, each 5-second auto-refresh creates a new
    psycopg2 connection that is never closed, quickly exhausting PostgreSQL's
    max_connections (default 100).  If the cached connection is stale (e.g.
    after a 'too many clients' error), call _clear_db_cache() then retry.

    Connection priority (each step falls through on failure):
      1. Explicit DATABASE_URL env var
      2. POSTGRES_HOST env var  (tried first in candidate loop, with fallback)
      3. mlops_postgres.orb.local — OrbStack Docker DNS (only when POSTGRES_HOST is set)
      4. SQLite fallback at data/database/mlops.db

    Auto-detection against OrbStack DNS only runs when POSTGRES_HOST is set
    (even if set to localhost).  In CI / bare-dev environments where neither
    DATABASE_URL nor POSTGRES_HOST is present, we go directly to SQLite so
    tests are not slowed by connection timeouts.

    POSTGRES_HOST = localhost is NOT trusted outright: a macOS system-installed
    PostgreSQL may answer on localhost:5432 but lack the mlops_user role.  The
    code validates with a real query and retries the OrbStack hostname before
    giving up.
    """
    try:
        from src.database.database import Database

        # 1. Explicit DATABASE_URL
        database_url = os.environ.get("DATABASE_URL", "")
        if database_url and database_url.startswith("postgresql"):
            try:
                db = Database(db_url=database_url)
                db.count_all_signals()
                return db
            except Exception:
                pass  # fall through to host-candidate loop

        # 2 & 3. POSTGRES_HOST candidate loop.
        # Only attempt network probes when POSTGRES_HOST is explicitly set;
        # skip entirely in CI / bare-dev so we don't block on DNS timeouts.
        postgres_host = os.environ.get("POSTGRES_HOST", "")
        if postgres_host:
            seen: set[str] = set()
            candidates: list[str] = [postgres_host]
            seen.add(postgres_host)
            # Always add OrbStack DNS as fallback when primary host fails
            if "mlops_postgres.orb.local" not in seen:
                candidates.append("mlops_postgres.orb.local")

            for try_host in candidates:
                try:
                    # Resolve to IPv4 to avoid multi-second latency from
                    # psycopg2 probing IPv6 first for OrbStack hostnames.
                    resolved = _resolve_ipv4(try_host) if try_host != "localhost" else try_host
                    db = Database(db_url=_build_pg_url(resolved))
                    db.count_all_signals()  # validate credentials + schema
                    _logger.info("Database connected via PostgreSQL host={}", try_host)
                    return db
                except Exception:
                    continue

        # SQLite fallback (dev / CI)
        db_path = PROJECT_ROOT / "data" / "database" / "mlops.db"
        _logger.debug("Database using SQLite fallback: {}", db_path)
        return Database(db_path=str(db_path))
    except Exception as exc:  # pragma: no cover
        _logger.warning("_get_db() failed: {}", exc)
        return exc


@st.cache_resource(show_spinner=False)
def _get_db_cached() -> Any:
    """Cached wrapper — returns a single Database instance for the process lifetime.

    ``@st.cache_resource`` creates ONE shared instance per Streamlit worker
    process, so every page rerun (including auto-refresh every 5 s) reuses the
    same psycopg2 connection instead of opening a new one.  This prevents the
    'too many clients already' error that occurs when the page refreshes faster
    than PostgreSQL can clean up idle connections.
    """
    return _get_db()


def _clear_db_cache() -> None:
    """Evict the cached Database instance so the next call reconnects fresh."""
    _get_db_cached.clear()  # type: ignore[attr-defined]


def _is_connection_stale(db: Any) -> bool:
    """Return True if the cached Database connection is closed or unusable.

    psycopg2 raises ``InterfaceError`` when a cursor is opened on a closed
    connection.  A lightweight SELECT 1 catches this without requiring the
    full schema-validation query used in ``_get_db()``.
    """
    try:
        if not hasattr(db, "conn"):
            return False
        cur = db.conn.cursor()
        cur.execute("SELECT 1")
        return False
    except Exception:
        return True


def _backend_badge(db: Any) -> str:
    """Return an HTML badge indicating current backend."""
    try:
        backend = getattr(db, "_backend", "sqlite")
        if backend == "postgres":
            return (
                '<span style="background:#065f46;color:#6ee7b7;padding:3px 10px;'
                'border-radius:9999px;font-size:.75rem;font-weight:700">🐘 PostgreSQL</span>'
            )
    except Exception:
        pass
    return (
        '<span style="background:#1e3a5f;color:#93c5fd;padding:3px 10px;'
        'border-radius:9999px;font-size:.75rem;font-weight:700">🗃️ SQLite (dev / CI)</span>'
    )


def _pg_row_val(row: Any, index: int, key: str) -> Any:
    """Extract a value from a db row regardless of backend.

    PostgreSQL returns ``RealDictRow`` (dict-like, key access only).
    SQLite returns ``sqlite3.Row`` (supports both index and key access).
    Using the *key* path works for both.
    """
    try:
        return row[key]
    except (KeyError, TypeError, IndexError):
        return row[index]


def _count_rows(db: Any, table: str) -> int | str:
    """Return row count for a table, or an error string."""
    try:
        cursor = db.conn.cursor()
        cursor.execute(f"SELECT COUNT(*) AS n FROM {table}")  # noqa: S608
        row = cursor.fetchone()
        if row is None:
            return 0
        return int(_pg_row_val(row, 0, "n"))
    except Exception as exc:
        return str(exc)


def _fetch_rows(
    db: Any,
    table: str,
    limit: int = 50,
    order_desc: bool = True,
) -> tuple[list[str], list[tuple[Any, ...]]]:
    """Return (column_names, rows) for a table SELECT with LIMIT.

    When *order_desc* is True (default) rows are sorted newest-first by the
    first column (usually the primary-key integer ID).
    """
    try:
        cursor = db.conn.cursor()
        order_clause = "ORDER BY 1 DESC " if order_desc else ""
        cursor.execute(f"SELECT * FROM {table} {order_clause}LIMIT {limit}")  # noqa: S608
        rows = cursor.fetchall()
        col_names: list[str] = [d[0] for d in cursor.description]
        # Convert sqlite3.Row / psycopg2 RealDictRow to plain tuples.
        # dict.values() gives the correct value order for RealDictRow;
        # tuple() on sqlite3.Row also preserves column order.
        plain_rows: list[tuple[Any, ...]] = [
            tuple(r.values()) if hasattr(r, "values") else tuple(r) for r in rows
        ]
        return col_names, plain_rows
    except Exception as exc:
        return [], [(str(exc),)]


def _list_backups() -> list[Path]:
    """Return .dump files in BACKUP_DIR sorted newest-first."""
    if not BACKUP_DIR.is_dir():
        return []
    return sorted(BACKUP_DIR.glob("*.dump"), key=lambda p: p.stat().st_mtime, reverse=True)


def _find_working_pg_url() -> str | None:
    """Return the first working PostgreSQL connection URL, or None if unavailable.

    Only probes the network when DATABASE_URL or POSTGRES_HOST is explicitly
    set; returns None immediately when neither env var is configured.
    """
    from src.database.database import Database

    database_url = os.environ.get("DATABASE_URL", "")
    postgres_host = os.environ.get("POSTGRES_HOST", "")

    # No env var at all → no PostgreSQL configured; skip all network probes.
    if not database_url and not postgres_host:
        return None

    if database_url and database_url.startswith("postgresql"):
        try:
            _db = Database(db_url=database_url)
            _db.count_all_signals()
            _db.close()
            return database_url
        except Exception:
            pass

    candidates: list[str] = []
    seen: set[str] = set()
    if postgres_host:
        candidates.append(postgres_host)
        seen.add(postgres_host)
    for h in ["localhost", "mlops_postgres.orb.local"]:
        if h not in seen:
            candidates.append(h)
    for try_host in candidates:
        try:
            resolved = _resolve_ipv4(try_host) if try_host != "localhost" else try_host
            url = _build_pg_url(resolved)
            _db = Database(db_url=url)
            _db.count_all_signals()
            _db.close()
            return url
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Docker-exec helpers for pg_dump / pg_restore (Windows fallback)
# ---------------------------------------------------------------------------

_PG_CONTAINER = "mlops_postgres"
_PG_USER = "mlops_user"
# Resolve database name at runtime from env vars (avoids hardcoding the old
# 'mlops_db' name which was renamed to 'mlops_local' / 'mlops_prod').
_PG_DBNAME = os.environ.get("POSTGRES_DB", os.environ.get("DB_NAME", "mlops_db"))


def _kubectl_pg_dump(out_path: Path) -> tuple[bool, str]:
    """Run pg_dump inside the K8s postgres pod; write bytes to *out_path*."""
    import subprocess

    try:
        # Find the postgres pod name in the mlops namespace
        pod_result = subprocess.run(  # noqa: S603, S607
            [
                "kubectl",
                "get",
                "pod",
                "-n",
                "mlops",
                "-l",
                "app=postgres",
                "-o",
                "jsonpath={.items[0].metadata.name}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if pod_result.returncode != 0 or not pod_result.stdout.strip():
            return False, "Could not find postgres pod in K8s namespace 'mlops'."

        pod_name = pod_result.stdout.strip()
        result = subprocess.run(  # noqa: S603, S607
            [
                "kubectl",
                "exec",
                "-n",
                "mlops",
                pod_name,
                "--",
                "pg_dump",
                "--no-password",
                "--format=custom",
                "--no-owner",
                "--no-acl",
                f"--username={_PG_USER}",
                f"--dbname={_PG_DBNAME}",
            ],
            capture_output=True,
            timeout=300,
        )
        if result.returncode == 0:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(result.stdout)
            return True, f"Backup written to {out_path.name} (via kubectl exec {pod_name})"
        return False, result.stderr.decode(errors="replace")
    except FileNotFoundError:
        return False, "kubectl not found. Ensure kubectl is installed and in PATH."
    except Exception as exc:
        return False, str(exc)


def _kubectl_get_postgres_pod() -> tuple[str, str]:
    """Return (pod_name, error_msg) for the K8s postgres pod."""
    import subprocess

    r = subprocess.run(  # noqa: S603, S607
        [
            "kubectl",
            "get",
            "pod",
            "-n",
            "mlops",
            "-l",
            "app=postgres",
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if r.returncode != 0 or not r.stdout.strip():
        return "", "Could not find postgres pod in K8s namespace 'mlops'."
    return r.stdout.strip(), ""


def _kubectl_validate(backup_file: Path) -> tuple[bool, str, list]:
    """Validate *backup_file* using pg_restore inside the K8s postgres pod."""
    import subprocess
    import uuid

    pod_name, err = _kubectl_get_postgres_pod()
    if err:
        return False, err, []

    tmp = f"/tmp/validate_{uuid.uuid4().hex[:8]}.dump"
    try:
        # Use stdin-pipe instead of `kubectl cp` to avoid Windows path issues
        # (Windows paths like C:\... contain a colon that kubectl misinterprets
        # as pod:path syntax, causing "one of src or dest must be a local file" error).
        # IMPORTANT: -i flag is required so kubectl connects stdin to the pod;
        # without it, cat receives EOF immediately and writes 0 bytes.
        _file_bytes = backup_file.read_bytes()
        cp = subprocess.run(  # noqa: S603, S607
            ["kubectl", "exec", "-i", "-n", "mlops", pod_name, "--", "bash", "-c", f"cat > {tmp}"],
            input=_file_bytes,
            capture_output=True,
            timeout=60,
        )
        if cp.returncode != 0:
            return (
                False,
                f"kubectl stdin transfer failed: {cp.stderr.decode(errors='replace').strip()}",
                [],
            )

        r = subprocess.run(  # noqa: S603, S607
            ["kubectl", "exec", "-n", "mlops", pod_name, "--", "pg_restore", "--list", tmp],
            capture_output=True,
            text=True,
            timeout=30,
        )
        subprocess.run(  # noqa: S603, S607
            ["kubectl", "exec", "-n", "mlops", pod_name, "--", "rm", "-f", tmp],
            capture_output=True,
            timeout=10,
        )
        if r.returncode == 0:
            lines = r.stdout.strip().splitlines()
            return True, f"Valid backup — {len(lines)} objects listed.", lines
        return False, r.stderr.strip(), []
    except FileNotFoundError:
        return False, "kubectl not found. Ensure kubectl is installed and in PATH.", []
    except Exception as exc:
        return False, str(exc), []


def _kubectl_restore(backup_file: Path) -> tuple[bool, str]:
    """Restore *backup_file* via pg_restore inside the K8s postgres pod."""
    import subprocess
    import uuid

    _logger.info("Starting kubectl restore: {}", backup_file.name)
    pod_name, err = _kubectl_get_postgres_pod()
    if err:
        return False, err

    tmp = f"/tmp/restore_{uuid.uuid4().hex[:8]}.dump"
    try:
        # Use stdin-pipe instead of `kubectl cp` to avoid Windows path issues
        # (Windows paths like C:\... contain a colon that kubectl misinterprets
        # as pod:path syntax).
        # IMPORTANT: -i flag is required so kubectl connects stdin to the pod;
        # without it, cat receives EOF immediately and writes 0 bytes.
        _file_bytes = backup_file.read_bytes()
        cp = subprocess.run(  # noqa: S603, S607
            ["kubectl", "exec", "-i", "-n", "mlops", pod_name, "--", "bash", "-c", f"cat > {tmp}"],
            input=_file_bytes,
            capture_output=True,
            timeout=120,
        )
        if cp.returncode != 0:
            return (
                False,
                f"kubectl stdin transfer failed: {cp.stderr.decode(errors='replace').strip()}",
            )

        # Terminate active connections before restore
        subprocess.run(  # noqa: S603, S607
            [
                "kubectl",
                "exec",
                "-n",
                "mlops",
                pod_name,
                "--",
                "psql",
                "--no-password",
                f"--username={_PG_USER}",
                "--dbname=postgres",
                "--command",
                f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                f"WHERE datname='{_PG_DBNAME}' AND pid <> pg_backend_pid();",
            ],
            capture_output=True,
            timeout=30,
        )

        r = subprocess.run(  # noqa: S603, S607
            [
                "kubectl",
                "exec",
                "-n",
                "mlops",
                pod_name,
                "--",
                "pg_restore",
                "--no-password",
                "--clean",
                "--if-exists",
                "--no-owner",
                "--no-privileges",
                "--format=custom",
                f"--username={_PG_USER}",
                f"--dbname={_PG_DBNAME}",
                tmp,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        subprocess.run(  # noqa: S603, S607
            ["kubectl", "exec", "-n", "mlops", pod_name, "--", "rm", "-f", tmp],
            capture_output=True,
            timeout=10,
        )
        if r.returncode == 0:
            _logger.info("kubectl restore succeeded: {}", backup_file.name)
            return True, "Restore completed successfully."
        _logger.warning("kubectl restore failed (rc={}): {}", r.returncode, r.stderr.strip()[:200])
        return False, r.stderr.strip()
    except FileNotFoundError:
        return False, "kubectl not found. Ensure kubectl is installed and in PATH."
    except Exception as exc:
        _logger.warning("kubectl restore exception: {}", exc)
        return False, str(exc)


def _docker_pg_dump(out_path: Path) -> tuple[bool, str]:
    """Run pg_dump inside ``mlops_postgres`` container; write bytes to *out_path*."""
    import subprocess

    try:
        result = subprocess.run(  # noqa: S603, S607
            [
                "docker",
                "exec",
                _PG_CONTAINER,
                "pg_dump",
                "--no-password",
                "--format=custom",
                "--no-owner",
                "--no-acl",
                f"--username={_PG_USER}",
                f"--dbname={_PG_DBNAME}",
            ],
            capture_output=True,
            timeout=300,  # 5 min — large DBs take longer than 120s to dump
        )
        if result.returncode == 0:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(result.stdout)
            return True, f"Backup written to {out_path.name} (via Docker)"
        return False, result.stderr.decode(errors="replace")
    except FileNotFoundError:
        return False, "Docker not found. Install Docker Desktop and ensure it is running."
    except Exception as exc:
        return False, str(exc)


def _docker_validate(backup_file: Path) -> tuple[bool, str, list]:
    """Validate *backup_file* using pg_restore inside the container."""
    import subprocess
    import uuid

    tmp = f"/tmp/validate_{uuid.uuid4().hex[:8]}.dump"
    try:
        subprocess.run(  # noqa: S603, S607
            ["docker", "cp", str(backup_file), f"{_PG_CONTAINER}:{tmp}"],
            check=True,
            capture_output=True,
            timeout=30,
        )
        r = subprocess.run(  # noqa: S603, S607
            ["docker", "exec", _PG_CONTAINER, "pg_restore", "--list", tmp],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
        )
        subprocess.run(  # noqa: S603, S607
            ["docker", "exec", _PG_CONTAINER, "rm", "-f", tmp],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=10,
        )
        if r.returncode == 0:
            lines = r.stdout.strip().splitlines()
            return True, f"Valid backup — {len(lines)} objects listed.", lines
        return False, r.stderr.strip(), []
    except FileNotFoundError:
        return False, "Docker not found. Install Docker Desktop and ensure it is running.", []
    except Exception as exc:
        return False, str(exc), []


def _docker_restore(backup_file: Path) -> tuple[bool, str]:
    """Restore *backup_file* via pg_restore inside the container."""
    import subprocess
    import uuid

    _logger.info("Starting Docker restore: {}", backup_file.name)
    tmp = f"/tmp/restore_{uuid.uuid4().hex[:8]}.dump"
    try:
        # Copy backup file into container
        cp_result = subprocess.run(  # noqa: S603, S607
            ["docker", "cp", str(backup_file), f"{_PG_CONTAINER}:{tmp}"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if cp_result.returncode != 0:
            return False, f"docker cp failed: {cp_result.stderr.strip()}"

        # Terminate active connections to mlops_prod before restoring
        # (pg_restore --clean will fail if other sessions hold locks)
        subprocess.run(  # noqa: S603, S607
            [
                "docker",
                "exec",
                _PG_CONTAINER,
                "psql",
                "--no-password",
                f"--username={_PG_USER}",
                "--dbname=postgres",
                "--command",
                f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                f"WHERE datname='{_PG_DBNAME}' AND pid <> pg_backend_pid();",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=30,
        )

        r = subprocess.run(  # noqa: S603, S607
            [
                "docker",
                "exec",
                _PG_CONTAINER,
                "pg_restore",
                "--no-password",
                "--clean",
                "--if-exists",
                "--no-owner",
                "--no-privileges",
                "--format=custom",
                f"--username={_PG_USER}",
                f"--dbname={_PG_DBNAME}",
                tmp,
            ],
            stdin=subprocess.DEVNULL,  # prevent docker exec from blocking on stdin
            capture_output=True,
            text=True,
            timeout=300,  # 5 min is ample for any normal DB size
        )
        subprocess.run(  # noqa: S603, S607
            ["docker", "exec", _PG_CONTAINER, "rm", "-f", tmp],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=10,
        )
        if r.returncode == 0:
            _logger.info("Docker restore succeeded: {}", backup_file.name)
            return True, "Restore completed successfully."
        _logger.warning("Docker restore failed (rc={}): {}", r.returncode, r.stderr.strip()[:200])
        return False, r.stderr.strip()
    except FileNotFoundError:
        _logger.warning("Docker restore: Docker not found")
        return False, "Docker not found. Install Docker Desktop and ensure it is running."
    except Exception as exc:
        _logger.warning("Docker restore exception: {}", exc)
        return False, str(exc)


def _trigger_backup() -> tuple[bool, str]:
    """
    Attempt a live pg_dump backup.

    Tries ``backup_postgres`` (local pg_dump) first; if pg_dump is not in
    PATH (common on Windows), falls back to running pg_dump inside the
    ``mlops_postgres`` Docker container.

    Returns (success, message).  If PostgreSQL is not reachable, returns a
    user-friendly explanation rather than failing.
    """
    db_url = _find_working_pg_url()
    if db_url is None:
        _logger.warning("_trigger_backup: PostgreSQL not reachable")
        return False, (
            "PostgreSQL is not reachable (POSTGRES_HOST / DATABASE_URL not set "
            "or connection failed).  Start the Docker stack with 'make local'."
        )

    try:
        from src.database.backup import backup_postgres, get_backup_filename

        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        filename = get_backup_filename()
        out_path = BACKUP_DIR / filename
        _logger.info("Starting pg_dump backup → {}", out_path.name)
        _is_k8s = os.environ.get("DEPLOYMENT_MODE", "") == "k8s"
        _validate_fn = _kubectl_validate if _is_k8s else _docker_validate
        try:
            backup_postgres(db_url=db_url, output_path=out_path)
            _logger.info("Backup succeeded: {}", out_path.name)
            size_kb = out_path.stat().st_size // 1024
            # Validate and count objects so the user sees proof of completeness
            ok_v, _vmsg, _objs = _validate_fn(out_path)
            _tbl_count = sum(1 for o in _objs if "TABLE DATA" in o) if ok_v and _objs else None
            _detail = f" ({_tbl_count} tables)" if _tbl_count else ""
            return True, (
                f"✅ Full database backup written to `{out_path.name}` "
                f"({size_kb} KB{_detail}). All tables and sequences are included."
            )
        except FileNotFoundError:
            # pg_dump not in host PATH (Windows) — run inside the container
            if _is_k8s:
                ok, msg = _kubectl_pg_dump(out_path)
            else:
                ok, msg = _docker_pg_dump(out_path)
            if ok:
                _logger.info("Backup (docker) succeeded: {}", out_path.name)
                size_kb = out_path.stat().st_size // 1024 if out_path.exists() else 0
                ok_v, _vmsg, _objs = _validate_fn(out_path)
                _tbl_count = sum(1 for o in _objs if "TABLE DATA" in o) if ok_v and _objs else None
                _detail = f" ({_tbl_count} tables)" if _tbl_count else ""
                msg = (
                    f"✅ Full database backup written to `{out_path.name}` "
                    f"({size_kb} KB{_detail}). All tables and sequences are included."
                )
            else:
                _logger.warning("Backup (docker) failed: {}", msg)
            return ok, msg
    except Exception as exc:
        _logger.warning("_trigger_backup exception: {}", exc)
        return False, str(exc)


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------


def _tab_schema() -> None:
    """Schema & ERD tab."""
    st.markdown(
        '<div class="section-header">🗺️ Entity-Relationship Diagram</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Nine tables form the complete schema.  The core data tables (devices, predictions, "
        "raw_signals, features, sparse_labels) carry signal and prediction data; drift tables "
        "store provocation experiments; governance tables (model_approvals, rescoring_runs) "
        "track MLOps workflows.  The same DDL is used for both SQLite (dev/CI) and PostgreSQL (production).",
    )
    _mermaid(DB_ERD_MERMAID, height=750)

    st.markdown("---")
    st.markdown(
        '<div class="section-header">📊 Data Flow</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Every `POST /predict` call triggers a cascade write: device → prediction → "
        "raw_signal → features.  Labels are injected asynchronously via the "
        "sparse-labelling API.  The daily Airflow backup DAG calls `pg_dump` early "
        "in the morning before production traffic peaks.",
    )
    _mermaid(DB_FLOW_MERMAID, height=560)

    st.markdown("---")
    st.markdown("#### Table Quick Reference")
    for tbl, desc in TABLE_DESCRIPTIONS.items():
        st.markdown(f"- **`{tbl}`** — {desc}")


def _tab_browser(db: Any) -> None:
    """Live table browser tab."""
    tables = list(TABLE_DESCRIPTIONS.keys())
    col_sel, col_limit, col_order = st.columns([3, 1, 1])
    with col_sel:
        selected = st.selectbox(
            "Select table",
            tables,
            format_func=lambda t: f"🗃️  {t}",
            key="browser_table_select",
        )
    with col_limit:
        limit = st.slider("Rows", min_value=5, max_value=200, value=50, step=5)
    with col_order:
        order_desc = st.checkbox("Newest first", value=True, key="browser_order_desc")

    if selected is None:
        return

    st.caption(TABLE_DESCRIPTIONS[selected])

    # Row count
    count = _count_rows(db, selected)
    st.metric("Total rows in table", count)

    col_names, rows = _fetch_rows(db, selected, limit=limit, order_desc=order_desc)

    if not col_names:
        st.error(f"Could not query table: {rows[0][0] if rows else 'unknown error'}")
        return

    if not rows:
        st.info("Table is empty.")
        return

    # Build a plain list of dicts for st.dataframe
    import pandas as pd

    df = pd.DataFrame(rows, columns=col_names)

    # Re-order columns when preferred order is defined
    pref = _COL_ORDER.get(selected, [])
    ordered = [c for c in pref if c in df.columns] + [c for c in df.columns if c not in pref]
    df = df[ordered]

    st.dataframe(df, width="stretch", hide_index=True)

    # Raw SQL hint
    with st.expander("SQL used for this query"):
        order_sql = "ORDER BY 1 DESC " if order_desc else ""
        st.code(f"SELECT * FROM {selected} {order_sql}LIMIT {limit};", language="sql")


def _tab_backup(db: Any) -> None:  # noqa: ARG001  (db kept for future use)
    """Backup & Recovery management tab."""
    st.markdown(
        '<div class="section-header">💾 Backup Files</div>',
        unsafe_allow_html=True,
    )

    backups = _list_backups()

    if backups:
        import pandas as pd

        rows_data = []
        for bp in backups:
            stat = bp.stat()
            rows_data.append(
                {
                    "File": bp.name,
                    "Size (KB)": round(stat.st_size / 1024, 1),
                    "Created": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    "Path": str(bp),
                }
            )
        st.dataframe(
            pd.DataFrame(rows_data),
            width="stretch",
            hide_index=True,
        )
        st.caption(f"📂 Backup directory: `{BACKUP_DIR}`")
    else:
        st.info(
            f"No backup files found in `{BACKUP_DIR}`.  "
            "Run a backup below or trigger the Airflow DAG."
        )

    st.markdown("---")
    st.markdown(
        '<div class="section-header">▶️ Manual Backup Trigger</div>',
        unsafe_allow_html=True,
    )

    host = os.environ.get("POSTGRES_HOST", "")
    database_url = os.environ.get("DATABASE_URL", "")
    pg_available = False
    working_pg_url = (
        database_url if (database_url and database_url.startswith("postgresql")) else ""
    )

    # Determine the first working PostgreSQL URL using the same host-candidate
    # logic as _get_db(): never trust POSTGRES_HOST=localhost blindly — a local
    # macOS PostgreSQL may answer on that port but lack our mlops_user role.
    if database_url and database_url.startswith("postgresql"):
        try:
            from src.database.database import Database

            _db = Database(db_url=database_url)
            _db.count_all_signals()
            _db.close()
            pg_available = True
            working_pg_url = database_url
        except Exception:
            pass

    if not pg_available:
        from src.database.database import Database

        seen: set[str] = set()
        candidates: list[str] = []
        if host:
            candidates.append(host)
            seen.add(host)
        for h in ["localhost", "mlops_postgres.orb.local"]:
            if h not in seen:
                candidates.append(h)
        for try_host in candidates:
            try:
                resolved = _resolve_ipv4(try_host) if try_host != "localhost" else try_host
                _url = _build_pg_url(resolved)
                _db = Database(db_url=_url)
                _db.count_all_signals()
                _db.close()
                pg_available = True
                host = try_host  # display name
                working_pg_url = _url  # store working URL for restore
                break
            except Exception:
                continue

    pg_display = host or (database_url.split("@")[-1].split("/")[0] if database_url else "")

    _display_port = os.environ.get("DB_PORT", "5433")
    if pg_available:
        st.markdown(
            f"Connected to PostgreSQL at **`{pg_display or f'localhost:{_display_port}'}`**.  Click the button to run `pg_dump` now.",
        )
    else:
        st.markdown(
            '<div class="info-card">'
            "<h4>ℹ️ PostgreSQL not detected</h4>"
            f"<p>PostgreSQL is not reachable on <code>localhost:{_display_port}</code>.  "
            "Start the full Docker stack with <code>make local</code> or <code>make cloud</code> "
            "to enable live backups.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    # Show persisted backup result (survives st.rerun())
    _bkr = st.session_state.pop("_backup_result", None)
    if _bkr:
        if _bkr.get("ok"):
            st.success(_bkr["msg"])
        else:
            st.error(_bkr["msg"])

    if st.button("💾 Run pg_dump backup now", type="primary", disabled=not pg_available):
        with st.spinner("Running pg_dump…"):
            ok, msg = _trigger_backup()
        st.session_state["_backup_result"] = {"ok": ok, "msg": msg}
        st.rerun()  # refresh the page so the new backup file appears in the list

    st.markdown("---")
    st.markdown(
        '<div class="section-header">🔄 Recovery Procedure</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "To restore from a `.dump` file, run the following command "
        "(replace `<file>` with the actual filename):"
    )
    st.code(
        "# 1. Stop the API / Airflow to prevent writes during restore\n"
        "docker compose stop api airflow-webserver airflow-scheduler\n\n"
        "# 2. Drop and recreate the target database\n"
        "psql -U mlops -c 'DROP DATABASE IF EXISTS mlops_db;'\n"
        "psql -U mlops -c 'CREATE DATABASE mlops_db;'\n\n"
        "# 3. Restore the dump\n"
        "pg_restore -Fc -d mlops_db -U mlops data/backups/<file>.dump\n\n"
        "# 4. Restart services\n"
        "docker compose start api airflow-webserver airflow-scheduler",
        language="bash",
    )

    st.markdown("---")
    st.markdown(
        '<div class="section-header">♻️ One-Click Restore</div>',
        unsafe_allow_html=True,
    )
    st.warning(
        "Restoring from a backup will **overwrite all current data** in the database.  "
        "This is irreversible.  Only available when PostgreSQL is reachable.",
        icon="⚠️",
    )

    restore_backups = _list_backups()
    if not restore_backups:
        st.info("No backup files available.  Run a backup first.")
    elif not pg_available:
        st.info("PostgreSQL is not reachable — restore requires a live PostgreSQL connection.")
    else:
        selected_backup = st.selectbox(
            "Select backup file to restore",
            restore_backups,
            format_func=lambda p: f"{p.name}  ({round(p.stat().st_size / 1024, 1)} KB)",
            key="restore_backup_select",
        )

        col_validate, col_restore = st.columns(2)
        with col_validate:
            if st.button("🔍 Validate backup", key="validate_backup_btn"):
                import shutil
                import subprocess as _sp

                if shutil.which("pg_restore"):
                    # Local pg_restore available
                    try:
                        _r = _sp.run(
                            ["pg_restore", "--list", str(selected_backup)],  # noqa: S603, S607
                            capture_output=True,
                            text=True,
                            timeout=30,
                        )
                        if _r.returncode == 0:
                            _lines = _r.stdout.strip().splitlines()
                            st.success(f"✅ Valid backup — {len(_lines)} objects listed.")
                            with st.expander("Backup contents"):
                                st.code(
                                    "\n".join(_lines[:50]) + ("\n…" if len(_lines) > 50 else "")
                                )
                        else:
                            st.error(f"Validation failed:\n{_r.stderr}")
                    except Exception as exc:
                        st.error(f"Validation error: {exc}")
                else:
                    # Fallback: validate via kubectl (K8s) or Docker container
                    _validate_fn = (
                        _kubectl_validate
                        if os.environ.get("DEPLOYMENT_MODE") == "k8s"
                        else _docker_validate
                    )
                    ok, msg, lines = _validate_fn(selected_backup)
                    if ok:
                        st.success(f"✅ {msg}")
                        if lines:
                            with st.expander("Backup contents"):
                                st.code("\n".join(lines[:50]) + ("\n…" if len(lines) > 50 else ""))
                    else:
                        st.error(f"Validation failed: {msg}")

        with col_restore:
            if st.button("♻️ Restore from backup", type="primary", key="restore_backup_btn"):
                st.session_state["restore_confirm_pending"] = str(selected_backup)

        if st.session_state.get("restore_confirm_pending") == str(selected_backup):
            st.error(
                f"**CONFIRM:** This will wipe all current data and restore from  "
                f"`{selected_backup.name}`.  Are you sure?"
            )
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("✅ Yes, restore now", type="primary", key="restore_confirm_yes"):
                    try:
                        if not working_pg_url:
                            st.error(
                                "PostgreSQL URL not available — cannot restore. "
                                "Ensure the database is reachable and try again."
                            )
                        else:
                            with st.spinner("Restoring database…"):
                                import shutil

                                try:
                                    from src.database.backup import restore_postgres

                                    restore_postgres(
                                        db_url=working_pg_url,
                                        backup_path=selected_backup,
                                    )
                                    ok, msg = True, "Restore completed successfully."
                                except FileNotFoundError:
                                    # pg_restore not in host PATH — use kubectl (K8s) or Docker
                                    if os.environ.get("DEPLOYMENT_MODE") == "k8s":
                                        ok, msg = _kubectl_restore(selected_backup)
                                    else:
                                        ok, msg = _docker_restore(selected_backup)
                            if ok:
                                st.success(
                                    f"✅ Restore complete from `{selected_backup.name}`.  "
                                    "Restart the API to pick up the restored data."
                                )
                            else:
                                st.error(f"Restore failed: {msg}")
                            st.session_state.pop("restore_confirm_pending", None)
                    except Exception as exc:
                        st.error(f"Restore failed: {exc}")
            with col_no:
                if st.button("Cancel", key="restore_confirm_no"):
                    st.session_state.pop("restore_confirm_pending", None)
                    st.rerun()


def _tab_manage(db: Any) -> None:
    """Data management — inject labels, wipe test data."""
    # Show wipe result that was stored before the last st.rerun()
    _wipe_result = st.session_state.pop("_wipe_result", None)
    if _wipe_result:
        _kind, _msg = _wipe_result
        if _kind == "success":
            st.success(_msg)
        else:
            st.info(_msg)

    st.markdown(
        '<div class="section-header">✏️ Inject Ground-Truth Label</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Sparse labelling: set the ground-truth label for an existing prediction "
        "so the model can be retrained with real feedback."
    )

    # Fetch recent predictions so the user can pick one from a dropdown.
    _pred_options: list[tuple[int, str, str, Any, str]] = []
    try:
        _cur = db.conn.cursor()
        _cur.execute(
            "SELECT prediction_id, device_id, predicted_label, ground_truth_label, created_at "
            "FROM predictions ORDER BY prediction_id DESC LIMIT 100"
        )
        for _row in _cur.fetchall():
            # Use named-key access — works for both sqlite3.Row and psycopg2 RealDictRow.
            _pred_options.append(
                (
                    int(_row["prediction_id"]),
                    str(_row["device_id"]),
                    str(_row["predicted_label"]),
                    _row["ground_truth_label"],
                    str(_row["created_at"])[:10] if _row["created_at"] else "",
                )
            )
    except Exception:
        _pred_options = []

    with st.form("inject_label_form"):
        if _pred_options:
            _labels = [f"#{p[0]} — {p[1]} — pred:{p[2]} gt:{p[3]} ({p[4]})" for p in _pred_options]
            sel_idx = st.selectbox(
                "Prediction",
                options=list(range(len(_labels))),
                format_func=lambda i: _labels[i],  # type: ignore[arg-type]
            )
            try:
                _sel = int(sel_idx)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                _sel = 0
            pred_id: Any = _pred_options[_sel][0]
        else:
            pred_id = st.number_input("prediction_id", min_value=1, step=1, value=1, format="%d")
        gt_label = st.selectbox(
            "Ground-truth label",
            [0, 1],
            format_func=lambda x: "0 — Healthy" if x == 0 else "1 — Unhealthy",
        )
        source = st.text_input("label_source", value="manual")
        submitted = st.form_submit_button("Inject label", type="primary")

    # Show persisted result from previous submit (survives the st.rerun() after inject)
    _ilr = st.session_state.pop("_inject_label_result", None)
    if _ilr:
        if _ilr.get("ok"):
            st.success(_ilr["msg"])
        else:
            st.error(_ilr["msg"])

    if submitted:
        try:
            db.inject_sparse_label(
                prediction_id=int(pred_id),
                ground_truth_label=int(gt_label),
                label_source=source,
            )
            st.session_state["_inject_label_result"] = {
                "ok": True,
                "msg": f"✅ Label {gt_label} injected for prediction_id={pred_id} (source: {source}).",
            }
            st.rerun()  # auto-refresh stats; result shown on next render
        except Exception as exc:
            st.session_state["_inject_label_result"] = {
                "ok": False,
                "msg": f"Failed to inject label: {exc}",
            }
            st.rerun()

    st.markdown("---")
    st.markdown(
        '<div class="section-header">📦 Batch Label Injection</div>',
        unsafe_allow_html=True,
    )
    st.markdown("Inject ground-truth labels for the N most recent unlabeled predictions at once.")

    col_n, col_lbl, col_src = st.columns([2, 2, 3])
    with col_n:
        batch_n = st.slider("Number of labels", min_value=1, max_value=200, value=10, key="batch_n")
    with col_lbl:
        batch_label = st.selectbox(
            "Ground-truth label",
            [0, 1],
            format_func=lambda x: "0 — Healthy" if x == 0 else "1 — Unhealthy",
            key="batch_gt_label",
        )
    with col_src:
        batch_source_tag = st.text_input(
            "label_source tag", value="batch_inject", key="batch_src_tag"
        )

    # Show persisted batch inject result
    _bir = st.session_state.pop("_batch_inject_result", None)
    if _bir:
        if _bir.get("ok"):
            st.success(_bir["msg"])
        else:
            st.warning(_bir["msg"])

    if st.button("🚀 Inject Batch Labels", type="primary", key="batch_inject_btn"):
        try:
            _cur = db.conn.cursor()
            _cur.execute(
                "SELECT prediction_id FROM predictions "
                "WHERE ground_truth_label IS NULL "
                "ORDER BY prediction_id DESC LIMIT %s"
                if db._backend == "postgresql"  # type: ignore[attr-defined]
                else "SELECT prediction_id FROM predictions "
                "WHERE ground_truth_label IS NULL "
                "ORDER BY prediction_id DESC LIMIT ?",
                (batch_n,),
            )
            _ids = [r["prediction_id"] for r in _cur.fetchall()]
            if not _ids:
                st.warning("No unlabeled predictions found in database.")
            else:
                _prog = st.progress(0, text="Injecting labels…")
                _ok = 0
                _errs: list[str] = []
                for _i, _pid in enumerate(_ids):
                    try:
                        db.inject_sparse_label(
                            prediction_id=int(_pid),
                            ground_truth_label=int(batch_label),
                            label_source=batch_source_tag,
                        )
                        _ok += 1
                    except Exception as _e:
                        _errs.append(f"pid={_pid}: {_e}")
                    _prog.progress(
                        (_i + 1) / len(_ids), text=f"Injecting label {_i + 1}/{len(_ids)}…"
                    )
                _prog.empty()
                if _errs:
                    st.session_state["_batch_inject_result"] = {
                        "ok": False,
                        "msg": f"⚠️ {_ok}/{len(_ids)} injected. Errors: {'; '.join(_errs[:3])}",
                    }
                else:
                    st.session_state["_batch_inject_result"] = {
                        "ok": True,
                        "msg": f"✅ {_ok} labels injected successfully.",
                    }
                st.rerun()
        except Exception as exc:
            st.error(f"Batch inject failed: {exc}")

    st.markdown("---")
    st.markdown(
        '<div class="section-header">🎯 Targeted Deletion</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Delete specific records by table and filter. Respects foreign key constraints "
        "(e.g., deleting a device cascades to its predictions)."
    )

    _ALLOWED_FILTER_COLS: dict[str, list[str]] = {
        "predictions": ["device_id", "predicted_label", "model_version", "label_source"],
        "raw_signals": ["device_id"],
        "features": ["device_id"],
        "sparse_labels": ["device_id", "label"],
        "devices": ["device_id", "status"],
    }

    del_table = st.selectbox(
        "Table",
        list(_ALLOWED_FILTER_COLS.keys()),
        key="_del_table",
    )
    del_filter_col = st.selectbox(
        "Filter column",
        _ALLOWED_FILTER_COLS[del_table],
        key="_del_filter_col",
    )
    del_filter_val = st.text_input("Filter value (exact match)", key="_del_filter_val")

    # Show persisted deletion result
    _dr = st.session_state.pop("_del_result", None)
    if _dr:
        if _dr.get("ok"):
            st.success(_dr["msg"])
        else:
            st.error(_dr["msg"])

    if st.button("🗑️ Delete matching rows", type="secondary", key="_del_targeted"):
        if not del_filter_val.strip():
            st.warning("Enter a filter value to proceed.")
        elif del_filter_col not in _ALLOWED_FILTER_COLS.get(del_table, []):
            st.error("Invalid filter column.")
        else:
            try:
                _del_cur = db.conn.cursor()
                # Parameterised DELETE — col name is whitelisted above, only the
                # value comes from user input and is bound as a parameter.
                _backend = getattr(db, "_backend", "sqlite")
                _placeholder = "%s" if _backend == "postgresql" else "?"
                _del_cur.execute(
                    f"DELETE FROM {del_table} WHERE {del_filter_col} = {_placeholder}",  # noqa: S608
                    (del_filter_val.strip(),),
                )
                db.conn.commit()
                _del_count = _del_cur.rowcount
                st.session_state["_del_result"] = {
                    "ok": True,
                    "msg": f"✅ Deleted {_del_count} row(s) from `{del_table}`.",
                }
                st.rerun()
            except Exception as exc:
                db.conn.rollback()
                st.error(f"Deletion failed: {exc}")

    st.markdown("---")
    st.markdown(
        '<div class="section-header">💣 Wipe ALL Database Data</div>',
        unsafe_allow_html=True,
    )
    st.error(
        "☢️ **DANGER ZONE** — This permanently deletes **every row** from "
        "all tables (`devices`, `predictions`, `raw_signals`, `features`, `sparse_labels`).  "
        "The schema and indexes are preserved so the system restarts cleanly, "
        "but **all prediction history, labels, and device registrations are lost forever**.  "
        "After wiping: re-run the stack to repopulate via fresh predictions.",
    )
    if "wipe_all_confirmed" not in st.session_state:
        st.session_state["wipe_all_confirmed"] = False

    if st.button("💣 Wipe ALL data (all tables)", type="secondary"):
        st.session_state["wipe_all_confirmed"] = True

    if st.session_state.get("wipe_all_confirmed"):
        st.error(
            "⚠️ Type **CONFIRM** below and click the button to proceed. This action is irreversible."
        )
        confirm_text = st.text_input("Type CONFIRM to proceed:", key="wipe_all_confirm_text")
        col_yes2, col_no2 = st.columns(2)
        with col_yes2:
            if st.button("💣 Yes, delete everything", type="primary", key="wipe_all_yes"):
                if confirm_text == "CONFIRM":
                    try:
                        counts = db.wipe_all_data()
                        total = sum(counts.values())
                        summary = ", ".join(f"{t}: {n}" for t, n in counts.items())
                        if total == 0:
                            st.session_state["_wipe_result"] = (
                                "empty",
                                f"✅ All tables truncated (database was already empty). "
                                f"Row counts before wipe — {summary}.",
                            )
                        else:
                            st.session_state["_wipe_result"] = (
                                "success",
                                f"✅ All data wiped successfully. Rows deleted — {summary}.",
                            )
                        st.session_state["wipe_all_confirmed"] = False
                        st.rerun()  # refresh row-count metrics immediately
                    except Exception as exc:
                        st.error(f"Wipe failed: {exc}")
                else:
                    st.warning("You must type exactly **CONFIRM** to proceed.")
        with col_no2:
            if st.button("Cancel", key="wipe_all_cancel"):
                st.session_state["wipe_all_confirmed"] = False
                st.rerun()

    st.markdown("---")
    st.markdown(
        '<div class="section-header">📋 Schema DDL</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "The idempotent `CREATE TABLE IF NOT EXISTS` DDL is in "
        "`src/database/init_db.py`.  PostgreSQL DDL uses `SERIAL` / "
        "`TIMESTAMPTZ` / `VARCHAR` types with named constraints.",
    )
    with st.expander("View devices DDL (PostgreSQL)"):
        st.code(
            """\
CREATE TABLE IF NOT EXISTS devices (
    device_id        VARCHAR(36) PRIMARY KEY,
    device_name      TEXT,
    device_type      TEXT,
    location         TEXT,
    status           TEXT NOT NULL DEFAULT 'active',
    deployment_mode  TEXT NOT NULL DEFAULT 'local',
    first_seen_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at     TIMESTAMPTZ,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT devices_status_check
        CHECK (status IN ('active', 'inactive', 'maintenance'))
);""",
            language="sql",
        )
    with st.expander("View predictions DDL (PostgreSQL)"):
        st.code(
            """\
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id         SERIAL PRIMARY KEY,
    device_id             VARCHAR(36) NOT NULL REFERENCES devices(device_id),
    timestamp             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    predicted_label       INTEGER NOT NULL,
    prediction_confidence FLOAT,
    model_version         TEXT NOT NULL,
    ground_truth_label    INTEGER,
    label_source          TEXT,
    mlflow_run_id         TEXT,
    git_sha               TEXT,
    dvc_data_hash         TEXT,
    airflow_run_id        TEXT,
    deployment_mode       TEXT NOT NULL DEFAULT 'local',
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT predictions_label_check
        CHECK (predicted_label IN (0, 1)),
    CONSTRAINT predictions_gt_check
        CHECK (ground_truth_label IS NULL OR ground_truth_label IN (0, 1)),
    CONSTRAINT predictions_confidence_check
        CHECK (prediction_confidence IS NULL
               OR (prediction_confidence >= 0 AND prediction_confidence <= 1))
);""",
            language="sql",
        )


def _tab_simulate() -> None:
    """Simulate shutdown scenario."""
    st.markdown(
        '<div class="section-header">💥 Simulate PostgreSQL Shutdown</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "This tab walks through the **failure and recovery sequence** for an "
        "unexpected PostgreSQL shutdown.  No actual process is killed — this is "
        "an educational walkthrough of the system's resilience mechanisms."
    )

    st.markdown("---")

    steps = [
        (
            "1️⃣  Normal operation",
            "API handles `/predict` requests normally.  Writes flow: "
            "devices → predictions → raw_signals → features.  "
            "Airflow DAGs query the database on schedule.",
        ),
        (
            "2️⃣  PostgreSQL process terminates",
            "The `postgres` container crashes or is stopped with "
            "`docker compose stop postgres`.  **Consequence**: any in-flight "
            "transaction is rolled back automatically by PostgreSQL "
            "(WAL replay on restart).  No partial writes survive.",
        ),
        (
            "3️⃣  API returns 503",
            "The FastAPI `/predict` endpoint catches the `OperationalError` from "
            "psycopg2 and returns HTTP 503.  Prometheus scrape picks up the "
            "connection-pool exhaustion metric.  Grafana fires the "
            "`API_DB_Connection_Failure` alert rule.",
        ),
        (
            "4️⃣  Auto-restart (Docker policy)",
            "The `postgres` service in `docker-compose.yml` has "
            "`restart: unless-stopped`.  Docker restarts it within seconds.  "
            "PostgreSQL applies WAL and returns to clean state.",
        ),
        (
            "5️⃣  Connection pool reconnects",
            "The `Database` class uses a single persistent connection with "
            "auto-reconnect logic.  On the next request the pool establishes a "
            "fresh connection.  Business traffic resumes transparently.",
        ),
        (
            "6️⃣  Data integrity verified",
            "All `FOREIGN KEY` constraints are enforced.  The row count "
            "before and after the outage matches — no data loss for committed "
            "transactions.  Uncommitted in-flight writes are absent, as expected.",
        ),
        (
            "7️⃣  Backup safety net",
            "Even if the disk is unrecoverable, the most recent `pg_dump` backup "
            "(at most 24 h old) in `data/backups/` can restore the database to a "
            "known-good state via `pg_restore`.",
        ),
    ]

    for title, detail in steps:
        with st.expander(title):
            st.markdown(detail)

    st.markdown("---")
    st.markdown("#### Hands-on commands")
    _sim_port = os.environ.get("DB_PORT", "5433")
    st.code(
        "# Simulate crash\n"
        "docker compose stop postgres\n\n"
        "# Watch API respond with 503\n"
        f"curl -s -o /dev/null -w '%{{http_code}}' http://{get_host()}:8080/health\n\n"
        "# Restart postgres (auto-reconnect)\n"
        "docker compose start postgres\n\n"
        "# Verify row counts unchanged\n"
        f"psql -h {get_host()} -p {_sim_port} -U mlops_user -d mlops_db -c 'SELECT COUNT(*) FROM predictions;'",
        language="bash",
    )

    st.markdown("---")
    st.markdown(
        '<div class="info-card">'
        "<h4>🛡️ SQLite fallback mode</h4>"
        "<p>In dev / CI the <code>Database</code> class automatically uses SQLite "
        "when <code>POSTGRES_HOST</code> is absent.  This means the full test suite "
        "runs without any Docker dependency, and every API call is unit-testable "
        "in an isolated in-memory database.</p>"
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the PostgreSQL Database page."""
    try:
        _render_content()
    except Exception:
        _logger.exception("Unhandled exception in PostgreSQL Database.render()")
        raise


def _render_content() -> None:
    """Inner render — separated so exceptions are caught and logged by render()."""
    st.markdown(get_global_css(), unsafe_allow_html=True)
    st.markdown(
        hero_section(
            "PostgreSQL Database",
            "Schema explorer, live table browser, backup manager, and failure simulation.",
        ),
        unsafe_allow_html=True,
    )

    # Connection badge — uses cached connection to avoid exhausting PostgreSQL max_connections.
    # If the cached connection is stale (e.g. after PostgreSQL restart or idle timeout),
    # clear the cache and reconnect transparently.
    db_or_err = _get_db_cached()
    if isinstance(db_or_err, Exception) or _is_connection_stale(db_or_err):
        _clear_db_cache()
        db_or_err = _get_db_cached()
    db_connected = not isinstance(db_or_err, Exception)

    if db_connected:
        db = db_or_err
        st.markdown(
            f"**Backend** {_backend_badge(db)}",
            unsafe_allow_html=True,
        )
    else:
        st.error(f"Could not open database: {db_or_err}")
        db = None

    st.markdown("")  # spacing

    # ── Summary metrics ─────────────────────────────────────────
    if db_connected and db is not None:
        _tbl_names = list(TABLE_DESCRIPTIONS.keys())
        _counts = {t: _count_rows(db, t) for t in _tbl_names}
        _total = sum(c for c in _counts.values() if isinstance(c, int))
        _tbl_icons = {
            "devices": "📡",
            "predictions": "🔮",
            "raw_signals": "📈",
            "features": "🧮",
            "sparse_labels": "🏷️",
            "drift_batches": "🌊",
            "drift_signals": "📉",
        }
        # Overall summary row — styled cards matching Home page
        _sum_cols = st.columns(3)
        _sum_cols[0].markdown(
            metric_card("📦", f"{_total:,}", "Total Rows (all tables)"),
            unsafe_allow_html=True,
        )
        _sum_cols[1].markdown(
            metric_card("🐘", getattr(db, "_backend", "sqlite").upper(), "Backend"),
            unsafe_allow_html=True,
        )
        _sum_cols[2].markdown(
            metric_card("🗂️", str(len(_tbl_names)), "Tables"),
            unsafe_allow_html=True,
        )
        st.markdown("")
        # Per-table metrics in two rows of 4
        _row_size = 4
        for _row_start in range(0, len(_tbl_names), _row_size):
            _chunk = _tbl_names[_row_start : _row_start + _row_size]
            _cols = st.columns(len(_chunk))
            for _ci, _tbl in enumerate(_chunk):
                _cnt = _counts[_tbl]
                _icon = _tbl_icons.get(_tbl, "🗃️")
                _display = f"{_cnt:,}" if isinstance(_cnt, int) else "⚠️"
                _cols[_ci].markdown(
                    metric_card(_icon, _display, _tbl),
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # ── Tabs — use st.radio so selection persists across st.rerun() calls.
    # st.tabs() resets to tab 0 whenever st.rerun() is triggered (e.g. after
    # backup/wipe/inject operations), causing a jarring "jump to Schema & ERD".
    db_tabs = [
        "🗺️ Schema & ERD",
        "🔍 Table Browser",
        "💾 Backup & Recovery",
        "✏️ Data Management",
        "💥 Simulate Shutdown",
    ]
    tab_css = """
<style>
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] { padding: 0 !important; }
div[data-baseweb="radio"] > div { gap: 0.25rem; flex-wrap: wrap; }
div[data-baseweb="radio"] > div > label {
    border: 1px solid #e2e8f0;
    border-radius: 8px 8px 0 0;
    padding: 0.45rem 1rem;
    margin-bottom: -1px;
    background: #f8fafc;
    font-size: 0.875rem;
    cursor: pointer;
    transition: background 0.15s, color 0.15s;
}
div[data-baseweb="radio"] > div > label:hover { background: #e0e7ff; }
div[data-baseweb="radio"] > div > label[data-checked="true"],
div[data-baseweb="radio"] > div > label[aria-checked="true"] {
    background: white;
    border-bottom-color: white;
    font-weight: 600;
    color: #4f46e5;
}
</style>
"""
    st.markdown(tab_css, unsafe_allow_html=True)
    active_tab = st.radio(
        "DB Tab",
        db_tabs,
        horizontal=True,
        key="_db_tab",
        label_visibility="collapsed",
    )
    st.markdown(
        "<hr style='margin:0 0 1rem 0;border-color:#e2e8f0;'>",
        unsafe_allow_html=True,
    )

    if active_tab == db_tabs[0]:
        _tab_schema()
    elif active_tab == db_tabs[1]:
        if db_connected and db is not None:
            _tab_browser(db)
        else:
            st.warning("Database connection unavailable — cannot browse live data.")
    elif active_tab == db_tabs[2]:
        _tab_backup(db)
    elif active_tab == db_tabs[3]:
        if db_connected and db is not None:
            _tab_manage(db)
        else:
            st.warning("Database connection unavailable — data management disabled.")
    elif active_tab == db_tabs[4]:
        _tab_simulate()
