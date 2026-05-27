"""
Airflow DAG for batch re-scoring historical predictions.

When a new champion model is promoted, this DAG re-runs predictions on the
last N days of stored signals using the new model so that the database always
reflects the current model's view of historical data.

Workflow:
  1. Load champion model from disk / MLflow registry
  2. Fetch historical predictions (last N days) that were made with a different model version
  3. Re-score each prediction using the champion model's pipeline
  4. Update the predictions table with the new labels / confidence scores
  5. Write a rescoring_runs audit record
  6. Notify on completion

Schedule: Manual trigger only (also callable from model_promotion DAG on promote).
Params:
  - lookback_days: int — how many days of predictions to re-score (default 30)
  - model_version: str — target model version (default "champion")
  - dry_run: bool — if True, compute changes but do not write to DB
"""

import contextlib
import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow.models import Param
from airflow.operators.python import PythonOperator

from airflow import DAG

_DAG_ID = "batch_rescoring"

_DEFAULT_LOOKBACK_DAYS = 30
_CHAMPION_PKL = Path("/opt/airflow/models/champion_model.pkl")
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ──────────────────────────────────────────────────────────────────────────────
# Task functions
# ──────────────────────────────────────────────────────────────────────────────


class _NumpyCompatUnpickler:
    """Unpickler that remaps numpy 2.x private modules to numpy 1.x equivalents.

    Models trained on Python 3.12 + numpy 2.x embed ``numpy._core`` in their
    pickles.  The Airflow container runs Python 3.8 + numpy 1.x, where that
    module does not exist.  This unpickler transparently rewrites the module
    path so deserialization succeeds.
    """

    def __new__(cls, fh):
        import pickle

        class _Compat(pickle.Unpickler):
            def find_class(self, module: str, name: str):
                if module.startswith("numpy._core"):
                    module = module.replace("numpy._core", "numpy.core", 1)
                return super().find_class(module, name)

        return _Compat(fh)


def load_champion_model(**context) -> dict:
    """Load the champion model bundle from disk or MLflow registry."""
    params = context.get("params", {})
    _model_version_hint = params.get("model_version", "champion")

    # Try file-based champion first (search common paths in order of preference)
    candidates = [
        _CHAMPION_PKL,
        _PROJECT_ROOT / "models" / "champion_model.pkl",
        _PROJECT_ROOT / "models" / "champion" / "model.pkl",
        # If file-based promotion was skipped (MLflow-only), the challenger is
        # the newest model — use it as a rescoring baseline
        _PROJECT_ROOT / "models" / "challenger.pkl",
        _PROJECT_ROOT / "models" / "bootstrap_model.pkl",
        Path("/opt/airflow/models/challenger.pkl"),
    ]
    for candidate in candidates:
        if candidate.exists():
            with open(candidate, "rb") as fh:
                bundle = _NumpyCompatUnpickler(fh).load()
            actual_version = bundle.get("model_version", str(candidate))
            # Prefer the MLflow registry's canonical champion version over the
            # (potentially stale) version string embedded in the pickle bundle.
            try:
                import mlflow as _mlflow

                _mlflow.set_tracking_uri(
                    os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_buffer:5000")
                )
                _client = _mlflow.tracking.MlflowClient()
                _model_name = os.getenv("MODEL_REGISTRY_NAME", "device_health_classifier")
                try:
                    # MLflow v3: resolve by alias
                    _mv = _client.get_model_version_by_alias(_model_name, "champion")
                    actual_version = _mv.version
                    # Normalize to "v{N}" to match predictions table format
                    if str(actual_version).lstrip("0123456789") == "":
                        actual_version = f"v{actual_version}"
                except Exception:
                    _champ_versions = _client.get_latest_versions(
                        _model_name, stages=["Production"]
                    )
                    if _champ_versions:
                        actual_version = _champ_versions[0].version
                        if str(actual_version).lstrip("0123456789") == "":
                            actual_version = f"v{actual_version}"
            except Exception as _exc:
                print(f"⚠️ MLflow version lookup skipped (using bundle metadata): {_exc}")
            print(f"✓ Loaded champion model from {candidate} (version={actual_version})")
            return {"model_path": str(candidate), "model_version": actual_version}

    # Try MLflow registry
    try:
        import mlflow

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_buffer:5000"))
        client = mlflow.tracking.MlflowClient()
        model_name = os.getenv("MODEL_REGISTRY_NAME", "device_health_classifier")
        # MLflow 3.x uses aliases instead of legacy stages.  Try alias "champion"
        # first; fall back to legacy stages for older server versions.
        champion_mv = None
        try:
            champion_mv = client.get_model_version_by_alias(model_name, "champion")
        except Exception:
            legacy = client.get_latest_versions(model_name, stages=["Production"])
            champion_mv = legacy[0] if legacy else None
        if champion_mv:
            mv = champion_mv
            local_dir = mlflow.artifacts.download_artifacts(
                run_id=mv.run_id,
                artifact_path="model",
                dst_path=str(_PROJECT_ROOT / "models" / "_rescoring_download"),
            )
            pkl_path = next(Path(local_dir).rglob("*.pkl"), None)
            if pkl_path:
                with open(pkl_path, "rb") as fh:
                    bundle = _NumpyCompatUnpickler(fh).load()
                print(f"✓ Loaded champion via MLflow registry (v{mv.version})")
                return {"model_path": str(pkl_path), "model_version": mv.version}
    except Exception as exc:
        print(f"⚠️ MLflow registry lookup failed: {exc}")

    raise FileNotFoundError(
        "Champion model not found. Ensure a champion model exists at "
        f"{_CHAMPION_PKL} or in the MLflow Production registry."
    )


def fetch_predictions_to_rescore(**context) -> dict:
    """
    Fetch historical predictions that differ from the current champion version.

    Returns a list of (prediction_id, device_id, signal features) tuples.
    """
    import sqlite3

    params = context.get("params", {})
    lookback_days = int(params.get("lookback_days", _DEFAULT_LOOKBACK_DAYS))
    ti = context.get("task_instance")
    model_info = ti.xcom_pull(task_ids="load_champion") if ti else {}
    champion_version = (model_info or {}).get("model_version", "champion")

    # Normalize version format: MLflow registry returns raw integers (e.g. "3"),
    # but predictions are stored as "v3". Canonicalize to "v{N}" so the
    # inequality filter correctly excludes already-rescored predictions.
    if (
        champion_version
        and champion_version.lstrip("0123456789").strip() == ""
        and not champion_version.startswith("v")
    ):
        champion_version = f"v{champion_version}"

    since_iso = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat()

    pg_url = os.environ.get("DATABASE_URL", "")
    rows = []

    if pg_url:
        try:
            import psycopg2

            conn = psycopg2.connect(pg_url)
            with conn, conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT p.prediction_id, p.device_id, p.model_version,
                           p.predicted_label,
                           f.fwhm, f.peak_height, f.peak_area, f.noise_level,
                           f.snr, f.peak_center
                    FROM predictions p
                    LEFT JOIN features f ON f.prediction_id = p.prediction_id
                    WHERE p.created_at >= %s
                      AND p.model_version != %s
                    ORDER BY p.created_at DESC
                    LIMIT 10000
                    """,
                    (since_iso, champion_version),
                )
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
            conn.close()
            print(f"✓ Fetched {len(rows)} predictions from PostgreSQL for re-scoring")
        except Exception as exc:
            print(f"⚠️ PostgreSQL fetch failed: {exc} — falling back to SQLite")

    if not rows:
        db_path = _PROJECT_ROOT / "data" / "mlops.db"
        if db_path.exists():
            with contextlib.closing(sqlite3.connect(str(db_path))) as con:
                cur2 = con.execute(
                    """
                    SELECT p.prediction_id, p.device_id, p.model_version,
                           p.predicted_label,
                           f.fwhm, f.peak_height, f.peak_area, f.noise_level,
                           f.snr, f.peak_center
                    FROM predictions p
                    LEFT JOIN features f ON f.prediction_id = p.prediction_id
                    WHERE p.created_at >= ?
                      AND p.model_version != ?
                    ORDER BY p.created_at DESC
                    LIMIT 10000
                    """,
                    (since_iso, champion_version),
                )
                cols = [d[0] for d in cur2.description]
                rows = cur2.fetchall()
        print(f"✓ Fetched {len(rows)} predictions from SQLite for re-scoring")

    records = [dict(zip(cols, r)) for r in rows]  # noqa: B905 (Python 3.8 compat)
    return {
        "n_predictions": len(records),
        "champion_version": champion_version,
        "records": records,
        "lookback_days": lookback_days,
    }


def rescore_predictions(**context) -> dict:
    """
    Re-score each prediction with the champion model and compute change rate.
    If dry_run=True, computes changes without writing to the database.
    """
    import sqlite3

    params = context.get("params", {})
    dry_run = bool(params.get("dry_run", False))

    ti = context.get("task_instance")
    model_info = ti.xcom_pull(task_ids="load_champion") if ti else {}
    fetch_info = ti.xcom_pull(task_ids="fetch_predictions") if ti else {}

    model_path = (model_info or {}).get("model_path", "")
    records = (fetch_info or {}).get("records", [])
    champion_version = (fetch_info or {}).get("champion_version", "champion")

    if not records:
        print("ℹ️ No predictions to re-score.")
        return {
            "n_predictions": 0,
            "n_changed": 0,
            "change_rate": 0.0,
            "champion_version": champion_version,
        }

    # Load model bundle (using compat unpickler to handle numpy 2.x → 1.x pickle differences)
    with open(model_path, "rb") as fh:
        bundle = _NumpyCompatUnpickler(fh).load()
    model = bundle["model"]
    scaler = bundle["scaler"]
    # Sklearn cross-version compat: 1.8.0 removed multi_class attribute but
    # 1.7.x predict_proba still reads self.multi_class. Patch if missing.
    if hasattr(model, "predict_proba") and not hasattr(model, "multi_class"):
        model.multi_class = "auto"
    feature_names = bundle.get(
        "feature_names", ["fwhm", "peak_height", "peak_area", "noise_level", "snr", "peak_center"]
    )

    # Build feature matrix
    import pandas as pd

    feature_rows = []
    valid_ids = []
    for r in records:
        row = {k: (r.get(k) or 0.0) for k in feature_names if k in r}
        if len(row) == len(feature_names):
            feature_rows.append(row)
            valid_ids.append(r["prediction_id"])

    if not feature_rows:
        print("⚠️ No complete feature rows found for re-scoring")
        return {"n_predictions": 0, "n_changed": 0, "change_rate": 0.0}

    X = pd.DataFrame(feature_rows)
    Xs = scaler.transform(X)
    new_labels = model.predict(Xs).tolist()
    new_confs = model.predict_proba(Xs).max(axis=1).tolist()

    # Compare with existing labels (from fetch_info records)
    existing_labels = {r["prediction_id"]: int(r.get("predicted_label") or -1) for r in records}
    n_changed = sum(
        1
        for pid, new_lbl in zip(valid_ids, new_labels)  # noqa: B905 (Python 3.8 compat)
        if existing_labels.get(pid, -1) != new_lbl
    )
    change_rate = n_changed / len(valid_ids) if valid_ids else 0.0

    print(
        f"{'[DRY RUN] ' if dry_run else ''}Re-scored {len(valid_ids)} predictions: "
        f"{n_changed} changed ({change_rate:.1%})"
    )

    if not dry_run:
        pg_url = os.environ.get("DATABASE_URL", "")
        updates = list(zip(new_labels, new_confs, [champion_version] * len(valid_ids), valid_ids))  # noqa: B905 (Python 3.8 compat)

        if pg_url:
            try:
                import psycopg2

                conn = psycopg2.connect(pg_url)
                with conn, conn.cursor() as cur:
                    cur.executemany(
                        "UPDATE predictions SET predicted_label=%s, prediction_confidence=%s, "
                        "model_version=%s WHERE prediction_id=%s",
                        updates,
                    )
                conn.close()
                print("✓ Updated predictions in PostgreSQL")
            except Exception as exc:
                print(f"⚠️ PostgreSQL update failed: {exc} — falling back to SQLite")
                pg_url = ""  # force SQLite path

        if not pg_url:
            db_path = _PROJECT_ROOT / "data" / "mlops.db"
            if db_path.exists():
                with contextlib.closing(sqlite3.connect(str(db_path))) as con:
                    con.executemany(
                        "UPDATE predictions SET predicted_label=?, prediction_confidence=?, "
                        "model_version=? WHERE prediction_id=?",
                        updates,
                    )
                    con.commit()
                print("✓ Updated predictions in SQLite")

    return {
        "n_predictions": len(valid_ids),
        "n_changed": n_changed,
        "change_rate": change_rate,
        "dry_run": dry_run,
        "champion_version": champion_version,
    }


def write_rescoring_audit(**context) -> dict:
    """Write a rescoring_runs audit record to the database."""
    import contextlib
    import sqlite3

    ti = context.get("task_instance")
    rescore_result = ti.xcom_pull(task_ids="rescore") if ti else {}
    _fetch_info = ti.xcom_pull(task_ids="fetch_predictions") if ti else {}

    n_predictions = (rescore_result or {}).get("n_predictions", 0)
    n_changed = (rescore_result or {}).get("n_changed", 0)
    change_rate = (rescore_result or {}).get("change_rate", 0.0)
    champion_version = (rescore_result or {}).get("champion_version", "unknown")
    dry_run = (rescore_result or {}).get("dry_run", False)
    dag_run = context.get("dag_run")
    triggered_by = getattr(dag_run, "run_type", "manual") if dag_run else "manual"

    status = "completed" if not dry_run else "completed"
    pg_url = os.environ.get("DATABASE_URL", "")
    audit_id = None

    if pg_url:
        try:
            import psycopg2

            conn = psycopg2.connect(pg_url)
            with conn, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO rescoring_runs
                        (model_version, n_predictions, n_changed, change_rate, triggered_by, status)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (champion_version, n_predictions, n_changed, change_rate, triggered_by, status),
                )
                audit_id = cur.fetchone()[0]
            conn.close()
            print(f"✓ Audit record written to PostgreSQL (id={audit_id})")
        except Exception as exc:
            print(f"⚠️ PostgreSQL audit write failed: {exc}")

    if audit_id is None:
        db_path = _PROJECT_ROOT / "data" / "mlops.db"
        if db_path.exists():
            with contextlib.closing(sqlite3.connect(str(db_path))) as con:
                cur2 = con.execute(
                    """
                    INSERT INTO rescoring_runs
                        (model_version, n_predictions, n_changed, change_rate, triggered_by, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (champion_version, n_predictions, n_changed, change_rate, triggered_by, status),
                )
                audit_id = cur2.lastrowid
                con.commit()
            print(f"✓ Audit record written to SQLite (id={audit_id})")

    return {"audit_id": audit_id, "n_predictions": n_predictions, "n_changed": n_changed}


# ──────────────────────────────────────────────────────────────────────────────
# DAG Definition
# ──────────────────────────────────────────────────────────────────────────────

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id=_DAG_ID,
    description="Batch re-scoring: re-run champion predictions on historical data after model promotion",
    default_args=default_args,
    schedule_interval=None,  # Manual trigger only
    start_date=datetime(2026, 4, 6),
    catchup=False,
    is_paused_upon_creation=False,
    tags=["rescoring", "batch", "champion-challenger"],
    max_active_runs=1,
    params={
        "lookback_days": Param(
            default=_DEFAULT_LOOKBACK_DAYS,
            type="integer",
            description="Number of days of historical predictions to re-score",
        ),
        "model_version": Param(
            default="champion",
            type="string",
            description="Model version to use for re-scoring (default: champion)",
        ),
        "dry_run": Param(
            default=False,
            type="boolean",
            description="If True, compute change stats without writing to the database",
        ),
    },
) as dag:
    load_task = PythonOperator(
        task_id="load_champion",
        python_callable=load_champion_model,
        provide_context=True,
    )

    fetch_task = PythonOperator(
        task_id="fetch_predictions",
        python_callable=fetch_predictions_to_rescore,
        provide_context=True,
    )

    rescore_task = PythonOperator(
        task_id="rescore",
        python_callable=rescore_predictions,
        provide_context=True,
    )

    audit_task = PythonOperator(
        task_id="write_audit",
        python_callable=write_rescoring_audit,
        provide_context=True,
    )

    load_task >> fetch_task >> rescore_task >> audit_task
