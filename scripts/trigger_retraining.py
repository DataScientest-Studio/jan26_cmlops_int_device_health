#!/usr/bin/env python
"""
UC-4: Automated Retraining Trigger.

Simulates the conditions that trigger automated retraining:
  1. Checks for labelled samples (needs ≥ 100 for training)
  2. Checks recent prediction accuracy (< 85% triggers retraining)
  3. If conditions met, invokes the training pipeline locally

When the full Docker stack is available, this script also attempts to
trigger the Airflow 'automated_retraining' DAG via the REST API.

Usage:
    python scripts/trigger_retraining.py
    python scripts/trigger_retraining.py --force          # skip threshold checks
    python scripts/trigger_retraining.py --check-only     # only report status
    python scripts/trigger_retraining.py --airflow-url http://localhost:8081
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _check_database_conditions() -> dict[str, object]:
    """Query the database for retraining trigger conditions."""
    result: dict[str, object] = {
        "labelled_count": 0,
        "recent_accuracy": None,
        "should_retrain": False,
        "reason": "",
    }
    try:
        from src.database.database import Database

        db_url = os.environ.get("DATABASE_URL", "")
        pg_host = os.environ.get("POSTGRES_HOST", "")
        if db_url and db_url.startswith("postgresql"):
            db = Database(db_url=db_url)
        elif pg_host:
            user = os.environ.get("POSTGRES_USER", "mlops_user")
            pw = os.environ.get("POSTGRES_PASSWORD", "changeme")
            port = os.environ.get("POSTGRES_PORT", "5432")
            dbname = os.environ.get("POSTGRES_DB", "mlops_db")
            db = Database(db_url=f"postgresql://{user}:{pw}@{pg_host}:{port}/{dbname}")
        else:
            db_path = PROJECT_ROOT / "data" / "database" / "mlops.db"
            db = Database(db_path=str(db_path))

        cursor = db.conn.cursor()

        # Count labelled predictions
        cursor.execute("SELECT COUNT(*) AS n FROM predictions WHERE ground_truth_label IS NOT NULL")
        row = cursor.fetchone()
        labelled_count = int(row["n"] if row else 0)
        result["labelled_count"] = labelled_count

        # Compute realized accuracy
        cursor.execute(
            """
            SELECT
                SUM(CASE WHEN predicted_label = ground_truth_label THEN 1 ELSE 0 END) * 100.0
                    / COUNT(*) AS accuracy
            FROM predictions
            WHERE ground_truth_label IS NOT NULL
            """
        )
        row = cursor.fetchone()
        accuracy = float(row["accuracy"]) if row and row["accuracy"] is not None else None
        result["recent_accuracy"] = accuracy

        db.close()

        # Evaluate trigger conditions
        if labelled_count < 100:
            result["should_retrain"] = False
            result["reason"] = f"Only {labelled_count} labelled samples (need ≥ 100)."
        elif accuracy is not None and accuracy < 85.0:
            result["should_retrain"] = True
            result["reason"] = (
                f"Accuracy {accuracy:.1f}% < 85% threshold with {labelled_count} labels."
            )
        else:
            result["should_retrain"] = False
            result["reason"] = (
                f"Accuracy {accuracy:.1f}% ≥ 85% — no retraining needed."
                if accuracy is not None
                else "No accuracy data."
            )

    except Exception as exc:
        result["reason"] = f"Could not query database: {exc}"

    return result


def _export_production_training_data(
    tmp_path: Path,
    lookback_days: int = 90,
    min_samples: int = 100,
) -> Path | None:
    """Export recent labeled production signals from the DB into a training JSON.

    The resulting file has the same schema as ``dataset_baseline_full.json``::

        {"n_samples": N, "signals": [{"id": ..., "time": [...], "amplitude": [...],
                                       "shape_type": ..., "metadata": {}, "label": 0|1}]}

    Returns the file path when ≥ ``min_samples`` labeled signals are available,
    or ``None`` to fall back to the static baseline file.
    """
    import json as _json
    import os as _os

    try:
        from src.database.database import Database

        db_url = _os.environ.get("DATABASE_URL", "")
        pg_host = _os.environ.get("POSTGRES_HOST", "")
        if db_url and db_url.startswith("postgresql"):
            db = Database(db_url=db_url)
        elif pg_host:
            user = _os.environ.get("POSTGRES_USER", "mlops_user")
            pw = _os.environ.get("POSTGRES_PASSWORD", "changeme")
            port = _os.environ.get("POSTGRES_PORT", "5432")
            dbname = _os.environ.get("POSTGRES_DB", "mlops_db")
            db = Database(db_url=f"postgresql://{user}:{pw}@{pg_host}:{port}/{dbname}")
        else:
            # OrbStack default
            db = Database(
                db_url="postgresql://mlops_user:local_dev_password@mlops_postgres.orb.local:5432/mlops_db"
            )

        cursor = db.conn.cursor()
        cursor.execute(
            """
            SELECT
                s.signal_id,
                p.prediction_id,
                p.device_id,
                s.time_values,
                s.amplitude_values,
                s.shape_type,
                p.ground_truth_label
            FROM predictions p
            JOIN raw_signals s ON p.prediction_id = s.prediction_id
            WHERE p.ground_truth_label IS NOT NULL
              AND p.created_at >= NOW() - INTERVAL '%s days'
            ORDER BY p.created_at DESC
            LIMIT 5000
            """,
            (lookback_days,),
        )
        rows = cursor.fetchall()

        if len(rows) < min_samples:
            print(
                f"[INFO] Only {len(rows)} labeled production samples in last"
                f" {lookback_days} days (need {min_samples}) — using baseline data."
            )
            return None

        signals = []
        for row in rows:
            try:
                time_vals = _json.loads(row[3]) if isinstance(row[3], str) else row[3]
                amp_vals = _json.loads(row[4]) if isinstance(row[4], str) else row[4]
                label = int(row[6]) if row[6] is not None else 0
                signals.append(
                    {
                        "id": str(row[1]),
                        "time": time_vals,
                        "amplitude": amp_vals,
                        "shape_type": row[5] or "unknown",
                        "metadata": {"device_id": str(row[2]), "source": "production"},
                        "label": label,
                    }
                )
            except Exception:
                continue

        if len(signals) < min_samples:
            print(
                f"[INFO] Successfully parsed {len(signals)} signals (need {min_samples})"
                " — using baseline data."
            )
            return None

        out_path = tmp_path / "production_training_data.json"
        with open(out_path, "w") as f:
            _json.dump({"n_samples": len(signals), "signals": signals}, f)

        print(
            f"[INFO] Exported {len(signals)} labeled production signals"
            f" (last {lookback_days} days) → {out_path}"
        )
        return out_path

    except Exception as exc:
        print(f"[WARN] Could not export production data: {exc} — using baseline data.")
        return None


def _run_local_training() -> int:
    """Call the training pipeline directly (no Airflow needed).

    Trains with MLflow enabled (pointing to the local Docker MLflow server on
    port 5001) so the trained model is registered in the Model Registry.
    The first trained model is auto-promoted to Production; subsequent runs
    produce a Staging challenger for UC-05 champion/challenger comparison.

    Training data priority:
      1. Recent labeled production signals from the database (rolling window)
         → reflects real drift; challenger may score differently than champion
      2. Static baseline file ``data/raw/dataset_baseline_full.json`` (fallback)
    """
    import tempfile

    print("\n[INFO] Running local training pipeline…")

    # ── 1. Try production data first (rolling window) ──────────────────────
    with tempfile.TemporaryDirectory() as _tmp:
        _tmp_path = Path(_tmp)
        prod_data_path = _export_production_training_data(
            _tmp_path, lookback_days=90, min_samples=100
        )
        if prod_data_path:
            data_path = prod_data_path
            print(f"[INFO] Training on production data: {data_path}")
        else:
            # ── 2. Fall back to static baseline ────────────────────────────
            data_path = PROJECT_ROOT / "data" / "raw" / "dataset_baseline_full.json"
            if not data_path.exists():
                print(f"[WARN] Training data not found at {data_path}.")
                print(
                    "       Run: python scripts/generate_data.py generate"
                    " --output-dir data/raw --drift-scenario baseline"
                )

                # Try to generate it
                import subprocess

                gen_result = subprocess.run(
                    [
                        sys.executable,
                        "scripts/generate_data.py",
                        "generate",
                        "--output-dir",
                        str(data_path.parent),
                        "--drift-scenario",
                        "baseline",
                    ],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                )
                if gen_result.returncode != 0:
                    print(f"[ERROR] Data generation failed: {gen_result.stderr}")
                    return 1
                print("[INFO] Data generated successfully.")
            print(f"[INFO] Training on baseline data: {data_path}")

        return _do_train(data_path)


def _do_train(data_path: Path) -> int:
    """Run the actual training + MLflow registration.  Shared by both paths."""
    import os as _os

    import mlflow

    from src.training.train import train_model

    # ── Determine MLflow tracking URI ─────────────────────────────────────
    # Cloud mode: use DagsHub tracking URI from env (MLFLOW_TRACKING_URI will
    # be set to https://dagshub.com/<user>/<repo>.mlflow).
    # Local mode: use localhost:5001 (local Docker container).
    # Fall back to localhost:5001 when the env var is empty.
    _raw_uri = _os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    _current_mode_file = PROJECT_ROOT / ".current_mode"
    _current_mode = (
        _current_mode_file.read_text().strip() if _current_mode_file.exists() else "local"
    )

    if _current_mode == "cloud" and _raw_uri and _raw_uri.startswith("https://"):
        # Cloud mode with a valid DagsHub URI — also forward credentials
        _mlflow_uri = _raw_uri
        _dagshub_user = _os.environ.get("MLFLOW_TRACKING_USERNAME", "") or _os.environ.get(
            "DAGSHUB_USER", ""
        )
        _dagshub_token = _os.environ.get("MLFLOW_TRACKING_PASSWORD", "") or _os.environ.get(
            "DAGSHUB_TOKEN", ""
        )
        if _dagshub_user and _dagshub_token:
            _os.environ["MLFLOW_TRACKING_USERNAME"] = _dagshub_user
            _os.environ["MLFLOW_TRACKING_PASSWORD"] = _dagshub_token
            print(f"[INFO] Cloud mode — DagsHub MLflow as {_dagshub_user}")
    else:
        # Local mode or fallback
        _mlflow_uri = "http://localhost:5001"

    mlflow.set_tracking_uri(_mlflow_uri)
    print(f"[INFO] MLflow tracking URI: {_mlflow_uri}")

    try:
        model_output_path = PROJECT_ROOT / "models" / "retrained_model.pkl"
        result = train_model(
            train_data_path=str(data_path),
            model_output_path=str(model_output_path),
            use_mlflow=True,
            model_version="retrained-local",
        )
        saved_path = result.get("model_path", str(model_output_path))
        print(f"[OK]   Training complete — model saved to {saved_path}")
        # Log validation set summary
        if result.get("gold_standard_path"):
            print(f"[OK]   Labeled validation set (gold standard): {result['gold_standard_path']}")
        if result.get("test_accuracy") is not None:
            print(f"[OK]   Validation accuracy: {result['test_accuracy']:.4f}")
        if result.get("test_f1_score") is not None:
            print(f"[OK]   Validation F1-score: {result['test_f1_score']:.4f}")

        # Register the model in MLflow and auto-assign stage
        run_id = result.get("mlflow_run_id")
        if run_id:
            from mlflow.tracking import MlflowClient

            from src.training.registry import (
                get_production_models,
                promote_model,
            )

            client = MlflowClient()
            model_name = os.environ.get("MODEL_REGISTRY_NAME", "device_health_classifier")

            # Ensure the registered model exists
            try:
                client.get_registered_model(model_name)
            except Exception:
                client.create_registered_model(model_name)

            # Register model version using a runs:/ URI. The artifact file
            # is not actually stored in MLflow's artifact store (the store is
            # inside Docker and not reachable from the local machine), but the
            # registry entry is still created so UC-05 promotion and UC-11
            # lineage queries work correctly.
            mv = client.create_model_version(
                name=model_name,
                source=f"runs:/{run_id}/model",
                run_id=run_id,
                description=f"Local retrained model — run {run_id[:8]}",
            )
            version = int(mv.version)
            print(f"[OK]   Registered as {model_name} v{version}")

            # If no Production model exists yet, promote this one directly to
            # Production as the initial baseline.  Otherwise put it in Staging
            # for UC-05 champion/challenger comparison.
            existing_prod = get_production_models(model_name)
            stage = "Production" if not existing_prod else "Staging"
            promote_model(model_name, version=version, stage=stage)
            print(f"[OK]   Promoted to {stage} ({model_name} v{version})")
        else:
            print("[WARN] No MLflow run_id returned — skipping model registration.")

        return 0
    except Exception as exc:
        print(f"[ERROR] Training failed: {exc}")
        return 1


def _trigger_airflow_dag(airflow_url: str) -> bool:
    """Trigger Airflow DAG via REST API (requires basic auth admin/admin)."""
    try:
        import base64
        import json
        import urllib.request

        url = f"{airflow_url.rstrip('/')}/api/v1/dags/automated_retraining/dagRuns"
        data = json.dumps({"conf": {"triggered_by": "trigger_retraining_script"}}).encode()
        auth = base64.b64encode(b"admin:admin").decode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "Authorization": f"Basic {auth}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            body = json.loads(resp.read())
            run_id = body.get("dag_run_id", "?")
            print(f"[OK]   Airflow DAG triggered — run_id: {run_id}")
            return True
    except Exception as exc:
        print(f"[WARN] Could not trigger Airflow DAG: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="UC-4: Automated Retraining Trigger")
    parser.add_argument(
        "--force", action="store_true", help="Skip threshold checks and retrain immediately."
    )
    parser.add_argument(
        "--check-only", action="store_true", help="Report conditions without triggering retraining."
    )
    parser.add_argument(
        "--airflow-url", default="http://localhost:8081", help="Airflow web UI URL."
    )
    args = parser.parse_args()

    print("UC-4: Automated Retraining Trigger")
    print("=" * 50)

    # Step 1 — Check database conditions
    print("\n[1/3] Checking database conditions…")
    conds = _check_database_conditions()
    print(f"      Labelled samples : {conds['labelled_count']}")
    print(f"      Recent accuracy  : {conds['recent_accuracy']}")
    print(f"      Should retrain   : {conds['should_retrain']}")
    print(f"      Reason           : {conds['reason']}")

    if args.check_only:
        return 0

    should_retrain = args.force or conds["should_retrain"]

    if not should_retrain:
        print("\n[INFO] Retraining conditions not met.  Use --force to override.")
        return 0

    # Step 2 — Try Airflow first
    print(f"\n[2/3] Attempting Airflow DAG trigger at {args.airflow_url}…")
    airflow_ok = _trigger_airflow_dag(args.airflow_url)

    if not airflow_ok:
        # Step 3 — Fall back to local training
        print("\n[3/3] Falling back to local training pipeline…")
        return _run_local_training()

    print("\n[3/3] Airflow DAG triggered — monitor progress at the Airflow UI.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
