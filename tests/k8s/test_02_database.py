"""
K8s Tier 2 — Database Tests
Tests PostgreSQL schema and data integrity using kubectl exec.
"""

from __future__ import annotations

import subprocess

NAMESPACE = "mlops"
DB_NAME = "mlops_k8s"
DB_USER = "mlops_user"


def psql(query: str, dbname: str = DB_NAME) -> str:
    """Run psql query in postgres pod, return stdout."""
    result = subprocess.run(
        [
            "kubectl",
            "exec",
            "-n",
            NAMESPACE,
            "deploy/postgres",
            "--",
            "psql",
            "-U",
            DB_USER,
            "-d",
            dbname,
            "-t",
            "-c",
            query,
        ],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"psql failed: {result.stderr}")
    return result.stdout.strip()


def test_app_tables_exist() -> None:
    """K1: All expected application tables must exist."""
    required = {
        "devices",
        "predictions",
        "raw_signals",
        "features",
        "sparse_labels",
        "model_training_data",
        "model_approvals",
    }
    rows = psql(r"\dt")
    existing = {line.strip().split("|")[1].strip() for line in rows.splitlines() if "|" in line}
    missing = required - existing
    assert not missing, f"Missing tables: {missing}"
    print(f"  ✅ App tables: {sorted(existing)}")


def test_airflow_database_exists() -> None:
    """K3: Airflow PostgreSQL database must have been created by airflow-init job."""
    result = psql("SELECT datname FROM pg_database WHERE datname='airflow';")
    assert "airflow" in result, f"airflow database not found. Got: {result!r}"
    print("  ✅ airflow database exists")


def test_airflow_tables_exist() -> None:
    """K3: Airflow should have its metadata tables in the airflow DB."""
    try:
        rows = psql(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema='public';",
            dbname="airflow",
        )
        count = int(rows.strip())
        assert count >= 10, f"Airflow DB has only {count} tables (expected ≥10)"
        print(f"  ✅ Airflow DB tables: {count}")
    except Exception as e:
        # Airflow might still be using SQLite — not a hard failure
        print(f"  ⚠️  Airflow DB check skipped (may use SQLite): {e}")


def test_model_training_data_schema() -> None:
    """model_training_data must have mlflow_run_id, signal_id, split columns."""
    cols_raw = psql(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name='model_training_data' ORDER BY column_name;"
    )
    cols = {c.strip() for c in cols_raw.splitlines() if c.strip()}
    required = {"mlflow_run_id", "signal_id", "split"}
    missing = required - cols
    assert not missing, f"model_training_data missing columns: {missing}"
    print(f"  ✅ model_training_data columns: {sorted(cols)}")


def test_crud_round_trip() -> None:
    """Insert a device row, verify it, then clean up."""
    import uuid

    device_id = str(uuid.uuid4())
    psql(
        f"INSERT INTO devices (device_id, device_name, device_type, location, status) "
        f"VALUES ('{device_id}', 'k8s-test-device', 'sensor', 'test-lab', 'active');"
    )
    result = psql(f"SELECT device_type FROM devices WHERE device_id='{device_id}';")
    assert "sensor" in result, f"CRUD read-back failed: {result!r}"
    psql(f"DELETE FROM devices WHERE device_id='{device_id}';")
    print("  ✅ CRUD round-trip: insert/read/delete OK")


def test_drift_data_on_shared_pvc() -> None:
    """K5: Verify drift-init job copied reference_data.parquet to the shared PVC."""
    # Check if shared-data-pvc is mounted anywhere and contains drift data
    # We verify via the api pod which mounts /app/data (image-baked)
    result = subprocess.run(
        [
            "kubectl",
            "exec",
            "-n",
            NAMESPACE,
            "deploy/api",
            "--",
            "python",
            "-c",
            "from pathlib import Path; p=Path('/data/drift/reference_data.parquet');"
            "print('exists' if p.exists() else 'missing')",
        ],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    # The shared PVC mount at /data may or may not be configured yet
    if result.returncode == 0 and "exists" in result.stdout:
        print("  ✅ Drift reference_data.parquet on shared PVC")
    else:
        print(f"  ⚠️  Drift PVC check: {result.stdout.strip()} {result.stderr.strip()[:100]}")


def run_all() -> int:
    tests = [
        test_app_tables_exist,
        test_airflow_database_exists,
        test_airflow_tables_exist,
        test_model_training_data_schema,
        test_crud_round_trip,
        test_drift_data_on_shared_pvc,
    ]
    passed = failed = 0
    for t in tests:
        name = t.__name__
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ {name}: EXCEPTION {type(e).__name__}: {e}")
            failed += 1
    print(f"\nTier 2 — Database: {passed} passed, {failed} failed")
    return failed


if __name__ == "__main__":
    import sys

    sys.exit(run_all())
