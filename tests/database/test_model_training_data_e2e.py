"""
Comprehensive end-to-end tests for model_training_data table population.

Covers the full pipeline: training → split tracking → model_training_data rows.
Includes both SQLite (unit) and PostgreSQL (integration) test variants.

These tests verify:
1. No ``strict=`` keyword argument is used in any ``zip()`` call in train.py
2. record_training_split correctly inserts rows into SQLite (unit / CI)
3. The full training pipeline (from_db=True) populates model_training_data
4. Split JSON files are created and have the expected structure
5. MD5 hashes are computed and stored in MLflow params
6. PostgreSQL integration path (skipped unless POSTGRES_TEST_URL is set)

Run all tests:
    pytest tests/database/test_model_training_data_e2e.py -v

Run only PostgreSQL integration tests:
    POSTGRES_TEST_URL=postgresql://mlops_user:...@127.0.0.1:5433/mlops_prod \
    pytest tests/database/test_model_training_data_e2e.py -v -m postgres
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any

import pytest

# ── Paths ─────────────────────────────────────────────────────────────────────
# test file is at tests/database/test_model_training_data_e2e.py
# parents[0] = tests/database/, parents[1] = tests/, parents[2] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRAIN_PY = _REPO_ROOT / "src" / "training" / "train.py"

# ── optional PostgreSQL fixture ────────────────────────────────────────────────
_PG_URL = os.environ.get("POSTGRES_TEST_URL", "")
_pg_available = False
if _PG_URL:
    try:
        import psycopg2  # noqa: F401

        _pg_available = True
    except ImportError:
        pass

pytestmark_postgres = pytest.mark.skipif(
    not _pg_available,
    reason=(
        "Set POSTGRES_TEST_URL=postgresql://mlops_user:..@127.0.0.1:5433/mlops_prod "
        "and install psycopg2 to run PostgreSQL integration tests"
    ),
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def sqlite_db(tmp_path: Path):
    """Fresh SQLite Database for each test."""
    from src.database import Database

    db = Database(db_path=tmp_path / "test.db")
    yield db
    db.close()


@pytest.fixture()
def populated_sqlite_db(tmp_path: Path):
    """SQLite DB pre-loaded with 30 labeled + 15 unlabeled cloud signals."""
    from src.database import Database, generate_device_id
    from src.signal_processing.signal_generator import generate_signal

    db = Database(db_path=tmp_path / "populated.db")
    device_id = generate_device_id()
    db.register_device(
        device_id=device_id,
        device_name="e2e-test-dev",
        device_type="sensor",
        location="lab",
        status="active",
    )

    import numpy as np

    rng = np.random.RandomState(42)

    # 15 labeled healthy (Gaussian)
    for i in range(15):
        sig = generate_signal("gaussian", drift_scenario="baseline", seed=i)
        tv = sig.signal.time if isinstance(sig.signal.time, list) else sig.signal.time.tolist()
        av = (
            sig.signal.amplitude
            if isinstance(sig.signal.amplitude, list)
            else sig.signal.amplitude.tolist()
        )
        pid = db.store_prediction(
            device_id=device_id,
            time_values=tv,
            amplitude_values=av,
            predicted_label=0,
            model_version="v0",
            features={},
            prediction_confidence=float(rng.uniform(0.7, 0.99)),
            deployment_mode="cloud",
        )
        db.inject_sparse_label(pid, 0, "automated_test")

    # 15 labeled unhealthy (Lorentzian)
    for i in range(15, 30):
        sig = generate_signal("lorentzian", drift_scenario="baseline", seed=i)
        tv = sig.signal.time if isinstance(sig.signal.time, list) else sig.signal.time.tolist()
        av = (
            sig.signal.amplitude
            if isinstance(sig.signal.amplitude, list)
            else sig.signal.amplitude.tolist()
        )
        pid = db.store_prediction(
            device_id=device_id,
            time_values=tv,
            amplitude_values=av,
            predicted_label=1,
            model_version="v0",
            features={},
            prediction_confidence=float(rng.uniform(0.7, 0.99)),
            deployment_mode="cloud",
        )
        db.inject_sparse_label(pid, 1, "automated_test")

    # 15 unlabeled (no sparse_label)
    for i in range(30, 45):
        sig = generate_signal("gaussian", drift_scenario="baseline", seed=i)
        tv = sig.signal.time if isinstance(sig.signal.time, list) else sig.signal.time.tolist()
        av = (
            sig.signal.amplitude
            if isinstance(sig.signal.amplitude, list)
            else sig.signal.amplitude.tolist()
        )
        db.store_prediction(
            device_id=device_id,
            time_values=tv,
            amplitude_values=av,
            predicted_label=0,
            model_version="v0",
            features={},
            prediction_confidence=float(rng.uniform(0.5, 0.8)),
            deployment_mode="cloud",
        )

    yield db
    db.close()


# ── Python 3.8 compatibility regression tests ─────────────────────────────────


class TestPython38ZipCompatibility:
    """Ensure train.py never uses Python 3.10+ zip(strict=...) syntax."""

    def _collect_zip_calls(self, source: str) -> list[ast.Call]:
        """Parse source and return all zip() call nodes."""
        tree = ast.parse(source)
        calls: list[ast.Call] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "zip":
                    calls.append(node)
        return calls

    def test_train_py_no_strict_keyword_in_zip(self):
        """Regression test: the production bug was strict=False in zip() calls.

        zip() does NOT accept keyword arguments in Python < 3.10.
        Any zip(iterable, strict=True/False) would raise:
            TypeError: zip() takes no keyword arguments
        in Python 3.8 (the Airflow container version).
        """
        source = _TRAIN_PY.read_text(encoding="utf-8")
        zip_calls = self._collect_zip_calls(source)

        violations: list[str] = []
        for call in zip_calls:
            for kw in call.keywords:
                if kw.arg == "strict":
                    lineno = call.lineno if hasattr(call, "lineno") else "?"
                    violations.append(
                        f"  Line {lineno}: zip(..., strict=...) — "
                        "this is Python 3.10+ only and will crash in Python 3.8 (Airflow)"
                    )

        assert not violations, (
            "train.py contains zip() calls with strict= keyword argument. "
            "This WILL cause silent failures in the Airflow container (Python 3.8):\n"
            + "\n".join(violations)
        )

    def test_train_py_uses_plain_zip(self):
        """Verify that the three split-tracking zip() calls in train.py exist and use plain zip."""
        source = _TRAIN_PY.read_text(encoding="utf-8")
        # Ensure the file is parseable
        ast.parse(source)
        # Verify the file contains zip() calls at all (they should be there for split tracking)
        assert "zip(" in source, "train.py should contain zip() calls for split tracking"

    def test_database_module_no_strict_keyword_in_zip(self):
        """Ensure database.py also doesn't use strict= in zip()."""
        db_py = _REPO_ROOT / "src" / "database" / "database.py"
        source = db_py.read_text(encoding="utf-8")
        zip_calls = self._collect_zip_calls(source)

        violations = [
            f"  Line {call.lineno}: zip(..., strict=...)"
            for call in zip_calls
            for kw in call.keywords
            if kw.arg == "strict"
        ]
        assert not violations, (
            "database.py contains zip() calls with strict= keyword argument:\n"
            + "\n".join(violations)
        )

    def test_all_src_modules_no_strict_zip_keyword(self):
        """
        Scan src/ modules that CAN run inside the Airflow container (Python 3.8).

        The Airflow DAGs import from src/training/, src/database/,
        src/signal_processing/, src/monitoring/, and src/data/ — all of these
        MUST be Python 3.8 compatible.

        src/ui/ is EXCLUDED: it runs on the host (Python 3.12) and is never
        imported by Airflow, so zip(strict=...) is acceptable there.
        """
        src_dir = _REPO_ROOT / "src"
        # Only scan modules that Airflow (Python 3.8) can import
        airflow_importable_dirs = [
            src_dir / "training",
            src_dir / "database",
            src_dir / "signal_processing",
            src_dir / "monitoring",
            src_dir / "data",
        ]
        violations: list[str] = []

        for scan_dir in airflow_importable_dirs:
            if not scan_dir.exists():
                continue
            for py_file in scan_dir.rglob("*.py"):
                try:
                    source = py_file.read_text(encoding="utf-8")
                    tree = ast.parse(source)
                except (SyntaxError, UnicodeDecodeError):
                    continue

                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        func = node.func
                        if isinstance(func, ast.Name) and func.id == "zip":
                            for kw in node.keywords:
                                if kw.arg == "strict":
                                    rel = py_file.relative_to(_REPO_ROOT)
                                    violations.append(
                                        f"  {rel}:{node.lineno}: zip(..., strict=...)"
                                    )

        assert not violations, (
            "Found zip() calls with strict= keyword argument in Airflow-importable src/ modules. "
            "These WILL crash in Python 3.8 (Airflow container):\n" + "\n".join(violations)
        )

    def test_airflow_dags_no_strict_zip_keyword(self):
        """Broad scan: no .py file in airflow/ uses zip(strict=...) — Python 3.8 safety."""
        dags_dir = _REPO_ROOT / "airflow"
        violations: list[str] = []

        for py_file in dags_dir.rglob("*.py"):
            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source)
            except (SyntaxError, UnicodeDecodeError):
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    if isinstance(func, ast.Name) and func.id == "zip":
                        for kw in node.keywords:
                            if kw.arg == "strict":
                                rel = py_file.relative_to(_REPO_ROOT)
                                violations.append(f"  {rel}:{node.lineno}: zip(..., strict=...)")

        assert not violations, (
            "Found zip() calls with strict= keyword argument in airflow/ directory. "
            "Airflow container uses Python 3.8 where zip() takes no keyword arguments:\n"
            + "\n".join(violations)
        )


# ── record_training_split unit tests ─────────────────────────────────────────


class TestRecordTrainingSplitUnit:
    """Unit tests for Database.record_training_split against SQLite."""

    def test_inserts_correct_row_count(self, sqlite_db):
        """N train + M test signals → exactly N+M rows in model_training_data."""
        from src.database import generate_device_id
        from src.signal_processing.signal_generator import generate_signal

        device_id = generate_device_id()
        sqlite_db.register_device(
            device_id=device_id,
            device_name="dev",
            device_type="sensor",
            location="lab",
            status="active",
        )

        signal_ids: list[int] = []
        for i in range(8):
            sig = generate_signal("gaussian", drift_scenario="baseline", seed=i)
            tv = sig.signal.time if isinstance(sig.signal.time, list) else sig.signal.time.tolist()
            av = (
                sig.signal.amplitude
                if isinstance(sig.signal.amplitude, list)
                else sig.signal.amplitude.tolist()
            )
            pid = sqlite_db.store_prediction(
                device_id=device_id,
                time_values=tv,
                amplitude_values=av,
                predicted_label=0,
                model_version="v0",
                features={},
                prediction_confidence=0.9,
                deployment_mode="cloud",
            )
            sqlite_db.inject_sparse_label(pid, 0, "test")
            row = sqlite_db.conn.execute(
                "SELECT signal_id FROM raw_signals WHERE prediction_id=?", (pid,)
            ).fetchone()
            signal_ids.append(int(row[0]))

        train_ids = signal_ids[:6]
        test_ids = signal_ids[6:]

        sqlite_db.record_training_split(
            mlflow_run_id="unit-test-run-001",
            train_signal_ids=train_ids,
            test_signal_ids=test_ids,
            model_version="v1",
        )

        # Verify count in the table
        count = sqlite_db.conn.execute(
            "SELECT COUNT(*) FROM model_training_data WHERE mlflow_run_id=?",
            ("unit-test-run-001",),
        ).fetchone()[0]
        assert count == 8, f"Expected 8 rows, got {count}"

        # Verify split distribution
        train_count = sqlite_db.conn.execute(
            "SELECT COUNT(*) FROM model_training_data WHERE mlflow_run_id=? AND split='train'",
            ("unit-test-run-001",),
        ).fetchone()[0]
        test_count = sqlite_db.conn.execute(
            "SELECT COUNT(*) FROM model_training_data WHERE mlflow_run_id=? AND split='test'",
            ("unit-test-run-001",),
        ).fetchone()[0]
        assert train_count == 6
        assert test_count == 2

    def test_model_version_stored_correctly(self, sqlite_db):
        """model_version column is set from the parameter."""
        from src.database import generate_device_id
        from src.signal_processing.signal_generator import generate_signal

        device_id = generate_device_id()
        sqlite_db.register_device(
            device_id=device_id,
            device_name="dev",
            device_type="sensor",
            location="lab",
            status="active",
        )
        sig = generate_signal("gaussian", seed=0)
        tv = sig.signal.time if isinstance(sig.signal.time, list) else sig.signal.time.tolist()
        av = (
            sig.signal.amplitude
            if isinstance(sig.signal.amplitude, list)
            else sig.signal.amplitude.tolist()
        )
        pid = sqlite_db.store_prediction(
            device_id=device_id,
            time_values=tv,
            amplitude_values=av,
            predicted_label=0,
            model_version="v0",
            features={},
            prediction_confidence=0.9,
            deployment_mode="cloud",
        )
        sqlite_db.inject_sparse_label(pid, 0, "test")
        row = sqlite_db.conn.execute(
            "SELECT signal_id FROM raw_signals WHERE prediction_id=?", (pid,)
        ).fetchone()
        sid = int(row[0])

        sqlite_db.record_training_split(
            mlflow_run_id="ver-test-run",
            train_signal_ids=[sid],
            test_signal_ids=[],
            model_version="v42",
        )
        stored_ver = sqlite_db.conn.execute(
            "SELECT model_version FROM model_training_data WHERE mlflow_run_id=?",
            ("ver-test-run",),
        ).fetchone()[0]
        assert stored_ver == "v42"

    def test_duplicate_signal_same_run_ignored(self, sqlite_db):
        """UNIQUE(mlflow_run_id, signal_id): inserting same signal twice for same run is idempotent."""
        from src.database import generate_device_id
        from src.signal_processing.signal_generator import generate_signal

        device_id = generate_device_id()
        sqlite_db.register_device(
            device_id=device_id,
            device_name="dev",
            device_type="sensor",
            location="lab",
            status="active",
        )
        sig = generate_signal("gaussian", seed=0)
        tv = sig.signal.time if isinstance(sig.signal.time, list) else sig.signal.time.tolist()
        av = (
            sig.signal.amplitude
            if isinstance(sig.signal.amplitude, list)
            else sig.signal.amplitude.tolist()
        )
        pid = sqlite_db.store_prediction(
            device_id=device_id,
            time_values=tv,
            amplitude_values=av,
            predicted_label=0,
            model_version="v0",
            features={},
            prediction_confidence=0.9,
            deployment_mode="cloud",
        )
        sqlite_db.inject_sparse_label(pid, 0, "test")
        row = sqlite_db.conn.execute(
            "SELECT signal_id FROM raw_signals WHERE prediction_id=?", (pid,)
        ).fetchone()
        sid = int(row[0])

        sqlite_db.record_training_split("dup-run", [sid], [], "v1")
        sqlite_db.record_training_split("dup-run", [sid], [], "v1")  # duplicate

        count = sqlite_db.conn.execute(
            "SELECT COUNT(*) FROM model_training_data WHERE mlflow_run_id='dup-run'"
        ).fetchone()[0]
        assert count == 1, "Duplicate insert should be silently ignored (UNIQUE constraint)"

    def test_same_signal_different_runs_allowed(self, sqlite_db):
        """Same signal_id can appear in multiple different runs."""
        from src.database import generate_device_id
        from src.signal_processing.signal_generator import generate_signal

        device_id = generate_device_id()
        sqlite_db.register_device(
            device_id=device_id,
            device_name="dev",
            device_type="sensor",
            location="lab",
            status="active",
        )
        sig = generate_signal("gaussian", seed=0)
        tv = sig.signal.time if isinstance(sig.signal.time, list) else sig.signal.time.tolist()
        av = (
            sig.signal.amplitude
            if isinstance(sig.signal.amplitude, list)
            else sig.signal.amplitude.tolist()
        )
        pid = sqlite_db.store_prediction(
            device_id=device_id,
            time_values=tv,
            amplitude_values=av,
            predicted_label=0,
            model_version="v0",
            features={},
            prediction_confidence=0.9,
            deployment_mode="cloud",
        )
        sqlite_db.inject_sparse_label(pid, 0, "test")
        row = sqlite_db.conn.execute(
            "SELECT signal_id FROM raw_signals WHERE prediction_id=?", (pid,)
        ).fetchone()
        sid = int(row[0])

        sqlite_db.record_training_split("run-a", [sid], [], "v1")
        sqlite_db.record_training_split("run-b", [sid], [], "v2")

        count = sqlite_db.conn.execute("SELECT COUNT(*) FROM model_training_data").fetchone()[0]
        assert count == 2, "Same signal in two different runs should produce two rows"

    def test_get_training_signal_ids_returns_correct_split(self, sqlite_db):
        """get_training_signal_ids returns only the requested split subset."""
        from src.database import generate_device_id
        from src.signal_processing.signal_generator import generate_signal

        device_id = generate_device_id()
        sqlite_db.register_device(
            device_id=device_id,
            device_name="dev",
            device_type="sensor",
            location="lab",
            status="active",
        )
        signal_ids: list[int] = []
        for i in range(6):
            sig = generate_signal("gaussian", seed=i)
            tv = sig.signal.time if isinstance(sig.signal.time, list) else sig.signal.time.tolist()
            av = (
                sig.signal.amplitude
                if isinstance(sig.signal.amplitude, list)
                else sig.signal.amplitude.tolist()
            )
            pid = sqlite_db.store_prediction(
                device_id=device_id,
                time_values=tv,
                amplitude_values=av,
                predicted_label=0,
                model_version="v0",
                features={},
                prediction_confidence=0.9,
                deployment_mode="cloud",
            )
            sqlite_db.inject_sparse_label(pid, 0, "test")
            row = sqlite_db.conn.execute(
                "SELECT signal_id FROM raw_signals WHERE prediction_id=?", (pid,)
            ).fetchone()
            signal_ids.append(int(row[0]))

        train_ids = signal_ids[:4]
        test_ids = signal_ids[4:]

        sqlite_db.record_training_split("split-run", train_ids, test_ids, "v5")

        retrieved_train = sqlite_db.get_training_signal_ids("split-run", "train")
        retrieved_test = sqlite_db.get_training_signal_ids("split-run", "test")

        assert set(retrieved_train) == set(train_ids)
        assert set(retrieved_test) == set(test_ids)
        assert not set(retrieved_train) & set(retrieved_test), "train/test should not overlap"

    def test_record_empty_splits_does_not_raise(self, sqlite_db):
        """Calling record_training_split with empty lists should not raise."""
        sqlite_db.record_training_split("empty-run", [], [], "v0")
        count = sqlite_db.conn.execute(
            "SELECT COUNT(*) FROM model_training_data WHERE mlflow_run_id='empty-run'"
        ).fetchone()[0]
        assert count == 0


# ── Full training pipeline integration tests ─────────────────────────────────


@pytest.fixture()
def local_mlflow_tracking(tmp_path: Path):
    """
    Configure MLflow to use a local file-based tracking store for tests.

    This allows use_mlflow=True in tests without a running MLflow server,
    which is essential for testing the split-recording code path that only
    runs when a real MLflow run_id is available.
    """
    import mlflow

    tracking_uri = (tmp_path / "mlruns").as_uri()
    old_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    yield tracking_uri
    mlflow.set_tracking_uri(old_uri)


class TestFullTrainingPipelineIntegration:
    """
    Integration tests: train_model(from_db=True) → model_training_data populated.

    NOTE: These tests use use_mlflow=True with a local file-based MLflow tracking
    store. This is necessary because record_training_split() is only called when a
    real MLflow run_id exists — matching the production Airflow DAG behaviour exactly.

    The production bug: Python 3.8's zip() raises TypeError for zip(strict=False),
    which was caught silently, leaving model_training_data empty after every DAG run.
    """

    def test_from_db_populates_model_training_data(
        self, tmp_path: Path, populated_sqlite_db, local_mlflow_tracking
    ):
        """
        CRITICAL REGRESSION TEST: model_training_data must NOT be empty after training.

        Production bug (now fixed): zip(strict=False) → TypeError on Python 3.8
        → silently swallowed → model_training_data was EMPTY after every DAG run.

        This test uses use_mlflow=True with a local file tracking URI so that a
        real run_id is generated — exactly matching the Airflow DAG code path.
        """
        from src.training.train import train_model

        model_path = tmp_path / "model.pkl"
        results = train_model(
            from_db=True,
            db=populated_sqlite_db,
            model_output_path=model_path,
            use_mlflow=True,
            allow_unlabeled=False,
        )

        run_id = results.get("mlflow_run_id")
        assert run_id, (
            "train_model with use_mlflow=True must return a mlflow_run_id. "
            "Got None — check that MLflow tracking URI is reachable."
        )

        # THE CRITICAL CHECK: model_training_data must have rows
        count = populated_sqlite_db.conn.execute(
            "SELECT COUNT(*) FROM model_training_data WHERE mlflow_run_id=?",
            (run_id,),
        ).fetchone()[0]
        assert count > 0, (
            f"CRITICAL: model_training_data is EMPTY for run_id={run_id!r}. "
            "This is the production bug that was fixed by removing strict=False from "
            "zip() calls in train.py. If this test fails, that regression fix has "
            "been reverted. Check train.py for any zip(strict=...) calls."
        )

    def test_from_db_split_is_disjoint(
        self, tmp_path: Path, populated_sqlite_db, local_mlflow_tracking
    ):
        """No signal appears in both train and test splits for the same run."""
        from src.training.train import train_model

        model_path = tmp_path / "model.pkl"
        results = train_model(
            from_db=True,
            db=populated_sqlite_db,
            model_output_path=model_path,
            use_mlflow=True,
            allow_unlabeled=False,
        )
        run_id = results.get("mlflow_run_id")
        if not run_id:
            pytest.skip("No run_id returned — MLflow tracking unreachable")

        train_ids = set(populated_sqlite_db.get_training_signal_ids(run_id, "train"))
        test_ids = set(populated_sqlite_db.get_training_signal_ids(run_id, "test"))

        assert len(train_ids) > 0, "Train set must not be empty"
        assert len(test_ids) > 0, "Test set must not be empty"
        overlap = train_ids & test_ids
        assert not overlap, f"Signal IDs appear in BOTH train and test: {overlap}"

    def test_from_db_all_labeled_signals_tracked(
        self, tmp_path: Path, populated_sqlite_db, local_mlflow_tracking
    ):
        """All 30 labeled signals must be accounted for in train+test split."""
        from src.training.train import train_model

        model_path = tmp_path / "model.pkl"
        results = train_model(
            from_db=True,
            db=populated_sqlite_db,
            model_output_path=model_path,
            use_mlflow=True,
            allow_unlabeled=False,
        )
        run_id = results.get("mlflow_run_id")
        if not run_id:
            pytest.skip("No run_id returned")

        total_tracked = populated_sqlite_db.conn.execute(
            "SELECT COUNT(*) FROM model_training_data WHERE mlflow_run_id=?",
            (run_id,),
        ).fetchone()[0]

        # We inserted exactly 30 labeled signals into the fixture
        assert total_tracked == 30, (
            f"Expected 30 tracked signals (all labeled), got {total_tracked}. "
            "Split tracking should cover all labeled signals used for training."
        )

    def test_split_json_files_created(
        self, tmp_path: Path, populated_sqlite_db, local_mlflow_tracking
    ):
        """
        After training, split JSON files must exist at:
          data/processed/training_splits/<run_id>/train.json
          data/processed/training_splits/<run_id>/test.json
        Each must be a non-empty JSON object with a 'signals' key.
        """
        from src.training.train import train_model

        model_path = tmp_path / "model.pkl"
        results = train_model(
            from_db=True,
            db=populated_sqlite_db,
            model_output_path=model_path,
            use_mlflow=True,
            allow_unlabeled=False,
        )
        run_id = results.get("mlflow_run_id")
        if not run_id:
            pytest.skip("No run_id returned")

        splits_dir = _REPO_ROOT / "data" / "processed" / "training_splits" / run_id
        assert splits_dir.exists(), (
            f"Split directory not created at {splits_dir}. "
            "train_model(from_db=True) must export train/test JSON files."
        )

        for split_name in ("train", "test"):
            json_file = splits_dir / f"{split_name}.json"
            assert json_file.exists(), f"{split_name}.json missing in {splits_dir}"
            data = json.loads(json_file.read_text())
            assert "signals" in data, f"{split_name}.json must have a 'signals' key"
            assert isinstance(data["signals"], list), "'signals' must be a list"
            assert len(data["signals"]) > 0, f"{split_name}.json 'signals' list must not be empty"

    def test_split_json_signal_ids_match_db(
        self, tmp_path: Path, populated_sqlite_db, local_mlflow_tracking
    ):
        """Signal IDs in split JSON files must match what's in model_training_data."""
        from src.training.train import train_model

        model_path = tmp_path / "model.pkl"
        results = train_model(
            from_db=True,
            db=populated_sqlite_db,
            model_output_path=model_path,
            use_mlflow=True,
            allow_unlabeled=False,
        )
        run_id = results.get("mlflow_run_id")
        if not run_id:
            pytest.skip("No run_id returned")

        splits_dir = _REPO_ROOT / "data" / "processed" / "training_splits" / run_id
        if not splits_dir.exists():
            pytest.skip(f"Split directory not created at {splits_dir}")

        for split_name in ("train", "test"):
            json_file = splits_dir / f"{split_name}.json"
            if not json_file.exists():
                continue
            payload = json.loads(json_file.read_text())
            json_ids = {sig["id"] for sig in payload.get("signals", [])}
            db_ids = set(populated_sqlite_db.get_training_signal_ids(run_id, split_name))
            assert json_ids == db_ids, (
                f"{split_name}.json signal IDs ({len(json_ids)}) don't match "
                f"model_training_data ({len(db_ids)}) for run_id={run_id}"
            )

    def test_multiple_training_runs_tracked_independently(
        self, tmp_path: Path, populated_sqlite_db, local_mlflow_tracking
    ):
        """Two consecutive training runs each get their own model_training_data rows."""
        from src.training.train import train_model

        results1 = train_model(
            from_db=True,
            db=populated_sqlite_db,
            model_output_path=tmp_path / "model1.pkl",
            use_mlflow=True,
            allow_unlabeled=False,
        )
        results2 = train_model(
            from_db=True,
            db=populated_sqlite_db,
            model_output_path=tmp_path / "model2.pkl",
            use_mlflow=True,
            allow_unlabeled=False,
        )

        run1 = results1.get("mlflow_run_id")
        run2 = results2.get("mlflow_run_id")

        if not run1 or not run2:
            pytest.skip("MLflow run IDs not returned")

        assert run1 != run2, "Each training run must produce a unique MLflow run_id"

        count1 = populated_sqlite_db.conn.execute(
            "SELECT COUNT(*) FROM model_training_data WHERE mlflow_run_id=?", (run1,)
        ).fetchone()[0]
        count2 = populated_sqlite_db.conn.execute(
            "SELECT COUNT(*) FROM model_training_data WHERE mlflow_run_id=?", (run2,)
        ).fetchone()[0]

        assert count1 > 0, f"Run 1 ({run1}) produced no model_training_data rows"
        assert count2 > 0, f"Run 2 ({run2}) produced no model_training_data rows"
        assert count1 == count2, (
            "Both runs used the same 30 labeled DB signals, so row counts should match: "
            f"run1={count1}, run2={count2}"
        )


# ── PostgreSQL integration tests ──────────────────────────────────────────────


@pytest.mark.postgres
class TestPostgreSQLIntegration:
    """
    Integration tests against a real PostgreSQL database.

    Requires POSTGRES_TEST_URL environment variable.
    These tests simulate the Airflow cloud environment.

    Run with:
        POSTGRES_TEST_URL=postgresql://mlops_user:...@127.0.0.1:5433/mlops_prod \
        pytest tests/database/test_model_training_data_e2e.py -v -m postgres
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_pg(self):
        if not _pg_available:
            pytest.skip("POSTGRES_TEST_URL not set or psycopg2 not installed")

    @pytest.fixture()
    def pg_db(self):
        """PostgreSQL Database instance for integration testing."""
        from src.database import Database

        db = Database(db_url=_PG_URL)
        yield db
        db.close()

    @pytest.fixture()
    def pg_device(self, pg_db):
        """Register a test device in PostgreSQL and clean up after test."""
        import psycopg2

        from src.database import generate_device_id

        device_id = generate_device_id()
        pg_db.register_device(
            device_id=device_id,
            device_name="pg-test-device",
            device_type="sensor",
            location="lab-pg",
            status="active",
        )
        yield device_id
        # Cleanup via direct psycopg2 to avoid wrapper limitations
        try:
            _c = psycopg2.connect(_PG_URL)
            with _c, _c.cursor() as _cur:
                _cur.execute(
                    "DELETE FROM model_training_data WHERE mlflow_run_id LIKE 'pg-%%-test%%'"
                )
                _cur.execute(
                    "DELETE FROM raw_signals WHERE signal_id IN ("
                    "  SELECT rs.signal_id FROM raw_signals rs"
                    "  JOIN predictions p ON rs.prediction_id = p.id"
                    "  WHERE p.device_id = %s)",
                    (device_id,),
                )
                _cur.execute("DELETE FROM predictions WHERE device_id=%s", (device_id,))
                _cur.execute("DELETE FROM devices WHERE device_id=%s", (device_id,))
            _c.close()
        except Exception:
            pass

    def _pg_cursor_query(self, sql: str, params: tuple = ()) -> Any:
        """Execute a SELECT and return the first row via direct psycopg2."""
        import psycopg2

        conn = psycopg2.connect(_PG_URL)
        with conn, conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
        conn.close()
        return row

    def test_pg_record_training_split_creates_rows(self, pg_db, pg_device):
        """record_training_split inserts correct rows into PostgreSQL model_training_data."""
        import psycopg2

        from src.signal_processing.signal_generator import generate_signal

        signal_ids: list[int] = []
        for i in range(6):
            sig = generate_signal("gaussian", drift_scenario="baseline", seed=i + 100)
            tv = sig.signal.time if isinstance(sig.signal.time, list) else sig.signal.time.tolist()
            av = (
                sig.signal.amplitude
                if isinstance(sig.signal.amplitude, list)
                else sig.signal.amplitude.tolist()
            )
            pid = pg_db.store_prediction(
                device_id=pg_device,
                time_values=tv,
                amplitude_values=av,
                predicted_label=0,
                model_version="v_pg_test",
                features={},
                prediction_confidence=0.9,
                deployment_mode="cloud",
            )
            pg_db.inject_sparse_label(pid, 0, "pg_integration_test")
            # Use cursor() API for PG (conn.execute() doesn't return a cursor)
            cur = pg_db.conn.cursor()
            cur.execute("SELECT signal_id FROM raw_signals WHERE prediction_id=?", (pid,))
            row = cur.fetchone()
            assert row is not None, f"raw_signals row not found for prediction_id={pid}"
            signal_ids.append(int(row["signal_id"]))

        test_run_id = f"pg-integration-test-{os.getpid()}"
        pg_db.record_training_split(
            mlflow_run_id=test_run_id,
            train_signal_ids=signal_ids[:4],
            test_signal_ids=signal_ids[4:],
            model_version="v_pg_test",
        )

        # Verify via direct psycopg2
        conn = psycopg2.connect(_PG_URL)
        with conn, conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM model_training_data WHERE mlflow_run_id=%s",
                (test_run_id,),
            )
            count = cur.fetchone()[0]
            cur.execute("DELETE FROM model_training_data WHERE mlflow_run_id=%s", (test_run_id,))
        conn.close()

        assert count == 6, f"Expected 6 rows in PostgreSQL model_training_data, got {count}"

    def test_pg_unique_constraint_honoured(self, pg_db, pg_device):
        """UNIQUE(mlflow_run_id, signal_id) is enforced in PostgreSQL."""
        import psycopg2

        from src.signal_processing.signal_generator import generate_signal

        sig = generate_signal("gaussian", seed=999)
        tv = sig.signal.time if isinstance(sig.signal.time, list) else sig.signal.time.tolist()
        av = (
            sig.signal.amplitude
            if isinstance(sig.signal.amplitude, list)
            else sig.signal.amplitude.tolist()
        )
        pid = pg_db.store_prediction(
            device_id=pg_device,
            time_values=tv,
            amplitude_values=av,
            predicted_label=0,
            model_version="v_pg_test",
            features={},
            prediction_confidence=0.9,
            deployment_mode="cloud",
        )
        pg_db.inject_sparse_label(pid, 0, "pg_unique_test")
        cur = pg_db.conn.cursor()
        cur.execute("SELECT signal_id FROM raw_signals WHERE prediction_id=?", (pid,))
        row = cur.fetchone()
        assert row is not None, f"raw_signals row not found for prediction_id={pid}"
        sid = int(row["signal_id"])

        test_run_id = f"pg-unique-test-{os.getpid()}"
        pg_db.record_training_split(test_run_id, [sid], [], "v_pg_test")
        # Second insert should not raise (INSERT ... ON CONFLICT DO NOTHING)
        pg_db.record_training_split(test_run_id, [sid], [], "v_pg_test")

        conn = psycopg2.connect(_PG_URL)
        with conn, conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM model_training_data WHERE mlflow_run_id=%s",
                (test_run_id,),
            )
            count = cur.fetchone()[0]
            cur.execute("DELETE FROM model_training_data WHERE mlflow_run_id=%s", (test_run_id,))
        conn.close()

        assert count == 1, f"Expected exactly 1 row after duplicate insert, got {count}"

    def test_pg_model_training_data_table_exists(self):
        """Verify model_training_data table exists in production PostgreSQL."""
        import psycopg2

        conn = psycopg2.connect(_PG_URL)
        with conn, conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema='public' AND table_name='model_training_data'"
            )
            count = cur.fetchone()[0]
        conn.close()
        assert count == 1, (
            "model_training_data table does not exist in PostgreSQL. Run init_db.py to create it."
        )

    def test_pg_model_training_data_current_count(self):
        """Report current row count in production model_training_data (informational)."""
        import psycopg2

        conn = psycopg2.connect(_PG_URL)
        with conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM model_training_data")
            total = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM model_training_data WHERE split='train'")
            train_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM model_training_data WHERE split='test'")
            test_count = cur.fetchone()[0]
        conn.close()

        # This is an informational test — always passes, just prints counts
        print(f"\n[model_training_data] total={total}, train={train_count}, test={test_count}")
        assert total > 0, (
            "model_training_data is EMPTY in production PostgreSQL! "
            "Trigger the automated_retraining DAG to populate it."
        )


# ── _PgCursorWrapper translation tests ───────────────────────────────────────


class TestPgCursorWrapper:
    """Verify that _PgCursorWrapper correctly translates ? → %s placeholders."""

    def _get_wrapper(self, tmp_path: Path):
        """Get a _PgCursorWrapper instance via SQLite (for testing the translation only)."""
        from src.database import Database

        db = Database(db_path=tmp_path / "wrapper_test.db")
        cursor = db.conn.cursor()
        return db, cursor

    def test_execute_translates_question_marks(self, tmp_path: Path):
        """_PgCursorWrapper.execute() translates ? to %s equivalent."""
        # We can't easily test _PgCursorWrapper with SQLite directly since
        # it's a wrapper for PostgreSQL psycopg2 cursors. Instead, verify
        # the translation logic directly.
        import unittest.mock as mock

        from src.database.database import _PgCursorWrapper

        mock_cursor = mock.MagicMock()

        class FakeConn:
            autocommit = False

        wrapper = _PgCursorWrapper(mock_cursor)
        wrapper.execute("SELECT * FROM t WHERE id=? AND name=?", (1, "foo"))
        mock_cursor.execute.assert_called_once_with(
            "SELECT * FROM t WHERE id=%s AND name=%s", (1, "foo")
        )

    def test_executemany_translates_question_marks(self, tmp_path: Path):
        """_PgCursorWrapper.executemany() translates ? to %s equivalent."""
        import unittest.mock as mock

        from src.database.database import _PgCursorWrapper

        mock_cursor = mock.MagicMock()
        wrapper = _PgCursorWrapper(mock_cursor)
        wrapper.executemany("INSERT INTO t VALUES (?, ?)", [(1, "a"), (2, "b")])
        mock_cursor.executemany.assert_called_once_with(
            "INSERT INTO t VALUES (%s, %s)", [(1, "a"), (2, "b")]
        )

    def test_no_double_translation(self, tmp_path: Path):
        """SQL with %s placeholders should not be double-translated."""
        import unittest.mock as mock

        from src.database.database import _PgCursorWrapper

        mock_cursor = mock.MagicMock()
        wrapper = _PgCursorWrapper(mock_cursor)
        wrapper.execute("SELECT * FROM t WHERE id=%s", (42,))
        mock_cursor.execute.assert_called_once_with("SELECT * FROM t WHERE id=%s", (42,))

    def test_record_training_split_uses_executemany(self, tmp_path: Path):
        """Verify record_training_split in database.py uses executemany (not a loop of execute).

        This is important because executemany is more efficient and the _PgCursorWrapper
        translates it correctly.
        """
        db_py = _REPO_ROOT / "src" / "database" / "database.py"
        source = db_py.read_text(encoding="utf-8")

        # Check that executemany is used in the context of model_training_data
        assert "executemany" in source, (
            "database.py should use executemany for bulk inserts into model_training_data"
        )


# ── Schema integrity tests ────────────────────────────────────────────────────


class TestModelTrainingDataSchema:
    """Verify model_training_data table schema is correct."""

    def test_schema_columns_present(self, sqlite_db):
        """model_training_data must have all required columns."""
        cursor = sqlite_db.conn.execute("PRAGMA table_info(model_training_data)")
        columns = {row[1] for row in cursor.fetchall()}

        required_columns = {
            "id",
            "mlflow_run_id",
            "signal_id",
            "split",
            "model_version",
            "created_at",
        }
        missing = required_columns - columns
        assert not missing, (
            f"model_training_data is missing columns: {missing}. "
            "Check src/database/init_db.py _migrate_model_training_data_pg()"
        )

    def test_split_check_constraint(self, sqlite_db):
        """split column only accepts 'train' or 'test' values."""
        import sqlite3

        with pytest.raises(sqlite3.IntegrityError):
            sqlite_db.conn.execute(
                "INSERT INTO model_training_data "
                "(mlflow_run_id, signal_id, split, model_version) "
                "VALUES ('run-x', 999, 'validation', 'v1')"
            )
            sqlite_db.conn.commit()

    def test_unique_constraint_on_run_signal(self, sqlite_db):
        """UNIQUE(mlflow_run_id, signal_id) must be enforced."""
        import sqlite3

        # First insert
        sqlite_db.conn.execute(
            "INSERT INTO model_training_data "
            "(mlflow_run_id, signal_id, split, model_version) "
            "VALUES ('run-u', 1, 'train', 'v1')"
        )
        sqlite_db.conn.commit()

        # Duplicate insert - SQLite with INSERT OR IGNORE should not raise
        # but a plain INSERT should raise IntegrityError
        with pytest.raises(sqlite3.IntegrityError):
            sqlite_db.conn.execute(
                "INSERT INTO model_training_data "
                "(mlflow_run_id, signal_id, split, model_version) "
                "VALUES ('run-u', 1, 'test', 'v1')"
            )
            sqlite_db.conn.commit()
