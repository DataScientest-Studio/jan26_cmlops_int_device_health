"""
Comprehensive tests for FT-5: Bootstrap + Champion/Challenger model_training_data.

These tests verify that running bootstrap (``run_bootstrap()``) and
champion/challenger (which calls ``run_bootstrap(promote=False)``) correctly
populates the ``model_training_data`` table — the long-standing gap labelled
"Permanently deferred (FT-5)" that is now implemented.

Root cause recap
─────────────────
Previously, ``_step_store_to_database()`` ran AFTER ``_step_train()``.
Training used ``train_data_path`` (a JSON file) and never touched the DB,
so ``record_training_split()`` was never called — ``model_training_data``
stayed empty for bootstrap and champion/challenger runs.

What was fixed (FT-5)
──────────────────────
1. ``_step_store_to_database()`` now runs BEFORE ``_step_train()``, returns
   ``(labeled_signal_ids, unlabeled_signal_ids, n_labels)`` instead of just
   ``n_labels``.
2. ``_step_train()`` accepts a new ``db_signal_ids`` param; when provided it
   calls ``train_model(from_db=True, signal_ids_filter=...)``.
3. ``database.get_labeled_signal_ids()`` and ``get_unlabeled_signal_ids()``
   gain ``signal_ids_filter``.  When provided the ``deployment_mode = 'cloud'``
   constraint is skipped so bootstrap (any mode) works correctly.
4. ``database.get_signal_id_by_prediction_id()`` returns the ``signal_id``
   assigned by the DB after ``store_prediction()``.
5. ``database.update_model_version_by_run_id()`` updates the temporary
   placeholder version in both ``predictions`` and ``model_training_data``
   after ``_step_register()`` assigns the canonical ``v<N>`` label.

Test structure
──────────────
A. Unit tests for new database helpers (SQLite — no PostgreSQL required)
B. Integration tests for the full bootstrap pipeline (SQLite MLflow + SQLite DB)
C. Signal-IDs-filter correctness (ensures filter restricts correctly)
D. PostgreSQL integration tests (skipped unless POSTGRES_TEST_URL is set)
E. Regression tests — old behaviour was broken; new behaviour is verified

Run all tests:
    pytest tests/database/test_bootstrap_training_split.py -v

Run only PostgreSQL tests:
    POSTGRES_TEST_URL=postgresql://mlops_user:local_dev_password@127.0.0.1:5433/mlops_prod \\
    pytest tests/database/test_bootstrap_training_split.py -v -m postgres
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import numpy as np
import pytest

# ── optional PostgreSQL fixture ───────────────────────────────────────────────
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
        "Set POSTGRES_TEST_URL=postgresql://mlops_user:local_dev_password"
        "@127.0.0.1:5433/mlops_prod to enable PostgreSQL integration tests"
    ),
)

# ── helpers ───────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_signal_data(n: int = 200, seed: int = 0, shape: str = "gaussian") -> dict:
    """Return a dict with 'time', 'amplitude', 'shape_type', 'label' keys."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 100, n).tolist()  # SignalData requires time span [0, 100]
    mu = 50.0
    sigma = 5.0 + rng.uniform(0, 3)
    a = np.exp(-((np.linspace(0, 100, n) - mu) ** 2) / (2 * sigma**2)).tolist()
    # add noise
    a = [v + float(rng.normal(0, 0.02)) for v in a]
    label = 0 if shape == "gaussian" else 1
    return {
        "time": t,
        "amplitude": a,
        "shape_type": shape,
        "label": label,
    }


def _write_bootstrap_json(
    tmp_path: Path, n_labeled: int = 20, n_unlabeled: int = 10
) -> dict[str, Path]:
    """Write baseline_train.json, bootstrap_labeled.json and bootstrap_unlabeled.json to tmp_path.

    _step_store_to_database() uses ``baseline_train.json`` as the
    training dataset and returns its signal IDs as ``labeled_signal_ids``.  The
    bootstrap files are still stored for production prediction lineage.
    """
    # baseline_train: labeled signals used for model training (mirrors n_labeled count)
    baseline_train = {
        "signals": [
            _make_signal_data(seed=500 + i, shape="gaussian" if i % 3 != 0 else "lorentzian")
            for i in range(n_labeled)
        ]
    }
    labeled = {
        "signals": [
            _make_signal_data(seed=i, shape="gaussian" if i % 3 != 0 else "lorentzian")
            for i in range(n_labeled)
        ]
    }
    unlabeled_sigs = []
    for i in range(n_unlabeled):
        s = _make_signal_data(seed=1000 + i)
        del s["label"]  # unlabeled
        unlabeled_sigs.append(s)
    unlabeled = {"signals": unlabeled_sigs}

    btp = tmp_path / "baseline_train.json"
    lp = tmp_path / "bootstrap_labeled.json"
    up = tmp_path / "bootstrap_unlabeled.json"
    btp.write_text(json.dumps(baseline_train))
    lp.write_text(json.dumps(labeled))
    up.write_text(json.dumps(unlabeled))
    return {"baseline_train": btp, "bootstrap_labeled": lp, "bootstrap_unlabeled": up}


# ═══════════════════════════════════════════════════════════════════════════════
# A. Unit tests: new database helpers
# ═══════════════════════════════════════════════════════════════════════════════


class TestNewDatabaseHelpers:
    """Tests for get_signal_id_by_prediction_id and update_model_version_by_run_id."""

    def test_get_signal_id_by_prediction_id_returns_correct_id(self, tmp_path: Path) -> None:
        """After store_prediction(), the returned prediction_id maps to a valid signal_id."""
        from src.database.database import Database

        db = Database(db_path=tmp_path / "test.db")
        device_id = str(uuid.uuid4())
        db.register_device(device_id, "test-dev", "sensor", "lab", "active")

        sig = _make_signal_data(seed=1)
        pid = db.store_prediction(
            device_id=device_id,
            time_values=sig["time"],
            amplitude_values=sig["amplitude"],
            predicted_label=0,
            model_version="v_test",
            features={},
            deployment_mode="cloud",
        )
        sid = db.get_signal_id_by_prediction_id(pid)
        assert sid is not None
        assert isinstance(sid, int)
        assert sid > 0
        db.close()

    def test_get_signal_id_by_prediction_id_missing_returns_none(self, tmp_path: Path) -> None:
        """Non-existent prediction_id returns None, not an exception."""
        from src.database.database import Database

        db = Database(db_path=tmp_path / "test.db")
        result = db.get_signal_id_by_prediction_id(99999)
        assert result is None
        db.close()

    def test_update_model_version_by_run_id_updates_predictions(self, tmp_path: Path) -> None:
        """update_model_version_by_run_id changes model_version in predictions table."""
        from src.database.database import Database

        db = Database(db_path=tmp_path / "test.db")
        device_id = str(uuid.uuid4())
        db.register_device(device_id, "test-dev", "sensor", "lab", "active")
        run_id = "test_run_abc123"

        sig = _make_signal_data(seed=2)
        db.store_prediction(
            device_id=device_id,
            time_values=sig["time"],
            amplitude_values=sig["amplitude"],
            predicted_label=0,
            model_version="bootstrap_test_ru",
            features={},
            mlflow_run_id=run_id,
            deployment_mode="cloud",
        )

        n = db.update_model_version_by_run_id(run_id, "v42")
        assert n > 0

        cur = db.conn.cursor()
        cur.execute("SELECT model_version FROM predictions WHERE mlflow_run_id = ?", (run_id,))
        row = cur.fetchone()
        assert row is not None
        assert row["model_version"] == "v42"
        db.close()

    def test_update_model_version_updates_model_training_data(self, tmp_path: Path) -> None:
        """update_model_version_by_run_id also updates model_training_data rows."""
        from src.database.database import Database

        db = Database(db_path=tmp_path / "test.db")
        run_id = "test_run_xyz"

        # insert a fake model_training_data row manually
        cur = db.conn.cursor()
        cur.execute(
            "INSERT INTO model_training_data (mlflow_run_id, signal_id, split, model_version) "
            "VALUES (?, ?, ?, ?)",
            (run_id, 999, "train", "bootstrap_test_ru"),
        )
        db.conn.commit()

        db.update_model_version_by_run_id(run_id, "v5")

        cur.execute(
            "SELECT model_version FROM model_training_data WHERE mlflow_run_id = ?", (run_id,)
        )
        row = cur.fetchone()
        assert row is not None
        assert row["model_version"] == "v5"
        db.close()

    def test_update_model_version_nonexistent_run_returns_zero(self, tmp_path: Path) -> None:
        """update_model_version_by_run_id on an unknown run_id returns 0 and doesn't crash."""
        from src.database.database import Database

        db = Database(db_path=tmp_path / "test.db")
        n = db.update_model_version_by_run_id("does_not_exist", "v99")
        assert n == 0
        db.close()


# ═══════════════════════════════════════════════════════════════════════════════
# B. Unit tests: signal_ids_filter on get_labeled/unlabeled_signal_ids
# ═══════════════════════════════════════════════════════════════════════════════


class TestSignalIdsFilter:
    """Tests for signal_ids_filter parameter on get_labeled_signal_ids / get_unlabeled_signal_ids."""

    @pytest.fixture()
    def db_with_signals(self, tmp_path: Path):
        """SQLite DB pre-loaded with 10 labeled + 5 unlabeled cloud signals."""
        from src.database.database import Database

        db = Database(db_path=tmp_path / "filter_test.db")
        device_id = str(uuid.uuid4())
        db.register_device(device_id, "filter-dev", "sensor", "lab", "active")

        rng = np.random.RandomState(7)
        signal_ids: dict[str, list[int]] = {"labeled": [], "unlabeled": []}

        for i in range(10):
            sig = _make_signal_data(seed=i)
            pid = db.store_prediction(
                device_id=device_id,
                time_values=sig["time"],
                amplitude_values=sig["amplitude"],
                predicted_label=0,
                model_version="v_filter",
                features={},
                prediction_confidence=float(rng.uniform(0.7, 0.99)),
                deployment_mode="cloud",
            )
            db.inject_sparse_label(pid, sig["label"], "test")
            sid = db.get_signal_id_by_prediction_id(pid)
            if sid is not None:
                signal_ids["labeled"].append(sid)

        for i in range(5):
            sig = _make_signal_data(seed=100 + i)
            pid = db.store_prediction(
                device_id=device_id,
                time_values=sig["time"],
                amplitude_values=sig["amplitude"],
                predicted_label=0,
                model_version="v_filter",
                features={},
                deployment_mode="cloud",
            )
            sid = db.get_signal_id_by_prediction_id(pid)
            if sid is not None:
                signal_ids["unlabeled"].append(sid)

        yield db, signal_ids
        db.close()

    def test_no_filter_returns_all_labeled(self, db_with_signals) -> None:
        """Without filter, get_labeled_signal_ids returns all 10 labeled signals."""
        db, ids = db_with_signals
        result = db.get_labeled_signal_ids()
        assert len(result) == 10

    def test_no_filter_returns_all_unlabeled(self, db_with_signals) -> None:
        """Without filter, get_unlabeled_signal_ids returns all 5 unlabeled signals."""
        db, ids = db_with_signals
        result = db.get_unlabeled_signal_ids()
        assert len(result) == 5

    def test_filter_restricts_labeled_subset(self, db_with_signals) -> None:
        """signal_ids_filter restricts to only the first 3 labeled signals."""
        db, ids = db_with_signals
        first_3 = ids["labeled"][:3]
        result = db.get_labeled_signal_ids(signal_ids_filter=first_3)
        assert sorted(result) == sorted(first_3)

    def test_filter_restricts_unlabeled_subset(self, db_with_signals) -> None:
        """signal_ids_filter restricts to only first 2 unlabeled signals."""
        db, ids = db_with_signals
        first_2 = ids["unlabeled"][:2]
        result = db.get_unlabeled_signal_ids(signal_ids_filter=first_2)
        assert sorted(result) == sorted(first_2)

    def test_filter_with_mixed_ids_returns_correct_split(self, db_with_signals) -> None:
        """Mixing labeled and unlabeled IDs in a single filter works correctly."""
        db, ids = db_with_signals
        mixed = ids["labeled"][:3] + ids["unlabeled"][:2]
        labeled_result = db.get_labeled_signal_ids(signal_ids_filter=mixed)
        unlabeled_result = db.get_unlabeled_signal_ids(signal_ids_filter=mixed)
        # labeled returns only the 3 labeled ones from the mixed set
        assert len(labeled_result) == 3
        assert len(unlabeled_result) == 2
        # no overlap
        assert set(labeled_result).isdisjoint(set(unlabeled_result))

    def test_filter_empty_list_returns_empty(self, db_with_signals) -> None:
        """An empty signal_ids_filter returns an empty list (not all signals)."""
        db, ids = db_with_signals
        result = db.get_labeled_signal_ids(signal_ids_filter=[])
        assert result == []

    def test_filter_nonexistent_ids_returns_empty(self, db_with_signals) -> None:
        """Filter with IDs that don't exist in DB returns empty list."""
        db, ids = db_with_signals
        result = db.get_labeled_signal_ids(signal_ids_filter=[99998, 99999])
        assert result == []

    def test_filter_ignores_deployment_mode_constraint(self, tmp_path: Path) -> None:
        """When signal_ids_filter is provided, deployment_mode='local' signals are included.

        This is critical for bootstrap which stores signals with deployment_mode from cfg.mode,
        which may not be 'cloud' in all configurations.
        """
        from src.database.database import Database

        db = Database(db_path=tmp_path / "mode_test.db")
        device_id = str(uuid.uuid4())
        db.register_device(device_id, "local-dev", "sensor", "lab", "active")

        sig = _make_signal_data(seed=3)
        pid = db.store_prediction(
            device_id=device_id,
            time_values=sig["time"],
            amplitude_values=sig["amplitude"],
            predicted_label=0,
            model_version="v_local",
            features={},
            deployment_mode="local",  # ← NOT 'cloud'
        )
        db.inject_sparse_label(pid, sig["label"], "test")
        sid = db.get_signal_id_by_prediction_id(pid)
        assert sid is not None

        # Without filter: not returned (deployment_mode = 'cloud' constraint)
        result_no_filter = db.get_labeled_signal_ids()
        assert sid not in result_no_filter

        # With filter: returned because deployment_mode constraint is skipped
        result_with_filter = db.get_labeled_signal_ids(signal_ids_filter=[sid])
        assert sid in result_with_filter
        db.close()


# ═══════════════════════════════════════════════════════════════════════════════
# C. Integration tests: full bootstrap → model_training_data population
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def local_mlflow_tracking(tmp_path: Path):
    """Configure MLflow to use a local file-based tracking store for tests."""
    import mlflow

    tracking_uri = (tmp_path / "mlruns").as_uri()
    old_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    yield tracking_uri
    mlflow.set_tracking_uri(old_uri)


@pytest.fixture()
def local_deployment_mode():
    """Force deployment mode to 'local' for end-to-end tests.

    Temporarily writes "local" to .current_mode so that neither
    greenfield_init._detect_mode() nor train.py's inline mode-detection
    enter the cloud code path (e.g. dvc add subprocess, DagsHub push).
    The original content is restored after the test.
    """
    from scripts.greenfield_init import PROJECT_ROOT

    _mode_file = PROJECT_ROOT / ".current_mode"
    _orig = _mode_file.read_text() if _mode_file.exists() else None
    _mode_file.write_text("local")
    _orig_env = os.environ.get("DEPLOYMENT_MODE")
    os.environ["DEPLOYMENT_MODE"] = "local"
    yield
    if _orig is not None:
        _mode_file.write_text(_orig)
    else:
        _mode_file.unlink(missing_ok=True)
    if _orig_env is not None:
        os.environ["DEPLOYMENT_MODE"] = _orig_env
    else:
        os.environ.pop("DEPLOYMENT_MODE", None)


class TestStepStoreToDatabaseReturnValues:
    """Tests for _step_store_to_database() return value changes (FT-5 Phase A)."""

    def test_returns_tuple_of_three(self, tmp_path: Path) -> None:
        """_step_store_to_database returns (labeled_ids, unlabeled_ids, n_labels) tuple."""
        from scripts.greenfield_init import BootstrapConfig, _step_store_to_database

        paths = _write_bootstrap_json(tmp_path, n_labeled=10, n_unlabeled=5)
        cfg = BootstrapConfig(
            n_samples=15,
        )
        # Need a DB to actually store
        db_url_env = os.environ.get("POSTGRES_TEST_URL") or os.environ.get("DATABASE_URL")
        if not db_url_env:
            pytest.skip("No DB URL available for integration storage test")

        messages: list[str] = []

        def cb(step: str, msg: str, frac: float) -> None:
            messages.append(msg)

        stub_run_id = uuid.uuid4().hex
        cfg = BootstrapConfig(n_samples=15)
        result = _step_store_to_database(cfg, paths, stub_run_id, cb)
        assert isinstance(result, tuple)
        assert len(result) == 3
        labeled_ids, unlabeled_ids, n_labels = result
        assert isinstance(labeled_ids, list)
        assert isinstance(unlabeled_ids, list)
        assert isinstance(n_labels, int)

    def test_returns_empty_lists_without_db(self, tmp_path: Path, monkeypatch) -> None:
        """When no DB URL is available, returns ([], [], 0)."""
        from scripts.greenfield_init import BootstrapConfig, _step_store_to_database

        # Remove all DB env vars
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("POSTGRES_TEST_URL", raising=False)
        monkeypatch.delenv("POSTGRES_URL", raising=False)

        paths = _write_bootstrap_json(tmp_path)
        cfg = BootstrapConfig(n_samples=15)

        def cb(step: str, msg: str, frac: float) -> None:
            pass

        stub_run_id = uuid.uuid4().hex
        result = _step_store_to_database(cfg, paths, stub_run_id, cb)
        labeled_ids, unlabeled_ids, n_labels = result
        assert labeled_ids == []
        assert unlabeled_ids == []
        assert n_labels == 0


class TestBootstrapSplitTracking:
    """
    Full integration: run_bootstrap() → model_training_data populated.

    These tests require a PostgreSQL instance (POSTGRES_TEST_URL) because
    bootstrap is a cloud-mode operation that writes to PostgreSQL.
    """

    @pytest.mark.postgres
    def test_bootstrap_populates_model_training_data(self, tmp_path: Path) -> None:
        """After run_bootstrap(), model_training_data has rows for the run_id."""
        if not _pg_available:
            pytest.skip("PostgreSQL not available")

        import mlflow

        from scripts.greenfield_init import BootstrapConfig, run_bootstrap

        tracking_uri = (tmp_path / "mlruns").as_uri()
        old_uri = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)
        try:
            config = BootstrapConfig(
                n_samples=20,  # small for speed
                gaussian_fraction=0.7,
                seed=999,
                classifier="logistic_regression",
                promote=False,
                wipe=False,
            )

            log: list[str] = []

            def cb(step: str, msg: str, frac: float) -> None:
                log.append(f"[{step}] {msg}")

            result = run_bootstrap(config, progress_callback=cb)

            assert result.success, f"Bootstrap failed: {result.error}\nLog:\n" + "\n".join(log)
            assert result.mlflow_run_id, "No MLflow run_id produced"

            # Verify model_training_data was populated
            from src.database.database import Database

            db = Database(db_url=_PG_URL)
            all_ids = db.get_training_signal_ids(result.mlflow_run_id)
            db.close()

            assert len(all_ids) > 0, (
                f"model_training_data empty for run {result.mlflow_run_id}.\n"
                f"Bootstrap log:\n" + "\n".join(log)
            )
        finally:
            mlflow.set_tracking_uri(old_uri)

    @pytest.mark.postgres
    def test_bootstrap_split_is_disjoint(self, tmp_path: Path) -> None:
        """Train and test signal_ids must not overlap."""
        if not _pg_available:
            pytest.skip("PostgreSQL not available")

        import mlflow

        from scripts.greenfield_init import BootstrapConfig, run_bootstrap

        tracking_uri = (tmp_path / "mlruns2").as_uri()
        old_uri = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)
        try:
            config = BootstrapConfig(
                n_samples=25,
                seed=1234,
                classifier="logistic_regression",
                promote=False,
                wipe=False,
            )
            result = run_bootstrap(config)
            assert result.success, f"Bootstrap failed: {result.error}"

            from src.database.database import Database

            db = Database(db_url=_PG_URL)
            train_ids = db.get_training_signal_ids(result.mlflow_run_id, split="train")
            test_ids = db.get_training_signal_ids(result.mlflow_run_id, split="test")
            db.close()

            assert len(train_ids) > 0, "No train IDs recorded"
            assert len(test_ids) > 0, "No test IDs recorded"
            assert set(train_ids).isdisjoint(set(test_ids)), (
                f"Train and test sets overlap! Overlap: {set(train_ids) & set(test_ids)}"
            )
        finally:
            mlflow.set_tracking_uri(old_uri)

    @pytest.mark.postgres
    def test_bootstrap_signal_ids_are_real_db_rows(self, tmp_path: Path) -> None:
        """Every signal_id in model_training_data must exist in raw_signals."""
        if not _pg_available:
            pytest.skip("PostgreSQL not available")

        import mlflow

        from scripts.greenfield_init import BootstrapConfig, run_bootstrap

        tracking_uri = (tmp_path / "mlruns3").as_uri()
        old_uri = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)
        try:
            config = BootstrapConfig(
                n_samples=20,
                seed=5678,
                classifier="logistic_regression",
                promote=False,
                wipe=False,
            )
            result = run_bootstrap(config)
            assert result.success, f"Bootstrap failed: {result.error}"

            from src.database.database import Database

            db = Database(db_url=_PG_URL)
            tracked_ids = db.get_training_signal_ids(result.mlflow_run_id)

            cur = db.conn.cursor()
            orphaned = []
            for sid in tracked_ids:
                cur.execute("SELECT 1 FROM raw_signals WHERE signal_id = %s", (sid,))
                if cur.fetchone() is None:
                    orphaned.append(sid)
            db.close()

            assert orphaned == [], f"Orphaned signal_ids in model_training_data: {orphaned}"
        finally:
            mlflow.set_tracking_uri(old_uri)

    @pytest.mark.postgres
    def test_bootstrap_model_version_canonical_after_registration(self, tmp_path: Path) -> None:
        """After registration, model_version in model_training_data is 'v<N>' not a stub."""
        if not _pg_available:
            pytest.skip("PostgreSQL not available")

        import mlflow

        from scripts.greenfield_init import BootstrapConfig, run_bootstrap

        tracking_uri = (tmp_path / "mlruns4").as_uri()
        old_uri = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)
        try:
            config = BootstrapConfig(
                n_samples=20,
                seed=91011,
                classifier="logistic_regression",
                promote=False,
                wipe=False,
            )
            result = run_bootstrap(config)
            assert result.success, f"Bootstrap failed: {result.error}"

            from src.database.database import Database

            db = Database(db_url=_PG_URL)
            cur = db.conn.cursor()
            cur.execute(
                "SELECT DISTINCT model_version FROM model_training_data WHERE mlflow_run_id = %s",
                (result.mlflow_run_id,),
            )
            versions = [row["model_version"] for row in cur.fetchall()]
            db.close()

            assert len(versions) > 0, "No model_training_data rows found"
            for v in versions:
                # Must be canonical "v<N>" format, not a bootstrap stub
                assert v is not None and v.startswith("v"), (
                    f"model_version '{v}' is not canonical — update_model_version_by_run_id failed"
                )
        finally:
            mlflow.set_tracking_uri(old_uri)

    @pytest.mark.postgres
    def test_champion_challenger_populates_model_training_data(self, tmp_path: Path) -> None:
        """Champion/challenger (run_bootstrap with promote=False) also populates split table."""
        if not _pg_available:
            pytest.skip("PostgreSQL not available")

        import mlflow

        from scripts.greenfield_init import BootstrapConfig, run_bootstrap

        tracking_uri = (tmp_path / "mlruns_cc").as_uri()
        old_uri = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)
        try:
            # Mimic what champion_challenger.py does
            project_root = _REPO_ROOT
            bootstrap_labeled = project_root / "data" / "raw" / "bootstrap_labeled.json"

            config = BootstrapConfig(
                n_samples=20,
                gaussian_fraction=0.7,
                labeled_fraction=0.8,
                seed=4242,
                classifier="logistic_regression",
                promote=False,
                wipe=False,
            )
            if bootstrap_labeled.exists():
                config.train_data_path_override = bootstrap_labeled

            log: list[str] = []

            def cb(step: str, msg: str, frac: float) -> None:
                log.append(f"[{step}] {msg}")

            os.environ.setdefault("TRAINED_BY", "champion_challenger_training")
            try:
                result = run_bootstrap(config, progress_callback=cb)
            finally:
                if os.environ.get("TRAINED_BY") == "champion_challenger_training":
                    del os.environ["TRAINED_BY"]

            assert result.success, (
                f"Champion/challenger bootstrap failed: {result.error}\n" + "\n".join(log)
            )

            from src.database.database import Database

            db = Database(db_url=_PG_URL)
            all_ids = db.get_training_signal_ids(result.mlflow_run_id)
            db.close()

            assert len(all_ids) > 0, (
                f"model_training_data empty for champion/challenger run {result.mlflow_run_id}.\n"
                + "\n".join(log)
            )
        finally:
            mlflow.set_tracking_uri(old_uri)


# ═══════════════════════════════════════════════════════════════════════════════
# D. Integration tests using SQLite: full _step_store_to_database + _step_train
# (no PostgreSQL required — uses sqlite:// db_url)
# ═══════════════════════════════════════════════════════════════════════════════


class TestBootstrapSplitSQLite:
    """
    Tests that verify the FT-5 flow using SQLite as the database backend.

    These tests run in full CI without any external services.  They use
    local_mlflow_tracking so that record_training_split is actually called.
    """

    @pytest.fixture()
    def sqlite_db_url(self, tmp_path: Path) -> str:
        """Return a sqlite:// URL pointing to a fresh temp DB."""
        db_file = tmp_path / "bootstrap_test.db"
        return f"sqlite:///{db_file}"

    @pytest.fixture(autouse=True)
    def force_local_mode(self, monkeypatch) -> None:
        """Force deployment mode to local in BOTH env var and .current_mode file.

        train.py reads .current_mode first (takes priority over DEPLOYMENT_MODE
        env var), so setting only the env var is not enough on a machine where
        .current_mode contains 'cloud'.
        """
        from scripts.greenfield_init import PROJECT_ROOT

        monkeypatch.setenv("DEPLOYMENT_MODE", "local")
        _mode_file = PROJECT_ROOT / ".current_mode"
        _orig = _mode_file.read_text() if _mode_file.exists() else None
        _mode_file.write_text("local")
        yield
        if _orig is not None:
            _mode_file.write_text(_orig)
        else:
            _mode_file.unlink(missing_ok=True)

    @pytest.fixture(autouse=True)
    def patched_db_url(self, sqlite_db_url: str):
        """Patch gfi._get_db_url to return sqlite URL (autouse — applied to all tests)."""
        import scripts.greenfield_init as gfi

        original = gfi._get_db_url
        gfi._get_db_url = lambda: sqlite_db_url
        yield
        gfi._get_db_url = original

    def test_store_then_labeled_ids_filter_finds_inserted_signals(
        self, tmp_path: Path, sqlite_db_url: str
    ) -> None:
        """
        After _step_store_to_database, the returned labeled_signal_ids can be
        retrieved via get_labeled_signal_ids(signal_ids_filter=...).
        """
        from scripts.greenfield_init import BootstrapConfig, _step_store_to_database
        from src.database.database import Database

        paths = _write_bootstrap_json(tmp_path, n_labeled=12, n_unlabeled=6)
        cfg = BootstrapConfig(n_samples=18)
        stub_run_id = uuid.uuid4().hex

        def cb(step: str, msg: str, frac: float) -> None:
            pass

        labeled_ids, unlabeled_ids, n_labels = _step_store_to_database(cfg, paths, stub_run_id, cb)

        # After C1 revert: bootstrap_labeled → labeled_ids, bootstrap_unlabeled → unlabeled_ids.
        # Semi-supervised training mode. F1=0 when a class is absent from the test set is
        # acceptable by design; only a missing training class would be a real error.
        assert len(labeled_ids) > 0, "No labeled signal_ids returned (bootstrap_labeled not stored)"
        assert len(unlabeled_ids) > 0, (
            "No unlabeled signal_ids returned (bootstrap_unlabeled not stored)"
        )
        assert n_labels >= 0

        # Verify bootstrap_labeled IDs are in the DB and findable via the filter
        db = Database(db_url=sqlite_db_url)
        retrieved = db.get_labeled_signal_ids(signal_ids_filter=labeled_ids)
        assert sorted(retrieved) == sorted(labeled_ids), (
            f"Filter returned {retrieved}, expected {labeled_ids}"
        )
        db.close()

    def test_full_pipeline_sqlite_populates_model_training_data(
        self,
        tmp_path: Path,
        sqlite_db_url: str,
        local_mlflow_tracking,
    ) -> None:
        """
        Full bootstrap flow (SQLite backend): store → train → record_split.

        This is the most important test: it runs the COMPLETE new flow end-to-end
        using only local resources (SQLite + local MLflow file store).
        """
        from scripts.greenfield_init import BootstrapConfig, run_bootstrap

        config = BootstrapConfig(
            n_samples=30,
            gaussian_fraction=0.5,
            labeled_fraction=0.5,
            seed=777,
            classifier="logistic_regression",
            promote=False,
            wipe=False,
        )

        log: list[str] = []

        def cb(step: str, msg: str, frac: float) -> None:
            log.append(f"[{step}] {msg}")

        result = run_bootstrap(config, progress_callback=cb)

        assert result.success, f"SQLite bootstrap failed: {result.error}\nLog:\n" + "\n".join(log)
        assert result.mlflow_run_id, "No MLflow run_id"

        # The key assertion: model_training_data has rows
        from src.database.database import Database

        db = Database(db_url=sqlite_db_url)
        tracked_ids = db.get_training_signal_ids(result.mlflow_run_id)
        db.close()

        assert len(tracked_ids) > 0, (
            f"model_training_data EMPTY for run {result.mlflow_run_id} "
            f"(FT-5 not working).\nBootstrap log:\n" + "\n".join(log)
        )

    def test_full_pipeline_split_is_disjoint(
        self,
        tmp_path: Path,
        sqlite_db_url: str,
        local_mlflow_tracking,
    ) -> None:
        """Train and test signal_ids must not overlap (SQLite backend)."""
        from scripts.greenfield_init import BootstrapConfig, run_bootstrap

        config = BootstrapConfig(
            n_samples=30,
            gaussian_fraction=0.5,
            labeled_fraction=0.5,
            seed=888,
            classifier="logistic_regression",
            promote=False,
        )
        result = run_bootstrap(config)

        assert result.success, f"Bootstrap failed: {result.error}"

        from src.database.database import Database

        db = Database(db_url=sqlite_db_url)
        train_ids = db.get_training_signal_ids(result.mlflow_run_id, split="train")
        test_ids = db.get_training_signal_ids(result.mlflow_run_id, split="test")
        db.close()

        assert len(train_ids) > 0, "No train split rows"
        assert len(test_ids) > 0, "No test split rows"
        assert set(train_ids).isdisjoint(set(test_ids)), (
            f"Overlap: {set(train_ids) & set(test_ids)}"
        )

    def test_full_pipeline_model_version_updated_after_registration(
        self,
        tmp_path: Path,
        sqlite_db_url: str,
        local_mlflow_tracking,
    ) -> None:
        """model_version in model_training_data ends as 'v<N>', not stub (SQLite)."""
        from scripts.greenfield_init import BootstrapConfig, run_bootstrap

        config = BootstrapConfig(
            n_samples=30,
            gaussian_fraction=0.5,
            labeled_fraction=0.5,
            seed=1337,
            classifier="logistic_regression",
            promote=False,
        )
        result = run_bootstrap(config)

        assert result.success, f"Bootstrap failed: {result.error}"
        assert result.model_version is not None, "No model_version"

        from src.database.database import Database

        db = Database(db_url=sqlite_db_url)
        cur = db.conn.cursor()
        cur.execute(
            "SELECT DISTINCT model_version FROM model_training_data WHERE mlflow_run_id = ?",
            (result.mlflow_run_id,),
        )
        rows = cur.fetchall()
        db.close()

        assert len(rows) > 0, "No rows in model_training_data"
        for row in rows:
            v = row["model_version"]
            assert v is not None and v.startswith("v"), (
                f"model_version '{v}' is not canonical 'v<N>' format"
            )

    @pytest.mark.timeout(120)
    def test_two_sequential_runs_tracked_independently(
        self,
        tmp_path: Path,
        sqlite_db_url: str,
        local_mlflow_tracking,
    ) -> None:
        """Two sequential bootstrap runs produce separate model_training_data entries."""
        from scripts.greenfield_init import BootstrapConfig, run_bootstrap

        results = []
        for seed in (42, 43):
            config = BootstrapConfig(
                n_samples=25,
                gaussian_fraction=0.5,
                labeled_fraction=0.5,
                seed=seed,
                classifier="logistic_regression",
                promote=False,
            )
            res = run_bootstrap(config)
            assert res.success, f"Bootstrap seed={seed} failed: {res.error}"
            results.append(res)

        run_id_1, run_id_2 = results[0].mlflow_run_id, results[1].mlflow_run_id
        assert run_id_1 != run_id_2, "Both runs produced the same MLflow run_id"

        from src.database.database import Database

        db = Database(db_url=sqlite_db_url)
        ids_1 = db.get_training_signal_ids(run_id_1)
        ids_2 = db.get_training_signal_ids(run_id_2)
        db.close()

        assert len(ids_1) > 0, f"No rows for run_id_1 ({run_id_1})"
        assert len(ids_2) > 0, f"No rows for run_id_2 ({run_id_2})"
        # They use different signal_ids (each run stores its own signals)
        assert set(ids_1).isdisjoint(set(ids_2)), (
            "Two independent runs share signal_ids — that is incorrect"
        )

    def test_run_id_backfill_updates_predictions(
        self,
        tmp_path: Path,
        sqlite_db_url: str,
        local_mlflow_tracking,
    ) -> None:
        """After bootstrap completes, predictions.mlflow_run_id = real MLflow run_id."""
        from scripts.greenfield_init import BootstrapConfig, run_bootstrap

        config = BootstrapConfig(
            n_samples=30,
            gaussian_fraction=0.5,
            labeled_fraction=0.5,
            seed=2222,
            classifier="logistic_regression",
            promote=False,
        )
        result = run_bootstrap(config)

        assert result.success, f"Bootstrap failed: {result.error}"
        assert result.mlflow_run_id

        from src.database.database import Database

        db = Database(db_url=sqlite_db_url)
        cur = db.conn.cursor()
        cur.execute(
            "SELECT COUNT(*) AS n FROM predictions WHERE mlflow_run_id = ?",
            (result.mlflow_run_id,),
        )
        row = cur.fetchone()
        db.close()

        assert row is not None and row["n"] > 0, (
            f"No predictions with mlflow_run_id={result.mlflow_run_id} — "
            "run_id backfill may have failed"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# E. Regression tests: verify the OLD broken behaviour no longer occurs
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegressionOldBrokenBehaviour:
    """
    Regression tests ensuring the pre-FT-5 behaviour no longer occurs.

    The old broken flow:
    1. store_to_database was called AFTER train → no signal_ids during training
    2. train_model was called with a file path, never from_db=True
    3. record_training_split was never called for bootstrap/CC
    """

    def test_step_store_to_database_returns_tuple_not_int(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Verify _step_store_to_database no longer returns a plain int (old API)."""
        from scripts.greenfield_init import BootstrapConfig, _step_store_to_database

        # Remove DB URL so it returns early with empty IDs
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("POSTGRES_TEST_URL", raising=False)
        monkeypatch.delenv("POSTGRES_URL", raising=False)

        paths = _write_bootstrap_json(tmp_path)
        cfg = BootstrapConfig(n_samples=15)

        def cb(step: str, msg: str, frac: float) -> None:
            pass

        result = _step_store_to_database(cfg, paths, uuid.uuid4().hex, cb)

        # Must be a tuple of 3, NOT an int
        assert isinstance(result, tuple), (
            f"_step_store_to_database returned {type(result).__name__}, expected tuple"
        )
        assert len(result) == 3

    def test_step_train_accepts_db_signal_ids_param(self, tmp_path: Path, monkeypatch) -> None:
        """_step_train() must accept db_signal_ids without raising TypeError."""
        # We just check that the parameter is accepted (not that it actually trains)
        import inspect

        from scripts.greenfield_init import _step_train

        sig = inspect.signature(_step_train)
        assert "db_signal_ids" in sig.parameters, (
            "_step_train() does not have db_signal_ids parameter — Phase B not applied"
        )

    def test_train_model_accepts_signal_ids_filter_param(self) -> None:
        """train_model() must accept signal_ids_filter without raising TypeError."""
        import inspect

        from src.training.train import train_model

        sig = inspect.signature(train_model)
        assert "signal_ids_filter" in sig.parameters, (
            "train_model() does not have signal_ids_filter parameter"
        )

    def test_database_get_labeled_accepts_signal_ids_filter(self, tmp_path: Path) -> None:
        """get_labeled_signal_ids() must accept signal_ids_filter."""
        import inspect

        from src.database.database import Database

        sig = inspect.signature(Database.get_labeled_signal_ids)
        assert "signal_ids_filter" in sig.parameters

    def test_database_get_unlabeled_accepts_signal_ids_filter(self, tmp_path: Path) -> None:
        """get_unlabeled_signal_ids() must accept signal_ids_filter."""
        import inspect

        from src.database.database import Database

        sig = inspect.signature(Database.get_unlabeled_signal_ids)
        assert "signal_ids_filter" in sig.parameters

    def test_database_has_get_signal_id_by_prediction_id(self, tmp_path: Path) -> None:
        """Database must expose get_signal_id_by_prediction_id (Phase A helper)."""
        from src.database.database import Database

        assert hasattr(Database, "get_signal_id_by_prediction_id"), (
            "Database.get_signal_id_by_prediction_id missing — Phase A not applied"
        )

    def test_database_has_update_model_version_by_run_id(self) -> None:
        """Database must expose update_model_version_by_run_id (Phase D helper)."""
        from src.database.database import Database

        assert hasattr(Database, "update_model_version_by_run_id"), (
            "Database.update_model_version_by_run_id missing — Phase D not applied"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# F. PostgreSQL: schema checks for model_training_data
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.postgres
class TestModelTrainingDataSchemaPG:
    """PostgreSQL schema checks for model_training_data table."""

    @pytest.fixture(autouse=True)
    def skip_if_no_pg(self):
        if not _pg_available:
            pytest.skip("PostgreSQL not available")

    @pytest.fixture()
    def pg_db(self):
        from src.database.database import Database

        db = Database(db_url=_PG_URL)
        yield db
        db.close()

    def test_model_training_data_table_exists(self, pg_db) -> None:
        """model_training_data table must exist in PostgreSQL."""
        cur = pg_db.conn.cursor()
        cur.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_name = 'model_training_data' AND table_schema = 'public'"
        )
        assert cur.fetchone() is not None

    def test_get_signal_id_by_prediction_id_works_in_pg(self, pg_db) -> None:
        """get_signal_id_by_prediction_id returns None for non-existent ID (PG)."""
        result = pg_db.get_signal_id_by_prediction_id(999999999)
        assert result is None

    def test_update_model_version_nonexistent_run_pg(self, pg_db) -> None:
        """update_model_version_by_run_id on unknown run returns 0 in PostgreSQL."""
        n = pg_db.update_model_version_by_run_id("nonexistent_run_xyz", "v99")
        assert n == 0

    def test_signal_ids_filter_empty_returns_empty_pg(self, pg_db) -> None:
        """Empty signal_ids_filter returns empty list in PostgreSQL."""
        result = pg_db.get_labeled_signal_ids(signal_ids_filter=[])
        assert result == []
