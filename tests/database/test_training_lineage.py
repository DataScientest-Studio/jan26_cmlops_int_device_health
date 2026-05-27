"""
Tests for training data lineage features:
- get_labeled_signal_ids(window_days=...)  SQL window filter
- get_unlabeled_signal_ids(window_days=...) SQL window filter
- record_training_split / get_training_signal_ids
- cleanup_old_training_splits utility
- train_model(from_db=True) DB-backed training path
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.database import Database, generate_device_id

# ── helpers ───────────────────────────────────────────────────────────────────


def _register_and_store(db: Database, label: int | None = 1, age_days: int = 0) -> int:
    """Store a prediction + raw signal and optionally a sparse label.

    Returns the *signal_id* (not prediction_id) because the training pipeline
    works with signal_ids.
    """
    import numpy as np

    device_id = generate_device_id()
    db.register_device(
        device_id=device_id,
        device_name="test-dev",
        device_type="sensor",
        location="lab",
        status="active",
    )
    t = np.linspace(0, 100, 101).tolist()
    amp = np.exp(-((np.array(t) - 50) ** 2) / (2 * 5.0**2)).tolist()
    pred_id = db.store_prediction(
        device_id=device_id,
        time_values=t,
        amplitude_values=amp,
        predicted_label=label if label is not None else 0,
        model_version="v0",
        features={},
        prediction_confidence=0.9,
        deployment_mode="cloud",  # get_labeled/unlabeled_signal_ids filters on cloud
    )

    # Back-date created_at in raw_signals if requested (SQLite only in tests)
    if age_days:
        backdated = (datetime.now(timezone.utc) - timedelta(days=age_days)).isoformat()
        db.conn.execute(
            "UPDATE raw_signals SET created_at=? WHERE prediction_id=?",
            (backdated, pred_id),
        )
        db.conn.commit()

    if label is not None:
        db.inject_sparse_label(pred_id, label, "test")

    # Retrieve the signal_id
    row = db.conn.execute(
        "SELECT signal_id FROM raw_signals WHERE prediction_id=?", (pred_id,)
    ).fetchone()
    return int(row[0])


# ── window filter tests ───────────────────────────────────────────────────────


class TestWindowFilter:
    """SQL-level window filter on created_at."""

    def test_labeled_no_filter_returns_all(self, db: Database):
        sid1 = _register_and_store(db, label=1, age_days=0)
        sid2 = _register_and_store(db, label=0, age_days=5)
        sid3 = _register_and_store(db, label=1, age_days=40)
        result = db.get_labeled_signal_ids()
        assert sid1 in result
        assert sid2 in result
        assert sid3 in result

    def test_labeled_window_excludes_old(self, db: Database):
        recent_sid = _register_and_store(db, label=1, age_days=1)
        old_sid = _register_and_store(db, label=0, age_days=60)
        result = db.get_labeled_signal_ids(window_days=30)
        assert recent_sid in result
        assert old_sid not in result

    def test_unlabeled_window_excludes_old(self, db: Database):
        recent_sid = _register_and_store(db, label=None, age_days=1)
        old_sid = _register_and_store(db, label=None, age_days=60)
        result = db.get_unlabeled_signal_ids(window_days=30)
        assert recent_sid in result
        assert old_sid not in result

    def test_window_none_is_no_op(self, db: Database):
        sids = [_register_and_store(db, label=1, age_days=d) for d in (1, 15, 45)]
        result = db.get_labeled_signal_ids(window_days=None)
        for sid in sids:
            assert sid in result


# ── training split record/query ───────────────────────────────────────────────


class TestTrainingSplit:
    """record_training_split and get_training_signal_ids."""

    def test_record_and_retrieve(self, db: Database):
        train_ids = [_register_and_store(db, label=1) for _ in range(4)]
        test_ids = [_register_and_store(db, label=0) for _ in range(2)]
        db.record_training_split(
            mlflow_run_id="run-abc-123",
            train_signal_ids=train_ids,
            test_signal_ids=test_ids,
            model_version="v1",
        )
        stored_train = db.get_training_signal_ids("run-abc-123", "train")
        stored_test = db.get_training_signal_ids("run-abc-123", "test")
        assert set(stored_train) == set(train_ids)
        assert set(stored_test) == set(test_ids)

    def test_empty_split(self, db: Database):
        db.record_training_split(
            "run-empty", train_signal_ids=[], test_signal_ids=[], model_version="v0"
        )
        assert db.get_training_signal_ids("run-empty", "train") == []
        assert db.get_training_signal_ids("run-empty", "test") == []

    def test_unknown_run_returns_empty(self, db: Database):
        assert db.get_training_signal_ids("nonexistent-run", "train") == []

    def test_duplicate_insert_is_idempotent(self, db: Database):
        """record_training_split with INSERT OR IGNORE: duplicate rows silently ignored."""
        sid = _register_and_store(db, label=1)
        db.record_training_split(
            "run-dup", train_signal_ids=[sid], test_signal_ids=[], model_version="v1"
        )
        # second call with the same signal_id should not raise
        db.record_training_split(
            "run-dup", train_signal_ids=[sid], test_signal_ids=[], model_version="v1"
        )
        result = db.get_training_signal_ids("run-dup", "train")
        assert result.count(sid) == 1  # still exactly one entry


# ── cleanup utility ───────────────────────────────────────────────────────────


class TestCleanupOldTrainingSplits:
    """cleanup_old_training_splits() in train.py."""

    def test_keeps_recent_and_deletes_old(self, tmp_path: Path):
        from src.training.train import cleanup_old_training_splits

        splits_dir = tmp_path / "data" / "processed" / "training_splits"
        splits_dir.mkdir(parents=True)
        # Create 15 "run" directories
        dirs = []
        for i in range(15):
            d = splits_dir / f"run-{i:04d}"
            d.mkdir()
            (d / "train.json").write_text("{}")
            dirs.append(d)

        result = cleanup_old_training_splits(keep_n=5, repo_root=tmp_path)
        remaining = [d for d in splits_dir.iterdir() if d.is_dir()]
        assert len(remaining) == 5
        assert result["kept"] == 5
        assert result["deleted"] == 10

    def test_champion_always_kept(self, tmp_path: Path):
        from src.training.train import cleanup_old_training_splits

        splits_dir = tmp_path / "data" / "processed" / "training_splits"
        splits_dir.mkdir(parents=True)
        # Champion must be older so it doesn't rank in the top keep_n by mtime
        import time

        champ_dir = splits_dir / "champ-run"
        champ_dir.mkdir()
        time.sleep(0.02)  # ensure champ_dir is older than the others
        for i in range(12):
            d = splits_dir / f"run-{i:03d}"
            d.mkdir()

        result = cleanup_old_training_splits(
            keep_n=5, repo_root=tmp_path, champion_run_id="champ-run"
        )
        assert champ_dir.exists(), "Champion directory must never be deleted"
        assert result["kept"] == 6  # champion + 5 most-recent
        assert result["deleted"] == 7

    def test_no_splits_dir_returns_zeros(self, tmp_path: Path):
        from src.training.train import cleanup_old_training_splits

        result = cleanup_old_training_splits(keep_n=10, repo_root=tmp_path)
        assert result == {"kept": 0, "deleted": 0}


# ── MD5 hash helper ───────────────────────────────────────────────────────────


class TestMd5Hash:
    """_export_split_json + MD5 computation via hashlib.md5."""

    def test_md5_is_deterministic(self, tmp_path: Path):
        """Same content → same MD5."""
        from hashlib import md5

        data = b"train-data-content" + b"test-data-content"
        h1 = md5(data, usedforsecurity=False).hexdigest()
        h2 = md5(data, usedforsecurity=False).hexdigest()
        assert h1 == h2

    def test_md5_differs_for_different_content(self, tmp_path: Path):
        from hashlib import md5

        h1 = md5(b"data-a", usedforsecurity=False).hexdigest()
        h2 = md5(b"data-b", usedforsecurity=False).hexdigest()
        assert h1 != h2


# ── from_db=True training path ────────────────────────────────────────────────


class TestFromDbTraining:
    """train_model(from_db=True) DB-backed data loading."""

    @pytest.fixture
    def populated_db(self, tmp_path: Path) -> Database:
        """DB with 20 labeled signals (10 Gaussian + 10 Lorentzian for K-means diversity)."""
        from src.signal_processing.signal_generator import generate_signal

        db = Database(db_path=tmp_path / "test.db")
        device_id = generate_device_id()
        db.register_device(
            device_id=device_id,
            device_name="test-dev",
            device_type="sensor",
            location="lab",
            status="active",
        )
        for i in range(10):
            sig = generate_signal("gaussian", drift_scenario="baseline", seed=i)
            pred_id = db.store_prediction(
                device_id=device_id,
                time_values=sig.signal.time
                if isinstance(sig.signal.time, list)
                else sig.signal.time.tolist(),
                amplitude_values=sig.signal.amplitude
                if isinstance(sig.signal.amplitude, list)
                else sig.signal.amplitude.tolist(),
                predicted_label=0,
                model_version="v0",
                features={},
                prediction_confidence=0.9,
                deployment_mode="cloud",
            )
            db.inject_sparse_label(pred_id, 0, "test")
        for i in range(10, 20):
            sig = generate_signal("lorentzian", drift_scenario="baseline", seed=i)
            pred_id = db.store_prediction(
                device_id=device_id,
                time_values=sig.signal.time
                if isinstance(sig.signal.time, list)
                else sig.signal.time.tolist(),
                amplitude_values=sig.signal.amplitude
                if isinstance(sig.signal.amplitude, list)
                else sig.signal.amplitude.tolist(),
                predicted_label=1,
                model_version="v0",
                features={},
                prediction_confidence=0.9,
                deployment_mode="cloud",
            )
            db.inject_sparse_label(pred_id, 1, "test")
        return db

    def test_raises_without_db(self):
        from src.training.train import train_model

        with pytest.raises(ValueError, match="requires db parameter"):
            train_model(from_db=True, db=None, use_mlflow=False)

    def test_raises_no_labeled_signals(self, tmp_path: Path):
        from src.training.train import train_model

        empty_db = Database(db_path=tmp_path / "empty.db")
        try:
            with pytest.raises(ValueError):
                train_model(
                    from_db=True,
                    db=empty_db,
                    model_output_path=tmp_path / "m.pkl",
                    use_mlflow=False,
                )
        finally:
            empty_db.close()

    def test_trains_successfully_from_db(self, tmp_path: Path, populated_db: Database):
        from src.training.train import train_model

        model_path = tmp_path / "model_from_db.pkl"
        try:
            results = train_model(
                from_db=True,
                db=populated_db,
                model_output_path=model_path,
                use_mlflow=False,
                allow_unlabeled=False,
            )
        finally:
            populated_db.close()

        assert model_path.exists()
        assert results["train_samples"] > 0
        assert 0.0 <= results["train_accuracy"] <= 1.0

    def test_from_db_records_split(self, tmp_path: Path):
        from src.signal_processing.signal_generator import generate_signal
        from src.training.train import train_model

        db = Database(db_path=tmp_path / "lineage.db")
        device_id = generate_device_id()
        db.register_device(
            device_id=device_id,
            device_name="dev",
            device_type="sensor",
            location="lab",
            status="active",
        )
        for i in range(10):
            sig = generate_signal("gaussian", drift_scenario="baseline", seed=i)
            pred_id = db.store_prediction(
                device_id=device_id,
                time_values=sig.signal.time
                if isinstance(sig.signal.time, list)
                else sig.signal.time.tolist(),
                amplitude_values=sig.signal.amplitude
                if isinstance(sig.signal.amplitude, list)
                else sig.signal.amplitude.tolist(),
                predicted_label=0,
                model_version="v0",
                features={},
                prediction_confidence=0.9,
                deployment_mode="cloud",
            )
            db.inject_sparse_label(pred_id, 0, "test")
        for i in range(10, 20):
            sig = generate_signal("lorentzian", drift_scenario="baseline", seed=i)
            pred_id = db.store_prediction(
                device_id=device_id,
                time_values=sig.signal.time
                if isinstance(sig.signal.time, list)
                else sig.signal.time.tolist(),
                amplitude_values=sig.signal.amplitude
                if isinstance(sig.signal.amplitude, list)
                else sig.signal.amplitude.tolist(),
                predicted_label=1,
                model_version="v0",
                features={},
                prediction_confidence=0.9,
                deployment_mode="cloud",
            )
            db.inject_sparse_label(pred_id, 1, "test")

        model_path = tmp_path / "m.pkl"
        try:
            results = train_model(
                from_db=True,
                db=db,
                model_output_path=model_path,
                use_mlflow=False,
                allow_unlabeled=False,
            )
        finally:
            pass  # keep open to query

        run_id = results.get("mlflow_run_id")
        if run_id:
            train_ids = db.get_training_signal_ids(run_id, "train")
            test_ids = db.get_training_signal_ids(run_id, "test")
            assert len(train_ids) + len(test_ids) > 0
        db.close()
