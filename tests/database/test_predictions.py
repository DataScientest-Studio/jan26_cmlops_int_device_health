"""
Tests for prediction storage and retrieval.

Validates:
- store_prediction creates records with correct fields
- get_prediction retrieves stored data
- get_predictions_by_device returns filtered results
- calculate_realized_accuracy with labeled data
- Prediction lineage fields (mlflow_run_id, git_sha, etc.)
"""

import pytest


class TestStorePrediction:
    """Tests for Database.store_prediction()."""

    def test_store_prediction_returns_id(self, db, device_id, sample_signal):
        """store_prediction returns positive integer ID."""
        t, amp = sample_signal
        pid = db.store_prediction(
            device_id=device_id,
            time_values=t,
            amplitude_values=amp,
            predicted_label=0,
            model_version="v1.0",
            features={"noise_std": 0.01},
            prediction_confidence=0.88,
        )
        assert isinstance(pid, int)
        assert pid > 0

    def test_store_prediction_with_lineage(self, db, device_id, sample_signal):
        """Prediction stores full lineage fields."""
        t, amp = sample_signal
        pid = db.store_prediction(
            device_id=device_id,
            time_values=t,
            amplitude_values=amp,
            predicted_label=1,
            model_version="v2.0",
            features={"snr": 12.0},
            prediction_confidence=0.75,
            mlflow_run_id="abc123",
            git_sha="deadbeef",
            dvc_data_hash="hash456",
            airflow_run_id="dag_run_1",
        )
        record = db.get_prediction(pid)
        assert record is not None
        assert record["mlflow_run_id"] == "abc123"
        assert record["git_sha"] == "deadbeef"
        assert record["dvc_data_hash"] == "hash456"
        assert record["airflow_run_id"] == "dag_run_1"

    def test_store_multiple_predictions(self, db, device_id, sample_signal):
        """Multiple predictions get unique IDs."""
        t, amp = sample_signal
        ids = []
        for i in range(5):
            pid = db.store_prediction(
                device_id=device_id,
                time_values=t,
                amplitude_values=amp,
                predicted_label=i % 2,
                model_version="v1.0",
                features={"noise_std": 0.01 * i},
                prediction_confidence=0.8 + i * 0.02,
            )
            ids.append(pid)
        assert len(set(ids)) == 5


class TestGetPrediction:
    """Tests for Database.get_prediction()."""

    def test_get_prediction_found(self, db, stored_prediction):
        """Retrieve existing prediction by ID."""
        record = db.get_prediction(stored_prediction)
        assert record is not None
        assert record["prediction_id"] == stored_prediction
        assert record["predicted_label"] == 0
        assert record["model_version"] == "test-v1.0"

    def test_get_prediction_not_found(self, db):
        """Non-existent prediction returns None."""
        assert db.get_prediction(99999) is None

    def test_get_prediction_has_timestamp(self, db, stored_prediction):
        """Stored prediction includes a timestamp."""
        record = db.get_prediction(stored_prediction)
        assert record["timestamp"] is not None

    def test_get_predictions_by_device(self, db, device_id, sample_signal):
        """Filter predictions by device ID."""
        t, amp = sample_signal
        for _ in range(3):
            db.store_prediction(
                device_id=device_id,
                time_values=t,
                amplitude_values=amp,
                predicted_label=0,
                model_version="v1.0",
                features={},
                prediction_confidence=0.9,
            )
        results = db.get_predictions_by_device(device_id)
        assert len(results) >= 3


class TestRealizedAccuracy:
    """Tests for Database.calculate_realized_accuracy()."""

    def test_accuracy_no_labels(self, db):
        """No labeled predictions returns zero accuracy."""
        result = db.calculate_realized_accuracy(lookback_days=30)
        assert result["total_labeled"] == 0

    def test_accuracy_with_labels(self, db, device_id, sample_signal):
        """Accuracy computed correctly from labeled predictions."""
        t, amp = sample_signal
        # Store predictions and inject correct labels
        for label in [0, 0, 1, 1, 0]:
            pid = db.store_prediction(
                device_id=device_id,
                time_values=t,
                amplitude_values=amp,
                predicted_label=label,
                model_version="v1.0",
                features={},
                prediction_confidence=0.9,
            )
            db.inject_sparse_label(
                prediction_id=pid,
                ground_truth_label=label,  # All correct
                label_source="test",
            )

        result = db.calculate_realized_accuracy(lookback_days=30)
        assert result["total_labeled"] == 5
        assert result["accuracy"] == 1.0

    def test_accuracy_partial_correctness(self, db, device_id, sample_signal):
        """Accuracy reflects partial correctness."""
        t, amp = sample_signal
        predictions = [0, 0, 1, 1]
        ground_truth = [0, 1, 1, 0]  # 2 correct out of 4

        for pred, gt in zip(predictions, ground_truth, strict=False):
            pid = db.store_prediction(
                device_id=device_id,
                time_values=t,
                amplitude_values=amp,
                predicted_label=pred,
                model_version="v1.0",
                features={},
                prediction_confidence=0.8,
            )
            db.inject_sparse_label(
                prediction_id=pid,
                ground_truth_label=gt,
                label_source="test",
            )

        result = db.calculate_realized_accuracy(lookback_days=30)
        assert result["accuracy"] == pytest.approx(0.5, abs=0.01)
