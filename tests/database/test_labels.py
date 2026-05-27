"""
Tests for sparse label injection and coverage metrics.

Validates:
- inject_sparse_label stores ground truth
- get_label_coverage computes coverage percentage
- Label injection for non-existent predictions handled
- Label source tracking
- Device registration and management
"""

import pytest

from src.database import generate_device_id


class TestInjectSparseLabel:
    """Tests for Database.inject_sparse_label()."""

    def test_inject_label_success(self, db, stored_prediction):
        """Successfully inject a ground truth label."""
        label_id = db.inject_sparse_label(
            prediction_id=stored_prediction,
            ground_truth_label=0,
            label_source="human_expert",
        )
        assert label_id is not None

    def test_inject_label_updates_prediction(self, db, stored_prediction):
        """Injected label is retrievable from prediction record."""
        db.inject_sparse_label(
            prediction_id=stored_prediction,
            ground_truth_label=1,
            label_source="automated_test",
        )
        record = db.get_prediction(stored_prediction)
        assert record["ground_truth_label"] == 1

    def test_inject_label_source_tracked(self, db, stored_prediction):
        """Label source is stored with the injection."""
        db.inject_sparse_label(
            prediction_id=stored_prediction,
            ground_truth_label=0,
            label_source="operator_review",
        )
        record = db.get_prediction(stored_prediction)
        assert record["label_source"] == "operator_review"

    def test_inject_label_nonexistent_prediction(self, db):
        """Injecting label for non-existent prediction raises error."""
        with pytest.raises(Exception):  # noqa: B017
            db.inject_sparse_label(
                prediction_id=99999,
                ground_truth_label=0,
                label_source="test",
            )


class TestLabelCoverage:
    """Tests for Database.get_label_coverage()."""

    def test_coverage_no_predictions(self, db):
        """Empty database returns zero coverage."""
        result = db.get_label_coverage()
        assert result["total_predictions"] == 0
        assert result["label_coverage"] == 0.0

    def test_coverage_no_labels(self, db, device_id, sample_signal):
        """Predictions without labels show zero coverage."""
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
        result = db.get_label_coverage()
        assert result["total_predictions"] == 3
        assert result["labeled_predictions"] == 0
        assert result["label_coverage"] == 0.0

    def test_coverage_partial_labels(self, db, device_id, sample_signal):
        """Coverage reflects fraction of labeled predictions."""
        t, amp = sample_signal
        pids = []
        for _ in range(4):
            pid = db.store_prediction(
                device_id=device_id,
                time_values=t,
                amplitude_values=amp,
                predicted_label=0,
                model_version="v1.0",
                features={},
                prediction_confidence=0.9,
            )
            pids.append(pid)

        # Label only 2 of 4
        db.inject_sparse_label(prediction_id=pids[0], ground_truth_label=0, label_source="test")
        db.inject_sparse_label(prediction_id=pids[1], ground_truth_label=1, label_source="test")

        result = db.get_label_coverage()
        assert result["total_predictions"] == 4
        assert result["labeled_predictions"] == 2
        assert result["label_coverage"] == pytest.approx(0.5, abs=0.01)


class TestDeviceManagement:
    """Tests for device registration and queries."""

    def test_register_device(self, db):
        """Register a device with all fields."""
        did = generate_device_id()
        result = db.register_device(
            device_id=did,
            device_name="Alpha Device",
            device_type="Sensor-B",
            location="Floor-2",
            status="active",
        )
        assert result == did

    def test_register_device_auto_id(self, db):
        """Device ID auto-generated when not provided."""
        did = db.register_device(device_name="Auto Device", status="active")
        assert did is not None
        assert len(did) == 36
        assert did.count("-") == 4

    def test_register_device_invalid_status(self, db):
        """Invalid device status raises ValueError."""
        with pytest.raises(ValueError, match="Invalid status"):
            db.register_device(device_id=generate_device_id(), status="broken")

    def test_generate_device_id_format(self):
        """generate_device_id returns UUID-format string."""
        did = generate_device_id()
        assert len(did) == 36
        assert did.count("-") == 4
