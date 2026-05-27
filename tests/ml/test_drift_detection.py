"""
Tests for drift detection: DriftDetector, data/target/prediction drift, reports, data loading.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.monitoring.drift_detection import (
    DriftDetector,
    load_reference_data,
)


@pytest.fixture
def reference_data():
    return pd.DataFrame(
        {
            "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0] * 20,
            "feature_2": [0.5, 1.5, 2.5, 3.5, 4.5] * 20,
            "feature_3": [10, 20, 30, 40, 50] * 20,
            "ground_truth_label": [0, 1, 0, 1, 0] * 20,
            "predicted_label": [0, 1, 0, 1, 0] * 20,
        }
    )


@pytest.fixture
def drifted_data():
    return pd.DataFrame(
        {
            "feature_1": [10.0, 20.0, 30.0, 40.0, 50.0] * 20,
            "feature_2": [5.0, 10.0, 15.0, 20.0, 25.0] * 20,
            "feature_3": [100, 200, 300, 400, 500] * 20,
            "ground_truth_label": [0, 1, 0, 1, 0] * 20,
            "predicted_label": [1, 1, 1, 1, 1] * 20,
        }
    )


@pytest.fixture
def detector():
    return DriftDetector(
        feature_columns=["feature_1", "feature_2", "feature_3"],
        target_column="ground_truth_label",
        prediction_column="predicted_label",
    )


class TestDriftDetectorInit:
    """Initialization."""

    def test_all_columns(self):
        d = DriftDetector(feature_columns=["f1"], target_column="t", prediction_column="p")
        assert d.feature_columns == ["f1"]
        assert d.target_column == "t"
        assert d.prediction_column == "p"

    def test_optional_columns(self):
        d = DriftDetector(feature_columns=["f1"], target_column=None, prediction_column=None)
        assert d.target_column is None


class TestDataDrift:
    """Data drift detection."""

    def test_no_drift_structure(self, detector, reference_data):
        result = detector.detect_data_drift(reference_data, reference_data.copy())
        assert "drift_detected" in result
        assert "drift_share" in result
        assert result["n_features"] == 3

    def test_with_drift(self, detector, reference_data, drifted_data):
        result = detector.detect_data_drift(reference_data, drifted_data)
        assert "drift_detected" in result
        assert result["n_features"] == 3

    def test_custom_threshold(self, detector, reference_data):
        result = detector.detect_data_drift(
            reference_data, reference_data.copy(), stattest_threshold=0.01
        )
        assert result["threshold"] == 0.01


class TestTargetDrift:
    """Target drift detection."""

    def test_no_drift(self, detector, reference_data):
        result = detector.detect_target_drift(reference_data, reference_data.copy())
        assert "drift_detected" in result
        assert result["target_column"] == "ground_truth_label"

    def test_no_target_column_raises(self, reference_data):
        d = DriftDetector(feature_columns=["feature_1"], target_column=None)
        with pytest.raises(ValueError, match="target_column must be set"):
            d.detect_target_drift(reference_data, reference_data)


class TestPredictionDrift:
    """Prediction drift detection."""

    def test_no_drift(self, detector):
        data = pd.DataFrame(
            {
                "feature_1": [1.0, 2.0, 3.0] * 30,
                "feature_2": [0.5, 1.5, 2.5] * 30,
                "feature_3": [10, 20, 30] * 30,
                "predicted_label": [0, 1, 0] * 30,
            }
        )
        result = detector.detect_prediction_drift(data, data.copy())
        assert "drift_detected" in result
        assert result["prediction_column"] == "predicted_label"

    def test_no_prediction_column_raises(self, reference_data):
        d = DriftDetector(feature_columns=["feature_1"], prediction_column=None)
        with pytest.raises(ValueError, match="prediction_column must be set"):
            d.detect_prediction_drift(reference_data, reference_data)


class TestDriftReport:
    """Report generation and summary saving."""

    def test_generate_report(self, detector, reference_data, tmp_path):
        output = tmp_path / "report.html"
        summary = detector.generate_drift_report(reference_data, reference_data.copy(), output)
        assert output.exists()
        assert "data_drift" in summary

    def test_save_summary(self, detector, tmp_path):
        summary = {"drift_detected": True, "drift_share": 0.33}
        out = tmp_path / "summary.json"
        detector.save_drift_summary(summary, out)
        assert out.exists()
        with open(out) as f:
            assert json.load(f) == summary


class TestDataLoading:
    """load_reference_data / load_current_data."""

    def test_csv(self, tmp_path):
        df = pd.DataFrame({"f1": [1, 2, 3], "label": [0, 1, 0]})
        path = tmp_path / "ref.csv"
        df.to_csv(path, index=False)
        pd.testing.assert_frame_equal(load_reference_data(path), df)

    def test_parquet(self, tmp_path):
        df = pd.DataFrame({"f1": [1, 2, 3], "label": [0, 1, 0]})
        path = tmp_path / "ref.parquet"
        df.to_parquet(path, index=False)
        pd.testing.assert_frame_equal(load_reference_data(path), df)

    def test_unsupported_format(self, tmp_path):
        path = tmp_path / "ref.txt"
        path.touch()
        with pytest.raises(ValueError, match="Unsupported format"):
            load_reference_data(path)
