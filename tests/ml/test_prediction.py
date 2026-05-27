"""
Tests for prediction pipeline: single predict, batch, probabilities, error handling.
"""

import pytest

from src.signal_processing.signal_generator import generate_signal
from src.training import predict, predict_batch


class TestPredictSingle:
    """Single-signal prediction."""

    def test_basic_prediction(self, trained_model):
        signal = generate_signal("gaussian", drift_scenario="baseline", seed=42).signal
        result = predict(
            time_values=signal.time,
            amplitude_values=signal.amplitude,
            model_path=trained_model,
            return_probabilities=True,
        )
        assert result["predicted_label"] in (0, 1)
        assert 0.0 <= result["confidence"] <= 1.0
        assert "healthy" in result["probabilities"]
        assert "unhealthy" in result["probabilities"]
        assert result["model_version"] == "test_v1"

    def test_without_probabilities(self, trained_model):
        signal = generate_signal("gaussian", drift_scenario="baseline", seed=42).signal
        result = predict(
            time_values=signal.time,
            amplitude_values=signal.amplitude,
            model_path=trained_model,
            return_probabilities=False,
        )
        assert "predicted_label" in result
        assert "probabilities" not in result

    def test_with_nan_values(self, trained_model):
        signal = generate_signal("gaussian", drift_scenario="baseline", seed=42).signal
        amplitude = list(signal.amplitude)
        amplitude[10] = None
        amplitude[50] = None
        result = predict(
            time_values=signal.time,
            amplitude_values=amplitude,
            model_path=trained_model,
        )
        assert result["predicted_label"] in (0, 1)

    def test_length_mismatch_raises(self, trained_model):
        with pytest.raises(ValueError, match="must have same length"):
            predict(
                time_values=[float(i) for i in range(100)],
                amplitude_values=[2.5] * 101,
                model_path=trained_model,
            )

    def test_signal_too_short_raises(self, trained_model):
        with pytest.raises(ValueError, match="Signal too short"):
            predict(
                time_values=[float(i) for i in range(50)],
                amplitude_values=[2.5] * 50,
                model_path=trained_model,
            )

    def test_features_in_result(self, trained_model):
        signal = generate_signal("gaussian", drift_scenario="baseline", seed=42).signal
        result = predict(
            time_values=signal.time,
            amplitude_values=signal.amplitude,
            model_path=trained_model,
        )
        assert "features" in result
        for key in ("fwhm", "peak_height", "peak_area", "noise_level", "snr", "peak_center"):
            assert key in result["features"]


class TestPredictBatch:
    """Batch prediction."""

    def test_batch_prediction(self, trained_model):
        signals = [
            (
                generate_signal("gaussian", seed=i).signal.time,
                generate_signal("gaussian", seed=i).signal.amplitude,
            )
            for i in range(5)
        ]
        results = predict_batch(signals=signals, model_path=trained_model)
        assert len(results) == 5
        for r in results:
            assert r["predicted_label"] in (0, 1)
