"""
Tests for prediction latency thresholds.

Validates:
- Feature extraction completes within time limits
- Model prediction completes within time limits
- Signal generation performance is acceptable
"""

import time

import numpy as np

from src.signal_processing.signal_generator import generate_gaussian_peak
from src.signal_processing.signal_models import SignalData


class TestFeatureExtractionLatency:
    """Ensure feature extraction meets latency targets."""

    def test_single_signal_under_200ms(self, sample_time_array):
        """Feature extraction for one signal < 200ms."""
        from src.signal_processing.feature_extractor import extract_features

        amplitude = generate_gaussian_peak(sample_time_array, mu=50.0, sigma=5.0, height=1.0)
        signal = SignalData(
            time=sample_time_array.tolist(), amplitude=amplitude.tolist(), shape_type="gaussian"
        )

        start = time.perf_counter()
        features = extract_features(signal)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 200, f"Feature extraction took {elapsed_ms:.1f}ms (limit: 200ms)"
        assert features is not None

    def test_batch_10_signals_under_2s(self, sample_time_array):
        """Batch feature extraction for 10 signals < 2 seconds."""
        from src.signal_processing.feature_extractor import extract_features

        signals = [
            SignalData(
                time=sample_time_array.tolist(),
                amplitude=generate_gaussian_peak(
                    sample_time_array, mu=50.0 + i, sigma=5.0, height=1.0
                ).tolist(),
                shape_type="gaussian",
            )
            for i in range(10)
        ]

        start = time.perf_counter()
        for sig in signals:
            extract_features(sig)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 2000, f"Batch extraction took {elapsed_ms:.1f}ms (limit: 2000ms)"


class TestModelPredictionLatency:
    """Ensure model prediction meets latency targets."""

    def test_prediction_under_500ms(self, bootstrap_model_path, sample_time_array):
        """Single prediction < 500ms (includes feature extraction)."""
        from src.training import predict

        amplitude = generate_gaussian_peak(sample_time_array, mu=5.0, sigma=0.5, height=1.0)

        start = time.perf_counter()
        result = predict(
            time_values=sample_time_array,
            amplitude_values=amplitude,
            model_path=str(bootstrap_model_path),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 500, f"Prediction took {elapsed_ms:.1f}ms (limit: 500ms)"
        assert "predicted_label" in result


class TestSignalGenerationPerformance:
    """Ensure signal generation is fast enough for testing."""

    def test_generate_100_signals_under_2s(self):
        """Generating 100 signals < 2 seconds."""
        t = np.linspace(0, 10, 101)

        start = time.perf_counter()
        for _ in range(100):
            generate_gaussian_peak(t, mu=5.0, sigma=0.5, height=1.0)
        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, f"Generating 100 signals took {elapsed:.2f}s (limit: 2s)"
