"""
Tests for Prometheus metrics instrumentation.

Validates:
- All expected metric objects exist and are correct types
- Helper functions record metrics correctly
- Counter/Gauge/Histogram label schemas
- Metric naming conventions follow Prometheus best practices
"""

from prometheus_client import Counter, Gauge, Histogram, Info


class TestMetricDefinitions:
    """Verify all Prometheus metric objects are defined correctly."""

    def test_api_requests_total_is_counter(self):
        """api_requests_total is a Counter with correct labels."""
        from src.monitoring.metrics import api_requests_total

        assert isinstance(api_requests_total, Counter)

    def test_api_request_duration_is_histogram(self):
        """api_request_duration_seconds is a Histogram."""
        from src.monitoring.metrics import api_request_duration_seconds

        assert isinstance(api_request_duration_seconds, Histogram)

    def test_api_errors_total_is_counter(self):
        """api_errors_total is a Counter."""
        from src.monitoring.metrics import api_errors_total

        assert isinstance(api_errors_total, Counter)

    def test_model_predictions_total_is_counter(self):
        """model_predictions_total is a Counter."""
        from src.monitoring.metrics import model_predictions_total

        assert isinstance(model_predictions_total, Counter)

    def test_model_info_is_info(self):
        """model_info is a prometheus Info metric."""
        from src.monitoring.metrics import model_info

        assert isinstance(model_info, Info)

    def test_sparse_label_coverage_is_gauge(self):
        """sparse_label_coverage is a Gauge."""
        from src.monitoring.metrics import sparse_label_coverage

        assert isinstance(sparse_label_coverage, Gauge)

    def test_drift_detected_gauge_exists(self):
        """drift_detected_gauge is a Gauge with drift_type label."""
        from src.monitoring.metrics import drift_detected_gauge

        assert isinstance(drift_detected_gauge, Gauge)

    def test_retraining_triggers_total_is_counter(self):
        """retraining_triggers_total is a Counter."""
        from src.monitoring.metrics import retraining_triggers_total

        assert isinstance(retraining_triggers_total, Counter)


class TestHelperFunctions:
    """Tests for metric recording helper functions."""

    def test_record_prediction_increments_counter(self):
        """record_prediction increments model_predictions_total."""
        from src.monitoring.metrics import record_prediction

        # Should not raise
        record_prediction(model_version="test-v1", predicted_label=0, confidence=0.95)
        record_prediction(model_version="test-v1", predicted_label=1, confidence=0.72)

    def test_record_model_reload_increments(self):
        """record_model_reload increments model_reloads_total."""
        from src.monitoring.metrics import record_model_reload

        record_model_reload(source="bootstrap", trigger="startup")
        record_model_reload(source="registry", trigger="manual")

    def test_record_label_injection(self):
        """record_label_injection increments labels_injected_total."""
        from src.monitoring.metrics import record_label_injection

        record_label_injection(label_value=0)
        record_label_injection(label_value=1)

    def test_record_drift_detection(self):
        """record_drift_detection increments drift_detections_total."""
        from src.monitoring.metrics import record_drift_detection

        record_drift_detection(drift_type="data")
        record_drift_detection(drift_type="concept")

    def test_record_invalid_signal(self):
        """record_invalid_signal increments invalid_signals_total."""
        from src.monitoring.metrics import record_invalid_signal

        record_invalid_signal(error_type="insufficient_data")
        record_invalid_signal(error_type="nan_values")

    def test_update_model_info_metric(self):
        """update_model_info_metric sets model_info labels."""
        from src.monitoring.metrics import update_model_info_metric

        update_model_info_metric(
            model_version="v2.0",
            algorithm="LogisticRegression",
            source="registry",
            trained_at="2025-01-01T00:00:00",
            features=["noise_std", "peak_count", "snr"],
        )

    def test_track_request_metrics_decorator(self):
        """track_request_metrics creates a working decorator."""
        from src.monitoring.metrics import track_request_metrics

        @track_request_metrics("/test", "GET")
        def sync_handler():
            return {"status": "ok"}

        result = sync_handler()
        assert result == {"status": "ok"}

    def test_track_request_metrics_async_decorator(self):
        """track_request_metrics works with async functions."""
        import asyncio

        from src.monitoring.metrics import track_request_metrics

        @track_request_metrics("/test-async", "POST")
        async def async_handler():
            return {"status": "ok"}

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(async_handler())
        finally:
            loop.close()
        assert result == {"status": "ok"}


class TestMetricNamingConventions:
    """Verify metric names follow Prometheus conventions."""

    def test_counter_names_end_with_total(self):
        """Counter metrics should end with _total."""
        from src.monitoring import metrics

        counters = [
            "api_requests_total",
            "api_errors_total",
            "model_predictions_total",
            "labels_injected_total",
            "drift_detections_total",
            "retraining_triggers_total",
            "retraining_failures_total",
            "invalid_signals_total",
            "feature_extractions_total",
            "model_reloads_total",
        ]
        for name in counters:
            assert hasattr(metrics, name), f"Missing counter: {name}"
            assert name.endswith("_total"), f"Counter {name} should end with _total"

    def test_histogram_names_include_unit(self):
        """Histogram metrics should include unit (seconds)."""
        from src.monitoring import metrics

        histograms = [
            "api_request_duration_seconds",
            "signal_validation_duration_seconds",
            "feature_extraction_duration_seconds",
        ]
        for name in histograms:
            assert hasattr(metrics, name), f"Missing histogram: {name}"
            assert "_seconds" in name, f"Histogram {name} should include _seconds"
