"""
Monitoring Module

This module provides:
- Prometheus metrics exporters
- EvidentlyAI drift detection (to be implemented)
- Performance monitoring and alerting
- Model health checks
"""

from .metrics import (
    api_errors_total,
    api_request_duration_seconds,
    api_requests_in_progress,
    # System metrics
    api_requests_total,
    drift_detected_gauge,
    drift_detections_total,
    drift_reports_total,
    feature_extraction_duration_seconds,
    # Feature metrics
    feature_extractions_total,
    # Data quality metrics
    invalid_signals_total,
    labels_in_db_total,
    # Business metrics
    labels_injected_total,
    model_accuracy_gauge,
    model_info,
    model_last_reload_timestamp_seconds,
    model_load_time_seconds,
    model_prediction_confidence,
    # Model metrics
    model_predictions_total,
    model_reloads_total,
    predictions_in_db_total,
    record_drift_detection,
    record_invalid_signal,
    record_label_injection,
    record_model_reload,
    record_prediction,
    record_retraining_trigger,
    retraining_failures_total,
    retraining_triggers_total,
    signal_validation_duration_seconds,
    sparse_label_coverage,
    # Helper functions
    track_request_metrics,
    update_model_info_metric,
)

__all__ = [
    # System metrics
    "api_requests_total",
    "api_request_duration_seconds",
    "api_errors_total",
    "api_requests_in_progress",
    # Model metrics
    "model_predictions_total",
    "model_prediction_confidence",
    "model_info",
    "model_reloads_total",
    "model_load_time_seconds",
    "model_last_reload_timestamp_seconds",
    "model_accuracy_gauge",
    # Business metrics
    "labels_injected_total",
    "labels_in_db_total",
    "predictions_in_db_total",
    "drift_detections_total",
    "drift_reports_total",
    "drift_detected_gauge",
    "retraining_triggers_total",
    "retraining_failures_total",
    "sparse_label_coverage",
    # Data quality metrics
    "invalid_signals_total",
    "signal_validation_duration_seconds",
    # Feature metrics
    "feature_extractions_total",
    "feature_extraction_duration_seconds",
    # Helper functions
    "track_request_metrics",
    "update_model_info_metric",
    "record_prediction",
    "record_model_reload",
    "record_label_injection",
    "record_drift_detection",
    "record_retraining_trigger",
    "record_invalid_signal",
]
