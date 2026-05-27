"""
Prometheus metrics instrumentation for MLOps Device Health API.

This module defines all Prometheus metrics for monitoring:
- System metrics: Request counts, latencies, errors
- Model metrics: Predictions, confidence, version tracking
- Business metrics: Labels, drift detection, retraining triggers
"""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import wraps

from prometheus_client import Counter, Gauge, Histogram, Info

# ==============================================================================
# System Metrics
# ==============================================================================

# Request counter by endpoint and status
api_requests_total = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status_code"],
)

# Request latency histogram (in seconds)
api_request_duration_seconds = Histogram(
    "api_request_duration_seconds",
    "API request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# Error counter by endpoint and error type
api_errors_total = Counter(
    "api_errors_total",
    "Total number of API errors",
    ["method", "endpoint", "error_type"],
)

# Active requests gauge
api_requests_in_progress = Gauge(
    "api_requests_in_progress",
    "Number of API requests currently being processed",
    ["method", "endpoint"],
)

# ==============================================================================
# Model Metrics
# ==============================================================================

# Prediction counter by label (0=healthy, 1=unhealthy)
model_predictions_total = Counter(
    "model_predictions_total",
    "Total number of predictions made",
    ["model_version", "predicted_label"],
)

# Prediction confidence histogram
model_prediction_confidence = Histogram(
    "model_prediction_confidence",
    "Prediction confidence (probability)",
    ["model_version", "predicted_label"],
    buckets=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0),
)

# Model info (version, algorithm, source)
model_info = Info(
    "model_info",
    "Information about the current model",
)

# Model reload counter
model_reloads_total = Counter(
    "model_reloads_total",
    "Total number of model reloads",
    ["source", "trigger"],
)

# Model load time gauge (seconds)
model_load_time_seconds = Gauge(
    "model_load_time_seconds",
    "Time taken to load the model in seconds",
)

# Model last reload timestamp — set whenever the model is loaded into memory.
# Used by ModelVersionStale and NoRetrainingInWeek alert rules.
model_last_reload_timestamp_seconds = Gauge(
    "model_last_reload_timestamp_seconds",
    "Unix timestamp (seconds since epoch) of the last successful model reload",
)

# ==============================================================================
# Business Metrics
# ==============================================================================

# Labels injected counter
labels_injected_total = Counter(
    "labels_injected_total",
    "Total number of ground truth labels injected",
    ["label_value"],
)

# Data drift detection counter
drift_detections_total = Counter(
    "drift_detections_total",
    "Total number of drift detections",
    ["drift_type"],
)

# Retraining trigger counter
retraining_triggers_total = Counter(
    "retraining_triggers_total",
    "Total number of retraining workflow triggers",
    ["trigger_reason"],
)

# Retraining failure counter — increment in the retraining DAG when a failure occurs.
# Used by the RetrainingFailed alert rule.
retraining_failures_total = Counter(
    "retraining_failures_total",
    "Total number of retraining pipeline failures",
    ["reason"],
)

# Sparse label coverage (percentage of predictions with labels)
sparse_label_coverage = Gauge(
    "sparse_label_coverage",
    "Percentage of predictions that have ground truth labels",
)

# DB-derived gauges — refreshed at /metrics scrape time to reflect actual DB state.
# These solve the cross-process problem: scripts that run outside the API process
# cannot increment Counter objects in the API process.
labels_in_db_total = Gauge(
    "labels_in_db_total",
    "Total predictions with a ground_truth_label in the database",
)

predictions_in_db_total = Gauge(
    "predictions_in_db_total",
    "Total predictions stored in the database",
)

drift_reports_total = Gauge(
    "drift_reports_total",
    "Total drift report JSON files found in reports/drift/",
)

# Gauge set from drift_summary JSON files at /metrics scrape time.
# Labeled by drift_type ("data", "concept", "feature", "prior_probability") and
# value is the count of reports where that drift type was detected.  This bridges
# the cross-process gap: detect_drift.py writes JSON files; the API reads them.
drift_detected_gauge = Gauge(
    "drift_detected_gauge",
    "Count of drift summary reports where drift was detected, by drift type",
    ["drift_type"],
)

# Realized model accuracy gauge — refreshed at /metrics scrape time using
# recent (30-day) labeled predictions from the database.
model_accuracy_gauge = Gauge(
    "model_accuracy_gauge",
    "Realized model accuracy on labeled predictions (last 30 days)",
)

# ==============================================================================
# KPI Governance Metrics  (Task 1)
# ==============================================================================

# Time from retraining trigger to production promotion (seconds).
# Set in the model_promotion DAG after a successful promotion.
model_deploy_time_seconds = Gauge(
    "model_deploy_time_seconds",
    "Time in seconds from retraining trigger to production model promotion",
)

# Fraction of retraining runs that were triggered automatically (0.0-1.0).
# Set in the automated_retraining DAG on each run completion.
automation_rate_gauge = Gauge(
    "automation_rate_gauge",
    "Fraction of retraining runs triggered automatically vs manually (0.0-1.0)",
)

# Mean time to drift detection (seconds between drift events).
# Set in the evidently_drift_detection DAG after each detection.
mttd_seconds = Gauge(
    "mttd_seconds",
    "Mean time to drift detection in seconds",
)

# ==============================================================================
# Data Quality Metrics
# ==============================================================================

# Invalid signal counter
invalid_signals_total = Counter(
    "invalid_signals_total",
    "Total number of invalid signals rejected",
    ["validation_error"],
)

# Signal validation time
signal_validation_duration_seconds = Histogram(
    "signal_validation_duration_seconds",
    "Signal validation latency in seconds",
    buckets=(0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1),
)

# ==============================================================================
# Feature Engineering Metrics
# ==============================================================================

# Feature extraction counter
feature_extractions_total = Counter(
    "feature_extractions_total",
    "Total number of feature extractions performed",
    ["success"],
)

# Feature extraction time
feature_extraction_duration_seconds = Histogram(
    "feature_extraction_duration_seconds",
    "Feature extraction latency in seconds",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)

# ==============================================================================
# Helper Functions
# ==============================================================================


def track_request_metrics(endpoint: str, method: str = "GET") -> Callable:
    """
    Decorator to track request metrics (count, latency, errors).

    Args:
        endpoint: API endpoint name (e.g., '/predict', '/health')
        method: HTTP method (GET, POST, etc.)

    Returns:
        Decorated function that tracks metrics

    Example:
        @track_request_metrics('/predict', 'POST')
        async def predict_endpoint(signal: Signal):
            return {"prediction": 0}
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Increment in-progress gauge
            api_requests_in_progress.labels(method=method, endpoint=endpoint).inc()

            start_time = time.time()
            status_code = 200
            error_type = None

            try:
                # Execute the function
                result = await func(*args, **kwargs)
                return result

            except Exception as e:
                # Track error
                status_code = 500
                error_type = type(e).__name__
                api_errors_total.labels(
                    method=method, endpoint=endpoint, error_type=error_type
                ).inc()
                raise

            finally:
                # Track latency
                duration = time.time() - start_time
                api_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(
                    duration
                )

                # Track request count
                api_requests_total.labels(
                    method=method, endpoint=endpoint, status_code=status_code
                ).inc()

                # Decrement in-progress gauge
                api_requests_in_progress.labels(method=method, endpoint=endpoint).dec()

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Increment in-progress gauge
            api_requests_in_progress.labels(method=method, endpoint=endpoint).inc()

            start_time = time.time()
            status_code = 200
            error_type = None

            try:
                # Execute the function
                result = func(*args, **kwargs)
                return result

            except Exception as e:
                # Track error
                status_code = 500
                error_type = type(e).__name__
                api_errors_total.labels(
                    method=method, endpoint=endpoint, error_type=error_type
                ).inc()
                raise

            finally:
                # Track latency
                duration = time.time() - start_time
                api_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(
                    duration
                )

                # Track request count
                api_requests_total.labels(
                    method=method, endpoint=endpoint, status_code=status_code
                ).inc()

                # Decrement in-progress gauge
                api_requests_in_progress.labels(method=method, endpoint=endpoint).dec()

        # Return appropriate wrapper based on function type
        import inspect

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def update_model_info_metric(
    model_version: str,
    algorithm: str,
    source: str,
    trained_at: str,
    features: list[str],
) -> None:
    """
    Update the model_info metric with current model metadata.

    Args:
        model_version: Model version identifier
        algorithm: ML algorithm name (e.g., 'LogisticRegression')
        source: Model source ('registry' or 'bootstrap')
        trained_at: Training timestamp
        features: List of feature names used by model
    """
    model_info.info(
        {
            "version": model_version,
            "algorithm": algorithm,
            "source": source,
            "trained_at": trained_at,
            "features": ",".join(features),
        }
    )


def record_prediction(model_version: str, predicted_label: int, confidence: float) -> None:
    """
    Record a prediction event with label and confidence.

    Args:
        model_version: Model version that made the prediction
        predicted_label: Predicted class (0=healthy, 1=unhealthy)
        confidence: Prediction confidence/probability
    """
    label_str = str(predicted_label)

    # Increment prediction counter
    model_predictions_total.labels(model_version=model_version, predicted_label=label_str).inc()

    # Observe confidence
    model_prediction_confidence.labels(
        model_version=model_version, predicted_label=label_str
    ).observe(confidence)


def record_model_reload(source: str, trigger: str = "manual") -> None:
    """
    Record a model reload event.

    Args:
        source: Model source after reload ('registry' or 'bootstrap')
        trigger: What triggered the reload ('manual', 'automated', 'startup')
    """
    model_reloads_total.labels(source=source, trigger=trigger).inc()
    model_last_reload_timestamp_seconds.set(time.time())


def record_label_injection(label_value: int) -> None:
    """
    Record a ground truth label injection.

    Args:
        label_value: Label value (0=healthy, 1=unhealthy)
    """
    labels_injected_total.labels(label_value=str(label_value)).inc()


def record_drift_detection(drift_type: str) -> None:
    """
    Record a drift detection event.

    Args:
        drift_type: Type of drift detected ('data', 'prediction', 'concept', 'feature')
    """
    drift_detections_total.labels(drift_type=drift_type).inc()


def record_retraining_trigger(reason: str) -> None:
    """
    Record a retraining workflow trigger.

    Args:
        reason: Reason for retraining ('drift', 'schedule', 'manual', 'accuracy_drop')
    """
    retraining_triggers_total.labels(trigger_reason=reason).inc()


def record_invalid_signal(error_type: str) -> None:
    """
    Record an invalid signal rejection.

    Args:
        error_type: Type of validation error (e.g., 'insufficient_data', 'nan_values')
    """
    invalid_signals_total.labels(validation_error=error_type).inc()
