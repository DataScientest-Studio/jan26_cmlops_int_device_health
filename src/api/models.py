"""
Pydantic models for API request and response validation.

Request/Response schemas for:
- Device health predictions
- Sparse label injection
- Model information
- Health checks
"""

import re
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ======================================
# Request Models
# ======================================

# Allowed device_id format: empty (auto-generated), UUID, or safe alphanumeric name.
_SAFE_DEVICE_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9\-_]{0,35}$")


class PredictRequest(BaseModel):
    """Request model for device health prediction."""

    device_id: str = Field(
        ...,
        description="Device UUID (empty string triggers auto-generation of UUID)",
        min_length=0,
        max_length=36,
    )

    @field_validator("device_id")
    @classmethod
    def validate_device_id(cls, v: str) -> str:
        """Reject device_id values that are neither empty nor safe identifiers.

        An empty string triggers auto-generation of a UUID in the API layer.
        Non-empty values must match the safe pattern (alphanumeric, hyphens,
        underscores; 1-36 chars) to prevent injection attacks.
        """
        if v and not _SAFE_DEVICE_ID.match(v):
            raise ValueError(
                "device_id must be empty (auto-generated) or a safe identifier "
                "(letters, digits, hyphens, underscores; max 36 characters)"
            )
        return v

    device_name: str | None = Field(
        None,
        description="Optional device name (e.g., 'Device-Alpha-001')",
    )
    device_type: str | None = Field(
        None,
        description="Optional device type (e.g., 'Sensor-A')",
    )
    location: str | None = Field(
        None,
        description="Optional location (e.g., 'Building-3-Floor-2')",
    )
    time_values: list[float] = Field(
        ...,
        description="Time array (e.g., [0.0, 1.0, ..., 100.0])",
        min_length=51,
    )
    amplitude_values: list[float | None] = Field(
        ...,
        description="Amplitude array (None represents NaN)",
        min_length=51,
    )

    @field_validator("amplitude_values")
    @classmethod
    def validate_amplitude_length(cls, v, info):
        """Ensure amplitude_values matches time_values length."""
        if "time_values" in info.data:
            time_len = len(info.data["time_values"])
            if len(v) != time_len:
                raise ValueError(
                    f"amplitude_values length ({len(v)}) must match time_values length ({time_len})"
                )
        return v

    @field_validator("amplitude_values")
    @classmethod
    def validate_nan_fraction(cls, v):
        """Ensure < 5% NaN values."""
        nan_count = sum(1 for val in v if val is None)
        max_nan = int(len(v) * 0.05)
        if nan_count > max_nan:
            raise ValueError(f"Too many NaN values: {nan_count} > {max_nan} (5% of {len(v)})")
        return v


class InjectLabelRequest(BaseModel):
    """Request model for injecting sparse label."""

    prediction_id: int = Field(
        ...,
        description="Prediction ID to label",
        gt=0,
    )
    ground_truth_label: int = Field(
        ...,
        description="Ground truth label (0=healthy, 1=unhealthy)",
        ge=0,
        le=1,
    )
    label_source: str = Field(
        default="manual",
        description="Source of label (e.g., 'manual', 'automated_test')",
    )
    injected_by: str | None = Field(
        None,
        description="User/system identifier",
    )


class RefreshTokenRequest(BaseModel):
    """Request model for refreshing access token."""

    refresh_token: str = Field(
        ...,
        description="Valid refresh token obtained from /auth/token",
    )


# ======================================
# Response Models
# ======================================


class PredictResponse(BaseModel):
    """Response model for device health prediction."""

    prediction_id: int = Field(..., description="Unique prediction ID")
    device_id: str = Field(..., description="Device UUID")
    timestamp: str = Field(..., description="Prediction timestamp (ISO 8601)")
    predicted_label: int = Field(..., description="Predicted label (0=healthy, 1=unhealthy)")
    prediction_confidence: float = Field(..., description="Model confidence [0,1]")
    model_version: str = Field(..., description="Model version used")
    probabilities: dict[str, float] = Field(
        ...,
        description="Class probabilities",
    )
    features: dict[str, float | None] = Field(
        ...,
        description="Extracted signal features",
    )
    mlflow_run_id: str | None = Field(
        None,
        description="MLflow run ID of the model's training run",
    )
    git_sha: str | None = Field(
        None,
        description="Git commit hash of the code that trained the model",
    )
    dvc_data_hash: str | None = Field(
        None,
        description="DVC hash of the training data used",
    )
    airflow_run_id: str | None = Field(
        None,
        description="Airflow DAG run ID that triggered model training",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction_id": 12345,
                "device_id": "550e8400-e29b-41d4-a716-446655440000",
                "timestamp": "2026-02-10T12:00:00Z",
                "predicted_label": 0,
                "prediction_confidence": 0.95,
                "model_version": "bootstrap_v1",
                "probabilities": {"healthy": 0.95, "unhealthy": 0.05},
                "features": {
                    "fwhm": 10.5,
                    "peak_height": 3.2,
                    "peak_area": 150.0,
                    "noise_level": 0.02,
                    "snr": 160.0,
                    "peak_center": 50.0,
                },
                "mlflow_run_id": "a1b2c3d4e5f6",
                "git_sha": "abc1234",
                "dvc_data_hash": "d41d8cd98f00b204e9800998ecf8427e",
            }
        }
    )


class InjectLabelResponse(BaseModel):
    """Response model for sparse label injection."""

    label_id: int = Field(..., description="Unique label ID")
    prediction_id: int = Field(..., description="Prediction ID that was labeled")
    ground_truth_label: int = Field(..., description="Ground truth label")
    label_source: str = Field(..., description="Label source")
    injected_at: str = Field(..., description="Injection timestamp (ISO 8601)")
    message: str = Field(..., description="Success message")


class PredictionLineageResponse(BaseModel):
    """Response model for prediction lineage/traceability lookup."""

    prediction_id: int = Field(..., description="Unique prediction ID")
    device_id: str = Field(..., description="Device UUID")
    timestamp: str = Field(..., description="Prediction timestamp (ISO 8601)")
    predicted_label: int = Field(..., description="Predicted label (0=healthy, 1=unhealthy)")
    prediction_confidence: float | None = Field(None, description="Model confidence [0,1]")
    model_version: str = Field(..., description="Model version used")
    mlflow_run_id: str | None = Field(None, description="MLflow run ID of the training run")
    git_sha: str | None = Field(None, description="Git commit hash of training code")
    dvc_data_hash: str | None = Field(None, description="DVC hash of training data")
    airflow_run_id: str | None = Field(None, description="Airflow DAG run ID")
    ground_truth_label: int | None = Field(None, description="Ground truth label (if available)")
    label_source: str | None = Field(None, description="Label source (if labeled)")
    created_at: datetime | str | None = Field(None, description="Record creation timestamp")


class HealthCheckResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(
        ..., description="Overall system status: 'healthy', 'degraded', or 'unhealthy'"
    )
    timestamp: str = Field(..., description="Check timestamp (ISO 8601)")
    version: str = Field(..., description="API version")
    deployment_mode: str = Field(..., description="Active deployment mode: 'local' or 'cloud'")
    database_connected: bool = Field(..., description="Database connection status")
    model_loaded: bool = Field(..., description="Model loaded status")
    model_valid: bool = Field(..., description="Model structure validation status")
    mlflow_accessible: bool = Field(..., description="MLflow tracking server accessibility")
    dvc_remote_accessible: bool | None = Field(
        None, description="DVC remote accessibility (if configured)"
    )
    services: dict[str, str] = Field(..., description="Individual service statuses")


class ModelInfoResponse(BaseModel):
    """Response model for model information."""

    model_version: str = Field(..., description="Current model version")
    model_path: str = Field(..., description="Model file path")
    algorithm: str = Field(..., description="Model algorithm")
    trained_at: str = Field(..., description="Training timestamp (ISO 8601)")
    features_used: list[str] = Field(..., description="Feature names")
    source: str = Field(..., description="Model source: 'registry' or 'bootstrap'")
    registry_version: int | None = Field(
        None, description="MLflow Registry version (if from registry)"
    )
    registry_stage: str | None = Field(None, description="MLflow Registry stage (if from registry)")


class ReloadModelResponse(BaseModel):
    """Response model for model reload operation."""

    success: bool = Field(..., description="Whether reload was successful")
    message: str = Field(..., description="Reload status message")
    previous_source: str = Field(..., description="Previous model source")
    new_source: str = Field(..., description="New model source after reload")
    model_version: str = Field(..., description="Current model version after reload")
    registry_version: int | None = Field(None, description="Registry version (if applicable)")


class MetricsResponse(BaseModel):
    """Response model for metrics."""

    total_predictions: int = Field(..., description="Total predictions made")
    total_labeled: int = Field(..., description="Total predictions with labels")
    label_coverage: float = Field(..., description="Percentage of predictions labeled")
    healthy_count: int = Field(0, description="Predictions classified as healthy (label=0)")
    unhealthy_count: int = Field(0, description="Predictions classified as unhealthy (label=1)")
    realized_accuracy: float | None = Field(
        None,
        description="Accuracy on labeled predictions (None if no labels)",
    )
    healthy_accuracy: float | None = Field(
        None,
        description="Accuracy on healthy predictions",
    )
    unhealthy_accuracy: float | None = Field(
        None,
        description="Accuracy on unhealthy predictions",
    )
    lookback_days: int = Field(..., description="Metrics lookback period (days)")


# ======================================
# Batch Evaluation Models
# ======================================


class EvaluateSignal(BaseModel):
    """Single signal entry in a batch /evaluate request."""

    device_id: str = Field(
        default="",
        description="Device ID (empty = auto-generated per signal)",
        max_length=36,
    )
    time_values: list[float] = Field(
        ...,
        description="Time array",
        min_length=51,
    )
    amplitude_values: list[float | None] = Field(
        ...,
        description="Amplitude array (None = NaN)",
        min_length=51,
    )
    expected_label: int | None = Field(
        None,
        description="Ground truth label (0=healthy, 1=unhealthy) — enables accuracy computation",
        ge=0,
        le=1,
    )

    @field_validator("amplitude_values")
    @classmethod
    def validate_amplitude_length(cls, v, info):
        if "time_values" in info.data and len(v) != len(info.data["time_values"]):
            raise ValueError("amplitude_values length must match time_values length")
        return v


class EvaluateRequest(BaseModel):
    """Batch evaluation request — send N signals in one round trip."""

    signals: list[EvaluateSignal] = Field(
        ...,
        description="Signals to evaluate (1–1 000)",
        min_length=1,
        max_length=1000,
    )
    store_predictions: bool = Field(
        default=False,
        description="Persist each prediction to the database (slower; useful for audit trails)",
    )


class EvaluatePrediction(BaseModel):
    """Per-signal result inside an EvaluateResponse."""

    signal_index: int = Field(..., description="0-based index of the signal in the request")
    prediction_id: int | None = Field(None, description="DB prediction ID (None if not stored)")
    predicted_label: int = Field(..., description="Model prediction (0=healthy, 1=unhealthy)")
    prediction_confidence: float = Field(..., description="Model confidence [0,1]")
    model_version: str = Field(..., description="Model version used")
    probabilities: dict[str, float] = Field(..., description="Class probabilities")
    features: dict[str, float | None] = Field(..., description="Extracted signal features")
    latency_ms: float = Field(..., description="Per-signal inference latency (ms)")
    expected_label: int | None = Field(None, description="Provided ground truth (if any)")
    correct: bool | None = Field(
        None,
        description="Whether prediction matches expected_label (None if no label provided)",
    )


class EvaluateResponse(BaseModel):
    """Batch evaluation response with per-signal results and aggregate statistics."""

    n_signals: int = Field(..., description="Number of signals evaluated")
    n_errors: int = Field(..., description="Number of inference failures")
    model_version: str = Field(..., description="Model version used for all predictions")
    predictions: list[EvaluatePrediction] = Field(..., description="Per-signal results")
    # Aggregate statistics
    accuracy: float | None = Field(
        None,
        description="Fraction correct (None if no expected_labels provided)",
    )
    healthy_count: int = Field(..., description="Signals predicted as healthy (label=0)")
    unhealthy_count: int = Field(..., description="Signals predicted as unhealthy (label=1)")
    mean_latency_ms: float = Field(..., description="Mean per-signal inference latency (ms)")
    p95_latency_ms: float = Field(..., description="95th-percentile inference latency (ms)")
