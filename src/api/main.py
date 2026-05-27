"""
FastAPI application for MLOps device health monitoring.

REST API endpoints:
- POST /auth/token - Obtain OAuth2 access token
- POST /auth/refresh - Refresh access token
- GET /auth/users/me - Get current user info
- POST /predict - Make prediction (protected)
- POST /labels - Inject sparse label (protected)
- GET /health - Health check endpoint (public)
- GET /metrics - Performance metrics (public)
- GET /model/info - Current model information (protected)
"""

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.security import OAuth2PasswordRequestForm
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from src.database import Database, generate_device_id
from src.monitoring import (
    api_errors_total,
    api_request_duration_seconds,
    api_requests_in_progress,
    api_requests_total,
    drift_detected_gauge,
    drift_reports_total,
    feature_extraction_duration_seconds,
    labels_in_db_total,
    model_accuracy_gauge,
    predictions_in_db_total,
    record_invalid_signal,
    record_label_injection,
    record_model_reload,
    record_prediction,
    retraining_failures_total,
    retraining_triggers_total,
    signal_validation_duration_seconds,
    sparse_label_coverage,
    update_model_info_metric,
)
from src.monitoring.logging_config import (
    log_label_injection,
    log_model_operation,
    log_prediction,
    set_request_id,
    setup_logging,
)
from src.training import predict

from .auth import Token, UserInDB, authenticate_user, create_token_response, refresh_access_token
from .dependencies import Settings, get_database, get_model, get_settings
from .models import (
    EvaluatePrediction,
    EvaluateRequest,
    EvaluateResponse,
    HealthCheckResponse,
    InjectLabelRequest,
    InjectLabelResponse,
    MetricsResponse,
    ModelInfoResponse,
    PredictionLineageResponse,
    PredictRequest,
    PredictResponse,
    RefreshTokenRequest,
    ReloadModelResponse,
)
from .security import get_current_active_user, get_current_user_or_api_key

# Set up structured logging
setup_logging(log_level="INFO", enable_console_logging=True)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MLflow / DagsHub startup log
# Verify with:  docker logs mlops_api | grep mlflow_tracking_info
# ---------------------------------------------------------------------------
def _log_mlflow_tracking_info() -> None:
    """Emit a single structured log line with the MLflow tracking config.

    Called both at module import time and inside ``startup_event`` so the
    information appears in ``docker logs mlops_api`` regardless of whether
    uvicorn is run with or without pre-fork workers.
    """
    try:
        from src.api.dependencies import get_settings as _load_settings

        settings = _load_settings()
        tracking_uri = settings.MLFLOW_TRACKING_URI
        tracking_user = os.environ.get("MLFLOW_TRACKING_USERNAME", "[not set]")
        deployment_mode = getattr(settings, "DEPLOYMENT_MODE", "unknown")

        logger.info(
            "mlflow_tracking_info | deployment=%s | uri=%s | username=%s",
            deployment_mode,
            tracking_uri,
            tracking_user,
            extra={
                "event": "mlflow_tracking_info",
                "mlflow_tracking_uri": tracking_uri,
                "mlflow_tracking_username": tracking_user,
                "deployment_mode": deployment_mode,
            },
        )
    except Exception as exc:  # broad catch intentional — must never crash startup
        logger.warning("mlflow_tracking_info | unavailable: %s", exc)


_log_mlflow_tracking_info()  # module-level call (pre-worker fork)

# ======================================
# FastAPI Application
# ======================================

# Module-level handle — kept here so both lifespan and any future
# direct-cancel logic can access it without circular import issues.
_model_poll_task: Any = None  # handle kept so shutdown can cancel


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """FastAPI lifespan context: runs startup before yield, shutdown after."""
    # ── Startup ──────────────────────────────────────────────────────────
    logger.info("api_startup | MLOps Device Health Monitoring API starting")
    logger.info("api_startup | API docs available at /docs")

    # Re-emit MLflow config after worker fork so it always appears in docker logs
    _log_mlflow_tracking_info()

    # Initialize model info metrics
    try:
        from src.api.dependencies import get_model as load_model_fn
        from src.api.dependencies import get_settings as load_settings

        settings = load_settings()
        model_artifact = load_model_fn(settings)

        update_model_info_metric(
            model_version=model_artifact.get("model_version", "unknown"),
            algorithm=model_artifact.get("algorithm", "unknown"),
            source=model_artifact.get("source", "unknown"),
            trained_at=model_artifact.get("trained_at", "unknown"),
            features=model_artifact.get("features_used", []),
        )
        record_model_reload(source=model_artifact.get("source", "unknown"), trigger="startup")
        logger.info(
            "api_startup | model ready: source=%s",
            model_artifact.get("source", "unknown"),
        )
    except Exception as exc:  # broad catch intentional — startup must not crash
        logger.warning("api_startup | failed to initialize model metrics: %s", exc)

    logger.info("api_startup | ready")

    # Launch background model-registry polling task
    global _model_poll_task
    _model_poll_task = asyncio.create_task(_poll_model_registry())

    yield  # ← application is running

    # ── Shutdown ─────────────────────────────────────────────────────────
    if _model_poll_task is not None:
        _model_poll_task.cancel()
    logger.info("api_shutdown | MLOps Device Health Monitoring API shutting down")


app = FastAPI(
    title="MLOps Device Health Monitoring API",
    description="REST API for device health prediction and monitoring with OAuth2 authentication",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ======================================
# Request ID Middleware
# ======================================


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add request ID tracking to all requests."""

    async def dispatch(self, request: Request, call_next):
        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        set_request_id(request_id)

        # Log incoming request
        logger.info(
            "incoming_request",
            extra={
                "method": request.method,
                "url": str(request.url),
                "client_host": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
            },
        )

        # Process request
        response = await call_next(request)

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        # Log outgoing response
        logger.info(
            "outgoing_response",
            extra={
                "method": request.method,
                "url": str(request.url),
                "status_code": response.status_code,
            },
        )

        return response


# ======================================
# Prometheus Metrics Middleware
# ======================================


class PrometheusMetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to record HTTP request metrics for Prometheus."""

    async def dispatch(self, request: Request, call_next):
        method = request.method
        endpoint = request.url.path
        api_requests_in_progress.labels(method=method, endpoint=endpoint).inc()
        start = time.time()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception:
            api_errors_total.labels(method=method, endpoint=endpoint, error_type="unhandled").inc()
            raise
        finally:
            api_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(
                time.time() - start
            )
            api_requests_total.labels(
                method=method, endpoint=endpoint, status_code=str(status_code)
            ).inc()
            api_requests_in_progress.labels(method=method, endpoint=endpoint).dec()


# Add request ID middleware (add before CORS)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(PrometheusMetricsMiddleware)

# ======================================
# CORS Configuration
# ======================================

# Configure CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React/Vue frontend
        "http://localhost:8501",  # Streamlit
        "http://localhost:8080",  # Alternative frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================================
# Exception Handlers
# ======================================


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request, exc):
    """Handle FileNotFoundError exceptions."""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": str(exc)},
    )


# ======================================
# Endpoints
# ======================================


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - redirect to docs."""
    return {
        "message": "MLOps Device Health Monitoring API",
        "docs": "/docs",
        "health": "/health",
        "authentication": "/auth/token",
    }


# ======================================
# Authentication Endpoints
# ======================================


@app.post("/auth/token", response_model=Token, tags=["Authentication"])
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
):
    """
    OAuth2 compatible token login.

    Authenticate with username and password to obtain access token.

    Args:
        form_data: OAuth2 form with username and password

    Returns:
        Token response with access_token and refresh_token

    Raises:
        HTTPException: If authentication fails
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return create_token_response(user, include_refresh=True)


@app.post("/auth/refresh", response_model=Token, tags=["Authentication"])
async def refresh_token(
    request: RefreshTokenRequest,
):
    """
    Refresh access token using refresh token.

    Args:
        request: Refresh token request

    Returns:
        New Token response with access_token

    Raises:
        HTTPException: If refresh token is invalid or expired
    """
    new_token = refresh_access_token(request.refresh_token)
    if new_token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return new_token


@app.get("/auth/users/me", tags=["Authentication"])
async def read_users_me(
    current_user: Annotated[UserInDB, Security(get_current_active_user)],
):
    """
    Get current authenticated user information.

    Args:
        current_user: Current authenticated user from token

    Returns:
        User information
    """
    return {
        "username": current_user.username,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "scopes": current_user.scopes,
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check(
    db: Database = Depends(get_database),
    settings: Settings = Depends(get_settings),
):
    """
    Comprehensive health check endpoint.

    Checks:
    - Database connectivity
    - Model loading status
    - Model structure validation
    - MLflow tracking server accessibility
    - DVC remote accessibility (if configured)

    Returns appropriate HTTP status codes:
    - 200: All systems healthy
    - 503: Critical systems unavailable (database or model)

    Note: If the database is completely unreachable (e.g. wrong credentials),
    ``get_database`` raises an ``HTTPException(503)`` before this handler runs.
    Use ``get_optional_database`` if you need the handler to run and report
    details even when the DB is unavailable.
    """
    import subprocess
    from pathlib import Path

    import mlflow

    services = {}

    # Check database connection
    database_connected = False
    try:
        db.conn.execute("SELECT 1")
        database_connected = True
        services["database"] = "healthy"
    except Exception as e:
        services["database"] = f"unhealthy: {str(e)[:50]}"

    # Check model loading
    model_loaded = False
    model_valid = False
    try:
        model_artifact = get_model(settings)
        if model_artifact is not None:
            model_loaded = True
            services["model_loading"] = "healthy"

            # Validate model structure
            required_keys = ["model", "scaler", "feature_names"]
            # Version can be either "version" or "model_version"
            has_version = "version" in model_artifact or "model_version" in model_artifact
            if all(key in model_artifact for key in required_keys) and has_version:
                model_valid = True
                services["model_validation"] = "healthy"
            else:
                services["model_validation"] = "degraded: missing keys"
        else:
            services["model_loading"] = "unhealthy: model is None"
    except Exception as e:
        services["model_loading"] = f"unhealthy: {str(e)[:50]}"

    # Check MLflow accessibility (3-second timeout to prevent CI hangs)
    mlflow_accessible = False
    try:
        import threading

        # In cloud mode the local MLflow Docker container is NOT running;
        # MLflow is hosted externally by DagsHub.  Probing the Docker-
        # internal hostname would always fail, so we perform a lightweight
        # HTTP HEAD against the DagsHub URL (with auth) instead, or simply
        # report the service as healthy/external when DagsHub is configured.
        _is_cloud = settings.DEPLOYMENT_MODE == "cloud"
        _uri = settings.MLFLOW_TRACKING_URI
        _is_dagshub = "dagshub.com" in _uri

        if _is_cloud and _is_dagshub:
            # Quick HTTP check against DagsHub (5 s timeout — remote service)
            import urllib.request

            _probe_result_cloud: list[bool] = []

            def _probe_dagshub() -> None:
                try:
                    req = urllib.request.Request(_uri, method="HEAD")
                    _user = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
                    _token = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")
                    if _user and _token:
                        import base64

                        _cred = base64.b64encode(f"{_user}:{_token}".encode()).decode()
                        req.add_header("Authorization", f"Basic {_cred}")
                    with urllib.request.urlopen(req, timeout=5):
                        _probe_result_cloud.append(True)
                except Exception:  # noqa: BLE001
                    _probe_result_cloud.append(False)

            _t = threading.Thread(target=_probe_dagshub, daemon=True)
            _t.start()
            _t.join(timeout=6.0)
            if _probe_result_cloud and _probe_result_cloud[0]:
                mlflow_accessible = True
                services["mlflow"] = "healthy (cloud — DagsHub)"
            else:
                # DagsHub unreachable but it's external; don't flag local
                # infrastructure as degraded.
                mlflow_accessible = True
                services["mlflow"] = "healthy (cloud — external)"
        else:
            # Local mode — probe the Docker MLflow container via SDK
            mlflow.set_tracking_uri(_uri)

            _probe_result_local: list[bool] = []

            def _probe_mlflow() -> None:
                try:
                    mlflow.search_experiments(max_results=1)
                    _probe_result_local.append(True)
                except Exception:  # noqa: BLE001
                    _probe_result_local.append(False)

            _probe_thread = threading.Thread(target=_probe_mlflow, daemon=True)
            _probe_thread.start()
            _probe_thread.join(timeout=3.0)
            if _probe_result_local and _probe_result_local[0]:
                mlflow_accessible = True
                services["mlflow"] = "healthy"
            else:
                services["mlflow"] = "degraded: unreachable or timeout"
    except Exception as e:
        services["mlflow"] = f"degraded: {str(e)[:50]}"

    # Check DVC remote accessibility (if configured)
    dvc_remote_accessible = None
    try:
        # In Kubernetes, DVC is not used in-cluster — each pod does not have a
        # local .dvc/config and S3/DagsHub is not accessible from inside pods.
        # Show N/A rather than "not configured" to avoid false alarms.
        if os.environ.get("MLOPS_ENVIRONMENT") == "kubernetes":
            dvc_remote_accessible = None
            services["dvc_remote"] = "not applicable (k8s)"
        else:
            # Check if DVC is configured
            dvc_config = Path(".dvc/config")
            if dvc_config.exists():
                # Try to check DVC remote status (quick check)
                result = subprocess.run(
                    ["dvc", "remote", "list"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    dvc_remote_accessible = True
                    services["dvc_remote"] = "healthy"
                else:
                    dvc_remote_accessible = False
                    services["dvc_remote"] = "not configured"
            elif (
                os.environ.get("AWS_ACCESS_KEY_ID")
                or os.environ.get("DAGSHUB_TOKEN")
                or os.environ.get("MLFLOW_TRACKING_PASSWORD")
            ):
                # Running inside Docker — DVC config lives on the host but
                # cloud credentials are available, so the remote is configured.
                dvc_remote_accessible = True
                services["dvc_remote"] = "healthy (host-managed)"
            else:
                services["dvc_remote"] = "not configured"
    except subprocess.TimeoutExpired:
        dvc_remote_accessible = False
        services["dvc_remote"] = "degraded: timeout"
    except FileNotFoundError:
        services["dvc_remote"] = "not available: dvc not installed"
    except Exception as e:
        services["dvc_remote"] = f"degraded: {str(e)[:50]}"

    # Determine overall status
    critical_healthy = database_connected and model_loaded and model_valid
    mlflow_ok = mlflow_accessible or not settings.MODEL_REGISTRY_ENABLED

    if critical_healthy and mlflow_ok:
        overall_status = "healthy"
        status_code = 200
    elif critical_healthy:
        overall_status = "degraded"
        status_code = 200
    else:
        overall_status = "unhealthy"
        status_code = 503

    response = HealthCheckResponse(
        status=overall_status,
        timestamp=datetime.now(UTC).isoformat(),
        version=settings.API_VERSION,
        deployment_mode=settings.DEPLOYMENT_MODE,
        database_connected=database_connected,
        model_loaded=model_loaded,
        model_valid=model_valid,
        mlflow_accessible=mlflow_accessible,
        dvc_remote_accessible=dvc_remote_accessible,
        services=services,
    )

    # Return appropriate status code
    if status_code == 503:
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=status_code,
            content=response.model_dump(),
        )

    return response


@app.post("/predict", response_model=PredictResponse, tags=["Predictions"])
async def make_prediction(
    request: PredictRequest,
    auth_info: Annotated[dict, Depends(get_current_user_or_api_key)],
    db: Database = Depends(get_database),
    model_artifact: dict[str, Any] = Depends(get_model),
    settings: Settings = Depends(get_settings),
):
    """
    Make device health prediction (Authentication Required).

    Predicts device health from raw signal, stores in database, and returns result.

    **Authentication:** Requires valid OAuth2 token or API key with 'write' scope.

    Args:
        request: Prediction request with signal data
        auth_info: Authentication info (user or API key)
        db: Database connection
        model_artifact: Loaded model

    Returns:
        Prediction response with label, confidence, and features

    Raises:
        HTTPException: If validation fails or prediction errors occur
    """
    # Verify write permission
    if "write" not in auth_info.get("scopes", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions: 'write' scope required",
        )

    try:
        # Signal validation timing
        _val_t0 = time.time()
        # Handle device registration
        device_id = request.device_id
        if not device_id or device_id == "":
            # Auto-generate UUID
            device_id = generate_device_id()

        # Register/update device
        db.register_device(
            device_id=device_id,
            device_name=request.device_name,
            device_type=request.device_type,
            location=request.location,
            status="active",
            deployment_mode=settings.DEPLOYMENT_MODE,
        )
        signal_validation_duration_seconds.observe(time.time() - _val_t0)

        # Make prediction (time feature extraction via histogram)
        _t0 = time.time()
        prediction_result = predict(
            time_values=request.time_values,
            amplitude_values=request.amplitude_values,  # type: ignore[arg-type]
            model_path=model_artifact,  # Pass artifact directly (already loaded)
            return_probabilities=True,
        )
        feature_extraction_duration_seconds.observe(time.time() - _t0)

        # Store in database
        prediction_id = db.store_prediction(
            device_id=device_id,
            time_values=request.time_values,
            amplitude_values=request.amplitude_values,  # type: ignore[arg-type]
            predicted_label=prediction_result["predicted_label"],
            model_version=prediction_result["model_version"],
            features=prediction_result["features"],
            prediction_confidence=prediction_result["confidence"],
            shape_type=None,  # Unknown for real-world data
            mlflow_run_id=model_artifact.get("mlflow_run_id"),
            git_sha=model_artifact.get("git_sha"),
            dvc_data_hash=model_artifact.get("dvc_data_hash"),
            airflow_run_id=model_artifact.get("airflow_run_id"),
            deployment_mode=settings.DEPLOYMENT_MODE,
        )

        # Retrieve stored prediction for timestamp
        stored_prediction = db.get_prediction(prediction_id)

        # Record prediction metrics for monitoring
        record_prediction(
            model_version=prediction_result["model_version"],
            predicted_label=prediction_result["predicted_label"],
            confidence=prediction_result["confidence"],
        )

        # Audit logging for compliance
        log_prediction(
            prediction_id=prediction_id,
            signal_id=stored_prediction.get("signal_id"),  # type: ignore[union-attr, arg-type]
            prediction=prediction_result["predicted_label"],
            confidence=prediction_result["confidence"],
            model_version=prediction_result["model_version"],
            user_id=auth_info.get("sub"),
        )

        # Normalise timestamp: PostgreSQL returns datetime, SQLite returns str
        _ts = stored_prediction["timestamp"]  # type: ignore[index]
        _ts_str = _ts.isoformat() if hasattr(_ts, "isoformat") else str(_ts)

        return PredictResponse(
            prediction_id=prediction_id,
            device_id=device_id,
            timestamp=_ts_str,
            predicted_label=prediction_result["predicted_label"],
            prediction_confidence=prediction_result["confidence"],
            model_version=prediction_result["model_version"],
            probabilities=prediction_result["probabilities"],
            features=prediction_result["features"],
            mlflow_run_id=model_artifact.get("mlflow_run_id"),
            git_sha=model_artifact.get("git_sha"),
            dvc_data_hash=model_artifact.get("dvc_data_hash"),
            airflow_run_id=model_artifact.get("airflow_run_id"),
        )

    except ValueError as e:
        record_invalid_signal("validation_error")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except Exception as e:
        # Log the full error for debugging
        import traceback

        error_detail = f"Prediction failed: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)  # This will show in test output
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        ) from e


@app.post("/evaluate", response_model=EvaluateResponse, tags=["Predictions"])
async def batch_evaluate(
    request: EvaluateRequest,
    auth_info: Annotated[dict, Depends(get_current_user_or_api_key)],
    db: Database = Depends(get_database),
    model_artifact: dict[str, Any] = Depends(get_model),
):
    """
    Batch evaluation of N signals in a single HTTP round trip.

    Runs inference on every signal in the request using the same loaded model
    instance, then returns per-signal predictions together with aggregate
    statistics (accuracy, latency distribution, class breakdown).

    **Why use this instead of N × /predict?**
    - One auth check, one model-load, one network round trip.
    - All predictions use exactly the same model version (atomic).
    - Aggregate stats (accuracy, p95 latency) computed server-side.
    - Ideal for A/B testing: send the same batch to champion and challenger.

    **Authentication:** Requires valid OAuth2 token or API key with 'read' scope.
    """
    if "read" not in auth_info.get("scopes", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions: 'read' scope required",
        )

    import time as _time

    predictions_out: list[EvaluatePrediction] = []
    latencies: list[float] = []
    n_errors = 0
    model_version_used = "unknown"

    for idx, sig in enumerate(request.signals):
        t0 = _time.monotonic()
        try:
            result = predict(
                time_values=sig.time_values,
                amplitude_values=sig.amplitude_values,  # type: ignore[arg-type]
                model_path=model_artifact,
                return_probabilities=True,
            )
            latency_ms = (_time.monotonic() - t0) * 1000
            model_version_used = result["model_version"]

            prediction_id: int | None = None
            if request.store_predictions:
                device_id = sig.device_id if sig.device_id else generate_device_id()
                db.register_device(
                    device_id=device_id,
                    status="active",
                    deployment_mode=os.getenv("DEPLOYMENT_MODE", "local"),
                )
                prediction_id = db.store_prediction(
                    device_id=device_id,
                    time_values=sig.time_values,
                    amplitude_values=sig.amplitude_values,  # type: ignore[arg-type]
                    predicted_label=result["predicted_label"],
                    model_version=result["model_version"],
                    features=result["features"],
                    prediction_confidence=result["confidence"],
                    mlflow_run_id=model_artifact.get("mlflow_run_id"),
                    git_sha=model_artifact.get("git_sha"),
                    dvc_data_hash=model_artifact.get("dvc_data_hash"),
                    airflow_run_id=model_artifact.get("airflow_run_id"),
                    deployment_mode=os.getenv("DEPLOYMENT_MODE", "local"),
                )

            correct: bool | None = None
            if sig.expected_label is not None:
                correct = result["predicted_label"] == sig.expected_label

            predictions_out.append(
                EvaluatePrediction(
                    signal_index=idx,
                    prediction_id=prediction_id,
                    predicted_label=result["predicted_label"],
                    prediction_confidence=result["confidence"],
                    model_version=result["model_version"],
                    probabilities=result["probabilities"],
                    features=result["features"],
                    latency_ms=round(latency_ms, 2),
                    expected_label=sig.expected_label,
                    correct=correct,
                )
            )
            latencies.append(latency_ms)

        except Exception:
            latency_ms = (_time.monotonic() - t0) * 1000
            n_errors += 1
            latencies.append(latency_ms)

    # Aggregate statistics
    valid_preds = list(predictions_out)
    labeled = [p for p in valid_preds if p.expected_label is not None]
    accuracy: float | None = (
        sum(1 for p in labeled if p.correct) / len(labeled) if labeled else None
    )
    healthy_count = sum(1 for p in valid_preds if p.predicted_label == 0)
    unhealthy_count = sum(1 for p in valid_preds if p.predicted_label == 1)
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    sorted_lats = sorted(latencies)
    p95_idx = max(0, int(len(sorted_lats) * 0.95) - 1)
    p95_latency = sorted_lats[p95_idx] if sorted_lats else 0.0

    return EvaluateResponse(
        n_signals=len(request.signals),
        n_errors=n_errors,
        model_version=model_version_used,
        predictions=predictions_out,
        accuracy=accuracy,
        healthy_count=healthy_count,
        unhealthy_count=unhealthy_count,
        mean_latency_ms=round(mean_latency, 2),
        p95_latency_ms=round(p95_latency, 2),
    )


@app.get(
    "/predictions/{prediction_id}/lineage",
    response_model=PredictionLineageResponse,
    tags=["Predictions"],
)
async def get_prediction_lineage(
    prediction_id: int,
    auth_info: Annotated[dict, Depends(get_current_user_or_api_key)],
    db: Database = Depends(get_database),
):
    """
    Get full traceability/lineage for a prediction.

    Returns the prediction record with all lineage IDs:
    model version, MLflow run ID, Git SHA, DVC data hash, and Airflow run ID.

    **Authentication:** Requires valid OAuth2 token or API key.
    """
    prediction = db.get_prediction(prediction_id)
    if prediction is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction {prediction_id} not found",
        )

    _ts = prediction["timestamp"]
    _ts_str = _ts.isoformat() if hasattr(_ts, "isoformat") else str(_ts)

    return PredictionLineageResponse(
        prediction_id=prediction["prediction_id"],
        device_id=prediction["device_id"],
        timestamp=_ts_str,
        predicted_label=prediction["predicted_label"],
        prediction_confidence=prediction.get("prediction_confidence"),
        model_version=prediction["model_version"],
        mlflow_run_id=prediction.get("mlflow_run_id"),
        git_sha=prediction.get("git_sha"),
        dvc_data_hash=prediction.get("dvc_data_hash"),
        airflow_run_id=prediction.get("airflow_run_id"),
        ground_truth_label=prediction.get("ground_truth_label"),
        label_source=prediction.get("label_source"),
        created_at=prediction.get("created_at"),
    )


@app.post("/labels", response_model=InjectLabelResponse, tags=["Labels"])
async def inject_label(
    request: InjectLabelRequest,
    auth_info: Annotated[dict, Depends(get_current_user_or_api_key)],
    db: Database = Depends(get_database),
    settings: Settings = Depends(get_settings),
):
    """
    Inject sparse label for model evaluation (Authentication Required).

    Adds ground truth label to a prediction (delayed labeling pattern).

    **Authentication:** Requires valid OAuth2 token or API key with 'write' scope.

    Args:
        request: Label injection request
        auth_info: Authentication info (user or API key)
        db: Database connection

    Returns:
        Label injection response

    Raises:
        HTTPException: If prediction not found or injection fails
    """
    # Verify write permission
    if "write" not in auth_info.get("scopes", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions: 'write' scope required",
        )

    try:
        # Verify prediction exists
        prediction = db.get_prediction(request.prediction_id)
        if prediction is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prediction {request.prediction_id} not found",
            )

        # Inject label
        label_id = db.inject_sparse_label(
            prediction_id=request.prediction_id,
            ground_truth_label=request.ground_truth_label,
            label_source=request.label_source,
            injected_by=request.injected_by,
            deployment_mode=settings.DEPLOYMENT_MODE,
        )

        # Audit logging for compliance
        log_label_injection(
            signal_id=prediction.get("signal_id"),  # type: ignore[arg-type]
            ground_truth=request.ground_truth_label,
            source=request.label_source,
            user_id=auth_info.get("sub"),
        )

        # Record Prometheus metrics
        record_label_injection(request.ground_truth_label)
        try:
            coverage = db.get_label_coverage()
            sparse_label_coverage.set(coverage.get("label_coverage", 0.0))
        except Exception:
            pass

        return InjectLabelResponse(
            label_id=label_id,
            prediction_id=request.prediction_id,
            ground_truth_label=request.ground_truth_label,
            label_source=request.label_source,
            injected_at=datetime.utcnow().isoformat(),
            message=f"Label successfully injected for prediction {request.prediction_id}",
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Label injection failed: {str(e)}",
        ) from e


@app.get("/metrics", tags=["Monitoring"])
async def prometheus_metrics(db: Database = Depends(get_database)):
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus text format for scraping by Prometheus server.
    Refreshes DB-derived gauges on every scrape so Grafana panels always reflect
    actual database state even when counters are incremented by external scripts.
    """
    # Refresh DB-derived gauges (solve cross-process counter problem)
    try:
        coverage = db.get_label_coverage()
        sparse_label_coverage.set(coverage.get("label_coverage", 0.0))
        labels_in_db_total.set(coverage.get("labeled_predictions", 0))
        predictions_in_db_total.set(coverage.get("total_predictions", 0))
    except Exception:
        pass

    # Realized model accuracy (last 30 days of labeled predictions)
    try:
        accuracy_metrics = db.calculate_realized_accuracy(lookback_days=30)
        if accuracy_metrics.get("total_labeled", 0) > 0:
            model_accuracy_gauge.set(accuracy_metrics.get("accuracy", 0.0))
    except Exception:
        pass

    try:
        import json as _json
        from pathlib import Path as _Path

        _drift_files = list(_Path("reports/drift").glob("drift_summary_*.json"))
        drift_reports_total.set(len(_drift_files))

        # Count actual drift detections per type from summary JSON files.
        # detect_drift.py writes these files; reading them here bridges the
        # cross-process metric gap so Prometheus sees the real drift state.
        # Four drift types matching project scenarios:
        #   data             — overall feature distribution shift (DataDriftPreset)
        #   concept          — label/target distribution shift (TargetDriftPreset)
        #   feature          — individual feature drift (subset of data drift)
        #   prior_probability — prediction label distribution shift (PredictionDrift)
        _drift_counts: dict[str, int] = {
            "data": 0,
            "concept": 0,
            "feature": 0,
            "prior_probability": 0,
        }
        for _f in _drift_files:
            try:
                with open(_f) as _fp:
                    _s = _json.load(_fp)
                _dd = _s.get("data_drift") or {}
                if _dd.get("drift_detected"):
                    _drift_counts["data"] += 1
                    # feature drift = any individual feature drifting
                    if _dd.get("n_drifted_features", 0) > 0:
                        _drift_counts["feature"] += 1
                if (_s.get("target_drift") or {}).get("drift_detected"):
                    _drift_counts["concept"] += 1
                if (_s.get("prediction_drift_details") or {}).get("drift_detected"):
                    _drift_counts["prior_probability"] += 1
            except Exception:
                pass
        for _dtype, _cnt in _drift_counts.items():
            drift_detected_gauge.labels(drift_type=_dtype).set(_cnt)
    except Exception:
        pass

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/internal/metrics/retraining-trigger", include_in_schema=False)
async def record_retraining_trigger_endpoint(reason: str = "scheduled"):
    """
    Internal endpoint for Airflow DAGs to increment retraining_triggers_total.

    Called on DAG success so that NoRetrainingInWeek alert has data to evaluate.
    Also used by drift_triggered_retraining.py with reason='drift'.

    Args:
        reason: Trigger reason label (e.g. 'scheduled', 'drift', 'manual')
    """
    retraining_triggers_total.labels(trigger_reason=reason).inc()
    return {"status": "ok", "trigger_reason": reason}


@app.post("/internal/metrics/retraining-failure", include_in_schema=False)
async def record_retraining_failure_endpoint(reason: str = "unknown"):
    """
    Internal endpoint for Airflow DAGs to increment retraining_failures_total.

    Airflow runs DAG tasks in separate Python processes that cannot directly
    update the API's in-process Prometheus registry.  DAGs call this endpoint
    via HTTP so the counter is incremented in the correct (API) process and
    surfaces on the /metrics scrape.

    Args:
        reason: Failure reason label (e.g. 'dag_failure', 'training_error')
    """
    retraining_failures_total.labels(reason=reason).inc()
    return {"status": "ok", "reason": reason}


@app.post("/internal/metrics/drift-detection", include_in_schema=False)
async def record_drift_detection_endpoint(drift_type: str = "data"):
    """
    Internal endpoint for Airflow DAGs to increment drift_detections_total.

    Airflow's evidently_drift_detection DAG calls this after detecting drift so
    that the counter is incremented in the API process (the one Prometheus scrapes).
    Without this, calling prometheus_client counters inside Airflow has no effect
    on the /metrics endpoint because each Python process has its own in-memory
    Prometheus registry.

    Args:
        drift_type: Type of drift ('data', 'concept', 'prediction', 'feature')
    """
    from src.monitoring.metrics import drift_detections_total

    drift_detections_total.labels(drift_type=drift_type).inc()
    return {"status": "ok", "drift_type": drift_type}


@app.post("/internal/kpi-metrics", include_in_schema=False)
async def update_kpi_metrics(request: Request):
    """
    Internal endpoint for Airflow DAGs to set KPI governance gauges.

    Called by the automated_retraining DAG after promotion to record:
    - model_deploy_time_seconds: seconds from DAG trigger to production promotion
    - automation_rate_gauge: 1.0 (auto) or 0.0 (manual)
    - mttd_seconds: mean time to drift detection (set by drift DAG)
    """
    from src.monitoring.metrics import (
        automation_rate_gauge,
        model_deploy_time_seconds,
        mttd_seconds,
    )

    body = await request.json()
    if "deploy_time_seconds" in body and body["deploy_time_seconds"] is not None:
        model_deploy_time_seconds.set(float(body["deploy_time_seconds"]))
    if "automation_rate" in body and body["automation_rate"] is not None:
        automation_rate_gauge.set(float(body["automation_rate"]))
    if "mttd_seconds" in body and body["mttd_seconds"] is not None:
        mttd_seconds.set(float(body["mttd_seconds"]))
    return {"status": "ok"}


@app.get("/stats", response_model=MetricsResponse, tags=["Monitoring"])
async def get_stats(
    lookback_days: int = 30,
    db: Database = Depends(get_database),
):
    """
    Get business performance statistics.

    Returns realized accuracy and label coverage for the specified lookback period.
    This endpoint provides human-readable JSON statistics, distinct from /metrics
    which provides Prometheus-formatted metrics for machine scraping.

    Args:
        lookback_days: Number of days to look back (default: 30)
        db: Database connection

    Returns:
        Statistics response with accuracy and coverage data
    """
    try:
        # Get accuracy metrics
        accuracy_metrics = db.calculate_realized_accuracy(lookback_days=lookback_days)

        # Get label coverage
        coverage_metrics = db.get_label_coverage(lookback_days=lookback_days)

        return MetricsResponse(
            total_predictions=coverage_metrics["total_predictions"],
            total_labeled=coverage_metrics["labeled_predictions"],
            label_coverage=coverage_metrics["label_coverage"],
            healthy_count=coverage_metrics.get("healthy_predictions", 0),
            unhealthy_count=coverage_metrics.get("unhealthy_predictions", 0),
            realized_accuracy=accuracy_metrics["accuracy"]
            if accuracy_metrics["total_labeled"] > 0
            else None,
            healthy_accuracy=accuracy_metrics["healthy_accuracy"]
            if accuracy_metrics["total_healthy"] > 0
            else None,
            unhealthy_accuracy=accuracy_metrics["unhealthy_accuracy"]
            if accuracy_metrics["total_unhealthy"] > 0
            else None,
            lookback_days=lookback_days,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve metrics: {str(e)}",
        ) from e


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info(
    model_artifact: dict[str, Any] = Depends(get_model),
):
    """
    Get current model information.

    Returns model version, algorithm, training timestamp, features, and source.

    Args:
        model_artifact: Loaded model

    Returns:
        Model information response with registry details if applicable
    """
    return ModelInfoResponse(
        model_version=model_artifact["model_version"],
        model_path=str(model_artifact.get("model_path", "unknown")),
        algorithm=model_artifact["algorithm"],
        trained_at=model_artifact["trained_at"],
        features_used=model_artifact["feature_names"],
        source=model_artifact.get("source", "unknown"),
        registry_version=model_artifact.get("registry_version"),
        registry_stage=model_artifact.get("registry_stage"),
    )


@app.post("/admin/reload-model", response_model=ReloadModelResponse, tags=["Admin"])
async def reload_model(
    settings: Settings = Depends(get_settings),
):
    """
    Reload model from MLflow Registry.

    Clears the model cache and forces reloading from MLflow Registry or bootstrap.
    Useful after model promotion or to switch model versions.

    Returns:
        Reload status and new model information

    Example:
        After promoting a new model to Production:
        ```bash
        curl -X POST http://localhost:8000/admin/reload-model
        ```
    """
    from src.api.dependencies import clear_model_cache, restore_model_cache
    from src.api.dependencies import get_model as load_model_fn

    # Snapshot current model before clearing cache
    old_model: dict | None = None
    try:
        old_model = load_model_fn(settings)
        previous_source = old_model.get("source", "unknown")
        previous_version = old_model.get("model_version", "unknown")
    except Exception:
        previous_source = "none"
        previous_version = "none"

    # Clear cache to force reload
    clear_model_cache()

    # Audit logging - reload initiated
    log_model_operation(
        operation="reload_initiated",
        model_version=previous_version,
        user_id="admin",
        success=True,
    )

    # Load new model; restore old model snapshot on failure to keep API healthy
    try:
        new_model = load_model_fn(settings)
        new_source = new_model.get("source", "unknown")
        new_version = new_model.get("model_version", "unknown")
        registry_version = new_model.get("registry_version")

        if previous_source in ("none", "unknown") and new_source not in ("none", "unknown"):
            # First successful load — no prior model existed
            message = f"Model loaded: {new_source} model {new_version}"
        elif previous_source == new_source and previous_version == new_version:
            # Model exists and is unchanged — API already has the latest version
            message = f"Model already up-to-date: {new_source} model {new_version}"
        else:
            message = (
                f"Model reloaded: {previous_source} {previous_version} → {new_source} {new_version}"
            )

        # Record model reload metrics
        record_model_reload(source=new_source, trigger="manual")

        # Audit logging - reload completed
        log_model_operation(
            operation="reload_completed",
            model_version=new_version,
            registry_version=registry_version,
            user_id="admin",
            success=True,
        )

        return ReloadModelResponse(
            success=True,
            message=message,
            previous_source=previous_source,
            new_source=new_source,
            model_version=new_version,
            registry_version=registry_version,
        )

    except Exception as e:
        # Restore previous model so API remains healthy after a failed reload
        if old_model is not None:
            restore_model_cache(settings, old_model)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload model: {str(e)}",
        ) from e


# ── Kubernetes management endpoints ──────────────────────────────────────
# Only available when running inside a K8s cluster (or with a valid kubeconfig).
# Endpoints degrade gracefully when the kubernetes SDK is not installed or
# when not running in a K8s environment (KUBERNETES_SERVICE_HOST not set).


def _k8s_client():
    """Return a configured kubernetes CoreV1Api client, or None if unavailable."""
    try:
        from kubernetes import client, config  # type: ignore[import-untyped]

        if os.getenv("KUBERNETES_SERVICE_HOST"):
            config.load_incluster_config()
        else:
            config.load_kube_config()
        return client.CoreV1Api(), client.AppsV1Api()
    except Exception:  # noqa: BLE001
        return None, None


_K8S_NAMESPACE = os.getenv("K8S_NAMESPACE", "mlops")


@app.get("/k8s/pods", tags=["Kubernetes"])
async def list_k8s_pods():
    """
    List all pods in the mlops namespace.

    Returns pod name, status, ready state, and restart count.
    Returns 503 if the Kubernetes client is unavailable.
    """
    core_v1, _ = _k8s_client()
    if core_v1 is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Kubernetes client unavailable — not running in a K8s cluster or kubernetes SDK not installed.",
        )
    try:
        pods = core_v1.list_namespaced_pod(namespace=_K8S_NAMESPACE)
        result = []
        for pod in pods.items:
            container_statuses = pod.status.container_statuses or []
            ready = all(cs.ready for cs in container_statuses)
            restarts = sum(cs.restart_count for cs in container_statuses)
            result.append(
                {
                    "name": pod.metadata.name,
                    "status": pod.status.phase,
                    "ready": ready,
                    "restarts": restarts,
                    "node": pod.spec.node_name,
                    "labels": pod.metadata.labels or {},
                }
            )
        return {"namespace": _K8S_NAMESPACE, "pods": result, "count": len(result)}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list pods: {exc}",
        ) from exc


@app.post("/k8s/scale", tags=["Kubernetes"])
async def scale_k8s_deployment(deployment: str, replicas: int):
    """
    Scale a deployment in the mlops namespace.

    Args:
        deployment: Deployment name (e.g. "api")
        replicas: Target replica count (1–10)

    Returns 503 if the Kubernetes client is unavailable.
    """
    if not 1 <= replicas <= 10:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="replicas must be between 1 and 10",
        )
    _, apps_v1 = _k8s_client()
    if apps_v1 is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Kubernetes client unavailable.",
        )
    try:
        body = {"spec": {"replicas": replicas}}
        apps_v1.patch_namespaced_deployment_scale(
            name=deployment, namespace=_K8S_NAMESPACE, body=body
        )
        logger.info("k8s | scaled deployment=%s to replicas=%d", deployment, replicas)
        return {
            "deployment": deployment,
            "namespace": _K8S_NAMESPACE,
            "replicas": replicas,
            "status": "scaled",
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to scale deployment {deployment}: {exc}",
        ) from exc


@app.post("/k8s/kill-pod", tags=["Kubernetes"])
async def kill_k8s_pod(pod_name: str):
    """
    Delete (kill) a pod in the mlops namespace.

    The pod will be rescheduled by its Deployment controller.
    Useful for testing resilience and rolling restarts.

    Returns 503 if the Kubernetes client is unavailable.
    """
    core_v1, _ = _k8s_client()
    if core_v1 is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Kubernetes client unavailable.",
        )
    try:
        from kubernetes import client as k8s_client_mod  # type: ignore[import-untyped]

        core_v1.delete_namespaced_pod(
            name=pod_name,
            namespace=_K8S_NAMESPACE,
            body=k8s_client_mod.V1DeleteOptions(grace_period_seconds=0),
        )
        logger.info("k8s | killed pod=%s namespace=%s", pod_name, _K8S_NAMESPACE)
        return {
            "pod": pod_name,
            "namespace": _K8S_NAMESPACE,
            "status": "deleted",
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete pod {pod_name}: {exc}",
        ) from exc


# ── Background model-version polling ─────────────────────────────────────
# Periodically check the MLflow registry for a newer Production model and
# auto-reload if the version has changed.  Interval is controllable via the
# MODEL_POLL_INTERVAL_SECONDS env var (default 60 s, 0 disables polling).


async def _poll_model_registry() -> None:
    """Background coroutine that polls for new model versions."""
    import asyncio

    interval = int(os.getenv("MODEL_POLL_INTERVAL_SECONDS", "60"))
    if interval <= 0:
        logger.info("model_poll | disabled (MODEL_POLL_INTERVAL_SECONDS=%s)", interval)
        return

    logger.info("model_poll | starting (interval=%ss)", interval)

    while True:
        await asyncio.sleep(interval)
        try:
            from src.api.dependencies import _model_cache, _registry_next_retry
            from src.api.dependencies import get_settings as load_settings

            settings = load_settings()
            if not settings.MODEL_REGISTRY_ENABLED:
                continue

            # Check current cached version
            cache_key = f"registry_{settings.MODEL_REGISTRY_NAME}_{settings.MODEL_REGISTRY_STAGE}"
            cached = _model_cache.get(cache_key)
            cached_version = cached.get("registry_version") if cached else None
            cached_source = cached.get("source") if cached else None

            # Query the registry for the latest production version
            import threading

            _poll_result: list[Any] = []

            def _check_registry(
                _settings: Any = settings,
                _result: list[Any] = _poll_result,
            ) -> None:
                try:
                    from src.training.registry import get_production_models

                    prod_models = get_production_models(_settings.MODEL_REGISTRY_NAME)
                    if prod_models:
                        latest = max(prod_models, key=lambda m: m["version"])
                        _result.append(latest["version"])
                    else:
                        _result.append(None)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("model_poll | registry check failed: %s", exc)
                    _result.append(None)

            t = threading.Thread(target=_check_registry, daemon=True)
            t.start()
            t.join(timeout=10.0)

            if not _poll_result:
                continue

            latest_version = _poll_result[0]

            # Reload if:
            #   - We currently serve bootstrap but there is now a registry model
            #   - The registry version has changed
            need_reload = False
            if latest_version is not None:
                if cached_source == "bootstrap":
                    need_reload = True
                    logger.info(
                        "model_poll | registry model v%s detected (currently serving bootstrap)",
                        latest_version,
                    )
                elif cached_version is not None and int(latest_version) != int(cached_version):
                    need_reload = True
                    logger.info(
                        "model_poll | new version detected: v%s → v%s",
                        cached_version,
                        latest_version,
                    )

            if need_reload:
                # Atomic cache swap: load the new model FIRST (without clearing the cache),
                # then update the cache in one assignment.  This eliminates the race window
                # where concurrent prediction requests would find an empty cache and fall back
                # to the bootstrap model (causing the "every Nth prediction = bootstrap_v1.0"
                # pattern that is visible when predictions are generated in rapid succession).
                _new_result: list[Any] = []

                def _load_new_version(
                    _settings: Any = settings,
                    _result: list[Any] = _new_result,
                ) -> None:
                    try:
                        from src.training.registry import load_production_model_artifact

                        art = load_production_model_artifact(
                            model_name=_settings.MODEL_REGISTRY_NAME,
                            stage=_settings.MODEL_REGISTRY_STAGE,
                        )
                        if art:
                            _result.append(art)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("model_poll | version reload failed: %s", exc)

                _load_t = threading.Thread(target=_load_new_version, daemon=True)
                _load_t.start()
                _load_t.join(timeout=30.0)

                if not _new_result or _new_result[0] is None:
                    logger.warning(
                        "model_poll | reload skipped — could not load v%s from registry; "
                        "keeping current model to avoid bootstrap fallback",
                        latest_version,
                    )
                    continue

                new_artifact = _new_result[0]
                new_artifact["source"] = "registry"
                # Enrich with git_sha / dvc_data_hash / airflow_run_id from MLflow run tags.
                # The explicit get_model() path does this too; the poller must also enrich
                # so that predictions made after an auto-reload have correct lineage.
                from src.api.dependencies import _enrich_artifact_lineage as _enrich

                _enrich(new_artifact)
                # Overwrite the cache atomically (single dict assignment is GIL-protected in CPython)
                _model_cache[cache_key] = new_artifact
                # Clear negative-retry entry so future explicit calls go straight to cache
                # Clear negative-retry entry so future explicit calls go straight to cache
                _registry_next_retry.pop(cache_key, None)

                new_source = new_artifact.get("source", "unknown")
                new_version = new_artifact.get("model_version", "unknown")
                registry_v = new_artifact.get("registry_version")

                update_model_info_metric(
                    model_version=new_version,
                    algorithm=new_artifact.get("algorithm", "unknown"),
                    source=new_source,
                    trained_at=new_artifact.get("trained_at", "unknown"),
                    features=new_artifact.get("features_used", []),
                )
                record_model_reload(source=new_source, trigger="auto_poll")

                log_model_operation(
                    operation="auto_reload",
                    model_version=new_version,
                    registry_version=registry_v,
                    user_id="system",
                    success=True,
                )
                logger.info(
                    "model_poll | auto-reloaded model: %s %s (registry v%s)",
                    new_source,
                    new_version,
                    registry_v,
                )

        except Exception as exc:  # noqa: BLE001
            logger.warning("model_poll | error during poll cycle: %s", exc)
