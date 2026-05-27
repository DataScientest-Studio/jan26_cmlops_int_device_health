"""
Dependency injection for FastAPI application.

Provides:
- Database connection management
- Model loading and caching (from MLflow Registry or bootstrap)
- Configuration management
"""

import os
from collections.abc import Generator
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import Depends, HTTPException, status

from src.database import Database

# ======================================
# Configuration
# ======================================


class Settings:
    """Application settings."""

    # Static values (not env-driven)
    API_VERSION: str = "1.0.0"
    API_TITLE: str = "MLOps Device Health Monitoring API"
    API_DESCRIPTION: str = """
    REST API for device health prediction and monitoring.

    Features:
    - Real-time device health predictions
    - Sparse label injection for model evaluation
    - Performance metrics and monitoring
    - Model information and versioning
    """

    # Static path defaults
    DATABASE_PATH: Path = Path("data/database/mlops.db")
    # No default model path — API starts in greenfield state (no model) until
    # the MLflow registry has a Production model or a champion_model.pkl is
    # written locally by the Airflow retraining pipeline.
    MODEL_PATH: Path | None = None

    # Metrics settings
    METRICS_LOOKBACK_DAYS: int = 30

    def __init__(self) -> None:
        # Deployment mode: 'local' (all Docker) or 'cloud' (DagsHub MLflow/DVC)
        # Set via DEPLOYMENT_MODE env var (written by .env.local / .env.cloud)
        self.DEPLOYMENT_MODE: str = os.getenv("DEPLOYMENT_MODE", "local")

        # MLflow tracking URI — varies by deployment mode
        # local: http://mlflow:5000  |  cloud: https://dagshub.com/...mlflow
        self.MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

        # Model registry settings
        self.MODEL_REGISTRY_ENABLED: bool = os.getenv("MODEL_REGISTRY_ENABLED", "true").lower() in (
            "true",
            "1",
            "yes",
        )
        self.MODEL_REGISTRY_NAME: str = os.getenv("MODEL_REGISTRY_NAME", "device_health_classifier")
        self.MODEL_REGISTRY_STAGE: str = os.getenv("MODEL_REGISTRY_STAGE", "Production")


@lru_cache
def get_settings() -> Settings:
    """Get application settings (cached)."""
    return Settings()


# ======================================
# Database Dependency
# ======================================


def get_database(settings: Settings = Depends(get_settings)) -> Generator[Database, None, None]:
    """
    Get database connection.

    Dependency that provides Database instance for request handlers.

    Selects the backend automatically:
    - If the ``DATABASE_URL`` environment variable is set to a PostgreSQL URL
      (``postgresql://...``), the PostgreSQL backend is used — this is the
      case in every Docker deployment (local or cloud mode).
    - Otherwise falls back to the SQLite file at ``settings.DATABASE_PATH``
      (used in unit / integration tests without Docker).

    Args:
        settings: Application settings

    Returns:
        Database instance

    Raises:
        HTTPException(503): If the database is unreachable (e.g. bad credentials,
            network error, or server not ready). Returns structured JSON so callers
            (including the /health endpoint) get a clean 503 instead of a 500.

    Note:
        Database connection is created per request and closed automatically.
    """
    db_url: str = os.getenv("DATABASE_URL", "")
    try:
        if db_url and db_url.startswith("postgresql"):
            db = Database(db_url=db_url)
        else:
            db = Database(settings.DATABASE_PATH)
    except Exception as exc:
        # Mask credential details from the URL before logging/returning
        safe_url = db_url.split("@")[-1] if "@" in db_url else (db_url or "sqlite")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database unavailable ({safe_url}): {exc}",
        ) from exc
    try:
        yield db
    finally:
        db.close()


def get_optional_database(
    settings: Settings = Depends(get_settings),
) -> Generator[Database | None, None, None]:
    """
    Get database connection — **never raises on failure**.

    Variant of :func:`get_database` intended for the ``/health`` endpoint.
    If the database is unreachable the dependency yields ``None`` instead of
    raising an :class:`HTTPException`, so the health handler can still run and
    return a structured JSON body describing the failure.

    Args:
        settings: Application settings

    Yields:
        Database instance on success, ``None`` if connection failed.
    """
    db_url: str = os.getenv("DATABASE_URL", "")
    db: Database | None = None
    try:
        if db_url and db_url.startswith("postgresql"):
            db = Database(db_url=db_url)
        else:
            db = Database(settings.DATABASE_PATH)
    except Exception:
        # Swallow the error — caller checks for None
        pass
    try:
        yield db
    finally:
        if db is not None:
            db.close()


# ======================================
# Model Dependency
# ======================================


_model_cache: dict[str, Any] = {}

# Negative cache: record the time of the last registry failure so we don't
# hammer the MLflow buffer repeatedly on health-check calls.
# Structure: { cache_key: next_retry_at_monotonic_time }
_registry_next_retry: dict[str, float] = {}
# 30 seconds is sufficient since the buffer is local; it only takes a few
# seconds to come up, unlike the previous 5-minute cooldown for DagsHub.
_REGISTRY_RETRY_COOLDOWN = 30  # seconds


def _enrich_artifact_lineage(artifact: dict[str, Any]) -> None:
    """Enrich a model artifact with git_sha/dvc_data_hash from its MLflow run tags."""
    run_id = artifact.get("mlflow_run_id")
    if not run_id:
        artifact["git_sha"] = None
        artifact["dvc_data_hash"] = None
        artifact["airflow_run_id"] = None
        return

    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        run = client.get_run(run_id)
        tags = run.data.tags
        # Use direct assignment (not setdefault) so we always populate from MLflow
        # tags rather than leaving None values from earlier bootstrap fallback.
        artifact["git_sha"] = tags.get("git_commit") or tags.get("git_sha")
        artifact["dvc_data_hash"] = (
            tags.get("dvc_data_version")
            or tags.get("dvc_data_hash")
            or run.data.params.get("dvc_data_hash")
        )
        artifact["airflow_run_id"] = tags.get("airflow_run_id")
    except Exception:  # noqa: BLE001
        artifact["git_sha"] = None
        artifact["dvc_data_hash"] = None
        artifact["airflow_run_id"] = None


def get_model(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """
    Get trained model (cached).

    Dependency that provides loaded model for predictions. Attempts to load
    from MLflow Registry first, falls back to bootstrap model if unavailable.

    Loading Strategy:
    1. If MODEL_REGISTRY_ENABLED=True:
       - Try loading from MLflow Registry (stage: Production/Staging)
       - Cache the registry model
    2. If registry unavailable or disabled:
       - Fall back to champion model from disk (MODEL_PATH or default locations)
       - Cache the bootstrap model

    Args:
        settings: Application settings

    Returns:
        Model artifact dict with:
        {
            "model": trained classifier,
            "scaler": StandardScaler,
            "feature_names": list[str],
            "model_version": str,
            "algorithm": str,
            "trained_at": str,
            "source": "registry" | "bootstrap",  # NEW: indicates model source
            "registry_version": int | None,      # NEW: registry version if applicable
            "registry_stage": str | None,        # NEW: registry stage if applicable
        }

    Raises:
        FileNotFoundError: If bootstrap model fallback also fails
    """
    import threading
    import time

    cache_key = f"registry_{settings.MODEL_REGISTRY_NAME}_{settings.MODEL_REGISTRY_STAGE}"

    # Check positive cache first
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    # Check negative cache: if the registry failed recently, skip the call
    # to avoid hammering DagsHub and triggering 429 rate limits.
    _now = time.monotonic()
    if cache_key in _registry_next_retry and _now < _registry_next_retry[cache_key]:
        _wait_left = int(_registry_next_retry[cache_key] - _now)
        print(f"⏳ Registry retry cooling down ({_wait_left}s left). Using cached fallback model.")
        # Return the bootstrap fallback if available (cached under a fallback key)
        _fallback_key = f"fallback_{cache_key}"
        if _fallback_key in _model_cache:
            return _model_cache[_fallback_key]
        # No fallback cached yet — attempt to load it
        return _load_bootstrap_fallback(settings, _fallback_key)

    model_artifact = None

    # Strategy 1: Try loading from MLflow Registry
    # IMPORTANT: MLflow network calls can block for up to MLFLOW_HTTP_REQUEST_TIMEOUT
    # seconds (default 120 s) when the tracking server is unreachable or the URI
    # is wrong.  A blocking call inside the async FastAPI event loop causes *all*
    # concurrent requests — including the Docker health-check — to time out.
    # We run the registry call in a daemon thread with a timeout controlled by
    # MODEL_REGISTRY_TIMEOUT (default 30 s).  Once the model is cached, subsequent
    # requests return instantly, so only the first call can be slow.
    _registry_timeout = int(os.getenv("MODEL_REGISTRY_TIMEOUT", "30"))
    if settings.MODEL_REGISTRY_ENABLED:
        _result: list[Any] = []

        def _load_from_registry() -> None:
            try:
                import mlflow

                tracking_uri = mlflow.get_tracking_uri()
                print(
                    f"ℹ Registry load: model={settings.MODEL_REGISTRY_NAME} "
                    f"stage={settings.MODEL_REGISTRY_STAGE} "
                    f"tracking_uri={tracking_uri}"
                )

                from src.training.registry import load_production_model_artifact

                artifact = load_production_model_artifact(
                    model_name=settings.MODEL_REGISTRY_NAME,
                    stage=settings.MODEL_REGISTRY_STAGE,
                )
                _result.append(artifact)
            except Exception as exc:  # noqa: BLE001
                print(f"⚠ Failed to load from MLflow Registry: {exc}")
                _result.append(None)

        _registry_thread = threading.Thread(target=_load_from_registry, daemon=True)
        _registry_thread.start()
        _registry_thread.join(timeout=_registry_timeout)

        if _result and _result[0] is not None:
            model_artifact = _result[0]
            model_artifact["source"] = "registry"
            # Enrich with git_sha/dvc_data_hash/airflow_run_id from MLflow run
            # tags.  We give the enrichment up to 10 s; on fast local connections
            # (Docker network) this completes in <100 ms.  If it times out the
            # fields stay None but predictions still work.
            _enrich_thread = threading.Thread(
                target=_enrich_artifact_lineage, args=(model_artifact,), daemon=True
            )
            _enrich_thread.start()
            _enrich_thread.join(timeout=10)
            print(
                f"✓ Loaded model from MLflow Registry: {settings.MODEL_REGISTRY_NAME} "
                f"v{model_artifact.get('registry_version')} ({settings.MODEL_REGISTRY_STAGE})"
            )
            _model_cache[cache_key] = model_artifact
            # Clear negative cache on success
            _registry_next_retry.pop(cache_key, None)
            return model_artifact
        elif not _registry_thread.is_alive():
            # Thread finished: either registry had no model, or an error occurred
            print(
                f"⚠ No {settings.MODEL_REGISTRY_STAGE} model found in registry "
                f"'{settings.MODEL_REGISTRY_NAME}', falling back to bootstrap model"
            )
        else:
            # Thread is still running (timeout hit) — MLflow is unreachable or slow
            print(
                f"⚠ MLflow registry call timed out after {_registry_timeout} s for "
                f"'{settings.MODEL_REGISTRY_NAME}'. Falling back to bootstrap model."
            )

    # Registry unavailable — record failure time to prevent retry for cooldown period
    _registry_next_retry[cache_key] = time.monotonic() + _REGISTRY_RETRY_COOLDOWN
    print(
        f"⏸  Registry unavailable. Will retry in {_REGISTRY_RETRY_COOLDOWN}s. "
        "Loading bootstrap fallback model."
    )

    # Strategy 2: Fall back to bootstrap model for degraded-but-functional operation.
    # The health check will return "degraded" (still 200) instead of "unhealthy" (503).
    _fallback_key = f"fallback_{cache_key}"
    return _load_bootstrap_fallback(settings, _fallback_key)


def _load_bootstrap_fallback(settings: "Settings", cache_key: str) -> dict[str, Any]:  # noqa: F821
    """Load bootstrap model pkl as a degraded fallback when the MLflow registry is unavailable."""
    import pickle
    from pathlib import Path

    # Check positive cache for fallback
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    # Look for a fallback model in MODEL_PATH or default locations.
    # In greenfield state (no training done yet) none of these exist, which
    # is the expected state — the caller handles FileNotFoundError gracefully.
    model_path = getattr(settings, "MODEL_PATH", None)
    candidates = []
    if model_path:
        candidates.append(Path(model_path))
    candidates += [
        Path("/app/models/champion_model.pkl"),
        Path("/opt/airflow/models/champion_model.pkl"),
        Path("models/champion_model.pkl"),
    ]

    for path in candidates:
        if path.exists():
            try:
                with open(path, "rb") as f:
                    artifact = pickle.load(f)  # noqa: S301  # trusted local file
                if isinstance(artifact, dict) and "model" in artifact:
                    artifact.setdefault("source", "bootstrap_fallback")
                    artifact.setdefault("model_version", "bootstrap_fallback")
                    artifact.setdefault("registry_version", None)
                    artifact.setdefault("registry_stage", None)
                    artifact.setdefault("git_sha", None)
                    artifact.setdefault("dvc_data_hash", None)
                    artifact.setdefault("airflow_run_id", None)
                    _model_cache[cache_key] = artifact
                    print(f"✓ Loaded bootstrap fallback model from {path}")
                    return artifact
            except Exception as exc:
                print(f"⚠ Failed to load fallback model from {path}: {exc}")

    raise FileNotFoundError(
        "No model available: MLflow registry is unreachable and no champion model found on disk. "
        "Run a Greenfield Bootstrap (use the Streamlit Greenfield use case or trigger the "
        "automated_retraining DAG) to train and register the first model."
    )


def clear_model_cache() -> None:
    """
    Clear model cache (for testing or reloading).

    Use this to force reloading models from MLflow Registry or disk.
    Useful after model promotion or for manual reload operations.

    Example:
        >>> from src.api.dependencies import clear_model_cache
        >>> clear_model_cache()  # Next request will reload model
    """
    global _model_cache, _registry_next_retry
    _model_cache.clear()
    _registry_next_retry.clear()
    print("✓ Model cache cleared (including negative cache — registry will be retried immediately)")


def restore_model_cache(settings: "Settings", model_artifact: dict[str, Any]) -> None:  # noqa: F821
    """
    Restore a previously loaded model artifact into the cache.

    Used by reload_model() to put the old model back when a reload attempt
    fails — this prevents the API from entering a model-less broken state.

    Args:
        settings: Application settings (used to derive the cache key)
        model_artifact: Model artifact dict previously returned by get_model()
    """
    cache_key = f"registry_{settings.MODEL_REGISTRY_NAME}_{settings.MODEL_REGISTRY_STAGE}"
    _model_cache[cache_key] = model_artifact
    # Clear any negative retry entries so the next reload attempt is immediate
    _registry_next_retry.pop(cache_key, None)
    print(
        f"✓ Model cache restored to {model_artifact.get('source', 'unknown')} "
        f"v{model_artifact.get('model_version', 'unknown')} after failed reload"
    )
