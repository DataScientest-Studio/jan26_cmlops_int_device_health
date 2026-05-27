"""
Shared fixtures for API tests.

Provides:
- FastAPI TestClient with dependency overrides
- In-memory SQLite database for isolation
- Mock model artifact
- Authentication helpers (tokens, API keys)
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.auth import create_access_token
from src.api.dependencies import Settings, get_database, get_model, get_settings
from src.api.main import app
from src.database import Database


@pytest.fixture
def test_settings():
    """Minimal settings for testing."""
    settings = Settings()
    settings.DEPLOYMENT_MODE = "local"
    settings.MLFLOW_TRACKING_URI = "http://localhost:5000"
    settings.MODEL_REGISTRY_ENABLED = False
    return settings


@pytest.fixture
def test_db(tmp_path):
    """In-memory SQLite database for test isolation."""
    db_path = tmp_path / "test.db"
    db = Database(db_path=db_path)
    yield db
    db.close()


@pytest.fixture
def mock_model_artifact():
    """Mock model artifact matching production structure."""
    from unittest.mock import MagicMock

    model = MagicMock()
    model.predict.return_value = np.array([0])
    model.predict_proba.return_value = np.array([[0.85, 0.15]])

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.mean_ = np.zeros(5)
    scaler.scale_ = np.ones(5)
    scaler.n_features_in_ = 5

    return {
        "model": model,
        "scaler": scaler,
        "feature_names": ["noise_std", "peak_count", "fwhm_mean", "area_total", "snr"],
        "model_version": "test-v1.0",
        "version": "1.0",
        "algorithm": "LogisticRegression",
        "trained_at": "2025-01-01T00:00:00",
        "source": "bootstrap",
        "model_path": "models/bootstrap_model.pkl",
        "mlflow_run_id": None,
        "git_sha": None,
        "dvc_data_hash": None,
        "airflow_run_id": None,
    }


@pytest.fixture
def client(test_db, mock_model_artifact, test_settings):
    """FastAPI TestClient with overridden dependencies."""

    def override_get_database():
        yield test_db

    def override_get_model():
        return mock_model_artifact

    def override_get_settings():
        return test_settings

    app.dependency_overrides[get_database] = override_get_database
    app.dependency_overrides[get_model] = override_get_model
    app.dependency_overrides[get_settings] = override_get_settings

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


@pytest.fixture
def admin_token():
    """Valid admin JWT token (read + write + admin scopes)."""
    return create_access_token(data={"sub": "admin", "scopes": ["read", "write", "admin"]})


@pytest.fixture
def user_token():
    """Valid user JWT token (read scope only)."""
    return create_access_token(data={"sub": "user", "scopes": ["read"]})


@pytest.fixture
def service_token():
    """Valid service JWT token (read + write scopes)."""
    return create_access_token(data={"sub": "service", "scopes": ["read", "write"]})


@pytest.fixture
def admin_headers(admin_token):
    """Authorization headers for admin user."""
    return {"Authorization": f"Bearer {admin_token}"}


@pytest.fixture
def user_headers(user_token):
    """Authorization headers for regular user."""
    return {"Authorization": f"Bearer {user_token}"}


@pytest.fixture
def service_headers(service_token):
    """Authorization headers for service account."""
    return {"Authorization": f"Bearer {service_token}"}


@pytest.fixture
def api_key_headers():
    """Headers with valid API key (read + write scopes)."""
    return {"X-API-Key": "dev-key-12345"}


@pytest.fixture
def readonly_api_key_headers():
    """Headers with read-only API key."""
    return {"X-API-Key": "monitoring-key-67890"}


@pytest.fixture
def sample_predict_request():
    """Valid prediction request payload."""
    from src.database import generate_device_id

    t = np.linspace(0, 10, 101).tolist()
    amplitude = (
        np.exp(-((np.array(t) - 5) ** 2) / (2 * 0.5**2)) + np.random.normal(0, 0.01, 101)
    ).tolist()
    return {
        "device_id": generate_device_id(),
        "device_name": "Test Device",
        "device_type": "Sensor-A",
        "location": "Lab-1",
        "time_values": t,
        "amplitude_values": amplitude,
    }
