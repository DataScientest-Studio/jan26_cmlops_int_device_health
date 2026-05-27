"""
Conftest for live Docker stack tests.

These tests require the full Docker stack running locally.
They are marked with @pytest.mark.live and excluded from CI.

Run manually with: uv run pytest tests/live/ -m live --timeout=30
"""

import os

import pytest
import requests


def _stack_is_running() -> bool:
    """Check if the Docker stack is accessible."""
    try:
        r = requests.get("http://localhost:8000/health", timeout=3)
        return r.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


@pytest.fixture(scope="session")
def stack_base_url() -> str:
    """Base URL for the API via Nginx."""
    return os.environ.get("MLOPS_API_URL", "http://localhost:8000")


@pytest.fixture(scope="session")
def api_direct_url() -> str:
    """Direct API URL (bypassing Nginx)."""
    return os.environ.get("MLOPS_API_DIRECT_URL", "http://localhost:8001")


@pytest.fixture(scope="session")
def prometheus_url() -> str:
    """Prometheus URL."""
    return "http://localhost:9090"


@pytest.fixture(scope="session")
def grafana_url() -> str:
    """Grafana URL."""
    return "http://localhost:3000"


@pytest.fixture(scope="session")
def mlflow_url() -> str:
    """MLflow buffer URL."""
    return "http://localhost:5000"


@pytest.fixture(scope="session", autouse=True)
def require_stack():
    """Skip all live tests if Docker stack is not running."""
    if not _stack_is_running():
        pytest.skip("Docker stack not running (API at localhost:8000 unreachable)")
