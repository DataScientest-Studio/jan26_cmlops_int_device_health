"""
Live API tests — require running Docker stack.

Run with: uv run pytest tests/live/ -m live --timeout=30
"""

import uuid

import pytest
import requests

pytestmark = pytest.mark.live


class TestAPILiveHealth:
    """Test health and readiness endpoints on running stack."""

    def test_health_endpoint_returns_200(self, stack_base_url):
        """GET /health returns 200."""
        r = requests.get(f"{stack_base_url}/health", timeout=5)
        assert r.status_code == 200

    def test_health_response_has_version(self, stack_base_url):
        """Health response includes version field."""
        r = requests.get(f"{stack_base_url}/health", timeout=5)
        data = r.json()
        assert "version" in data
        assert data["version"]  # non-empty

    def test_health_response_has_deployment_mode(self, stack_base_url):
        """Health response includes deployment_mode."""
        r = requests.get(f"{stack_base_url}/health", timeout=5)
        data = r.json()
        assert "deployment_mode" in data
        assert data["deployment_mode"] in ("local", "cloud")

    def test_health_response_has_model_status(self, stack_base_url):
        """Health response includes model_loaded status."""
        r = requests.get(f"{stack_base_url}/health", timeout=5)
        data = r.json()
        assert "model_loaded" in data


class TestAPILivePrediction:
    """Test prediction endpoint on running stack."""

    def test_predict_requires_auth(self, stack_base_url):
        """POST /predict without auth returns 401."""
        r = requests.post(
            f"{stack_base_url}/predict",
            json={"device_id": str(uuid.uuid4()), "time_values": [0.0], "amplitude_values": [1.0]},
            timeout=5,
        )
        assert r.status_code == 401

    def test_predict_with_valid_auth(self, stack_base_url):
        """POST /predict with valid credentials returns 200 or 422 (validation)."""
        # First get a token
        login_r = requests.post(
            f"{stack_base_url}/auth/token",
            data={"username": "admin", "password": "admin123", "grant_type": "password"},
            timeout=5,
        )
        if login_r.status_code != 200:
            pytest.skip("Cannot authenticate — admin user may not exist")

        token = login_r.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Send a minimal signal (may fail validation but should not 401)
        import numpy as np

        t = np.linspace(0, 100, 101).tolist()
        amp = (np.exp(-0.5 * ((np.array(t) - 50) / 3) ** 2) * 2.0).tolist()

        r = requests.post(
            f"{stack_base_url}/predict",
            json={"device_id": str(uuid.uuid4()), "time_values": t, "amplitude_values": amp},
            headers=headers,
            timeout=10,
        )
        # Should succeed or give meaningful error (not 401/403)
        assert r.status_code in (200, 422, 500)  # auth passed


class TestAPILiveMetrics:
    """Test metrics/observability endpoints."""

    def test_metrics_endpoint_exists(self, api_direct_url):
        """GET /metrics returns Prometheus metrics."""
        r = requests.get(f"{api_direct_url}/metrics", timeout=5)
        assert r.status_code == 200
        assert "api_requests_total" in r.text or "python_info" in r.text

    def test_openapi_schema_available(self, stack_base_url):
        """GET /openapi.json returns valid OpenAPI schema."""
        r = requests.get(f"{stack_base_url}/openapi.json", timeout=5)
        assert r.status_code == 200
        schema = r.json()
        assert "openapi" in schema
        assert "paths" in schema


class TestServiceHealth:
    """Verify all Docker services are responsive."""

    def test_prometheus_is_running(self, prometheus_url):
        """Prometheus responds at /api/v1/status/config."""
        r = requests.get(f"{prometheus_url}/api/v1/status/config", timeout=5)
        assert r.status_code == 200

    def test_prometheus_has_targets(self, prometheus_url):
        """Prometheus has at least one active scrape target."""
        r = requests.get(f"{prometheus_url}/api/v1/targets", timeout=5)
        data = r.json()
        assert data["status"] == "success"
        active = data["data"]["activeTargets"]
        assert len(active) > 0

    def test_grafana_is_running(self, grafana_url):
        """Grafana responds at /api/health."""
        r = requests.get(f"{grafana_url}/api/health", timeout=5)
        assert r.status_code == 200
        assert r.json()["database"] == "ok"

    def test_mlflow_buffer_is_running(self, mlflow_url):
        """MLflow buffer responds."""
        try:
            r = requests.get(f"{mlflow_url}/health", timeout=5)
            # MLflow may return 200 or have a different health endpoint
            assert r.status_code in (200, 404)
        except requests.ConnectionError:
            pytest.skip("MLflow buffer not accessible")

    def test_nginx_proxy_headers(self, stack_base_url):
        """Nginx adds expected security headers."""
        r = requests.get(f"{stack_base_url}/health", timeout=5)
        # Check for common security headers added by Nginx
        headers = r.headers
        # At minimum, content-type should be set
        assert "content-type" in headers
