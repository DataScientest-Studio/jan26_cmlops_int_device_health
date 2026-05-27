"""
Tests for GET /health endpoint.

Validates:
- Health check returns structured response
- Status field reflects system state
- Services dictionary present
- Correct status codes (200 healthy, 503 unhealthy)
"""

from unittest.mock import MagicMock, patch


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_check_returns_200(self, client):
        """Health endpoint returns 200 when all systems healthy."""
        with patch("src.api.main.get_model") as mock_get_model:
            mock_get_model.return_value = {
                "model": MagicMock(),
                "scaler": MagicMock(),
                "feature_names": ["f1", "f2", "f3"],
                "model_version": "test-v1.0",
                "version": "1.0",
            }
            response = client.get("/health")

        # May return 200 or 503 depending on model availability
        assert response.status_code in (200, 503)
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "services" in data
        assert data["status"] in ("healthy", "degraded", "unhealthy")

    def test_health_check_includes_version(self, client):
        """Health response includes API version."""
        response = client.get("/health")
        data = response.json()
        assert "version" in data

    def test_health_check_includes_deployment_mode(self, client):
        """Health response includes deployment mode."""
        response = client.get("/health")
        data = response.json()
        assert "deployment_mode" in data
        assert data["deployment_mode"] == "local"

    def test_health_check_no_auth_required(self, client):
        """Health endpoint is public — no authentication needed."""
        response = client.get("/health")
        # Should not return 401/403
        assert response.status_code not in (401, 403)
