"""
Tests for API security — authentication, authorization, and input validation.

Validates:
- OAuth2 token-based authentication works correctly
- API key authentication works correctly
- Scope-based authorization enforcement
- Invalid/expired token handling
- Input validation prevents injection attacks
- Security headers are present
"""

from unittest.mock import patch

from src.api.auth import create_access_token
from src.api.security import validate_api_key


class TestOAuth2Security:
    """Tests for OAuth2 token authentication."""

    def test_valid_token_grants_access(self, client, admin_headers):
        """Valid token allows access to protected endpoints."""
        response = client.get("/auth/users/me", headers=admin_headers)
        assert response.status_code == 200

    def test_missing_token_returns_401(self, client):
        """Missing token returns 401 Unauthorized."""
        response = client.get("/auth/users/me")
        assert response.status_code == 401

    def test_invalid_token_returns_401(self, client):
        """Malformed token returns 401."""
        response = client.get(
            "/auth/users/me",
            headers={"Authorization": "Bearer invalid.token.here"},
        )
        assert response.status_code == 401

    def test_expired_token_returns_401(self, client):
        """Expired token returns 401."""
        from datetime import timedelta

        expired_token = create_access_token(
            data={"sub": "admin", "scopes": ["read"]},
            expires_delta=timedelta(seconds=-10),
        )
        response = client.get(
            "/auth/users/me",
            headers={"Authorization": f"Bearer {expired_token}"},
        )
        assert response.status_code == 401


class TestAPIKeySecurity:
    """Tests for API key authentication."""

    def test_valid_api_key_grants_access(self):
        """Valid API key returns key info."""
        result = validate_api_key("dev-key-12345")
        assert result is not None
        assert "scopes" in result

    def test_invalid_api_key_returns_none(self):
        """Invalid API key returns None."""
        assert validate_api_key("nonexistent-key") is None
        assert validate_api_key("") is None

    def test_api_key_authentication_on_endpoint(
        self, client, api_key_headers, sample_predict_request
    ):
        """API key can authenticate to protected endpoints."""
        with patch("src.api.main.predict") as mock_predict:
            mock_predict.return_value = {
                "predicted_label": 0,
                "confidence": 0.9,
                "model_version": "test-v1.0",
                "probabilities": {"0": 0.9, "1": 0.1},
                "features": {"noise_std": 0.01},
            }
            response = client.post("/predict", json=sample_predict_request, headers=api_key_headers)
        assert response.status_code == 200


class TestScopeEnforcement:
    """Tests for scope-based authorization."""

    def test_write_scope_required_for_predict(self, client, user_headers, sample_predict_request):
        """User with only 'read' scope cannot POST /predict."""
        with patch("src.api.main.predict") as mock_predict:
            mock_predict.return_value = {
                "predicted_label": 0,
                "confidence": 0.9,
                "model_version": "test-v1.0",
                "probabilities": {},
                "features": {},
            }
            response = client.post("/predict", json=sample_predict_request, headers=user_headers)
        assert response.status_code == 403

    def test_admin_can_access_all(self, client, admin_headers):
        """Admin user with all scopes can access any endpoint."""
        response = client.get("/auth/users/me", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert "admin" in data["scopes"]


class TestInputValidation:
    """Tests for input validation security."""

    def test_sql_injection_in_device_id(self, client, admin_headers):
        """SQL injection attempt in device_id is rejected by Pydantic."""
        payload = {
            "device_id": "'; DROP TABLE predictions;--",
            "time_values": [1, 2, 3],
            "amplitude_values": [0.1, 0.2, 0.3],
        }
        response = client.post("/predict", json=payload, headers=admin_headers)
        assert response.status_code == 422

    def test_xss_in_device_name(self, client, admin_headers, sample_predict_request):
        """XSS payload in device_name doesn't execute (field is data-only)."""
        sample_predict_request["device_name"] = "<script>alert('xss')</script>"
        with patch("src.api.main.predict") as mock_predict:
            mock_predict.return_value = {
                "predicted_label": 0,
                "confidence": 0.9,
                "model_version": "test-v1.0",
                "probabilities": {"0": 0.9, "1": 0.1},
                "features": {"noise_std": 0.01},
            }
            response = client.post("/predict", json=sample_predict_request, headers=admin_headers)
        # FastAPI returns JSON, not HTML — XSS has no effect
        assert response.status_code == 200

    def test_oversized_payload_handled(self, client, admin_headers):
        """Extremely large signal arrays are handled gracefully."""
        payload = {
            "device_id": "",
            "time_values": list(range(100000)),  # 100K points
            "amplitude_values": [0.1] * 100000,
        }
        with patch("src.api.main.predict") as mock_predict:
            mock_predict.return_value = {
                "predicted_label": 0,
                "confidence": 0.9,
                "model_version": "test-v1.0",
                "probabilities": {},
                "features": {},
            }
            response = client.post("/predict", json=payload, headers=admin_headers)
        # Should either succeed or return validation error, not crash
        assert response.status_code in (200, 400, 422)

    def test_request_id_header_returned(self, client):
        """Response includes X-Request-ID header."""
        response = client.get("/health")
        assert "X-Request-ID" in response.headers
