"""
Tests for POST /predict endpoint.

Validates:
- Successful prediction with valid auth
- Response schema correctness
- Input validation (missing fields, invalid signals)
- Scope enforcement (write scope required)
- Device auto-registration
- Error handling
"""

from unittest.mock import patch

import numpy as np


class TestPredictEndpoint:
    """Tests for POST /predict."""

    def test_predict_success_with_token(self, client, admin_headers, sample_predict_request):
        """Successful prediction with valid admin token."""
        with patch("src.api.main.predict") as mock_predict:
            mock_predict.return_value = {
                "predicted_label": 0,
                "confidence": 0.92,
                "model_version": "test-v1.0",
                "probabilities": {"0": 0.92, "1": 0.08},
                "features": {
                    "noise_std": 0.01,
                    "peak_count": 1,
                    "fwhm_mean": 1.2,
                    "area_total": 0.9,
                    "snr": 45.0,
                },
            }
            response = client.post("/predict", json=sample_predict_request, headers=admin_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["predicted_label"] == 0
        assert data["prediction_confidence"] == 0.92
        assert data["model_version"] == "test-v1.0"
        assert data["device_id"] == sample_predict_request["device_id"]
        assert "prediction_id" in data
        assert "features" in data

    def test_predict_success_with_api_key(self, client, api_key_headers, sample_predict_request):
        """Successful prediction with valid API key."""
        with patch("src.api.main.predict") as mock_predict:
            mock_predict.return_value = {
                "predicted_label": 1,
                "confidence": 0.78,
                "model_version": "test-v1.0",
                "probabilities": {"0": 0.22, "1": 0.78},
                "features": {
                    "noise_std": 0.05,
                    "peak_count": 2,
                    "fwhm_mean": 2.5,
                    "area_total": 1.8,
                    "snr": 12.0,
                },
            }
            response = client.post("/predict", json=sample_predict_request, headers=api_key_headers)

        assert response.status_code == 200
        assert response.json()["predicted_label"] == 1

    def test_predict_no_auth_returns_401(self, client, sample_predict_request):
        """Request without authentication returns 401."""
        response = client.post("/predict", json=sample_predict_request)
        assert response.status_code == 401

    def test_predict_read_only_scope_returns_403(
        self, client, user_headers, sample_predict_request
    ):
        """User with only 'read' scope cannot create predictions."""
        with patch("src.api.main.predict") as mock_predict:
            mock_predict.return_value = {
                "predicted_label": 0,
                "confidence": 0.9,
                "model_version": "test-v1.0",
                "probabilities": {"0": 0.9, "1": 0.1},
                "features": {},
            }
            response = client.post("/predict", json=sample_predict_request, headers=user_headers)

        assert response.status_code == 403
        assert (
            "write" in response.json()["detail"].lower()
            or "permission" in response.json()["detail"].lower()
        )

    def test_predict_readonly_api_key_returns_403(
        self, client, readonly_api_key_headers, sample_predict_request
    ):
        """API key with only 'read' scope cannot create predictions."""
        with patch("src.api.main.predict") as mock_predict:
            mock_predict.return_value = {
                "predicted_label": 0,
                "confidence": 0.9,
                "model_version": "test-v1.0",
                "probabilities": {"0": 0.9, "1": 0.1},
                "features": {},
            }
            response = client.post(
                "/predict", json=sample_predict_request, headers=readonly_api_key_headers
            )

        assert response.status_code == 403

    def test_predict_empty_device_id_auto_generates(self, client, admin_headers):
        """Empty device_id triggers UUID auto-generation."""
        request = {
            "device_id": "",
            "time_values": np.linspace(0, 10, 101).tolist(),
            "amplitude_values": np.exp(-((np.linspace(0, 10, 101) - 5) ** 2) / 0.5).tolist(),
        }
        with patch("src.api.main.predict") as mock_predict:
            mock_predict.return_value = {
                "predicted_label": 0,
                "confidence": 0.88,
                "model_version": "test-v1.0",
                "probabilities": {"0": 0.88, "1": 0.12},
                "features": {"noise_std": 0.01},
            }
            response = client.post("/predict", json=request, headers=admin_headers)

        assert response.status_code == 200
        # Auto-generated UUID should be non-empty
        assert len(response.json()["device_id"]) > 0

    def test_predict_invalid_device_id_rejected(self, client, admin_headers):
        """Device ID with invalid characters is rejected by Pydantic."""
        request = {
            "device_id": "'; DROP TABLE--",
            "time_values": [1, 2, 3],
            "amplitude_values": [0.1, 0.2, 0.3],
        }
        response = client.post("/predict", json=request, headers=admin_headers)
        assert response.status_code == 422  # Pydantic validation error

    def test_predict_missing_time_values(self, client, admin_headers):
        """Missing required field returns 422."""
        request = {
            "device_id": "test-001",
            "amplitude_values": [0.1, 0.2, 0.3],
        }
        response = client.post("/predict", json=request, headers=admin_headers)
        assert response.status_code == 422
