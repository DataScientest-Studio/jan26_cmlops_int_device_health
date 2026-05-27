"""
Shared fixtures for security tests.

Re-exports fixtures from the API conftest since security tests
exercise the same FastAPI TestClient and auth helpers.
"""

from tests.api.conftest import (  # noqa: F401
    admin_headers,
    admin_token,
    api_key_headers,
    client,
    mock_model_artifact,
    readonly_api_key_headers,
    sample_predict_request,
    service_headers,
    service_token,
    test_db,
    test_settings,
    user_headers,
    user_token,
)
