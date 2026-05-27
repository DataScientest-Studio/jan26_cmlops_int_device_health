"""
Tests for authentication endpoints and JWT token handling.

Validates:
- POST /auth/token — login with username/password
- POST /auth/refresh — token refresh mechanism
- GET /auth/users/me — current user info
- Token expiration and invalid token handling
- Password verification
"""

from datetime import timedelta

from src.api.auth import (
    authenticate_user,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_password_hash,
    verify_password,
)


class TestAuthEndpoints:
    """Tests for /auth/* endpoints."""

    def test_login_success(self, client):
        """Successful login returns token."""
        response = client.post(
            "/auth/token",
            data={"username": "admin", "password": "secret"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data
        assert "refresh_token" in data

    def test_login_wrong_password(self, client):
        """Wrong password returns 401."""
        response = client.post(
            "/auth/token",
            data={"username": "admin", "password": "wrongpassword"},
        )
        assert response.status_code == 401

    def test_login_nonexistent_user(self, client):
        """Non-existent user returns 401."""
        response = client.post(
            "/auth/token",
            data={"username": "nobody", "password": "secret"},
        )
        assert response.status_code == 401

    def test_refresh_token_success(self, client):
        """Valid refresh token returns new access token."""
        # First login to get refresh token
        login_resp = client.post(
            "/auth/token",
            data={"username": "admin", "password": "secret"},
        )
        refresh_token = login_resp.json()["refresh_token"]

        # Use refresh token
        response = client.post(
            "/auth/refresh",
            json={"refresh_token": refresh_token},
        )
        assert response.status_code == 200
        assert "access_token" in response.json()

    def test_refresh_token_invalid(self, client):
        """Invalid refresh token returns 401."""
        response = client.post(
            "/auth/refresh",
            json={"refresh_token": "invalid-token-xyz"},
        )
        assert response.status_code == 401

    def test_users_me_authenticated(self, client, admin_headers):
        """GET /auth/users/me returns user info for authenticated user."""
        response = client.get("/auth/users/me", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "admin"
        assert "email" in data
        assert "scopes" in data
        assert "admin" in data["scopes"]

    def test_users_me_no_auth(self, client):
        """GET /auth/users/me without auth returns 401."""
        response = client.get("/auth/users/me")
        assert response.status_code == 401


class TestAuthFunctions:
    """Unit tests for auth module functions."""

    def test_verify_password_correct(self):
        """Correct password verification succeeds."""
        hashed = get_password_hash("testpassword")
        assert verify_password("testpassword", hashed) is True

    def test_verify_password_incorrect(self):
        """Incorrect password verification fails."""
        hashed = get_password_hash("testpassword")
        assert verify_password("wrongpassword", hashed) is False

    def test_authenticate_user_valid(self):
        """Valid credentials return user object."""
        user = authenticate_user("admin", "secret")
        assert user is not None
        assert user.username == "admin"

    def test_authenticate_user_invalid(self):
        """Invalid credentials return None."""
        assert authenticate_user("admin", "wrong") is None
        assert authenticate_user("nobody", "secret") is None

    def test_create_access_token(self):
        """Access token can be created and decoded."""
        token = create_access_token(data={"sub": "admin", "scopes": ["read", "write"]})
        decoded = decode_token(token)
        assert decoded is not None
        assert decoded.username == "admin"
        assert "read" in decoded.scopes
        assert "write" in decoded.scopes

    def test_create_refresh_token(self):
        """Refresh token encodes username and type."""
        token = create_refresh_token("admin")
        from jose import jwt

        from src.api.auth import ALGORITHM, SECRET_KEY

        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["sub"] == "admin"
        assert payload["type"] == "refresh"

    def test_decode_token_expired(self):
        """Expired token returns None."""
        token = create_access_token(
            data={"sub": "admin", "scopes": []},
            expires_delta=timedelta(seconds=-1),
        )
        assert decode_token(token) is None

    def test_decode_token_invalid(self):
        """Malformed token returns None."""
        assert decode_token("not.a.valid.token") is None
