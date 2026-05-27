"""
Security dependencies for FastAPI endpoints.

Provides dependency injection for:
- OAuth2 authentication with scopes
- API key authentication
- User authorization
"""

import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import (
    APIKeyHeader,
    OAuth2PasswordBearer,
    SecurityScopes,
)

from .auth import UserInDB, decode_token, get_user

# ======================================
# OAuth2 Configuration
# ======================================

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/auth/token",
    scopes={
        "read": "Read access to predictions and metrics",
        "write": "Write access to create predictions and labels",
        "admin": "Administrative access to all resources",
    },
    auto_error=False,  # Don't auto-raise error if token missing (allow API key)
)

# ======================================
# API Key Configuration
# ======================================

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Mock API keys database (Replace with PostgreSQL in production)
# Format: {"key": {"name": "Service Name", "scopes": ["read", "write"], "disabled": False}}
fake_api_keys_db = {
    "dev-key-12345": {
        "name": "Development Key",
        "scopes": ["read", "write"],
        "disabled": False,
    },
    "monitoring-key-67890": {
        "name": "Monitoring Service",
        "scopes": ["read"],
        "disabled": False,
    },
}


# ======================================
# API Key Management
# ======================================


def generate_api_key() -> str:
    """
    Generate a secure random API key.

    Returns:
        32-character random API key
    """
    return secrets.token_urlsafe(32)


def validate_api_key(api_key: str) -> dict | None:
    """
    Validate API key and return key info.

    Args:
        api_key: API key to validate

    Returns:
        Dictionary with key info if valid, None otherwise
    """
    if api_key in fake_api_keys_db:
        key_info = fake_api_keys_db[api_key]
        if not key_info.get("disabled", False):
            return key_info
    return None


# ======================================
# OAuth2 Dependencies
# ======================================


async def get_current_user(
    security_scopes: SecurityScopes,
    token: Annotated[str, Depends(oauth2_scheme)],
) -> UserInDB:
    """
    Get current user from JWT token.

    Args:
        security_scopes: Required scopes for this endpoint
        token: JWT access token

    Returns:
        Current authenticated user

    Raises:
        HTTPException: If token is invalid or user lacks required scopes
    """
    if security_scopes.scopes:
        authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
    else:
        authenticate_value = "Bearer"

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": authenticate_value},
    )

    # Check if token provided
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": authenticate_value},
        )

    # Decode token
    token_data = decode_token(token)
    if token_data is None or token_data.username is None:
        raise credentials_exception

    # Get user from database
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception

    # Check if user is disabled
    if user.disabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
        )

    # Verify scopes
    for scope in security_scopes.scopes:
        if scope not in token_data.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions",
                headers={"WWW-Authenticate": authenticate_value},
            )

    return user


async def get_current_active_user(
    current_user: Annotated[UserInDB, Security(get_current_user, scopes=[])],
) -> UserInDB:
    """
    Get current active user (simplified dependency without scope checking).

    Args:
        current_user: User from get_current_user dependency

    Returns:
        Current active user

    Raises:
        HTTPException: If user is disabled
    """
    if current_user.disabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
        )
    return current_user


# ======================================
# API Key Dependencies
# ======================================


async def get_api_key(
    api_key: Annotated[str | None, Depends(api_key_header)],
) -> str | None:
    """
    Extract API key from header.

    Args:
        api_key: API key from X-API-Key header

    Returns:
        API key if present, None otherwise
    """
    return api_key


async def require_api_key(
    api_key: Annotated[str | None, Depends(get_api_key)],
) -> dict:
    """
    Require valid API key for endpoint access.

    Args:
        api_key: API key from header

    Returns:
        API key info dictionary

    Raises:
        HTTPException: If API key is missing or invalid
    """
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    key_info = validate_api_key(api_key)
    if key_info is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    return key_info


async def require_api_key_with_scope(
    api_key_info: Annotated[dict, Depends(require_api_key)],
    required_scope: str = "read",
) -> dict:
    """
    Require API key with specific scope.

    Args:
        api_key_info: API key info from require_api_key dependency
        required_scope: Required scope (read, write, admin)

    Returns:
        API key info dictionary

    Raises:
        HTTPException: If API key lacks required scope
    """
    if required_scope not in api_key_info.get("scopes", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"API key lacks required scope: {required_scope}",
        )

    return api_key_info


# ======================================
# Combined Authentication (OAuth2 OR API Key)
# ======================================


async def get_current_user_or_api_key(
    token: Annotated[str | None, Depends(oauth2_scheme)] = None,
    api_key: Annotated[str | None, Depends(get_api_key)] = None,
) -> dict:
    """
    Allow authentication via OAuth2 token OR API key.

    Args:
        token: JWT token (optional)
        api_key: API key (optional)

    Returns:
        Dictionary with auth info (user or api_key_info)

    Raises:
        HTTPException: If neither authentication method provided or both invalid
    """
    # Try OAuth2 token first
    if token:
        try:
            token_data = decode_token(token)
            if token_data and token_data.username:
                user = get_user(username=token_data.username)
                if user and not user.disabled:
                    return {
                        "type": "user",
                        "username": user.username,
                        "scopes": user.scopes,
                    }
        except Exception:
            pass

    # Try API key
    if api_key:
        key_info = validate_api_key(api_key)
        if key_info:
            return {
                "type": "api_key",
                "name": key_info["name"],
                "scopes": key_info["scopes"],
            }

    # Neither authentication method succeeded
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer, ApiKey"},
    )
