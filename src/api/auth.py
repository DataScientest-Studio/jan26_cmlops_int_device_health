"""
Authentication module for OAuth2 with JWT tokens.

Provides:
- User authentication with username/password
- JWT token generation and validation
- Password hashing and verification
- Token refresh mechanism
"""

import logging
import os
from datetime import UTC, datetime, timedelta

from dotenv import load_dotenv
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# Load .env.secrets so the module works when run outside Docker
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env.secrets"), override=False)

_logger = logging.getLogger(__name__)

# ======================================
# Configuration
# ======================================

# JWT Configuration
_raw_secret = os.environ.get("API_SECRET_KEY")
if not _raw_secret:
    _logger.warning(
        "API_SECRET_KEY is not set — using insecure placeholder. "
        "Set API_SECRET_KEY in .env.secrets before deploying."
    )
    _raw_secret = "your-secret-key-change-in-production"  # noqa: S105 — intentional placeholder
SECRET_KEY: str = _raw_secret
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing context
# Note: bcrypt truncates passwords to 72 bytes
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
)


# ======================================
# Models
# ======================================


class Token(BaseModel):
    """OAuth2 token response."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: str | None = None


class TokenData(BaseModel):
    """Data stored in JWT token."""

    username: str | None = None
    scopes: list[str] = []
    exp: datetime | None = None


class User(BaseModel):
    """User model."""

    username: str
    email: str | None = None
    full_name: str | None = None
    disabled: bool = False
    scopes: list[str] = []


class UserInDB(User):
    """User model with hashed password."""

    hashed_password: str


# ======================================
# Mock User Database (Replace with real DB)
# ======================================

# In production, store users in PostgreSQL with proper schema
fake_users_db = {
    "admin": {
        "username": "admin",
        "full_name": "Admin User",
        "email": "admin@mlops-device-health.local",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
        "disabled": False,
        "scopes": ["read", "write", "admin"],
    },
    "user": {
        "username": "user",
        "full_name": "Regular User",
        "email": "user@mlops-device-health.local",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
        "disabled": False,
        "scopes": ["read"],
    },
    "service": {
        "username": "service",
        "full_name": "Service Account",
        "email": "service@mlops-device-health.local",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
        "disabled": False,
        "scopes": ["read", "write"],
    },
}


# ======================================
# Password Hashing
# ======================================


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.

    Note: bcrypt truncates passwords to 72 bytes.

    Args:
        plain_password: Plain text password
        hashed_password: Hashed password from database

    Returns:
        True if password matches, False otherwise
    """
    # Ensure password doesn't exceed bcrypt's 72-byte limit
    if len(plain_password.encode("utf-8")) > 72:
        plain_password = plain_password.encode("utf-8")[:72].decode("utf-8", errors="ignore")
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Hash a password for storing in database.

    Note: bcrypt truncates passwords to 72 bytes.

    Args:
        password: Plain text password

    Returns:
        Hashed password
    """
    # Ensure password doesn't exceed bcrypt's 72-byte limit
    if len(password.encode("utf-8")) > 72:
        password = password.encode("utf-8")[:72].decode("utf-8", errors="ignore")
    return pwd_context.hash(password)


# ======================================
# User Management
# ======================================


def get_user(username: str) -> UserInDB | None:
    """
    Get user from database by username.

    Args:
        username: Username to look up

    Returns:
        User object if found, None otherwise
    """
    if username in fake_users_db:
        user_dict = fake_users_db[username]
        return UserInDB(**user_dict)
    return None


def authenticate_user(username: str, password: str) -> UserInDB | None:
    """
    Authenticate user with username and password.

    Args:
        username: Username
        password: Plain text password

    Returns:
        User object if authentication successful, None otherwise
    """
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


# ======================================
# JWT Token Management
# ======================================


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """
    Create JWT access token.

    Args:
        data: Data to encode in token (username, scopes, etc.)
        expires_delta: Token expiration time (default: 30 minutes)

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(username: str) -> str:
    """
    Create JWT refresh token for token renewal.

    Args:
        username: Username to encode in token

    Returns:
        Encoded JWT refresh token (valid for 7 days)
    """
    expire = datetime.now(UTC) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode = {"sub": username, "exp": expire, "type": "refresh"}
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> TokenData | None:
    """
    Decode and validate JWT token.

    Args:
        token: JWT token to decode

    Returns:
        TokenData if valid, None if invalid or expired
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None

        scopes = payload.get("scopes", [])
        exp = payload.get("exp")
        if exp:
            exp = datetime.fromtimestamp(exp, tz=UTC)

        return TokenData(username=username, scopes=scopes, exp=exp)
    except JWTError:
        return None


def refresh_access_token(refresh_token: str) -> Token | None:
    """
    Generate new access token using refresh token.

    Args:
        refresh_token: Valid refresh token

    Returns:
        New Token object with access token, None if refresh token invalid
    """
    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        token_type: str = payload.get("type")

        if username is None or token_type != "refresh":
            return None

        # Get user to include current scopes
        user = get_user(username)
        if user is None or user.disabled:
            return None

        # Create new access token
        access_token = create_access_token(data={"sub": user.username, "scopes": user.scopes})

        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )
    except JWTError:
        return None


# ======================================
# Helper Functions
# ======================================


def create_token_response(user: UserInDB, include_refresh: bool = True) -> Token:
    """
    Create complete token response for authenticated user.

    Args:
        user: Authenticated user
        include_refresh: Whether to include refresh token

    Returns:
        Token response with access token and optional refresh token
    """
    access_token = create_access_token(data={"sub": user.username, "scopes": user.scopes})

    refresh_token = None
    if include_refresh:
        refresh_token = create_refresh_token(user.username)

    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        refresh_token=refresh_token,
    )
