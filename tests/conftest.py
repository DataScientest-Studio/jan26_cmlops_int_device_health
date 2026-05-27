"""
Root conftest.py — shared fixtures for the entire test suite.

Provides:
- Environment bootstrapping (.env.secrets)
- Common time arrays and signal parameters
- Basic signal fixtures used across all test categories
"""

import logging
import os
from pathlib import Path

import numpy as np
import pytest

from src.signal_processing.signal_models import (
    GaussianParameters,
    LorentzianParameters,
)

# ---------------------------------------------------------------------------
# Load .env.secrets so live tests pick up correct passwords.
# Do NOT load .env.local — it contains Docker-internal URIs unreachable
# from the host.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Force UTF-8 mode on Windows to prevent UnicodeEncodeError with ✓ in dependencies.py
os.environ.setdefault("PYTHONUTF8", "1")


def _load_env_file(path: Path) -> None:
    """Parse a simple KEY=VALUE env file and export to ``os.environ``."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if "#" in value:
            stripped = value.split("#", 1)[0]
            value = stripped.strip().strip("'\"")
        else:
            value = value.strip().strip("'\"")
        if key not in os.environ:
            os.environ[key] = value


_load_env_file(_PROJECT_ROOT / ".env.secrets")

# Suppress logging errors from background daemon threads during teardown.
logging.raiseExceptions = False


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def project_root() -> Path:
    """Path to project root directory."""
    return _PROJECT_ROOT


@pytest.fixture
def sample_time_array() -> np.ndarray:
    """Standard time array (101 points, evenly spaced 0–100)."""
    return np.linspace(0, 100, 101)


@pytest.fixture
def sample_time_minimal() -> np.ndarray:
    """Minimal time array (51 points — minimum acceptable)."""
    return np.linspace(0, 100, 51)


@pytest.fixture
def gaussian_params() -> GaussianParameters:
    """Standard Gaussian parameters (healthy baseline)."""
    return GaussianParameters(mu=50.0, sigma=3.0, height=2.0, noise_level=0.02)


@pytest.fixture
def lorentzian_params() -> LorentzianParameters:
    """Standard Lorentzian parameters (gamma ≈ 1.1775 * sigma=3.0)."""
    return LorentzianParameters(mu=50.0, gamma=3.5325, height=2.0, noise_level=0.02)


@pytest.fixture
def bootstrap_model_path(project_root) -> Path:
    """Path to the bootstrap model artifact."""
    return project_root / "models" / "bootstrap_model.pkl"
