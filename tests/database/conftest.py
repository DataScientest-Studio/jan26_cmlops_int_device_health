"""
Shared fixtures for database tests.

Provides:
- Clean SQLite database per test
- Sample device and prediction data
"""

import pytest

from src.database import Database, generate_device_id


@pytest.fixture
def db(tmp_path):
    """Fresh SQLite database for each test."""
    db_path = tmp_path / "test_db.sqlite"
    database = Database(db_path=db_path)
    yield database
    database.close()


@pytest.fixture
def device_id(db):
    """Register a test device and return its ID."""
    did = generate_device_id()  # UUID format required by DB CHECK constraint
    db.register_device(
        device_id=did,
        device_name="Test Device Alpha",
        device_type="Sensor-A",
        location="Lab-1",
        status="active",
    )
    return did


@pytest.fixture
def sample_signal():
    """Sample time/amplitude arrays for predictions."""
    import numpy as np

    t = np.linspace(0, 10, 101).tolist()
    amp = np.exp(-((np.array(t) - 5) ** 2) / (2 * 0.5**2)).tolist()
    return t, amp


@pytest.fixture
def stored_prediction(db, device_id, sample_signal):
    """Store a prediction and return its ID."""
    t, amp = sample_signal
    prediction_id = db.store_prediction(
        device_id=device_id,
        time_values=t,
        amplitude_values=amp,
        predicted_label=0,
        model_version="test-v1.0",
        features={"noise_std": 0.01, "peak_count": 1, "snr": 45.0},
        prediction_confidence=0.92,
    )
    return prediction_id
