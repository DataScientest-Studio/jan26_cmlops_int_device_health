"""
Core test conftest — signal-specific fixtures.

Provides pre-built signals, datasets, and noisy/NaN variants.
"""

import numpy as np
import pytest

from src.signal_processing.signal_generator import (
    generate_gaussian_peak,
    generate_lorentzian_peak,
    generate_signal,
)
from src.signal_processing.signal_models import SignalData


@pytest.fixture
def gaussian_signal(sample_time_array, gaussian_params) -> SignalData:
    """Clean Gaussian signal (no noise)."""
    amplitude = generate_gaussian_peak(
        sample_time_array, gaussian_params.mu, gaussian_params.sigma, gaussian_params.height
    )
    return SignalData(
        time=sample_time_array.tolist(),
        amplitude=amplitude.tolist(),
        shape_type="gaussian",
    )


@pytest.fixture
def lorentzian_signal(sample_time_array, lorentzian_params) -> SignalData:
    """Clean Lorentzian signal (no noise)."""
    amplitude = generate_lorentzian_peak(
        sample_time_array, lorentzian_params.mu, lorentzian_params.gamma, lorentzian_params.height
    )
    return SignalData(
        time=sample_time_array.tolist(),
        amplitude=amplitude.tolist(),
        shape_type="lorentzian",
    )


@pytest.fixture
def noisy_signal() -> SignalData:
    """Gaussian signal with noise_level=0.05."""
    return generate_signal(
        shape_type="gaussian",
        mu=50.0,
        width_param=3.0,
        height=2.0,
        noise_level=0.05,
        seed=42,
    ).signal


@pytest.fixture
def signal_with_nans(sample_time_array, gaussian_params) -> SignalData:
    """Signal with ~3% NaN values injected."""
    amplitude = generate_gaussian_peak(
        sample_time_array, gaussian_params.mu, gaussian_params.sigma, gaussian_params.height
    )
    amplitude_copy = amplitude.copy()
    rng = np.random.default_rng(42)
    nan_indices = rng.choice(len(amplitude), size=3, replace=False)
    amplitude_copy[nan_indices] = np.nan
    return SignalData(
        time=sample_time_array.tolist(),
        amplitude=amplitude_copy.tolist(),
        shape_type="gaussian",
    )


@pytest.fixture
def sample_dataset():
    """Mixed dataset: 5 Gaussian + 5 Lorentzian signals."""
    dataset = []
    for i in range(5):
        dataset.append(generate_signal(shape_type="gaussian", seed=100 + i))
    for i in range(5):
        dataset.append(generate_signal(shape_type="lorentzian", seed=200 + i))
    return dataset
