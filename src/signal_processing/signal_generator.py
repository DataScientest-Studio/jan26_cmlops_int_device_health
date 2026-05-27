"""
Signal generation module for synthetic time-series data.

Generates Gaussian and Lorentzian peaks with configurable noise and drift scenarios
for MLOps pipeline testing and model training.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .signal_models import (
    GaussianParameters,
    HealthClassificationRules,
    LabeledSignal,
    LorentzianParameters,
    SignalData,
)


def generate_gaussian_peak(time: np.ndarray, mu: float, sigma: float, height: float) -> np.ndarray:
    """
    Generate a Gaussian peak.

    f(t) = H * exp(-(t-μ)²/(2σ²))

    Args:
        time: Time coordinate array
        mu: Peak center
        sigma: Standard deviation (width parameter)
        height: Peak amplitude

    Returns:
        Signal amplitude array
    """
    return height * np.exp(-((time - mu) ** 2) / (2 * sigma**2))


def generate_lorentzian_peak(
    time: np.ndarray, mu: float, gamma: float, height: float
) -> np.ndarray:
    """
    Generate a Lorentzian peak.

    f(t) = H * γ² / ((t-μ)² + γ²)

    Args:
        time: Time coordinate array
        mu: Peak center
        gamma: Half-width at half-maximum (HWHM)
        height: Peak amplitude

    Returns:
        Signal amplitude array
    """
    return height * (gamma**2) / ((time - mu) ** 2 + gamma**2)


def add_gaussian_noise(
    signal: np.ndarray, noise_level: float, seed: int | None = None
) -> np.ndarray:
    """
    Add Gaussian noise to signal.

    Args:
        signal: Clean signal array
        noise_level: Standard deviation of noise
        seed: Random seed for reproducibility

    Returns:
        Noisy signal array
    """
    if seed is not None:
        np.random.seed(seed)

    noise = np.random.normal(0, noise_level, size=signal.shape)
    return signal + noise


def inject_nans(
    signal: np.ndarray, nan_fraction: float = 0.0, seed: int | None = None
) -> np.ndarray:
    """
    Inject NaN values at random positions.

    Args:
        signal: Signal array
        nan_fraction: Fraction of points to replace with NaN (0.0-0.05)
        seed: Random seed for reproducibility

    Returns:
        Signal with NaNs injected
    """
    if nan_fraction <= 0:
        return signal

    if nan_fraction > 0.05:
        raise ValueError("NaN fraction must be ≤ 0.05 (5% limit)")

    if seed is not None:
        np.random.seed(seed)

    signal_copy = signal.copy()
    n_nans = int(len(signal) * nan_fraction)
    nan_indices = np.random.choice(len(signal), size=n_nans, replace=False)
    signal_copy[nan_indices] = np.nan

    return signal_copy


def create_time_array(
    n_points: int = 101,
    spacing: Literal["uniform", "variable"] = "uniform",
    seed: int | None = None,
) -> np.ndarray:
    """
    Create time coordinate array.

    Args:
        n_points: Number of points (≥51, recommended 101)
        spacing: 'uniform' for evenly spaced, 'variable' for irregular
        seed: Random seed for variable spacing

    Returns:
        Time array spanning [0, 100]
    """
    if n_points < 51:
        raise ValueError("Minimum 51 points required")

    if spacing == "uniform":
        return np.linspace(0, 100, n_points)

    # Variable spacing: random points + ensure boundaries
    if seed is not None:
        np.random.seed(seed)

    # Generate random interior points
    interior_points = np.random.uniform(0.5, 99.5, n_points - 2)
    # Add boundaries and sort
    time = np.concatenate([[0.0], interior_points, [100.0]])
    return np.sort(time)


def generate_signal(
    shape_type: Literal["gaussian", "lorentzian"],
    mu: float | None = None,
    width_param: float | None = None,  # sigma for Gaussian, gamma for Lorentzian
    height: float | None = None,
    noise_level: float | None = None,
    n_points: int = 101,
    spacing: Literal["uniform", "variable"] = "uniform",
    nan_fraction: float = 0.0,
    drift_scenario: Literal["baseline", "data_drift", "concept_drift"] | None = None,
    seed: int | None = None,
) -> LabeledSignal:
    """
    Generate a complete labeled signal with optional drift scenarios.

    Args:
        shape_type: 'gaussian' or 'lorentzian'
        mu: Peak center (default: random based on shape and scenario)
        width_param: sigma (Gaussian) or gamma (Lorentzian) (default: random based on shape)
        height: Peak amplitude (default: random based on shape)
        noise_level: Noise std dev (default: random based on shape)
        n_points: Number of time points
        spacing: 'uniform' or 'variable'
        nan_fraction: Fraction of NaN values (0.0-0.05)
        drift_scenario: Drift type for testing/demo purposes
        seed: Random seed for reproducibility

    Returns:
        LabeledSignal with signal data, label, and metadata

    Parameter Ranges for Clear Separability:
        Healthy Gaussian:
            - \u03bc: [48, 52] (well-centered)
            - \u03c3: [2.0, 3.0] (narrow)
            - height: [2.5, 3.0] (high)
            - noise: [0.01, 0.02] (clean)

        Unhealthy Lorentzian:
            - \u03bc: [42, 47] or [53, 58] (off-center)
            - \u03b3: [4.5, 6.0] (wide)
            - height: [1.0, 1.5] (low)
            - noise: [0.06, 0.10] (noisy)

    Drift Scenarios:
        - baseline: Normal distribution (healthy Gaussian or unhealthy Lorentzian)
        - data_drift: Shift \u03bc outside range, increase noise (sensor degradation)
        - concept_drift: Change Gaussian/Lorentzian ratio (process change)
    """
    if seed is not None:
        np.random.seed(seed)

    # Apply drift scenarios and shape-specific parameter ranges
    if drift_scenario == "data_drift":
        # Simulates sensor degradation: shifted center, high noise
        mu_range = (35.0, 42.0) if mu is None else (mu, mu)
        noise_range = (0.08, 0.12)
        if shape_type == "gaussian":
            sigma_range = (3.5, 5.0)
            height_range = (1.5, 2.0)
        else:
            gamma_range = (5.5, 7.0)
            height_range = (0.8, 1.2)

    elif drift_scenario == "concept_drift":
        # Process change: parameters shift toward boundary conditions
        if shape_type == "gaussian":
            mu_range = (46.0, 54.0) if mu is None else (mu, mu)
            sigma_range = (2.5, 4.0)
            height_range = (2.0, 2.8)
            noise_range = (0.02, 0.04)
        else:
            mu_range = (45.0, 55.0) if mu is None else (mu, mu)
            gamma_range = (4.0, 5.5)
            height_range = (1.2, 1.8)
            noise_range = (0.04, 0.08)

    else:  # baseline - clear separation
        if shape_type == "gaussian":
            # Healthy Gaussian: centered, narrow, high, clean
            mu_range = (48.0, 52.0) if mu is None else (mu, mu)
            sigma_range = (2.0, 3.0)
            height_range = (2.5, 3.0)
            noise_range = (0.01, 0.02)
        else:
            # Unhealthy Lorentzian: off-center, wide, low, noisy
            # Randomly choose left or right offset
            if np.random.random() < 0.5:
                mu_range = (42.0, 47.0) if mu is None else (mu, mu)  # Left offset
            else:
                mu_range = (53.0, 58.0) if mu is None else (mu, mu)  # Right offset
            gamma_range = (4.5, 6.0)
            height_range = (1.0, 1.5)
            noise_range = (0.06, 0.10)

    # Sample parameters
    mu_final = np.random.uniform(*mu_range) if mu is None else mu
    noise_final = np.random.uniform(*noise_range) if noise_level is None else noise_level

    # Create time array
    time = create_time_array(n_points, spacing, seed=seed)

    # Generate peak
    if shape_type == "gaussian":
        sigma = np.random.uniform(*sigma_range) if width_param is None else width_param
        height_final = np.random.uniform(*height_range) if height is None else height
        params = GaussianParameters(
            mu=mu_final, sigma=sigma, height=height_final, noise_level=noise_final
        )
        clean_signal = generate_gaussian_peak(time, params.mu, params.sigma, params.height)
        label = HealthClassificationRules.classify_gaussian(params)
        metadata = {
            "shape_type": "gaussian",
            "mu": params.mu,
            "sigma": params.sigma,
            "height": params.height,
            "noise_level": params.noise_level,
            "drift_scenario": drift_scenario,
        }

    else:  # lorentzian
        gamma = np.random.uniform(*gamma_range) if width_param is None else width_param
        height_final = np.random.uniform(*height_range) if height is None else height

        params = LorentzianParameters(
            mu=mu_final, gamma=gamma, height=height_final, noise_level=noise_final
        )
        clean_signal = generate_lorentzian_peak(time, params.mu, params.gamma, params.height)
        label = HealthClassificationRules.classify_lorentzian(params)
        metadata = {
            "shape_type": "lorentzian",
            "mu": params.mu,
            "gamma": params.gamma,
            "height": params.height,
            "noise_level": params.noise_level,
            "drift_scenario": drift_scenario,
        }

    # Add noise
    noisy_signal = add_gaussian_noise(clean_signal, noise_final, seed=seed)

    # Inject NaNs
    final_signal = inject_nans(noisy_signal, nan_fraction, seed=seed)

    # Create signal data
    signal_data = SignalData(
        time=time.tolist(), amplitude=final_signal.tolist(), shape_type=shape_type
    )

    return LabeledSignal(signal=signal_data, label=label, metadata=metadata)


def generate_dataset(
    n_samples: int = 100,
    gaussian_fraction: float = 0.5,
    drift_scenario: Literal["baseline", "data_drift", "concept_drift"] | None = None,
    seed: int | None = None,
) -> list[LabeledSignal]:
    """
    Generate a complete dataset of signals.

    Args:
        n_samples: Total number of signals to generate
        gaussian_fraction: Fraction of Gaussian peaks (rest are Lorentzian)
        drift_scenario: Optional drift scenario to apply
        seed: Random seed for reproducibility

    Returns:
        List of LabeledSignal objects
    """
    if seed is not None:
        np.random.seed(seed)

    n_gaussian = int(n_samples * gaussian_fraction)
    n_lorentzian = n_samples - n_gaussian

    dataset = []

    # Generate Gaussian signals
    for i in range(n_gaussian):
        signal = generate_signal(
            shape_type="gaussian",
            drift_scenario=drift_scenario,
            seed=seed + i if seed is not None else None,
        )
        dataset.append(signal)

    # Generate Lorentzian signals
    for i in range(n_lorentzian):
        signal = generate_signal(
            shape_type="lorentzian",
            drift_scenario=drift_scenario,
            seed=seed + n_gaussian + i if seed is not None else None,
        )
        dataset.append(signal)

    # Shuffle dataset
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(dataset)

    return dataset
