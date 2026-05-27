"""
Pydantic models for signal data validation and type safety.

This module defines the data models used throughout the signal processing pipeline,
ensuring type safety and automatic validation.
"""

from __future__ import annotations

from typing import List, Literal, Tuple

import numpy as np
from pydantic import BaseModel, Field, field_validator


class GaussianParameters(BaseModel):
    """Parameters for Gaussian peak generation."""

    mu: float = Field(..., ge=35.0, le=60.0, description="Peak center position (time coordinate)")
    sigma: float = Field(..., ge=2.0, le=6.0, description="Standard deviation (width parameter)")
    height: float = Field(..., ge=1.0, le=3.5, description="Peak amplitude")
    noise_level: float = Field(
        ..., ge=0.0, le=0.15, description="Gaussian noise standard deviation"
    )

    model_config = {"frozen": True}  # Immutable for reproducibility


class LorentzianParameters(BaseModel):
    """Parameters for Lorentzian peak generation."""

    mu: float = Field(..., ge=35.0, le=60.0, description="Peak center position (time coordinate)")
    gamma: float = Field(..., ge=2.36, le=7.5, description="Half-width at half-maximum (HWHM)")
    height: float = Field(..., ge=0.8, le=3.5, description="Peak amplitude")
    noise_level: float = Field(
        ..., ge=0.0, le=0.15, description="Gaussian noise standard deviation"
    )

    model_config = {"frozen": True}

    @classmethod
    def from_gaussian_sigma(
        cls, mu: float, sigma: float, height: float, noise_level: float
    ) -> LorentzianParameters:
        """
        Create Lorentzian parameters with equivalent width to Gaussian.

        Uses the conversion: γ = 1.1775σ to match FWHM.
        """
        gamma = 1.1775 * sigma
        return cls(mu=mu, gamma=gamma, height=height, noise_level=noise_level)


class SignalData(BaseModel):
    """
    Validated signal data container.

    Ensures signal meets minimal quality requirements before processing.
    """

    time: List[float] = Field(..., min_length=51, description="Time coordinate array")
    amplitude: List[float] = Field(..., min_length=51, description="Signal amplitude array")
    shape_type: Literal["gaussian", "lorentzian"] = Field(..., description="Peak distribution type")

    @field_validator("time", "amplitude")
    @classmethod
    def check_equal_length(cls, v: List[float], info) -> List[float]:
        """Ensure time and amplitude have same length."""
        if (
            info.data
            and "time" in info.data
            and "amplitude" in info.data
            and len(info.data["time"]) != len(info.data["amplitude"])
        ):
            raise ValueError("Time and amplitude arrays must have equal length")
        return v

    @field_validator("time")
    @classmethod
    def check_time_range(cls, v: List[float]) -> List[float]:
        """Ensure time array covers [0, 100] range."""
        if min(v) > 0 or max(v) < 100:
            raise ValueError("Time array must span [0, 100]")
        return v

    @field_validator("amplitude")
    @classmethod
    def check_nan_limit(cls, v: List[float]) -> List[float]:
        """Ensure NaN values don't exceed 5% of total points."""
        nan_count = sum(1 for x in v if x is None or (isinstance(x, float) and np.isnan(x)))
        if nan_count / len(v) > 0.05:
            raise ValueError(f"NaN values ({nan_count}/{len(v)}) exceed 5% limit")
        return v

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert to numpy arrays, handling NaNs."""
        return np.array(self.time), np.array(self.amplitude)

    model_config = {"arbitrary_types_allowed": True}


class LabeledSignal(BaseModel):
    """Signal with ground truth health label (-1 = unlabeled, used for semi-supervised learning)."""

    signal: SignalData
    label: Literal[-1, 0, 1] = Field(..., description="-1=Unlabeled, 0=Healthy, 1=Unhealthy")
    metadata: dict = Field(
        default_factory=dict, description="Additional metadata (parameters, generation info)"
    )

    model_config = {"arbitrary_types_allowed": True}


class HealthClassificationRules(BaseModel):
    """
    Ground truth labeling rules for signal health classification.

    Designed for clear feature separability in MLOps demonstrations.

    Healthy (0) - ALL conditions must be met:
    - Peak shape: Gaussian
    - Well-centered: μ ∈ [48, 52]
    - High peak: height ∈ [2.5, 3.0]
    - Narrow width: σ ∈ [2.0, 3.0]
    - Clean signal: noise < 0.03

    Unhealthy (1) - ANY condition triggers:
    - Peak shape: Lorentzian (different physical process)
    - Off-center: μ outside [48, 52]
    - Low peak: height < 2.5
    - Wide peak: σ > 3.0
    - Noisy signal: noise ≥ 0.03
    """

    mu_healthy_range: Tuple[float, float] = (48.0, 52.0)
    sigma_healthy_range: Tuple[float, float] = (2.0, 3.0)
    height_healthy_range: Tuple[float, float] = (2.5, 3.0)
    noise_healthy_threshold: float = 0.03

    @staticmethod
    def classify_gaussian(params: GaussianParameters) -> int:
        """Classify Gaussian signal as healthy (0) or unhealthy (1)."""
        rules = HealthClassificationRules()

        # Check all conditions
        is_healthy = (
            rules.mu_healthy_range[0] <= params.mu <= rules.mu_healthy_range[1]
            and rules.sigma_healthy_range[0] <= params.sigma <= rules.sigma_healthy_range[1]
            and rules.height_healthy_range[0] <= params.height <= rules.height_healthy_range[1]
            and params.noise_level < rules.noise_healthy_threshold
        )

        return 0 if is_healthy else 1

    @staticmethod
    def classify_lorentzian(params: LorentzianParameters) -> int:
        """Classify Lorentzian signal (always unhealthy by definition)."""
        return 1  # Lorentzian indicates mechanical/electronic anomaly
