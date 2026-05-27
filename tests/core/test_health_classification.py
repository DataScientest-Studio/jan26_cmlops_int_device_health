"""
Tests for health classification rules: Gaussian and Lorentzian classification logic.
"""

from src.signal_processing.signal_models import (
    GaussianParameters,
    HealthClassificationRules,
    LorentzianParameters,
)


class TestGaussianClassification:
    """HealthClassificationRules.classify_gaussian."""

    def test_healthy_baseline(self):
        params = GaussianParameters(mu=50.0, sigma=2.5, height=2.8, noise_level=0.02)
        assert HealthClassificationRules.classify_gaussian(params) == 0

    def test_unhealthy_shifted_mu(self):
        params = GaussianParameters(mu=40.0, sigma=2.5, height=2.8, noise_level=0.02)
        assert HealthClassificationRules.classify_gaussian(params) == 1

    def test_unhealthy_high_noise(self):
        params = GaussianParameters(mu=50.0, sigma=2.5, height=2.8, noise_level=0.05)
        assert HealthClassificationRules.classify_gaussian(params) == 1

    def test_unhealthy_wrong_sigma(self):
        params = GaussianParameters(mu=50.0, sigma=5.0, height=2.8, noise_level=0.02)
        assert HealthClassificationRules.classify_gaussian(params) == 1


class TestLorentzianClassification:
    """HealthClassificationRules.classify_lorentzian — always unhealthy."""

    def test_always_unhealthy(self):
        params = LorentzianParameters(mu=50.0, gamma=3.5325, height=2.0, noise_level=0.02)
        assert HealthClassificationRules.classify_lorentzian(params) == 1

    def test_healthy_range_params_still_unhealthy(self):
        """Lorentzian shape is always classified as unhealthy regardless of params."""
        params = LorentzianParameters(mu=50.0, gamma=2.36, height=2.8, noise_level=0.01)
        assert HealthClassificationRules.classify_lorentzian(params) == 1
