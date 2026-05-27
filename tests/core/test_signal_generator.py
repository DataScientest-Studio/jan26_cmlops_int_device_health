"""
Tests for signal generation: peak shapes, noise, NaN injection, drift, datasets.
"""

import numpy as np
import pytest

from src.signal_processing.signal_generator import (
    add_gaussian_noise,
    create_time_array,
    generate_dataset,
    generate_gaussian_peak,
    generate_lorentzian_peak,
    generate_signal,
    inject_nans,
)
from src.signal_processing.signal_models import LorentzianParameters


class TestGaussianPeakGeneration:
    """Gaussian peak shape, height, width, and symmetry."""

    def test_peak_at_center(self, sample_time_array):
        amplitude = generate_gaussian_peak(sample_time_array, 50.0, 3.0, 2.0)
        assert amplitude[50] == pytest.approx(2.0, rel=0.01)

    def test_symmetry(self, sample_time_array):
        amplitude = generate_gaussian_peak(sample_time_array, 50.0, 3.0, 2.0)
        np.testing.assert_allclose(amplitude[:50], amplitude[51:][::-1], rtol=0.01)

    @pytest.mark.parametrize("height", [1.0, 2.0, 3.0])
    def test_achieves_specified_height(self, sample_time_array, height):
        amplitude = generate_gaussian_peak(sample_time_array, 50.0, 3.0, height)
        assert amplitude.max() == pytest.approx(height, rel=0.01)

    def test_wider_peak_decays_slower(self, sample_time_array):
        narrow = generate_gaussian_peak(sample_time_array, 50.0, 2.0, 2.0)
        wide = generate_gaussian_peak(sample_time_array, 50.0, 5.0, 2.0)
        assert wide[60] > narrow[60]


class TestLorentzianPeakGeneration:
    """Lorentzian peak shape, height, and heavier tails."""

    def test_peak_at_center(self, sample_time_array):
        amplitude = generate_lorentzian_peak(sample_time_array, 50.0, 3.5, 2.0)
        assert amplitude[50] == pytest.approx(2.0, rel=0.01)

    def test_symmetry(self, sample_time_array):
        amplitude = generate_lorentzian_peak(sample_time_array, 50.0, 3.5, 2.0)
        np.testing.assert_allclose(amplitude[:50], amplitude[51:][::-1], rtol=0.01)

    def test_heavier_tails_than_gaussian(self, sample_time_array):
        sigma = 3.0
        gamma = 1.1775 * sigma
        gaussian = generate_gaussian_peak(sample_time_array, 50.0, sigma, 2.0)
        lorentzian = generate_lorentzian_peak(sample_time_array, 50.0, gamma, 2.0)
        assert lorentzian[80] > gaussian[80]


class TestSigmaGammaConversion:
    """LorentzianParameters.from_gaussian_sigma conversion."""

    def test_gamma_equals_1_1775_times_sigma(self):
        params = LorentzianParameters.from_gaussian_sigma(
            mu=50.0, sigma=3.0, height=2.0, noise_level=0.02
        )
        assert params.gamma == pytest.approx(1.1775 * 3.0, rel=0.001)

    @pytest.mark.parametrize("sigma", [2.0, 3.0, 4.0, 5.0])
    def test_equivalent_fwhm_crossings(self, sample_time_array, sigma):
        gamma = 1.1775 * sigma
        gaussian = generate_gaussian_peak(sample_time_array, 50.0, sigma, 2.0)
        lorentzian = generate_lorentzian_peak(sample_time_array, 50.0, gamma, 2.0)
        g_cross = np.where(np.diff(np.sign(gaussian - 1.0)))[0]
        l_cross = np.where(np.diff(np.sign(lorentzian - 1.0)))[0]
        assert len(g_cross) == 2
        assert len(l_cross) == 2


class TestNoiseInjection:
    """Gaussian noise addition: effect, reproducibility, level scaling."""

    def test_noise_modifies_signal(self, sample_time_array):
        clean = np.ones(len(sample_time_array))
        noisy = add_gaussian_noise(clean, noise_level=0.1, seed=42)
        assert not np.array_equal(clean, noisy)

    def test_reproducibility_with_seed(self, sample_time_array):
        clean = np.ones(len(sample_time_array))
        a = add_gaussian_noise(clean, noise_level=0.1, seed=42)
        b = add_gaussian_noise(clean, noise_level=0.1, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_higher_noise_more_variation(self, sample_time_array):
        clean = np.ones(len(sample_time_array))
        low = add_gaussian_noise(clean, noise_level=0.01, seed=42)
        high = add_gaussian_noise(clean, noise_level=0.1, seed=43)
        assert np.std(low - clean) < np.std(high - clean)


class TestNaNInjection:
    """NaN injection: count, limit, reproducibility."""

    def test_correct_nan_count(self, sample_time_array):
        signal = np.ones(len(sample_time_array))
        result = inject_nans(signal, nan_fraction=0.03, seed=42)
        expected = int(len(signal) * 0.03)
        assert np.isnan(result).sum() == expected

    def test_limit_exceeded_raises(self, sample_time_array):
        signal = np.ones(len(sample_time_array))
        with pytest.raises(ValueError, match="NaN fraction must be"):
            inject_nans(signal, nan_fraction=0.1, seed=42)

    def test_reproducibility(self, sample_time_array):
        signal = np.ones(len(sample_time_array))
        a = inject_nans(signal, nan_fraction=0.03, seed=42)
        b = inject_nans(signal, nan_fraction=0.03, seed=42)
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))

    def test_zero_fraction_no_nans(self, sample_time_array):
        signal = np.ones(len(sample_time_array))
        result = inject_nans(signal, nan_fraction=0.0, seed=42)
        assert np.isnan(result).sum() == 0


class TestTimeArrayCreation:
    """Time coordinate array creation and spacing."""

    def test_uniform_spacing(self):
        time = create_time_array(n_points=101, spacing="uniform")
        assert len(time) == 101
        assert time[0] == 0.0
        assert time[-1] == 100.0
        np.testing.assert_allclose(np.diff(time), np.diff(time)[0])

    def test_variable_spacing(self):
        time = create_time_array(n_points=101, spacing="variable", seed=42)
        assert len(time) == 101
        assert not np.allclose(np.diff(time), np.diff(time)[0])

    def test_minimum_points_enforced(self):
        with pytest.raises(ValueError, match="Minimum 51 points"):
            create_time_array(n_points=50)


class TestSignalGeneration:
    """Complete signal generation with shape, seed, metadata."""

    def test_gaussian_shape_type(self):
        s = generate_signal(shape_type="gaussian", seed=42)
        assert s.signal.shape_type == "gaussian"
        assert len(s.signal.time) == 101
        assert s.label in [0, 1]

    def test_lorentzian_always_unhealthy(self):
        s = generate_signal(shape_type="lorentzian", seed=42)
        assert s.signal.shape_type == "lorentzian"
        assert s.label == 1

    def test_reproducibility(self):
        a = generate_signal(shape_type="gaussian", seed=42)
        b = generate_signal(shape_type="gaussian", seed=42)
        np.testing.assert_array_equal(a.signal.time, b.signal.time)
        np.testing.assert_array_equal(a.signal.amplitude, b.signal.amplitude)

    def test_metadata_populated(self):
        s = generate_signal(
            shape_type="gaussian",
            mu=50.0,
            width_param=3.0,
            height=2.0,
            noise_level=0.02,
            seed=42,
        )
        assert s.metadata["shape_type"] == "gaussian"
        assert s.metadata["mu"] == 50.0
        assert s.metadata["sigma"] == 3.0


class TestDriftScenarios:
    """Drift scenario generation: baseline, data_drift, concept_drift."""

    @pytest.mark.parametrize("scenario", ["baseline", "data_drift", "concept_drift"])
    def test_drift_scenario_metadata(self, scenario):
        s = generate_signal(shape_type="gaussian", drift_scenario=scenario, seed=42)
        assert s.metadata["drift_scenario"] == scenario


class TestDatasetGeneration:
    """Batch dataset generation: size, fraction, reproducibility."""

    def test_correct_size(self):
        assert len(generate_dataset(n_samples=20, seed=42)) == 20

    def test_gaussian_fraction(self):
        ds = generate_dataset(n_samples=100, gaussian_fraction=0.7, seed=42)
        n_g = sum(1 for s in ds if s.signal.shape_type == "gaussian")
        assert n_g == 70

    def test_reproducibility(self):
        a = generate_dataset(n_samples=10, seed=42)
        b = generate_dataset(n_samples=10, seed=42)
        for s1, s2 in zip(a, b, strict=False):
            np.testing.assert_array_equal(s1.signal.time, s2.signal.time)
