"""
Tests for feature extraction: noise estimation, peak finding, FWHM, area, SNR, batch.
"""

import numpy as np
import pytest

from src.signal_processing.feature_extractor import (
    compute_fwhm,
    compute_peak_area,
    compute_snr,
    estimate_baseline_noise,
    extract_features,
    extract_features_batch,
    find_primary_peak,
)
from src.signal_processing.signal_generator import generate_gaussian_peak, generate_lorentzian_peak
from src.signal_processing.signal_models import SignalData


class TestNoiseEstimation:
    """Baseline noise estimation."""

    def test_clean_signal_low_noise(self, sample_time_array):
        clean = generate_gaussian_peak(sample_time_array, 50.0, 3.0, 2.0)
        assert estimate_baseline_noise(clean) < 0.05

    def test_noisy_signal_higher(self, sample_time_array):
        clean = generate_gaussian_peak(sample_time_array, 50.0, 3.0, 2.0)
        rng = np.random.default_rng(42)
        noisy = clean + rng.normal(0, 0.1, size=len(clean))
        assert estimate_baseline_noise(noisy) > 0.05

    def test_handles_nans(self, sample_time_array):
        signal = generate_gaussian_peak(sample_time_array, 50.0, 3.0, 2.0)
        signal[::10] = np.nan
        assert estimate_baseline_noise(signal) >= 0

    def test_short_signal_fallback(self):
        short = np.random.default_rng(42).normal(0, 0.1, size=5)
        assert estimate_baseline_noise(short, window_length=11) >= 0


class TestPeakFinding:
    """Primary peak detection."""

    def test_find_gaussian_peak(self, sample_time_array):
        signal = generate_gaussian_peak(sample_time_array, 50.0, 3.0, 2.0)
        idx = find_primary_peak(signal)
        assert idx is not None
        assert 45 <= idx <= 55

    def test_find_lorentzian_peak(self, sample_time_array):
        signal = generate_lorentzian_peak(sample_time_array, 50.0, 3.5, 2.0)
        idx = find_primary_peak(signal)
        assert idx is not None
        assert 45 <= idx <= 55

    def test_flat_signal_returns_none(self, sample_time_array):
        flat = np.ones(len(sample_time_array)) * 0.1
        assert find_primary_peak(flat, height_threshold=0.5) is None

    def test_finds_highest_peak(self, sample_time_array):
        p1 = generate_gaussian_peak(sample_time_array, 30.0, 3.0, 1.5)
        p2 = generate_gaussian_peak(sample_time_array, 70.0, 3.0, 2.5)
        idx = find_primary_peak(p1 + p2)
        assert idx is not None
        assert 65 <= idx <= 75

    def test_handles_nans(self, sample_time_array):
        signal = generate_gaussian_peak(sample_time_array, 50.0, 3.0, 2.0)
        signal[::10] = np.nan
        assert find_primary_peak(signal) is not None


class TestFWHMComputation:
    """Full Width at Half Maximum."""

    def test_gaussian_theoretical_fwhm(self, sample_time_array):
        sigma = 3.0
        signal = generate_gaussian_peak(sample_time_array, 50.0, sigma, 2.0)
        fwhm = compute_fwhm(sample_time_array, signal, 50)
        assert fwhm == pytest.approx(2.355 * sigma, rel=0.1)

    def test_lorentzian_fwhm(self, sample_time_array):
        gamma = 3.5
        signal = generate_lorentzian_peak(sample_time_array, 50.0, gamma, 2.0)
        fwhm = compute_fwhm(sample_time_array, signal, 50)
        assert fwhm == pytest.approx(2 * gamma, rel=0.1)

    @pytest.mark.parametrize(
        "sigma,expected_fwhm",
        [(2.0, 4.71), (3.0, 7.07), (4.0, 9.42), (5.0, 11.77)],
    )
    def test_various_widths(self, sample_time_array, sigma, expected_fwhm):
        signal = generate_gaussian_peak(sample_time_array, 50.0, sigma, 2.0)
        fwhm = compute_fwhm(sample_time_array, signal, 50)
        assert fwhm == pytest.approx(expected_fwhm, rel=0.15)


class TestPeakAreaComputation:
    """Peak area integration."""

    def test_positive_area(self, sample_time_array):
        signal = generate_gaussian_peak(sample_time_array, 50.0, 3.0, 2.0)
        assert compute_peak_area(sample_time_array, signal) > 0

    def test_higher_peak_larger_area(self, sample_time_array):
        low = generate_gaussian_peak(sample_time_array, 50.0, 3.0, 1.0)
        high = generate_gaussian_peak(sample_time_array, 50.0, 3.0, 3.0)
        assert compute_peak_area(sample_time_array, high) > compute_peak_area(
            sample_time_array, low
        )

    def test_wider_peak_larger_area(self, sample_time_array):
        narrow = generate_gaussian_peak(sample_time_array, 50.0, 2.0, 2.0)
        wide = generate_gaussian_peak(sample_time_array, 50.0, 5.0, 2.0)
        assert compute_peak_area(sample_time_array, wide) > compute_peak_area(
            sample_time_array, narrow
        )

    def test_handles_nans(self, sample_time_array):
        signal = generate_gaussian_peak(sample_time_array, 50.0, 3.0, 2.0)
        signal[::10] = np.nan
        assert compute_peak_area(sample_time_array, signal) > 0


class TestSNRComputation:
    """Signal-to-Noise Ratio."""

    @pytest.mark.parametrize(
        "peak_height,noise_level,expected",
        [(2.0, 0.1, 20.0), (3.0, 0.1, 30.0), (2.0, 0.05, 40.0), (1.0, 0.5, 2.0)],
    )
    def test_snr_values(self, peak_height, noise_level, expected):
        assert compute_snr(np.array([]), peak_height, noise_level) == pytest.approx(expected)

    def test_zero_noise_returns_inf(self):
        assert compute_snr(np.array([]), peak_height=2.0, noise_level=0.0) == float("inf")


class TestFeatureExtraction:
    """Complete feature extraction from signals."""

    def test_all_features_present(self, gaussian_signal):
        features = extract_features(gaussian_signal)
        for key in ("fwhm", "peak_height", "peak_area", "noise_level", "snr", "peak_center"):
            assert key in features
            assert features[key] is not None

    def test_lorentzian_peak_height(self, lorentzian_signal):
        features = extract_features(lorentzian_signal)
        assert features["peak_height"] == pytest.approx(2.0, rel=0.01)

    def test_noisy_signal_features(self, noisy_signal):
        features = extract_features(noisy_signal)
        assert features["fwhm"] is not None
        assert features["snr"] is not None
        assert features["snr"] < 100

    def test_flat_signal_no_peak(self, sample_time_array):
        flat = np.ones(len(sample_time_array)) * 0.1
        sd = SignalData(
            time=sample_time_array.tolist(), amplitude=flat.tolist(), shape_type="gaussian"
        )
        features = extract_features(sd)
        assert features["fwhm"] is None
        assert features["peak_height"] is None
        assert features["noise_level"] is not None

    def test_feature_ranges(self, gaussian_signal):
        f = extract_features(gaussian_signal)
        assert f["fwhm"] > 0
        assert f["peak_height"] > 0
        assert f["peak_area"] > 0
        assert f["noise_level"] >= 0
        assert f["snr"] > 0
        assert 0 <= f["peak_center"] <= 100

    def test_handles_nans(self, signal_with_nans):
        features = extract_features(signal_with_nans)
        assert features["noise_level"] is not None


class TestBatchFeatureExtraction:
    """Batch feature extraction."""

    def test_batch_length_matches(self, sample_dataset):
        signals = [ls.signal for ls in sample_dataset]
        result = extract_features_batch(signals)
        assert len(result) == len(signals)
        for features in result:
            assert "fwhm" in features

    def test_batch_matches_single(self, gaussian_signal):
        single = extract_features(gaussian_signal)
        batch = extract_features_batch([gaussian_signal])
        assert len(batch) == 1
        for key in single:
            if single[key] is not None:
                assert batch[0][key] == pytest.approx(single[key], rel=0.01)

    def test_empty_batch(self):
        assert extract_features_batch([]) == []
