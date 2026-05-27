"""
Tests for signal validators: completeness, density, NaN limits, time range, peaks, amplitude.
"""

import numpy as np
import pytest
from pydantic_core import ValidationError

from src.signal_processing.signal_models import SignalData
from src.signal_processing.validators import (
    is_signal_valid,
    validate_amplitude_range,
    validate_nan_limit,
    validate_peak_count,
    validate_signal_all,
    validate_signal_completeness,
    validate_signal_density,
    validate_time_range,
)


class TestSignalCompleteness:
    """Signal length validation."""

    def test_valid_length(self, gaussian_signal):
        is_valid, msg = validate_signal_completeness(gaussian_signal, min_length=51)
        assert is_valid

    def test_too_short_rejected_by_pydantic(self):
        with pytest.raises(ValidationError):
            SignalData(time=list(range(30)), amplitude=[1.0] * 30, shape_type="gaussian")

    def test_boundary_case(self):
        signal = SignalData(
            time=np.linspace(0, 100, 51).tolist(),
            amplitude=np.ones(51).tolist(),
            shape_type="gaussian",
        )
        is_valid, _ = validate_signal_completeness(signal, min_length=51)
        assert is_valid


class TestSignalDensity:
    """Signal density validation."""

    def test_sufficient_density(self, gaussian_signal):
        is_valid, msg = validate_signal_density(gaussian_signal, min_density=101)
        assert is_valid

    def test_insufficient_density(self, sample_time_minimal):
        signal = SignalData(
            time=sample_time_minimal.tolist(),
            amplitude=np.ones(51).tolist(),
            shape_type="gaussian",
        )
        is_valid, msg = validate_signal_density(signal, min_density=101)
        assert not is_valid

    @pytest.mark.parametrize(
        "n_points,expected",
        [(51, False), (101, True), (150, True)],
    )
    def test_density_thresholds(self, n_points, expected):
        signal = SignalData(
            time=np.linspace(0, 100, n_points).tolist(),
            amplitude=np.ones(n_points).tolist(),
            shape_type="gaussian",
        )
        is_valid, _ = validate_signal_density(signal, min_density=101)
        assert is_valid == expected


class TestNaNLimit:
    """NaN fraction validation."""

    def test_no_nans_passes(self, gaussian_signal):
        is_valid, _ = validate_nan_limit(gaussian_signal)
        assert is_valid

    def test_acceptable_nans(self, signal_with_nans):
        is_valid, _ = validate_nan_limit(signal_with_nans, max_nan_fraction=0.05)
        assert is_valid

    def test_excessive_nans_rejected_by_pydantic(self, sample_time_array):
        amplitude = np.ones(len(sample_time_array))
        indices = np.arange(0, 10)
        amplitude[indices] = np.nan
        with pytest.raises(ValidationError):
            SignalData(
                time=sample_time_array.tolist(),
                amplitude=amplitude.tolist(),
                shape_type="gaussian",
            )


class TestTimeRange:
    """Time range [0, 100] validation."""

    def test_valid_range(self, gaussian_signal):
        is_valid, _ = validate_time_range(gaussian_signal)
        assert is_valid

    def test_starts_too_late(self):
        with pytest.raises(ValidationError):
            SignalData(
                time=np.linspace(10, 100, 101).tolist(),
                amplitude=np.ones(101).tolist(),
                shape_type="gaussian",
            )

    def test_ends_too_early(self):
        with pytest.raises(ValidationError):
            SignalData(
                time=np.linspace(0, 80, 101).tolist(),
                amplitude=np.ones(101).tolist(),
                shape_type="gaussian",
            )


class TestPeakCount:
    """Peak count validation."""

    def test_single_peak_valid(self, gaussian_signal):
        is_valid, _ = validate_peak_count(gaussian_signal, expected_count=1)
        assert is_valid

    def test_multiple_peaks_invalid(self, sample_time_array):
        p1 = 2.0 * np.exp(-((sample_time_array - 30) ** 2) / 18)
        p2 = 2.0 * np.exp(-((sample_time_array - 70) ** 2) / 18)
        signal = SignalData(
            time=sample_time_array.tolist(),
            amplitude=(p1 + p2).tolist(),
            shape_type="gaussian",
        )
        is_valid, msg = validate_peak_count(signal, expected_count=1)
        assert not is_valid

    def test_no_peaks_invalid(self, sample_time_array):
        signal = SignalData(
            time=sample_time_array.tolist(),
            amplitude=(np.ones(101) * 0.1).tolist(),
            shape_type="gaussian",
        )
        is_valid, _ = validate_peak_count(signal, expected_count=1)
        assert not is_valid


class TestAmplitudeRange:
    """Amplitude range validation."""

    def test_valid_amplitude(self, gaussian_signal):
        is_valid, _ = validate_amplitude_range(gaussian_signal)
        assert is_valid

    def test_too_high(self, sample_time_array):
        signal = SignalData(
            time=sample_time_array.tolist(),
            amplitude=(np.ones(101) * 10.0).tolist(),
            shape_type="gaussian",
        )
        is_valid, msg = validate_amplitude_range(signal, max_amplitude=5.0)
        assert not is_valid

    def test_too_low(self, sample_time_array):
        signal = SignalData(
            time=sample_time_array.tolist(),
            amplitude=(np.ones(101) * -1.0).tolist(),
            shape_type="gaussian",
        )
        is_valid, msg = validate_amplitude_range(signal, min_amplitude=0.0)
        assert not is_valid

    @pytest.mark.parametrize(
        "value,expected",
        [(2.0, True), (0.0, True), (5.0, True), (-0.5, False), (6.0, False)],
    )
    def test_boundaries(self, sample_time_array, value, expected):
        signal = SignalData(
            time=sample_time_array.tolist(),
            amplitude=(np.ones(101) * value).tolist(),
            shape_type="gaussian",
        )
        is_valid, _ = validate_amplitude_range(signal, min_amplitude=0.0, max_amplitude=5.0)
        assert is_valid == expected


class TestComprehensiveValidation:
    """validate_signal_all and is_signal_valid."""

    def test_all_validators_run(self, gaussian_signal):
        results = validate_signal_all(gaussian_signal)
        for key in (
            "completeness",
            "density",
            "nan_limit",
            "time_range",
            "peak_count",
            "amplitude_range",
        ):
            assert key in results
            assert results[key][0] is True

    def test_is_signal_valid_clean(self, gaussian_signal):
        assert is_signal_valid(gaussian_signal)
