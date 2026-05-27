"""
Validation functions for signal data quality.

Provides FastAPI-ready validators for signal completeness, density, NaN limits,
and peak detection.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from .signal_models import SignalData


def validate_signal_completeness(signal: SignalData, min_length: int = 51) -> tuple[bool, str]:
    """
    Validate signal has sufficient data points.

    Args:
        signal: SignalData instance
        min_length: Minimum required length

    Returns:
        Tuple of (is_valid, message)
    """
    signal_length = len(signal.time)

    if signal_length < min_length:
        return False, f"Signal too short: {signal_length} < {min_length}"

    return True, f"Signal length valid: {signal_length}"


def validate_signal_density(signal: SignalData, min_density: int = 101) -> tuple[bool, str]:
    """
    Validate signal has adequate point density.

    Args:
        signal: SignalData instance
        min_density: Minimum required number of points

    Returns:
        Tuple of (is_valid, message)
    """
    density = len(signal.time)

    if density < min_density:
        return False, f"Insufficient density: {density} < {min_density}"

    return True, f"Density adequate: {density} points"


def validate_nan_limit(signal: SignalData, max_nan_fraction: float = 0.05) -> tuple[bool, str]:
    """
    Validate NaN values are within acceptable limits.

    Args:
        signal: SignalData instance
        max_nan_fraction: Maximum allowed NaN fraction (default 5%)

    Returns:
        Tuple of (is_valid, message)
    """
    amplitude_arr = np.array(signal.amplitude)
    nan_count = np.isnan(amplitude_arr).sum()
    nan_fraction = nan_count / len(amplitude_arr)

    if nan_fraction > max_nan_fraction:
        return False, f"NaN fraction {nan_fraction:.2%} exceeds {max_nan_fraction:.2%}"

    return True, f"NaN fraction acceptable: {nan_fraction:.2%}"


def validate_time_range(
    signal: SignalData, expected_min: float = 0.0, expected_max: float = 100.0
) -> tuple[bool, str]:
    """
    Validate time array spans expected range.

    Args:
        signal: SignalData instance
        expected_min: Expected minimum time value
        expected_max: Expected maximum time value

    Returns:
        Tuple of (is_valid, message)
    """
    time_arr = np.array(signal.time)
    actual_min = time_arr.min()
    actual_max = time_arr.max()

    if actual_min > expected_min + 1.0:  # 1.0 tolerance
        return False, f"Time range starts late: {actual_min} > {expected_min}"

    if actual_max < expected_max - 1.0:
        return False, f"Time range ends early: {actual_max} < {expected_max}"

    return True, f"Time range valid: [{actual_min:.1f}, {actual_max:.1f}]"


def validate_peak_count(
    signal: SignalData,
    expected_count: int = 1,
    height_threshold: float = 0.5,
    prominence: float = 0.3,
) -> tuple[bool, str]:
    """
    Validate signal contains expected number of peaks.

    Single-peak signals are healthy. Multiple peaks suggest artifacts or interference.

    Args:
        signal: SignalData instance
        expected_count: Expected number of peaks (default 1)
        height_threshold: Minimum peak height
        prominence: Minimum peak prominence

    Returns:
        Tuple of (is_valid, message)
    """
    amplitude_arr = np.array(signal.amplitude)

    # Remove NaNs for peak detection
    valid_mask = ~np.isnan(amplitude_arr)
    valid_amplitude = amplitude_arr[valid_mask]

    if len(valid_amplitude) < 10:
        return False, "Too many NaNs for peak detection"

    # Find peaks
    peaks, properties = find_peaks(valid_amplitude, height=height_threshold, prominence=prominence)
    peak_count = len(peaks)

    if peak_count != expected_count:
        return False, f"Peak count mismatch: found {peak_count}, expected {expected_count}"

    return True, f"Peak count valid: {peak_count}"


def validate_amplitude_range(
    signal: SignalData, min_amplitude: float = 0.0, max_amplitude: float = 5.0
) -> tuple[bool, str]:
    """
    Validate amplitude values are within physical bounds.

    Args:
        signal: SignalData instance
        min_amplitude: Minimum allowed amplitude
        max_amplitude: Maximum allowed amplitude

    Returns:
        Tuple of (is_valid, message)
    """
    amplitude_arr = np.array(signal.amplitude)
    valid_amplitude = amplitude_arr[~np.isnan(amplitude_arr)]

    actual_min = valid_amplitude.min()
    actual_max = valid_amplitude.max()

    if actual_min < min_amplitude:
        return False, f"Amplitude too low: {actual_min:.3f} < {min_amplitude}"

    if actual_max > max_amplitude:
        return False, f"Amplitude too high: {actual_max:.3f} > {max_amplitude}"

    return True, f"Amplitude range valid: [{actual_min:.3f}, {actual_max:.3f}]"


def validate_signal_all(signal: SignalData) -> dict[str, tuple[bool, str]]:
    """
    Run all validation checks on a signal.

    Args:
        signal: SignalData instance

    Returns:
        Dictionary mapping validator name to (is_valid, message) tuple
    """
    validations = {
        "completeness": validate_signal_completeness(signal),
        "density": validate_signal_density(signal),
        "nan_limit": validate_nan_limit(signal),
        "time_range": validate_time_range(signal),
        "peak_count": validate_peak_count(signal),
        "amplitude_range": validate_amplitude_range(signal),
    }

    return validations


def is_signal_valid(signal: SignalData) -> bool:
    """
    Check if signal passes all validation checks.

    Args:
        signal: SignalData instance

    Returns:
        True if all validators pass, False otherwise
    """
    validations = validate_signal_all(signal)
    return all(is_valid for is_valid, _ in validations.values())
