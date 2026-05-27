"""
Feature extraction module for signal analysis.

Extracts interpretable features from signal data: FWHM, peak height, SNR, area, noise level.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import find_peaks, savgol_filter

from .signal_models import SignalData


def estimate_baseline_noise(
    signal: np.ndarray, window_length: int = 11, polyorder: int = 3
) -> float:
    """
    Estimate noise level using Savitzky-Golay filter residuals.

    Args:
        signal: Amplitude array
        window_length: Window size for Savitzky-Golay filter (must be odd)
        polyorder: Polynomial order for filter

    Returns:
        Estimated noise standard deviation
    """
    # Remove NaNs for filtering
    valid_mask = ~np.isnan(signal)
    valid_signal = signal[valid_mask]

    if len(valid_signal) < window_length:
        # Fallback: use simple std dev
        return float(np.nanstd(signal))

    # Ensure window_length is odd and valid
    if window_length % 2 == 0:
        window_length += 1

    if window_length > len(valid_signal):
        window_length = len(valid_signal) if len(valid_signal) % 2 == 1 else len(valid_signal) - 1

    # Apply Savitzky-Golay filter to smooth signal
    smoothed = savgol_filter(valid_signal, window_length=window_length, polyorder=polyorder)

    # Residuals represent noise
    residuals = valid_signal - smoothed
    noise_level = float(np.std(residuals))

    return noise_level


def find_primary_peak(signal: np.ndarray, height_threshold: float = 0.5) -> int | None:
    """
    Find the primary peak in signal.

    Args:
        signal: Amplitude array
        height_threshold: Minimum height for peak detection

    Returns:
        Index of primary peak, or None if no peaks found
    """
    # Remove NaNs
    valid_mask = ~np.isnan(signal)
    valid_signal = signal[valid_mask]
    valid_indices = np.where(valid_mask)[0]

    if len(valid_signal) < 5:
        return None

    # Find all peaks
    peaks, properties = find_peaks(valid_signal, height=height_threshold, prominence=0.2)

    if len(peaks) == 0:
        return None

    # Return highest peak
    peak_heights = valid_signal[peaks]
    primary_peak_idx = peaks[np.argmax(peak_heights)]

    # Map back to original indices
    return int(valid_indices[primary_peak_idx])


def compute_fwhm(time: np.ndarray, amplitude: np.ndarray, peak_idx: int) -> float | None:
    """
    Compute Full Width at Half Maximum (FWHM).

    Args:
        time: Time coordinate array
        amplitude: Amplitude array
        peak_idx: Index of peak center

    Returns:
        FWHM value, or None if cannot be computed
    """
    peak_height = amplitude[peak_idx]
    half_max = peak_height / 2.0

    # Find left half-maximum crossing
    left_idx = None
    for i in range(peak_idx, -1, -1):
        if not np.isnan(amplitude[i]) and amplitude[i] <= half_max:
            left_idx = i
            break

    # Find right half-maximum crossing
    right_idx = None
    for i in range(peak_idx, len(amplitude)):
        if not np.isnan(amplitude[i]) and amplitude[i] <= half_max:
            right_idx = i
            break

    if left_idx is None or right_idx is None:
        return None

    # Interpolate for more accurate crossing points
    # Left crossing
    if left_idx < len(time) - 1:
        t_left = np.interp(
            half_max,
            [amplitude[left_idx], amplitude[left_idx + 1]],
            [time[left_idx], time[left_idx + 1]],
        )
    else:
        t_left = time[left_idx]

    # Right crossing
    if right_idx > 0:
        t_right = np.interp(
            half_max,
            [amplitude[right_idx - 1], amplitude[right_idx]],
            [time[right_idx - 1], time[right_idx]],
        )
    else:
        t_right = time[right_idx]

    fwhm = abs(t_right - t_left)
    return float(fwhm)


def compute_peak_area(time: np.ndarray, amplitude: np.ndarray) -> float:
    """
    Compute peak area using trapezoidal integration.

    Args:
        time: Time coordinate array
        amplitude: Amplitude array

    Returns:
        Integrated area under curve
    """
    # Remove NaNs by interpolation
    valid_mask = ~np.isnan(amplitude)
    valid_time = time[valid_mask]
    valid_amplitude = amplitude[valid_mask]

    if len(valid_time) < 2:
        return 0.0

    # Integrate
    area = trapezoid(valid_amplitude, valid_time)
    return float(area)


def compute_snr(signal: np.ndarray, peak_height: float, noise_level: float) -> float:
    """
    Compute Signal-to-Noise Ratio (SNR).

    SNR = peak_height / noise_level

    Args:
        signal: Amplitude array (not used directly, kept for API consistency)
        peak_height: Maximum signal amplitude
        noise_level: Estimated noise standard deviation

    Returns:
        SNR value
    """
    if noise_level == 0:
        return float("inf")

    snr = peak_height / noise_level
    return float(snr)


def extract_features(signal_data: SignalData) -> dict[str, float | None]:
    """
    Extract all features from a signal.

    Features:
        - fwhm: Full width at half maximum
        - peak_height: Maximum amplitude
        - peak_area: Integrated area under curve
        - noise_level: Estimated noise std dev
        - snr: Signal-to-noise ratio
        - peak_center: Time coordinate of peak maximum

    Args:
        signal_data: SignalData instance

    Returns:
        Dictionary of feature names to values
    """
    time = np.array(signal_data.time)
    amplitude = np.array(signal_data.amplitude)

    # Estimate noise
    noise_level = estimate_baseline_noise(amplitude)

    # Find primary peak
    peak_idx = find_primary_peak(amplitude)

    if peak_idx is None:
        # Cannot extract features without peak
        return {
            "fwhm": None,
            "peak_height": None,
            "peak_area": None,
            "noise_level": noise_level,
            "snr": None,
            "peak_center": None,
        }

    # Peak height
    peak_height = float(amplitude[peak_idx])

    # Peak center
    peak_center = float(time[peak_idx])

    # FWHM
    fwhm = compute_fwhm(time, amplitude, peak_idx)

    # Peak area
    peak_area = compute_peak_area(time, amplitude)

    # SNR
    snr = compute_snr(amplitude, peak_height, noise_level)

    return {
        "fwhm": fwhm,
        "peak_height": peak_height,
        "peak_area": peak_area,
        "noise_level": noise_level,
        "snr": snr,
        "peak_center": peak_center,
    }


def extract_features_batch(signals: list[SignalData]) -> list[dict[str, float | None]]:
    """
    Extract features from multiple signals.

    Args:
        signals: List of SignalData instances

    Returns:
        List of feature dictionaries
    """
    return [extract_features(signal) for signal in signals]
