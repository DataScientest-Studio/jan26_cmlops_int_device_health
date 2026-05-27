"""Signal processing module for device health monitoring."""

from .feature_extractor import (
    compute_fwhm,
    compute_peak_area,
    compute_snr,
    estimate_baseline_noise,
    extract_features,
    extract_features_batch,
    find_primary_peak,
)
from .signal_generator import (
    add_gaussian_noise,
    create_time_array,
    generate_dataset,
    generate_gaussian_peak,
    generate_lorentzian_peak,
    generate_signal,
    inject_nans,
)
from .signal_models import (
    GaussianParameters,
    HealthClassificationRules,
    LabeledSignal,
    LorentzianParameters,
    SignalData,
)
from .validators import (
    is_signal_valid,
    validate_amplitude_range,
    validate_nan_limit,
    validate_peak_count,
    validate_signal_all,
    validate_signal_completeness,
    validate_signal_density,
    validate_time_range,
)

__all__ = [
    # Models
    "GaussianParameters",
    "LorentzianParameters",
    "SignalData",
    "LabeledSignal",
    "HealthClassificationRules",
    # Generators
    "generate_gaussian_peak",
    "generate_lorentzian_peak",
    "generate_signal",
    "generate_dataset",
    "add_gaussian_noise",
    "inject_nans",
    "create_time_array",
    # Validators
    "validate_signal_completeness",
    "validate_signal_density",
    "validate_nan_limit",
    "validate_time_range",
    "validate_peak_count",
    "validate_amplitude_range",
    "validate_signal_all",
    "is_signal_valid",
    # Feature extraction
    "estimate_baseline_noise",
    "find_primary_peak",
    "compute_fwhm",
    "compute_peak_area",
    "compute_snr",
    "extract_features",
    "extract_features_batch",
]
