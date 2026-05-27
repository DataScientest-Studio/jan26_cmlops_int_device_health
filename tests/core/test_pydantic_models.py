"""
Tests for Pydantic signal models: GaussianParameters, LorentzianParameters, SignalData, LabeledSignal.
"""

import numpy as np
import pytest
from pydantic_core import ValidationError

from src.signal_processing.signal_models import (
    GaussianParameters,
    LabeledSignal,
    LorentzianParameters,
    SignalData,
)


class TestGaussianParameters:
    """GaussianParameters field constraints and immutability."""

    def test_valid_construction(self):
        p = GaussianParameters(mu=50.0, sigma=3.0, height=2.0, noise_level=0.02)
        assert p.mu == 50.0
        assert p.sigma == 3.0

    @pytest.mark.parametrize(
        "field,value",
        [
            ("mu", 34.0),  # below ge=35.0
            ("mu", 61.0),  # above le=60.0
            ("sigma", 1.0),  # below ge=2.0
            ("sigma", 7.0),  # above le=6.0
            ("height", 0.5),  # below ge=1.0
            ("height", 4.0),  # above le=3.5
            ("noise_level", -0.01),  # below ge=0.0
            ("noise_level", 0.2),  # above le=0.15
        ],
    )
    def test_field_out_of_range(self, field, value):
        defaults = {"mu": 50.0, "sigma": 3.0, "height": 2.0, "noise_level": 0.02}
        defaults[field] = value
        with pytest.raises(ValidationError):
            GaussianParameters(**defaults)

    def test_frozen_immutable(self):
        p = GaussianParameters(mu=50.0, sigma=3.0, height=2.0, noise_level=0.02)
        with pytest.raises(ValidationError):
            p.mu = 60.0


class TestLorentzianParameters:
    """LorentzianParameters field constraints and from_gaussian_sigma."""

    def test_valid_construction(self):
        p = LorentzianParameters(mu=50.0, gamma=3.5325, height=2.0, noise_level=0.02)
        assert p.gamma == 3.5325

    def test_from_gaussian_sigma(self):
        p = LorentzianParameters.from_gaussian_sigma(
            mu=50.0, sigma=3.0, height=2.0, noise_level=0.02
        )
        assert p.gamma == pytest.approx(1.1775 * 3.0, rel=0.001)


class TestSignalData:
    """SignalData validators: length, time range, NaN limit, to_numpy."""

    def test_valid_construction(self):
        sd = SignalData(
            time=np.linspace(0, 100, 101).tolist(),
            amplitude=np.ones(101).tolist(),
            shape_type="gaussian",
        )
        assert len(sd.time) == 101

    def test_min_length_enforced(self):
        with pytest.raises(ValidationError):
            SignalData(time=list(range(50)), amplitude=[1.0] * 50, shape_type="gaussian")

    def test_time_amplitude_length_mismatch_accepted(self):
        """Pydantic field validators run per-field so length mismatch isn't caught at construction.
        Downstream code (extract_features, predict) validates this."""
        sd = SignalData(
            time=np.linspace(0, 100, 101).tolist(),
            amplitude=np.ones(100).tolist(),
            shape_type="gaussian",
        )
        assert len(sd.time) != len(sd.amplitude)

    def test_time_range_start_invalid(self):
        with pytest.raises(ValidationError):
            SignalData(
                time=np.linspace(5, 100, 101).tolist(),
                amplitude=np.ones(101).tolist(),
                shape_type="gaussian",
            )

    def test_nan_limit_enforced(self):
        amplitude = np.ones(101)
        amplitude[:7] = np.nan  # ~7% > 5% limit
        with pytest.raises(ValidationError):
            SignalData(
                time=np.linspace(0, 100, 101).tolist(),
                amplitude=amplitude.tolist(),
                shape_type="gaussian",
            )

    def test_to_numpy(self):
        sd = SignalData(
            time=np.linspace(0, 100, 101).tolist(),
            amplitude=np.ones(101).tolist(),
            shape_type="gaussian",
        )
        t, a = sd.to_numpy()
        assert isinstance(t, np.ndarray)
        assert isinstance(a, np.ndarray)
        assert len(t) == 101


class TestLabeledSignal:
    """LabeledSignal label values and metadata."""

    def test_valid_labels(self):
        sd = SignalData(
            time=np.linspace(0, 100, 101).tolist(),
            amplitude=np.ones(101).tolist(),
            shape_type="gaussian",
        )
        for label in (-1, 0, 1):
            ls = LabeledSignal(signal=sd, label=label)
            assert ls.label == label

    def test_metadata_default(self):
        sd = SignalData(
            time=np.linspace(0, 100, 101).tolist(),
            amplitude=np.ones(101).tolist(),
            shape_type="gaussian",
        )
        ls = LabeledSignal(signal=sd, label=0)
        assert ls.metadata == {}
