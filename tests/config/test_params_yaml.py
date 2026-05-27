"""
params.yaml validation tests.

Validates:
- params.yaml exists and is valid YAML
- Required sections and keys are present
- Parameter values are within reasonable ranges
- Consistency between params and source code expectations
"""

import pytest
import yaml


@pytest.fixture
def params(project_root) -> dict:
    """Load params.yaml as a dictionary."""
    params_path = project_root / "params.yaml"
    assert params_path.is_file(), "params.yaml not found"
    with open(params_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestParamsStructure:
    """Verify params.yaml has required sections."""

    def test_params_file_exists(self, project_root):
        """params.yaml exists in project root."""
        assert (project_root / "params.yaml").is_file()

    def test_has_generate_data_section(self, params):
        """params.yaml has generate_data section."""
        assert "generate_data" in params

    def test_has_preprocess_section(self, params):
        """params.yaml has preprocess section."""
        assert "preprocess" in params

    def test_has_train_section(self, params):
        """params.yaml has train section."""
        assert "train" in params

    def test_has_evaluate_section(self, params):
        """params.yaml has evaluate section."""
        assert "evaluate" in params

    def test_has_quality_gate_section(self, params):
        """params.yaml has quality_gate section."""
        assert "quality_gate" in params


class TestGenerateDataParams:
    """Validate data generation parameters."""

    def test_n_samples_is_positive(self, params):
        """n_samples must be positive integer."""
        n = params["generate_data"]["n_samples"]
        assert isinstance(n, int) and n > 0

    def test_gaussian_fraction_in_range(self, params):
        """gaussian_fraction must be in [0, 1]."""
        frac = params["generate_data"]["gaussian_fraction"]
        assert 0.0 <= frac <= 1.0

    def test_mu_range_is_valid(self, params):
        """mu_range must be [low, high] within [0, 100]."""
        mu = params["generate_data"]["mu_range"]
        assert len(mu) == 2
        assert 0 <= mu[0] < mu[1] <= 100

    def test_sigma_range_is_positive(self, params):
        """sigma_range values must be positive."""
        sigma = params["generate_data"]["sigma_range"]
        assert len(sigma) == 2
        assert sigma[0] > 0 and sigma[1] > sigma[0]

    def test_height_range_is_valid(self, params):
        """height_range values must be >= 1.0 (Pydantic constraint)."""
        height = params["generate_data"]["height_range"]
        assert len(height) == 2
        assert height[0] >= 1.0 and height[1] > height[0]


class TestTrainParams:
    """Validate training parameters."""

    def test_model_name_is_string(self, params):
        """model_name must be a non-empty string."""
        name = params["train"]["model_name"]
        assert isinstance(name, str) and len(name) > 0

    def test_c_is_positive(self, params):
        """SVM C parameter must be positive."""
        c = params["train"]["C"]
        assert c > 0

    def test_penalty_is_valid(self, params):
        """penalty must be a valid sklearn option."""
        penalty = params["train"]["penalty"]
        assert penalty in ("l1", "l2", "elasticnet", "none")

    def test_solver_is_valid(self, params):
        """solver must be a valid sklearn solver."""
        solver = params["train"]["solver"]
        valid_solvers = ("lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga")
        assert solver in valid_solvers

    def test_test_size_in_range(self, params):
        """test_size must be in (0, 1)."""
        ts = params["train"]["test_size"]
        assert 0.0 < ts < 1.0

    def test_random_state_is_set(self, params):
        """random_state should be set for reproducibility."""
        rs = params["train"]["random_state"]
        assert isinstance(rs, int) and rs >= 0


class TestQualityGateParams:
    """Validate quality gate thresholds."""

    def test_min_accuracy_in_range(self, params):
        """min_accuracy must be in (0, 1]."""
        acc = params["quality_gate"]["min_accuracy"]
        assert 0.0 < acc <= 1.0

    def test_min_f1_in_range(self, params):
        """min_f1 must be in (0, 1]."""
        f1 = params["quality_gate"]["min_f1"]
        assert 0.0 < f1 <= 1.0

    def test_thresholds_are_reasonable(self, params):
        """Thresholds should not be unrealistically high."""
        acc = params["quality_gate"]["min_accuracy"]
        f1 = params["quality_gate"]["min_f1"]
        # For synthetic data, 0.7-0.95 is reasonable
        assert acc <= 0.99, "min_accuracy too high (unrealistic for real data)"
        assert f1 <= 0.99, "min_f1 too high (unrealistic for real data)"


class TestPreprocessParams:
    """Validate preprocessing parameters."""

    def test_window_length_is_odd(self, params):
        """Savitzky-Golay window_length must be odd."""
        wl = params["preprocess"]["window_length"]
        assert wl % 2 == 1, "window_length must be odd for Savitzky-Golay"

    def test_polyorder_less_than_window(self, params):
        """polyorder must be less than window_length."""
        wl = params["preprocess"]["window_length"]
        po = params["preprocess"]["polyorder"]
        assert po < wl

    def test_peak_prominence_positive(self, params):
        """peak_prominence must be positive."""
        pp = params["preprocess"]["peak_prominence"]
        assert pp > 0
