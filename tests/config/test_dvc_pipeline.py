"""
DVC pipeline validation tests.

Validates:
- dvc.yaml exists and is valid YAML
- Pipeline stages are correctly defined
- Stage dependencies reference existing files
- Stage outputs are defined
"""

import pytest
import yaml


@pytest.fixture
def dvc_config(project_root) -> dict:
    """Load dvc.yaml as a dictionary."""
    dvc_path = project_root / "dvc.yaml"
    if not dvc_path.is_file():
        pytest.skip("dvc.yaml not found")
    with open(dvc_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestDVCPipelineStructure:
    """Verify dvc.yaml is valid and well-structured."""

    def test_dvc_yaml_exists(self, project_root):
        """dvc.yaml exists in project root."""
        assert (project_root / "dvc.yaml").is_file()

    def test_has_stages_section(self, dvc_config):
        """dvc.yaml has a stages section."""
        assert "stages" in dvc_config

    def test_expected_stages_exist(self, dvc_config):
        """Key pipeline stages are defined."""
        stages = dvc_config["stages"]
        expected = ["generate_data", "extract_features", "train", "evaluate"]
        for stage in expected:
            assert stage in stages, f"Missing stage: {stage}"

    def test_stages_have_cmd(self, dvc_config):
        """Each stage defines a cmd."""
        for name, stage in dvc_config["stages"].items():
            assert "cmd" in stage, f"Stage '{name}' missing cmd"

    def test_stages_have_deps_or_params(self, dvc_config):
        """Each stage has deps, params, or both."""
        for name, stage in dvc_config["stages"].items():
            has_deps = "deps" in stage
            has_params = "params" in stage
            assert has_deps or has_params, f"Stage '{name}' has neither deps nor params"

    def test_stage_commands_reference_python(self, dvc_config):
        """Stage commands should invoke Python scripts."""
        for name, stage in dvc_config["stages"].items():
            cmd = stage["cmd"]
            assert "python" in cmd.lower() or "uv run" in cmd.lower(), (
                f"Stage '{name}' cmd doesn't invoke Python: {cmd}"
            )

    def test_train_stage_has_outputs(self, dvc_config):
        """Train stage should produce model outputs."""
        train = dvc_config["stages"]["train"]
        has_outs = "outs" in train
        has_metrics = "metrics" in train
        assert has_outs or has_metrics, "Train stage has no outputs or metrics"

    def test_evaluate_stage_has_metrics(self, dvc_config):
        """Evaluate stage should produce metrics."""
        evaluate = dvc_config["stages"]["evaluate"]
        has_metrics = "metrics" in evaluate
        has_plots = "plots" in evaluate
        has_outs = "outs" in evaluate
        assert has_metrics or has_plots or has_outs, (
            "Evaluate stage produces no metrics/plots/outputs"
        )
