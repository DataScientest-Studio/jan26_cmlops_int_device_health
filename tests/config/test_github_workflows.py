"""
Tests for GitHub Actions workflow YAML files.

Validates:
- test.yml workflow exists and is valid YAML
- Required jobs are defined
- Python version matrix is configured
- Correct pytest command is used
- All workflows have valid structure (on, jobs sections)
"""

from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WORKFLOWS_DIR = PROJECT_ROOT / ".github" / "workflows"


class TestTestWorkflow:
    """Tests for .github/workflows/test.yml."""

    @pytest.fixture
    def test_workflow(self):
        """Load test.yml workflow."""
        path = WORKFLOWS_DIR / "test.yml"
        assert path.exists(), f"test.yml not found at {path}"
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_has_trigger_events(self, test_workflow):
        """Workflow has on/trigger section."""
        assert "on" in test_workflow or True in test_workflow  # YAML parses 'on' as True
        triggers = test_workflow.get("on") or test_workflow.get(True)
        assert triggers is not None

    def test_has_jobs_section(self, test_workflow):
        """Workflow has jobs section."""
        assert "jobs" in test_workflow
        assert len(test_workflow["jobs"]) > 0

    def test_test_job_exists(self, test_workflow):
        """A 'test' job is defined."""
        assert "test" in test_workflow["jobs"]

    def test_runs_on_ubuntu(self, test_workflow):
        """Test job runs on ubuntu."""
        test_job = test_workflow["jobs"]["test"]
        runs_on = test_job.get("runs-on", "")
        assert "ubuntu" in runs_on

    def test_uses_python_312(self, test_workflow):
        """Test matrix includes Python 3.12."""
        test_job = test_workflow["jobs"]["test"]
        strategy = test_job.get("strategy", {})
        matrix = strategy.get("matrix", {})
        python_versions = matrix.get("python-version", [])
        assert "3.12" in python_versions

    def test_pytest_command_excludes_live(self, test_workflow):
        """Pytest command uses '-m not live' marker filter."""
        test_job = test_workflow["jobs"]["test"]
        steps = test_job.get("steps", [])
        # Find step that runs pytest
        pytest_steps = [s for s in steps if "pytest" in str(s.get("run", ""))]
        assert len(pytest_steps) > 0
        pytest_cmd = pytest_steps[0]["run"]
        assert "not live" in pytest_cmd


class TestAllWorkflows:
    """Validate all workflow files are valid YAML."""

    def test_all_workflows_are_valid_yaml(self):
        """Every .yml file in workflows/ parses as valid YAML."""
        if not WORKFLOWS_DIR.exists():
            pytest.skip("No .github/workflows directory")

        for yml_file in WORKFLOWS_DIR.glob("*.yml"):
            with open(yml_file, encoding="utf-8") as f:
                config = yaml.safe_load(f)
            assert config is not None, f"{yml_file.name} is empty or invalid"
            assert "jobs" in config, f"{yml_file.name} missing 'jobs' section"

    def test_all_workflows_have_triggers(self):
        """Every workflow has trigger events defined."""
        if not WORKFLOWS_DIR.exists():
            pytest.skip("No .github/workflows directory")

        for yml_file in WORKFLOWS_DIR.glob("*.yml"):
            with open(yml_file, encoding="utf-8") as f:
                config = yaml.safe_load(f)
            # 'on' parses as True in YAML
            triggers = config.get("on") or config.get(True)
            assert triggers is not None, f"{yml_file.name} missing trigger events"
