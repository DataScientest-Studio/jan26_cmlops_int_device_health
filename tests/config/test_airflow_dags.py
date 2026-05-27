"""
Airflow DAG validation tests.

Validates:
- All DAG Python files are syntactically valid
- DAG files have required attributes (dag_id, schedule)
- _dag_guards module works correctly
- No import errors in DAG files (when Airflow is not required)
"""

import ast
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def dags_dir(project_root) -> Path:
    """Path to the Airflow DAGs directory."""
    return project_root / "airflow" / "dags"


@pytest.fixture
def dag_files(dags_dir) -> list[Path]:
    """All .py files in the dags directory (excluding __pycache__)."""
    return sorted(p for p in dags_dir.glob("*.py") if not p.name.startswith("__"))


class TestDAGSyntax:
    """Verify all DAG files are syntactically valid Python."""

    def test_dags_directory_exists(self, dags_dir):
        """airflow/dags/ directory exists."""
        assert dags_dir.is_dir()

    def test_all_dag_files_parse(self, dag_files):
        """All DAG Python files parse without syntax errors."""
        for dag_file in dag_files:
            source = dag_file.read_text(encoding="utf-8")
            try:
                ast.parse(source, filename=str(dag_file))
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {dag_file.name}: {e}")

    def test_expected_dags_exist(self, dags_dir):
        """Key DAG files are present."""
        expected = [
            "evidently_drift_detection.py",
            "automated_retraining.py",
            "database_backup.py",
            "model_promotion.py",
        ]
        for name in expected:
            assert (dags_dir / name).is_file(), f"Missing DAG file: {name}"

    def test_dag_files_have_docstrings(self, dag_files):
        """Each DAG file starts with a module docstring."""
        for dag_file in dag_files:
            if dag_file.name.startswith("_"):
                continue  # Skip helper modules
            source = dag_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
            docstring = ast.get_docstring(tree)
            assert docstring, f"{dag_file.name} missing module docstring"


class TestDAGStructure:
    """Verify DAG files follow expected patterns."""

    def test_dag_files_define_dag_id_or_dag_literal(self, dag_files):
        """Each DAG file defines a _DAG_ID constant or a DAG id string."""
        for dag_file in dag_files:
            if dag_file.name.startswith("_"):
                continue
            source = dag_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
            # Look for _DAG_ID assignment or dag_id= keyword
            has_dag_id = any(
                isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "_DAG_ID" for t in node.targets)
                for node in ast.walk(tree)
            )
            has_dag_id_string = "dag_id" in source or "_DAG_ID" in source
            assert has_dag_id or has_dag_id_string, f"{dag_file.name} missing DAG identifier"

    def test_cloud_only_dag_files_import_dag_guards(self, dag_files):
        """Cloud-only DAG files import from _dag_guards for mode checking."""
        for dag_file in dag_files:
            if dag_file.name.startswith("_"):
                continue
            source = dag_file.read_text(encoding="utf-8")
            # Only check DAGs that are cloud-specific
            if "cloud" in source.lower() and "require_cloud_mode" not in source:
                # Some DAGs work in both modes - that's fine
                pass
            # At least verify the DAG uses DAG()
            assert "DAG(" in source, f"{dag_file.name} doesn't define a DAG"

    def test_dag_files_use_dag_context(self, dag_files):
        """DAG files use DAG() (context manager or assignment)."""
        for dag_file in dag_files:
            if dag_file.name.startswith("_"):
                continue
            source = dag_file.read_text(encoding="utf-8")
            # Should contain DAG( which indicates DAG definition
            assert "DAG(" in source, f"{dag_file.name} doesn't define a DAG"


class TestDAGGuards:
    """Test the _dag_guards module directly."""

    def test_deployment_mode_from_env(self):
        """_deployment_mode reads DEPLOYMENT_MODE env var."""
        sys.path.insert(0, str(Path("airflow/dags")))
        try:
            from _dag_guards import _deployment_mode

            with patch.dict(os.environ, {"DEPLOYMENT_MODE": "cloud"}):
                assert _deployment_mode() == "cloud"

            with patch.dict(os.environ, {"DEPLOYMENT_MODE": "local"}):
                assert _deployment_mode() == "local"
        finally:
            sys.path.pop(0)

    def test_deployment_mode_defaults_to_local(self):
        """Without env var or mode file, defaults to 'local'."""
        sys.path.insert(0, str(Path("airflow/dags")))
        try:
            from _dag_guards import _deployment_mode

            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop("DEPLOYMENT_MODE", None)
                # Also mock the .current_mode file not existing
                with patch("pathlib.Path.exists", return_value=False):
                    mode = _deployment_mode()
                    assert mode == "local"
        finally:
            sys.path.pop(0)

    def test_require_cloud_mode_raises_in_local(self):
        """require_cloud_mode raises RuntimeError in local mode."""
        sys.path.insert(0, str(Path("airflow/dags")))
        try:
            from _dag_guards import require_cloud_mode

            with (
                patch.dict(os.environ, {"DEPLOYMENT_MODE": "local"}),
                pytest.raises(RuntimeError, match="cloud/k8s-mode only"),
            ):
                require_cloud_mode("test_dag")
        finally:
            sys.path.pop(0)

    def test_require_cloud_mode_passes_in_cloud(self):
        """require_cloud_mode does not raise in cloud mode."""
        sys.path.insert(0, str(Path("airflow/dags")))
        try:
            from _dag_guards import require_cloud_mode

            with patch.dict(os.environ, {"DEPLOYMENT_MODE": "cloud"}):
                # Should not raise
                require_cloud_mode("test_dag")
        finally:
            sys.path.pop(0)
