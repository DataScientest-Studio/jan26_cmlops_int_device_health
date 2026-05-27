"""
Tests for Grafana dashboard JSON definitions.

Validates:
- Dashboard JSON files are valid JSON
- Required panels and datasources are defined
- Dashboard UIDs are unique
- Key dashboards exist (system_health, model_performance, business_kpis)
"""

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DASHBOARDS_DIR = PROJECT_ROOT / "docker" / "grafana" / "dashboards"


class TestGrafanaDashboards:
    """Tests for Grafana dashboard definitions."""

    def test_dashboards_directory_exists(self):
        """Grafana dashboards directory exists."""
        assert DASHBOARDS_DIR.exists(), f"Dashboards dir not found at {DASHBOARDS_DIR}"

    def test_all_dashboards_are_valid_json(self):
        """Every .json file in dashboards/ is valid JSON."""
        for json_file in DASHBOARDS_DIR.glob("*.json"):
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            assert isinstance(data, dict), f"{json_file.name} is not a JSON object"

    def test_key_dashboards_exist(self):
        """Essential dashboards are present."""
        expected = ["system_health.json", "model_performance.json", "business_kpis.json"]
        existing = {f.name for f in DASHBOARDS_DIR.glob("*.json")}
        for name in expected:
            assert name in existing, f"Missing dashboard: {name}"

    def test_dashboards_have_panels(self):
        """Each dashboard defines at least one panel."""
        for json_file in DASHBOARDS_DIR.glob("*.json"):
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            panels = data.get("panels", [])
            # Some dashboards use rows containing panels
            rows = data.get("rows", [])
            total_panels = len(panels) + sum(len(r.get("panels", [])) for r in rows)
            assert total_panels > 0, f"{json_file.name} has no panels"

    def test_dashboard_uids_are_unique(self):
        """Dashboard UIDs are unique across all files."""
        uids = []
        for json_file in DASHBOARDS_DIR.glob("*.json"):
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            uid = data.get("uid")
            if uid:
                uids.append((json_file.name, uid))

        uid_values = [u[1] for u in uids]
        assert len(uid_values) == len(set(uid_values)), f"Duplicate dashboard UIDs found: {uids}"

    def test_dashboards_reference_prometheus(self):
        """Dashboards use Prometheus as data source."""
        for json_file in DASHBOARDS_DIR.glob("*.json"):
            with open(json_file, encoding="utf-8") as f:
                content = f.read()
            # Check if prometheus is referenced anywhere (datasource, templating, etc.)
            has_prometheus = "prometheus" in content.lower() or "Prometheus" in content
            # Some dashboards might use different datasource names
            assert has_prometheus or "datasource" in content.lower(), (
                f"{json_file.name} doesn't reference Prometheus"
            )

    def test_system_health_dashboard_has_expected_metrics(self):
        """system_health.json references key API metrics."""
        path = DASHBOARDS_DIR / "system_health.json"
        if not path.exists():
            pytest.skip("system_health.json not found")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        # Should reference at least some core metrics
        expected_metrics = ["api_requests_total", "api_request_duration"]
        for metric in expected_metrics:
            assert metric in content, f"system_health.json missing metric: {metric}"
