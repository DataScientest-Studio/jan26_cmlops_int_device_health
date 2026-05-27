"""
Tests for Prometheus configuration files.

Validates:
- prometheus.yml is valid YAML with required sections
- alerts.yml defines expected alert rules
- Scrape targets reference correct services
- Alert severity labels present
"""

from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TestPrometheusConfig:
    """Tests for docker/prometheus/prometheus.yml."""

    @pytest.fixture
    def prometheus_config(self):
        """Load Prometheus configuration."""
        config_path = PROJECT_ROOT / "docker" / "prometheus" / "prometheus.yml"
        assert config_path.exists(), f"Prometheus config not found at {config_path}"
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_global_section_exists(self, prometheus_config):
        """Config has global section with scrape_interval."""
        assert "global" in prometheus_config
        assert "scrape_interval" in prometheus_config["global"]

    def test_scrape_interval_reasonable(self, prometheus_config):
        """Scrape interval is between 5s and 60s."""
        interval = prometheus_config["global"]["scrape_interval"]
        # Parse "15s" → 15
        seconds = int(interval.replace("s", "").replace("m", ""))
        assert 5 <= seconds <= 60

    def test_scrape_configs_present(self, prometheus_config):
        """At least one scrape target is configured."""
        assert "scrape_configs" in prometheus_config
        assert len(prometheus_config["scrape_configs"]) > 0

    def test_mlops_api_job_defined(self, prometheus_config):
        """Scrape config includes mlops_api job targeting the API."""
        jobs = {c["job_name"] for c in prometheus_config["scrape_configs"]}
        assert "mlops_api" in jobs

    def test_alerting_section_exists(self, prometheus_config):
        """Alerting section references Alertmanager."""
        assert "alerting" in prometheus_config

    def test_rule_files_configured(self, prometheus_config):
        """Alert rule files are referenced."""
        assert "rule_files" in prometheus_config
        assert len(prometheus_config["rule_files"]) > 0


class TestAlertRules:
    """Tests for docker/prometheus/alerts.yml."""

    @pytest.fixture
    def alerts_config(self):
        """Load alert rules configuration."""
        alerts_path = PROJECT_ROOT / "docker" / "prometheus" / "alerts.yml"
        assert alerts_path.exists(), f"Alerts config not found at {alerts_path}"
        with open(alerts_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_groups_defined(self, alerts_config):
        """Alert rules have at least one group."""
        assert "groups" in alerts_config
        assert len(alerts_config["groups"]) > 0

    def test_api_alerts_group_exists(self, alerts_config):
        """api_alerts group is defined."""
        group_names = [g["name"] for g in alerts_config["groups"]]
        assert "api_alerts" in group_names

    def test_critical_alerts_defined(self, alerts_config):
        """Critical alerts (APIDown, HighErrorRate) are defined."""
        all_alerts = []
        for group in alerts_config["groups"]:
            for rule in group.get("rules", []):
                if "alert" in rule:
                    all_alerts.append(rule["alert"])

        assert "APIDown" in all_alerts
        assert "HighErrorRate" in all_alerts

    def test_alerts_have_severity_labels(self, alerts_config):
        """All alerts include a severity label."""
        for group in alerts_config["groups"]:
            for rule in group.get("rules", []):
                if "alert" in rule:
                    labels = rule.get("labels", {})
                    assert "severity" in labels, f"Alert {rule['alert']} missing severity label"

    def test_alerts_have_annotations(self, alerts_config):
        """All alerts include summary and description annotations."""
        for group in alerts_config["groups"]:
            for rule in group.get("rules", []):
                if "alert" in rule:
                    annotations = rule.get("annotations", {})
                    assert "summary" in annotations, (
                        f"Alert {rule['alert']} missing summary annotation"
                    )
                    assert "description" in annotations, (
                        f"Alert {rule['alert']} missing description annotation"
                    )


class TestDockerComposeConfig:
    """Tests for docker-compose.yml service definitions."""

    @pytest.fixture
    def compose_config(self):
        """Load main docker-compose.yml."""
        compose_path = PROJECT_ROOT / "docker-compose.yml"
        assert compose_path.exists(), f"docker-compose.yml not found at {compose_path}"
        with open(compose_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_services_defined(self, compose_config):
        """Docker compose has services section."""
        assert "services" in compose_config
        assert len(compose_config["services"]) > 0

    def test_api_service_defined(self, compose_config):
        """API service is defined in compose."""
        assert "api" in compose_config["services"]

    def test_api_service_has_healthcheck(self, compose_config):
        """API service has a healthcheck defined."""
        api = compose_config["services"]["api"]
        assert "healthcheck" in api

    def test_database_service_defined(self, compose_config):
        """Database (postgres) service is defined."""
        services = compose_config["services"]
        # Might be named postgres, db, or database
        db_names = {"postgres", "db", "database"}
        found = db_names & set(services.keys())
        assert len(found) > 0, f"No database service found. Services: {list(services.keys())}"

    def test_volumes_section_exists(self, compose_config):
        """Compose file defines named volumes."""
        # volumes may be at top level or within services
        has_top_level = "volumes" in compose_config
        has_service_volumes = any(
            "volumes" in svc for svc in compose_config.get("services", {}).values()
        )
        assert has_top_level or has_service_volumes
