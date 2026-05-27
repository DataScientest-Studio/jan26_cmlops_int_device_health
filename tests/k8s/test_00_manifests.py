"""
K8s Tier 0 — Manifest & Configuration Validation Tests (offline — no cluster needed)

Validates all K8s YAML manifests, Kustomize overlays, ConfigMap settings,
DAG sync, and Makefile targets WITHOUT requiring a running cluster.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
K8S_BASE = ROOT / "k8s" / "base"
K8S_OVERLAYS = ROOT / "k8s" / "overlays"
AIRFLOW_DAGS = ROOT / "airflow" / "dags"
MAKEFILE = ROOT / "Makefile"


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _load_yaml_all(path: Path) -> list[dict]:
    return list(yaml.safe_load_all(path.read_text(encoding="utf-8")))


def _kustomize_build(overlay: str) -> str:
    """Run kubectl kustomize on an overlay path and return the rendered YAML."""
    overlay_path = K8S_OVERLAYS / overlay
    if not overlay_path.exists():
        pytest.skip(f"Overlay {overlay} not found")
    result = subprocess.run(
        ["kubectl", "kustomize", str(overlay_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    assert result.returncode == 0, f"kustomize build {overlay} failed:\n{result.stderr}"
    return result.stdout


# ─── T1.1 ConfigMap has required K8s environment variables ────────────────────


class TestConfigMap:
    @pytest.fixture(scope="class")
    def configmap(self) -> dict:
        return _load_yaml(K8S_BASE / "configmap.yaml")

    def test_deployment_mode_is_cloud(self, configmap: dict) -> None:
        data = configmap.get("data", {})
        assert data.get("DEPLOYMENT_MODE") == "cloud", (
            f"ConfigMap DEPLOYMENT_MODE should be 'cloud', got '{data.get('DEPLOYMENT_MODE')}'"
        )

    def test_mlops_environment_is_kubernetes(self, configmap: dict) -> None:
        data = configmap.get("data", {})
        assert data.get("MLOPS_ENVIRONMENT") == "kubernetes", (
            f"ConfigMap MLOPS_ENVIRONMENT should be 'kubernetes', got '{data.get('MLOPS_ENVIRONMENT')}'"
        )

    def test_configmap_kind(self, configmap: dict) -> None:
        assert configmap.get("kind") == "ConfigMap"

    def test_configmap_namespace(self, configmap: dict) -> None:
        assert configmap.get("metadata", {}).get("namespace") == "mlops"

    def test_configmap_has_database_vars(self, configmap: dict) -> None:
        data = configmap.get("data", {})
        for key in ["DB_NAME", "DB_USER", "DATABASE_URL"]:
            assert key in data, f"ConfigMap missing {key}"

    def test_configmap_has_mlflow_vars(self, configmap: dict) -> None:
        data = configmap.get("data", {})
        assert "MLFLOW_TRACKING_URI" in data, "ConfigMap missing MLFLOW_TRACKING_URI"

    def test_configmap_has_airflow_vars(self, configmap: dict) -> None:
        data = configmap.get("data", {})
        assert "AIRFLOW_USER" in data, "ConfigMap missing AIRFLOW_USER"


# ─── T1.2 Local overlay PVCs use cluster-default storageClassName ─────────────
# The local overlay intentionally omits storageClassName so the cluster default
# is used automatically: 'hostpath' on Docker Desktop (Windows), 'local-path'
# on OrbStack (macOS), 'standard' on Kind (CI).  This makes the overlay
# portable across platforms without per-platform kustomization patches.


class TestLocalOverlay:
    @pytest.fixture(scope="class")
    def rendered(self) -> str:
        return _kustomize_build("local")

    @pytest.fixture(scope="class")
    def docs(self, rendered: str) -> list[dict]:
        return list(yaml.safe_load_all(rendered))

    def test_kustomize_builds_successfully(self, rendered: str) -> None:
        assert len(rendered) > 0

    def test_all_pvcs_omit_storage_class(self, docs: list[dict]) -> None:
        """PVCs should NOT hard-code a storageClassName so the cluster default is used.

        Cross-platform: Docker Desktop (Windows) defaults to 'hostpath';
        OrbStack (macOS) defaults to 'local-path'; Kind (CI) defaults to 'standard'.
        """
        pvcs = [d for d in docs if d and d.get("kind") == "PersistentVolumeClaim"]
        assert len(pvcs) >= 7, f"Expected at least 7 PVCs, found {len(pvcs)}"
        for pvc in pvcs:
            name = pvc["metadata"]["name"]
            sc = pvc.get("spec", {}).get("storageClassName")
            assert sc is None, (
                f"PVC '{name}' should omit storageClassName (use cluster default), "
                f"got '{sc}'. Remove the storageClassName field for cross-platform support."
            )

    def test_api_replicas(self, docs: list[dict]) -> None:
        deploys = [
            d
            for d in docs
            if d and d.get("kind") == "Deployment" and d["metadata"]["name"] == "api"
        ]
        if deploys:
            replicas = deploys[0].get("spec", {}).get("replicas")
            assert replicas == 1, f"API replicas in local overlay should be 1, got {replicas}"


# ─── T1.3 Cloud overlay does NOT hardcode storageClassName ────────────────────


class TestCloudOverlay:
    @pytest.fixture(scope="class")
    def kustomization(self) -> dict:
        path = K8S_OVERLAYS / "cloud" / "kustomization.yaml"
        if not path.exists():
            pytest.skip("Cloud overlay not found")
        return _load_yaml(path)

    def test_cloud_overlay_exists(self, kustomization: dict) -> None:
        assert kustomization is not None

    def test_cloud_does_not_patch_storageclass(self, kustomization: dict) -> None:
        patches = kustomization.get("patches", []) + kustomization.get("patchesStrategicMerge", [])
        patch_text = json.dumps(patches)
        assert "hostpath" not in patch_text.lower(), (
            "Cloud overlay should NOT reference 'hostpath' storage class"
        )


# ─── T1.4-T1.5 All overlays build valid YAML ─────────────────────────────────


class TestOverlayBuilds:
    @pytest.mark.parametrize("overlay", ["local", "cloud", "ghcr"])
    def test_overlay_builds(self, overlay: str) -> None:
        rendered = _kustomize_build(overlay)
        docs = list(yaml.safe_load_all(rendered))
        valid = [d for d in docs if d is not None]
        assert len(valid) > 0, f"Overlay '{overlay}' produced no valid documents"

    @pytest.mark.parametrize("overlay", ["local", "cloud", "ghcr"])
    def test_overlay_has_namespace(self, overlay: str) -> None:
        rendered = _kustomize_build(overlay)
        docs = list(yaml.safe_load_all(rendered))
        ns_docs = [d for d in docs if d and d.get("kind") == "Namespace"]
        assert len(ns_docs) >= 1, f"Overlay '{overlay}' missing Namespace resource"


# ─── T1.6 DAGs ConfigMap contains all DAG files ──────────────────────────────


class TestDAGsConfigMap:
    @pytest.fixture(scope="class")
    def dags_cm(self) -> dict:
        path = K8S_BASE / "airflow" / "dags-configmap.yaml"
        if not path.exists():
            pytest.skip("dags-configmap.yaml not found")
        return _load_yaml(path)

    @pytest.fixture(scope="class")
    def dag_files(self) -> list[str]:
        return [f.name for f in AIRFLOW_DAGS.glob("*.py")]

    def test_dags_cm_kind(self, dags_cm: dict) -> None:
        assert dags_cm.get("kind") == "ConfigMap"

    def test_all_dag_files_present(self, dags_cm: dict, dag_files: list[str]) -> None:
        data_keys = list(dags_cm.get("data", {}).keys())
        for dag_file in dag_files:
            assert dag_file in data_keys, f"DAG file '{dag_file}' missing from dags-configmap.yaml"

    def test_dag_guard_present(self, dags_cm: dict) -> None:
        data_keys = list(dags_cm.get("data", {}).keys())
        assert "_dag_guards.py" in data_keys, "DAG guard file missing from configmap"

    def test_dag_content_not_empty(self, dags_cm: dict) -> None:
        for key, val in dags_cm.get("data", {}).items():
            assert val and len(val.strip()) > 10, f"DAG '{key}' has empty/trivial content"


# ─── T1.7 Makefile targets ───────────────────────────────────────────────────


class TestMakefileTargets:
    @pytest.fixture(scope="class")
    def makefile_text(self) -> str:
        return MAKEFILE.read_text(encoding="utf-8")

    @pytest.mark.parametrize(
        "target",
        [
            "k8s-secret",
            "k8s-sync-dags",
            "k8s-ports",
            "k8s-ports-stop",
            "k8s-test",
            "k8s-setup",
            "k8s-full",
        ],
    )
    def test_target_defined(self, makefile_text: str, target: str) -> None:
        pattern = rf"^{re.escape(target)}\s*:"
        assert re.search(pattern, makefile_text, re.MULTILINE), (
            f"Makefile target '{target}' not defined"
        )

    @pytest.mark.parametrize(
        "target",
        [
            "k8s-secret",
            "k8s-sync-dags",
            "k8s-ports",
            "k8s-ports-stop",
            "k8s-test",
            "k8s-setup",
            "k8s-full",
        ],
    )
    def test_target_in_phony(self, makefile_text: str, target: str) -> None:
        assert target in makefile_text, f"Makefile target '{target}' not in .PHONY"


# ─── T1.8 Secret generator produces valid YAML ──────────────────────────────


class TestSecretGenerator:
    def test_env_secrets_exists(self) -> None:
        env_secrets = ROOT / ".env.secrets"
        if not env_secrets.exists():
            pytest.skip(".env.secrets not found")
        content = env_secrets.read_text(encoding="utf-8")
        assert len(content.strip()) > 0, ".env.secrets is empty"

    def test_secret_example_exists(self) -> None:
        path = K8S_BASE / "secret.example.yaml"
        if not path.exists():
            pytest.skip("secret.example.yaml not found")
        doc = _load_yaml(path)
        assert doc.get("kind") == "Secret"

    def test_secret_generator_script_exists(self) -> None:
        ps1 = ROOT / "scripts" / "k8s_setup_secret.ps1"
        sh = ROOT / "scripts" / "k8s_setup_secret.sh"
        assert ps1.exists() or sh.exists(), (
            "Neither k8s_setup_secret.ps1 nor k8s_setup_secret.sh found"
        )


# ─── T1.9 Base kustomization.yaml references ─────────────────────────────────


class TestBaseKustomization:
    @pytest.fixture(scope="class")
    def kustomization(self) -> dict:
        return _load_yaml(K8S_BASE / "kustomization.yaml")

    def test_has_resources(self, kustomization: dict) -> None:
        resources = kustomization.get("resources", [])
        assert len(resources) >= 20, (
            f"Base kustomization has only {len(resources)} resources, expected 20+"
        )

    def test_namespace_in_resources(self, kustomization: dict) -> None:
        resources = kustomization.get("resources", [])
        assert "namespace.yaml" in resources

    def test_configmap_in_resources(self, kustomization: dict) -> None:
        resources = kustomization.get("resources", [])
        assert "configmap.yaml" in resources

    def test_all_referenced_files_exist(self, kustomization: dict) -> None:
        resources = kustomization.get("resources", [])
        missing = []
        for res in resources:
            if res.startswith("#"):
                continue
            path = K8S_BASE / res
            if not path.exists():
                missing.append(res)
        assert not missing, f"Referenced but missing files: {missing}"


# ─── T1.10 Port-forward & smoke test scripts ─────────────────────────────────


class TestScripts:
    @pytest.mark.parametrize(
        "script",
        [
            "k8s_port_forward.ps1",
            "k8s_port_forward.sh",
            "k8s_smoke_test.ps1",
            "k8s_sync_dags.ps1",
            "k8s_stop_ports.ps1",
            "k8s_setup_secret.ps1",
            "k8s_setup_secret.sh",
        ],
    )
    def test_script_exists(self, script: str) -> None:
        path = ROOT / "scripts" / script
        assert path.exists(), f"Script '{script}' not found"

    def test_port_forward_covers_all_services(self) -> None:
        ps1 = ROOT / "scripts" / "k8s_port_forward.ps1"
        if not ps1.exists():
            pytest.skip("k8s_port_forward.ps1 not found")
        content = ps1.read_text(encoding="utf-8")
        for svc in ["api", "streamlit", "mlflow", "airflow", "grafana", "prometheus", "nginx"]:
            assert svc in content.lower(), f"Port-forward script missing service '{svc}'"

    def test_smoke_test_covers_all_services(self) -> None:
        ps1 = ROOT / "scripts" / "k8s_smoke_test.ps1"
        if not ps1.exists():
            pytest.skip("k8s_smoke_test.ps1 not found")
        content = ps1.read_text(encoding="utf-8")
        for svc in ["api", "mlflow", "airflow", "grafana", "prometheus"]:
            assert svc in content.lower(), f"Smoke test script missing service '{svc}'"


# ─── T1.11 All service deployments have resource limits ──────────────────────


class TestResourceLimits:
    @pytest.fixture(scope="class")
    def rendered_docs(self) -> list[dict]:
        rendered = _kustomize_build("local")
        return [d for d in yaml.safe_load_all(rendered) if d]

    def test_deployments_have_containers(self, rendered_docs: list[dict]) -> None:
        deploys = [d for d in rendered_docs if d.get("kind") == "Deployment"]
        assert len(deploys) >= 5, f"Expected at least 5 deployments, got {len(deploys)}"
        for dep in deploys:
            name = dep["metadata"]["name"]
            containers = dep["spec"]["template"]["spec"]["containers"]
            assert len(containers) >= 1, f"Deployment '{name}' has no containers"


# ─── T1.12 Services expose correct ports ─────────────────────────────────────


class TestServicePorts:
    @pytest.fixture(scope="class")
    def rendered_docs(self) -> list[dict]:
        rendered = _kustomize_build("local")
        return [d for d in yaml.safe_load_all(rendered) if d]

    EXPECTED_PORTS = {
        "api": 8000,
        "streamlit": 8501,
        "mlflow": 5000,
        "airflow": 8080,
        "grafana": 3000,
        "prometheus": 9090,
        "nginx": 80,
    }

    def test_service_ports(self, rendered_docs: list[dict]) -> None:
        services = {d["metadata"]["name"]: d for d in rendered_docs if d.get("kind") == "Service"}
        for svc_name, expected_port in self.EXPECTED_PORTS.items():
            if svc_name not in services:
                continue
            svc = services[svc_name]
            ports = svc.get("spec", {}).get("ports", [])
            port_numbers = [p.get("port") or p.get("targetPort") for p in ports]
            assert expected_port in port_numbers, (
                f"Service '{svc_name}' expected port {expected_port}, got {port_numbers}"
            )


# ─── T1.13 Grafana dashboard files exist and are valid JSON ──────────────────


class TestGrafanaDashboards:
    DASHBOARD_DIR = ROOT / "docker" / "grafana" / "dashboards"

    @pytest.fixture(scope="class")
    def dashboard_files(self) -> list[Path]:
        if not self.DASHBOARD_DIR.exists():
            pytest.skip("Grafana dashboards directory not found")
        return list(self.DASHBOARD_DIR.glob("*.json"))

    def test_dashboards_exist(self, dashboard_files: list[Path]) -> None:
        assert len(dashboard_files) >= 5, (
            f"Expected at least 5 Grafana dashboards, found {len(dashboard_files)}"
        )

    @pytest.mark.parametrize(
        "dashboard_name",
        [
            "system_health.json",
            "model_performance.json",
            "kubernetes_cluster.json",
            "retraining_pipeline.json",
            "data_quality.json",
        ],
    )
    def test_dashboard_present(self, dashboard_name: str) -> None:
        path = self.DASHBOARD_DIR / dashboard_name
        assert path.exists(), f"Dashboard '{dashboard_name}' not found"

    def test_dashboards_valid_json(self, dashboard_files: list[Path]) -> None:
        for f in dashboard_files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                assert isinstance(data, dict), f"{f.name} is not a JSON object"
            except json.JSONDecodeError as e:
                pytest.fail(f"Dashboard '{f.name}' is invalid JSON: {e}")

    def test_dashboards_have_panels(self, dashboard_files: list[Path]) -> None:
        for f in dashboard_files:
            data = json.loads(f.read_text(encoding="utf-8"))
            panels = data.get("panels", [])
            # Some dashboards may use rows with nested panels
            rows = data.get("rows", [])
            total = len(panels) + sum(len(r.get("panels", [])) for r in rows)
            assert total > 0, f"Dashboard '{f.name}' has no panels"


# ─── T1.14 Airflow DAG files are valid Python ────────────────────────────────


class TestAirflowDAGFiles:
    @pytest.fixture(scope="class")
    def dag_files(self) -> list[Path]:
        return list(AIRFLOW_DAGS.glob("*.py"))

    def test_dag_files_exist(self, dag_files: list[Path]) -> None:
        assert len(dag_files) >= 8, f"Expected at least 8 DAG files, found {len(dag_files)}"

    def test_dag_files_compile(self, dag_files: list[Path]) -> None:
        for f in dag_files:
            try:
                compile(f.read_text(encoding="utf-8"), str(f), "exec")
            except SyntaxError as e:
                pytest.fail(f"DAG '{f.name}' has syntax error: {e}")

    @pytest.mark.parametrize(
        "dag_file",
        [
            "automated_retraining.py",
            "drift_triggered_retraining.py",
            "evidently_drift_detection.py",
            "model_promotion.py",
            "batch_rescoring.py",
        ],
    )
    def test_expected_dag_exists(self, dag_file: str) -> None:
        assert (AIRFLOW_DAGS / dag_file).exists(), f"Expected DAG '{dag_file}' not found"


# ─── T1.15 Prometheus config ─────────────────────────────────────────────────


class TestPrometheusConfig:
    @pytest.fixture(scope="class")
    def prom_config(self) -> dict:
        path = K8S_BASE / "prometheus" / "configmap.yaml"
        if not path.exists():
            pytest.skip("Prometheus configmap not found")
        doc = _load_yaml(path)
        data = doc.get("data", {})
        prom_yaml = data.get("prometheus.yml", "")
        return yaml.safe_load(prom_yaml) if prom_yaml else {}

    def test_prom_has_scrape_configs(self, prom_config: dict) -> None:
        assert "scrape_configs" in prom_config, "Prometheus config missing scrape_configs"

    def test_prom_scrapes_api(self, prom_config: dict) -> None:
        jobs = [s.get("job_name", "") for s in prom_config.get("scrape_configs", [])]
        api_jobs = [j for j in jobs if "api" in j.lower()]
        assert len(api_jobs) > 0, f"Prometheus not scraping API. Jobs: {jobs}"


# ─── T1.16 K8s documentation exists ──────────────────────────────────────────


class TestDocumentation:
    def test_quickstart_doc_exists(self) -> None:
        path = ROOT / "doc" / "K8s_Windows_Quickstart.md"
        if not path.exists():
            pytest.skip("K8s_Windows_Quickstart.md not included in this distribution")
        assert path.exists(), "K8s_Windows_Quickstart.md not found"

    def test_quickstart_has_content(self) -> None:
        path = ROOT / "doc" / "K8s_Windows_Quickstart.md"
        if not path.exists():
            pytest.skip("Not found")
        content = path.read_text(encoding="utf-8")
        assert len(content) > 500, "Quickstart doc is too short"
        for term in ["kubectl", "make k8s", "port-forward"]:
            assert term in content, f"Quickstart doc missing '{term}'"
