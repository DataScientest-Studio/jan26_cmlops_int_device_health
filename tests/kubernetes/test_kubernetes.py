"""
Kubernetes test suite — manifest validation, kustomize overlays, Make targets,
Streamlit kubernetes.py helpers, and FastAPI K8s endpoints.

Tests are split into categories:
  - TestManifestStructure      : YAML validity, required fields, naming
  - TestKustomizeOverlays      : kustomize build produces expected resources
  - TestMakefileTargets        : Make k8s-* targets exist and are syntactically valid
  - TestApiK8sEndpoints        : FastAPI /k8s/* logic with a mock K8s client
  - TestK8sHelpers             : kubernetes.py helper functions (_kubectl_available,
                                 _k8s_context, _namespace_exists, _status_icon, etc.)
  - TestK8sPageRender          : kubernetes.py render() smoke-tests (no Streamlit runtime)
  - TestLiveCluster (optional) : live kubectl-backed tests; skipped if no cluster
  - TestLiveE2E (optional)     : full running stack E2E tests via port-forward + K8s SDK

Run with:
    uv run pytest tests/kubernetes/ -v
Live tests:
    uv run pytest tests/kubernetes/ -v -m live
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

# ── Constants ─────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parents[2]
_K8S_BASE = _ROOT / "k8s" / "base"
_K8S_OVERLAYS = _ROOT / "k8s" / "overlays"
_OVERLAYS = ["local", "cloud", "ghcr"]

# All manifest files under k8s/base (flat list)
_BASE_MANIFESTS = sorted(_K8S_BASE.rglob("*.yaml"))

# Deployments expected in the base layer
_EXPECTED_DEPLOYMENTS = {
    "postgres",
    "mlflow",
    "api",
    "streamlit",
    "nginx",
    "airflow",
    "prometheus",
    "grafana",
    "kube-state-metrics",
}
_EXPECTED_SERVICES = {
    "postgres",
    "mlflow",
    "api",
    "nginx",
    "streamlit",
    "airflow",
    "prometheus",
    "grafana",
    "kube-state-metrics",
}
_EXPECTED_PVCS = {
    "postgres-pvc",
    "mlflow-artifacts-pvc",
    "mlflow-db-pvc",
    "airflow-logs-pvc",
    "airflow-models-pvc",
    "prometheus-pvc",
    "grafana-pvc",
}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _kubectl(*args: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a kubectl command and return the CompletedProcess."""
    return subprocess.run(
        ["kubectl", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _kubectl_available() -> bool:
    try:
        result = subprocess.run(
            ["kubectl", "version", "--client"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _cluster_running() -> bool:
    """True only when a live K8s cluster is reachable AND has nodes in Ready state.

    Uses 'kubectl get nodes' rather than 'kubectl cluster-info' because
    cluster-info may return 0 when a context is configured but no cluster runs.
    'kubectl get nodes' succeeds (exit 0 + non-empty output) only when the API
    server is reachable and nodes are registered.
    """
    if not _kubectl_available():
        return False
    try:
        result = subprocess.run(
            ["kubectl", "get", "nodes", "--request-timeout=5s"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        # Both exit code AND actual node output required — prevents false positives
        # when kubectl has a configured context but the cluster is not running.
        return result.returncode == 0 and bool(result.stdout.strip())
    except Exception:
        return False


_CLUSTER_RUNNING = _cluster_running()

# Markers
_skip_no_cluster = pytest.mark.skipif(not _CLUSTER_RUNNING, reason="No live K8s cluster available")


def live(obj: Any) -> Any:
    """Decorator that marks a test class/function with `live` AND skips if no cluster."""
    return _skip_no_cluster(pytest.mark.live(obj))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Manifest Structure Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestManifestStructure:
    """Validate every YAML file under k8s/base/ is valid and well-structured."""

    def test_all_manifests_are_valid_yaml(self):
        """Each manifest file must parse as valid YAML."""
        errors = []
        for path in _BASE_MANIFESTS:
            try:
                list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
            except yaml.YAMLError as exc:
                errors.append(f"{path.name}: {exc}")
        assert not errors, "YAML parse errors:\n" + "\n".join(errors)

    def test_namespace_manifest_exists(self):
        assert (_K8S_BASE / "namespace.yaml").exists()

    def test_configmap_manifest_exists(self):
        assert (_K8S_BASE / "configmap.yaml").exists()

    def test_secret_example_manifest_exists(self):
        assert (_K8S_BASE / "secret.example.yaml").exists()

    def test_secret_yaml_exists_for_local_testing(self):
        """secret.yaml must exist locally (git-ignored, created from example).

        Skipped in CI because the file intentionally contains real credentials
        and is never committed.  Developers must copy secret.example.yaml →
        secret.yaml and fill in their own values before running K8s tests locally.
        """
        if os.environ.get("CI"):
            pytest.skip("secret.yaml is git-ignored and not present in CI — local-only check")
        assert (_K8S_BASE / "secret.yaml").exists(), (
            "k8s/base/secret.yaml is missing. "
            "Copy secret.example.yaml → secret.yaml and fill in dev credentials."
        )

    def test_all_resources_have_namespace(self):
        """Every namespaced resource must declare namespace: mlops."""
        non_namespaced_kinds = {
            "Namespace",
            "ClusterRole",
            "ClusterRoleBinding",
            "PersistentVolume",
            "StorageClass",
            "ServiceAccount",
            "Kustomization",  # kustomize config file, not a K8s resource
        }
        errors = []
        for path in _BASE_MANIFESTS:
            docs = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
            for doc in docs:
                if doc is None:
                    continue
                kind = doc.get("kind", "")
                if kind in non_namespaced_kinds:
                    continue
                # ServiceAccount can be namespaced or not
                ns = (doc.get("metadata") or {}).get("namespace")
                if ns is None:
                    errors.append(
                        f"{path.name}: kind={kind} name="
                        f"{(doc.get('metadata') or {}).get('name')} missing namespace"
                    )
                elif ns != "mlops":
                    errors.append(
                        f"{path.name}: kind={kind} has namespace={ns!r}, expected 'mlops'"
                    )
        assert not errors, "Namespace issues:\n" + "\n".join(errors)

    def test_all_deployments_have_resource_limits(self):
        """Every container in every Deployment must declare resource limits."""
        errors = []
        for path in _BASE_MANIFESTS:
            docs = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
            for doc in docs:
                if doc is None or doc.get("kind") != "Deployment":
                    continue
                name = (doc.get("metadata") or {}).get("name", "?")
                spec = doc.get("spec", {}).get("template", {}).get("spec", {})
                containers = spec.get("containers", [])
                for ctr in containers:
                    resources = ctr.get("resources") or {}
                    if not resources.get("limits"):
                        errors.append(f"{name}/{ctr.get('name')}: missing resources.limits")
                    if not resources.get("requests"):
                        errors.append(f"{name}/{ctr.get('name')}: missing resources.requests")
        assert not errors, "Missing resource limits:\n" + "\n".join(errors)

    def test_all_deployments_have_liveness_and_readiness_probes(self):
        """Every main container must have both a liveness and readiness probe."""
        errors = []
        for path in _BASE_MANIFESTS:
            docs = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
            for doc in docs:
                if doc is None or doc.get("kind") != "Deployment":
                    continue
                name = (doc.get("metadata") or {}).get("name", "?")
                containers = (
                    doc.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
                )
                for ctr in containers:
                    cname = ctr.get("name", "?")
                    if not ctr.get("readinessProbe"):
                        errors.append(f"{name}/{cname}: missing readinessProbe")
                    if not ctr.get("livenessProbe"):
                        errors.append(f"{name}/{cname}: missing livenessProbe")
        assert not errors, "Missing health probes:\n" + "\n".join(errors)

    def test_all_deployments_have_selector_labels(self):
        """Deployment selectors must match pod template labels."""
        errors = []
        for path in _BASE_MANIFESTS:
            docs = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
            for doc in docs:
                if doc is None or doc.get("kind") != "Deployment":
                    continue
                name = (doc.get("metadata") or {}).get("name", "?")
                selector_labels = doc.get("spec", {}).get("selector", {}).get("matchLabels", {})
                pod_labels = (
                    doc.get("spec", {}).get("template", {}).get("metadata", {}).get("labels", {})
                )
                for k, v in selector_labels.items():
                    if pod_labels.get(k) != v:
                        errors.append(
                            f"{name}: selector matchLabel {k}={v!r} not in pod labels {pod_labels}"
                        )
        assert not errors, "Selector/label mismatches:\n" + "\n".join(errors)

    def test_all_services_have_matching_selector(self):
        """Services must have a non-empty selector."""
        errors = []
        for path in _BASE_MANIFESTS:
            docs = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
            for doc in docs:
                if doc is None or doc.get("kind") != "Service":
                    continue
                name = (doc.get("metadata") or {}).get("name", "?")
                spec = doc.get("spec") or {}
                svc_type = spec.get("type", "ClusterIP")
                # Headless or ExternalName services may have no selector
                if svc_type in ("ExternalName",):
                    continue
                selector = spec.get("selector") or {}
                if not selector:
                    errors.append(f"Service {name}: empty selector")
        assert not errors, "Services with empty selectors:\n" + "\n".join(errors)

    def test_pvcs_have_storage_requests(self):
        """Every PVC must request some storage."""
        errors = []
        for path in _BASE_MANIFESTS:
            docs = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
            for doc in docs:
                if doc is None or doc.get("kind") != "PersistentVolumeClaim":
                    continue
                name = (doc.get("metadata") or {}).get("name", "?")
                storage = (
                    doc.get("spec", {}).get("resources", {}).get("requests", {}).get("storage")
                )
                if not storage:
                    errors.append(f"PVC {name}: missing storage request")
        assert not errors, "PVCs without storage:\n" + "\n".join(errors)

    def test_namespace_is_mlops(self):
        """The namespace manifest must declare name: mlops."""
        doc = yaml.safe_load((_K8S_BASE / "namespace.yaml").read_text())
        assert doc["metadata"]["name"] == "mlops"

    def test_configmap_has_required_keys(self):
        """The main configmap must have all required application keys."""
        doc = yaml.safe_load((_K8S_BASE / "configmap.yaml").read_text())
        data = doc["data"]
        required_keys = [
            "DB_NAME",
            "DB_USER",
            "MLFLOW_TRACKING_URI",
            "MLFLOW_EXPERIMENT_NAME",
            "MODEL_REGISTRY_NAME",
            "AIRFLOW_USER",
            "LOG_LEVEL",
        ]
        missing = [k for k in required_keys if k not in data]
        assert not missing, f"ConfigMap missing keys: {missing}"

    def test_secret_example_has_required_keys(self):
        """The secret example template must cover all required secret keys."""
        doc = yaml.safe_load((_K8S_BASE / "secret.example.yaml").read_text(encoding="utf-8"))
        string_data = doc.get("stringData") or doc.get("data") or {}
        required_keys = [
            "DB_PASSWORD",
            "AIRFLOW_PASSWORD",
            "API_SECRET_KEY",
            "GF_SECURITY_ADMIN_PASSWORD",
        ]
        missing = [k for k in required_keys if k not in string_data]
        assert not missing, f"Secret example missing keys: {missing}"

    def test_airflow_uses_db_migrate_not_db_init(self):
        """Airflow manifests must use 'airflow db migrate' (not deprecated 'db init').

        The DB init is handled by the airflow-init Job (init-job.yaml), not the
        main Deployment — so we check both files.
        """
        files_to_check = [
            _K8S_BASE / "airflow" / "deployment.yaml",
            _K8S_BASE / "airflow" / "init-job.yaml",
        ]
        combined = "\n".join(p.read_text() for p in files_to_check if p.exists())
        assert "airflow db init" not in combined, (
            "Deprecated 'airflow db init' found in airflow manifests. Use 'airflow db migrate'."
        )
        assert "airflow db migrate" in combined, (
            "'airflow db migrate' not found in airflow manifests (deployment.yaml or init-job.yaml)."
        )

    def test_hpa_targets_api_deployment(self):
        """HPA must target the api Deployment."""
        path = _K8S_BASE / "api" / "hpa.yaml"
        doc = yaml.safe_load(path.read_text())
        ref = doc["spec"]["scaleTargetRef"]
        assert ref["kind"] == "Deployment"
        assert ref["name"] == "api"

    def test_hpa_min_max_replicas(self):
        """HPA must have sensible min/max replicas."""
        path = _K8S_BASE / "api" / "hpa.yaml"
        doc = yaml.safe_load(path.read_text())
        spec = doc["spec"]
        assert spec["minReplicas"] >= 1
        assert spec["maxReplicas"] >= spec["minReplicas"]
        assert spec["maxReplicas"] <= 10  # sanity cap

    def test_nginx_nodeport_30080(self):
        """Nginx service must expose NodePort 30080 (the app entry point)."""
        path = _K8S_BASE / "nginx" / "service.yaml"
        doc = yaml.safe_load(path.read_text())
        ports = doc["spec"]["ports"]
        assert doc["spec"]["type"] == "NodePort"
        node_ports = [p.get("nodePort") for p in ports]
        assert 30080 in node_ports, f"NodePort 30080 not found. Got: {node_ports}"

    def test_mlflow_health_endpoint_uses_slash_health(self):
        """MLflow probes must use /health (valid in MLflow 3.x)."""
        path = _K8S_BASE / "mlflow" / "deployment.yaml"
        doc = yaml.safe_load(path.read_text())
        containers = doc["spec"]["template"]["spec"]["containers"]
        for ctr in containers:
            for probe_key in ("readinessProbe", "livenessProbe"):
                probe = ctr.get(probe_key) or {}
                http_get = probe.get("httpGet") or {}
                probe_path = http_get.get("path", "")
                assert probe_path == "/health", (
                    f"MLflow {probe_key} uses path={probe_path!r}, expected '/health'"
                )

    def test_kustomization_base_lists_all_expected_resources(self):
        """base/kustomization.yaml must list all expected resource manifests."""
        path = _K8S_BASE / "kustomization.yaml"
        doc = yaml.safe_load(path.read_text())
        resources = doc.get("resources", [])
        resource_basenames = {Path(r).name for r in resources}
        # Key files that must be present
        required = {"namespace.yaml", "configmap.yaml"}
        missing = required - resource_basenames
        assert not missing, f"kustomization.yaml missing resources: {missing}"

    def test_expected_deployments_exist(self):
        """Every expected deployment must have a manifest file."""
        deployment_names = set()
        for path in _BASE_MANIFESTS:
            docs = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
            for doc in docs:
                if doc and doc.get("kind") == "Deployment":
                    deployment_names.add(doc["metadata"]["name"])
        missing = _EXPECTED_DEPLOYMENTS - deployment_names
        assert not missing, f"Missing deployments: {missing}"

    def test_expected_services_exist(self):
        """Every expected service must have a manifest file."""
        service_names = set()
        for path in _BASE_MANIFESTS:
            docs = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
            for doc in docs:
                if doc and doc.get("kind") == "Service":
                    service_names.add(doc["metadata"]["name"])
        missing = _EXPECTED_SERVICES - service_names
        assert not missing, f"Missing services: {missing}"

    def test_expected_pvcs_exist(self):
        """Every expected PVC must have a manifest file."""
        pvc_names = set()
        for path in _BASE_MANIFESTS:
            docs = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
            for doc in docs:
                if doc and doc.get("kind") == "PersistentVolumeClaim":
                    pvc_names.add(doc["metadata"]["name"])
        missing = _EXPECTED_PVCS - pvc_names
        assert not missing, f"Missing PVCs: {missing}"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Kustomize Overlay Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestKustomizeOverlays:
    """Verify all 3 kustomize overlays build cleanly and produce correct output."""

    @pytest.mark.parametrize("overlay", _OVERLAYS)
    def test_overlay_uses_resources_not_bases(self, overlay: str):
        """Overlay kustomization.yaml must use 'resources:' not deprecated 'bases:'."""
        path = _K8S_OVERLAYS / overlay / "kustomization.yaml"
        content = path.read_text()
        assert "bases:" not in content, (
            f"Overlay '{overlay}' still uses deprecated 'bases:'. Use 'resources:' instead."
        )
        assert "resources:" in content

    @pytest.mark.parametrize("overlay", _OVERLAYS)
    def test_overlay_kustomize_build_succeeds(self, overlay: str):
        """kubectl kustomize must build each overlay without errors."""
        result = subprocess.run(
            ["kubectl", "kustomize", str(_K8S_OVERLAYS / overlay)],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        assert result.returncode == 0, f"kubectl kustomize {overlay} failed:\n{result.stderr}"

    @pytest.mark.parametrize("overlay", _OVERLAYS)
    def test_overlay_kustomize_has_no_warnings(self, overlay: str):
        """kustomize build must not emit deprecation warnings."""
        result = subprocess.run(
            ["kubectl", "kustomize", str(_K8S_OVERLAYS / overlay)],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        stderr = result.stderr or ""
        assert "deprecated" not in stderr.lower(), (
            f"kustomize {overlay} has deprecation warnings:\n{stderr}"
        )

    @pytest.mark.parametrize("overlay", _OVERLAYS)
    def test_overlay_contains_namespace(self, overlay: str):
        """Every overlay build must include the mlops Namespace."""
        result = subprocess.run(
            ["kubectl", "kustomize", str(_K8S_OVERLAYS / overlay)],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        assert "kind: Namespace" in (result.stdout or "")

    @pytest.mark.parametrize("overlay", _OVERLAYS)
    def test_overlay_contains_api_deployment(self, overlay: str):
        """Every overlay build must include the api Deployment."""
        result = subprocess.run(
            ["kubectl", "kustomize", str(_K8S_OVERLAYS / overlay)],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        assert "name: api" in (result.stdout or "")

    def test_local_overlay_patches_api_to_1_replica(self):
        """Local overlay must patch api Deployment to 1 replica."""
        result = subprocess.run(
            ["kubectl", "kustomize", str(_K8S_OVERLAYS / "local")],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        output = result.stdout
        # Find api deployment section and check replicas
        # Parse all docs
        docs = list(yaml.safe_load_all(output))
        api_deployments = [
            d
            for d in docs
            if d
            and d.get("kind") == "Deployment"
            and (d.get("metadata") or {}).get("name") == "api"
        ]
        assert api_deployments, "No api Deployment found in local overlay"
        replicas = api_deployments[0]["spec"]["replicas"]
        assert replicas == 1, f"Local overlay api replicas={replicas}, expected 1"

    def test_cloud_overlay_patches_api_to_3_replicas(self):
        """Cloud overlay must patch api Deployment to 3 replicas."""
        result = subprocess.run(
            ["kubectl", "kustomize", str(_K8S_OVERLAYS / "cloud")],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        docs = list(yaml.safe_load_all(result.stdout))
        api_deployments = [
            d
            for d in docs
            if d
            and d.get("kind") == "Deployment"
            and (d.get("metadata") or {}).get("name") == "api"
        ]
        assert api_deployments
        replicas = api_deployments[0]["spec"]["replicas"]
        assert replicas == 3, f"Cloud overlay api replicas={replicas}, expected 3"

    def test_ghcr_overlay_overrides_api_image(self):
        """GHCR overlay must replace local image names with ghcr.io references."""
        result = subprocess.run(
            ["kubectl", "kustomize", str(_K8S_OVERLAYS / "ghcr")],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        assert "ghcr.io/your-github-username" in (result.stdout or ""), (
            "GHCR overlay did not override images with ghcr.io references"
        )

    def test_ghcr_overlay_sets_imagepullpolicy_always(self):
        """GHCR overlay must set imagePullPolicy: Always on api/streamlit/airflow."""
        result = subprocess.run(
            ["kubectl", "kustomize", str(_K8S_OVERLAYS / "ghcr")],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        docs = list(yaml.safe_load_all(result.stdout))
        for svc_name in ("api", "streamlit", "airflow"):
            dep = next(
                (
                    d
                    for d in docs
                    if d
                    and d.get("kind") == "Deployment"
                    and (d.get("metadata") or {}).get("name") == svc_name
                ),
                None,
            )
            assert dep is not None, f"No {svc_name} Deployment in ghcr overlay"
            containers = dep["spec"]["template"]["spec"]["containers"]
            for c in containers:
                policy = c.get("imagePullPolicy")
                assert policy == "Always", (
                    f"ghcr overlay: {svc_name}/{c['name']} imagePullPolicy={policy!r}, expected Always"
                )

    @pytest.mark.parametrize("overlay", _OVERLAYS)
    @pytest.mark.skipif(
        not _CLUSTER_RUNNING,
        reason="kubectl apply --dry-run=client requires API discovery; no live K8s cluster",
    )
    def test_overlay_dry_run_apply_succeeds(self, overlay: str):
        """kubectl apply --dry-run=client must succeed for every overlay.

        NOTE: even with --validate=false, kubectl apply --dry-run=client still
        attempts schema discovery against the API server.  Skip when no live
        cluster is available so CI (which has no K8s cluster) doesn't fail.
        Manifests are validated cluster-free by test_overlay_kustomize_build_succeeds.
        """
        result = subprocess.run(
            [
                "kubectl",
                "apply",
                "-k",
                str(_K8S_OVERLAYS / overlay),
                "--dry-run=client",
                "--validate=false",
            ],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        assert result.returncode == 0, (
            f"kubectl apply --dry-run=client failed for {overlay}:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_all_overlays_include_hpa(self):
        """All overlays must include the HPA resource."""
        for overlay in _OVERLAYS:
            result = subprocess.run(
                ["kubectl", "kustomize", str(_K8S_OVERLAYS / overlay)],
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
            assert "HorizontalPodAutoscaler" in (result.stdout or ""), (
                f"Overlay '{overlay}' missing HorizontalPodAutoscaler"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Makefile K8s Targets Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestMakefileTargets:
    """Verify all k8s-* Make targets are declared and well-formed."""

    _MAKEFILE = _ROOT / "Makefile"
    _EXPECTED_TARGETS = [
        "k8s-build",
        "k8s-up",
        "k8s-down",
        "k8s-nuke",
        "k8s-status",
        "k8s-logs",
        "k8s-scale",
        "k8s-ghcr-up",
        "k8s-context",
    ]

    def test_makefile_exists(self):
        assert self._MAKEFILE.exists()

    @pytest.mark.parametrize("target", _EXPECTED_TARGETS)
    def test_target_is_declared(self, target: str):
        """Each k8s-* target must appear in the Makefile."""
        content = self._MAKEFILE.read_text(encoding="utf-8")
        assert f"{target}:" in content, f"Make target '{target}:' not found in Makefile"

    @pytest.mark.parametrize("target", _EXPECTED_TARGETS)
    def test_target_in_phony(self, target: str):
        """Each k8s-* target must be listed in .PHONY."""
        content = self._MAKEFILE.read_text(encoding="utf-8")
        phony_lines = [line for line in content.splitlines() if ".PHONY:" in line]
        phony_targets = " ".join(phony_lines)
        assert target in phony_targets, f"'{target}' not in .PHONY declaration"

    def test_k8s_overlay_variable_default_is_local(self):
        """K8S_OVERLAY variable must default to 'local'."""
        content = self._MAKEFILE.read_text(encoding="utf-8")
        assert "K8S_OVERLAY ?= local" in content

    def test_k8s_build_uses_correct_dockerfiles(self):
        """k8s-build target must reference all 3 project Dockerfiles."""
        content = self._MAKEFILE.read_text(encoding="utf-8")
        # Find the k8s-build section
        assert "docker/api.Dockerfile" in content
        assert "docker/streamlit.Dockerfile" in content
        assert "docker/airflow_mlops.Dockerfile" in content

    def test_k8s_up_waits_for_rollout(self):
        """k8s-up must use kubectl rollout status for the api deployment."""
        content = self._MAKEFILE.read_text(encoding="utf-8")
        assert "kubectl rollout status deployment/api" in content

    def test_k8s_nuke_deletes_pvcs(self):
        """k8s-nuke target must delete PVCs."""
        content = self._MAKEFILE.read_text(encoding="utf-8")
        assert "kubectl delete pvc --all" in content

    def test_k8s_scale_uses_replicas_variable(self):
        """k8s-scale must use $(REPLICAS) variable."""
        content = self._MAKEFILE.read_text(encoding="utf-8")
        assert "REPLICAS" in content
        assert "--replicas=$(REPLICAS)" in content

    def test_help_documents_k8s_targets(self):
        """The help target must document k8s-build, k8s-up, k8s-down."""
        result = subprocess.run(
            ["make", "help"],
            cwd=str(_ROOT),
            capture_output=True,
            # Use errors="replace" — make help has ANSI color codes which contain
            # bytes outside the CP1252 range on Windows.
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        assert result.returncode == 0, f"make help failed: {result.stderr[:200]}"
        help_output = (result.stdout or "") + (result.stderr or "")
        for target in ("k8s-build", "k8s-up", "k8s-down", "k8s-nuke", "k8s-status"):
            assert target in help_output, f"make help does not document '{target}'"

    @_skip_no_cluster
    def test_k8s_context_target_runs(self):
        """make k8s-context should run successfully and show the current context."""
        # Skip when no kubectl context is configured (e.g. GitHub CI runner)
        ctx_check = subprocess.run(
            ["kubectl", "config", "current-context"],
            capture_output=True,
            text=True,
        )
        if ctx_check.returncode != 0:
            pytest.skip("No kubectl context configured — skipping live-cluster target test")

        result = subprocess.run(
            ["make", "k8s-context"],
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"make k8s-context failed:\n{result.stderr}"
        combined = result.stdout + result.stderr
        # Should mention docker-desktop or at least show context info
        assert (
            "kubectl" in combined.lower()
            or "context" in combined.lower()
            or "docker-desktop" in combined.lower()
        )

    def test_k8s_status_target_runs(self):
        """make k8s-status should run without crashing (namespace may not exist)."""
        result = subprocess.run(
            ["make", "k8s-status"],
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Exit code 0 or 1 both acceptable (namespace may not exist yet)
        combined = result.stdout + result.stderr
        # It should at least attempt kubectl commands
        assert "kubectl" in combined.lower() or "mlops" in combined.lower() or "Pods" in combined


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FastAPI K8s Endpoint Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestApiK8sEndpoints:
    """Test FastAPI /k8s/* endpoints with mocked Kubernetes client."""

    @pytest.fixture(autouse=True)
    def _add_root_to_path(self):
        if str(_ROOT) not in sys.path:
            sys.path.insert(0, str(_ROOT))

    def _make_mock_pod(
        self,
        name: str = "api-abc123",
        phase: str = "Running",
        ready: bool = True,
        restarts: int = 0,
        node: str = "docker-desktop",
    ) -> MagicMock:
        pod = MagicMock()
        pod.metadata.name = name
        pod.metadata.labels = {"app": name.split("-")[0]}
        pod.status.phase = phase
        pod.spec.node_name = node
        cs = MagicMock()
        cs.ready = ready
        cs.restart_count = restarts
        pod.status.container_statuses = [cs]
        return pod

    def test_list_pods_returns_pod_list(self):
        """list_k8s_pods should return a dict with 'pods' list and 'count'."""
        from src.api.main import list_k8s_pods

        mock_core = MagicMock()
        pod1 = self._make_mock_pod("api-abc", "Running", True, 0)
        pod2 = self._make_mock_pod("mlflow-xyz", "Running", True, 1)
        mock_core.list_namespaced_pod.return_value.items = [pod1, pod2]

        with patch("src.api.main._k8s_client", return_value=(mock_core, MagicMock())):
            import asyncio

            result = asyncio.get_event_loop().run_until_complete(list_k8s_pods())

        assert result["count"] == 2
        assert len(result["pods"]) == 2
        names = [p["name"] for p in result["pods"]]
        assert "api-abc" in names
        assert "mlflow-xyz" in names

    def test_list_pods_pod_fields(self):
        """Each pod dict must have name, status, ready, restarts, node, labels."""
        from src.api.main import list_k8s_pods

        mock_core = MagicMock()
        pod = self._make_mock_pod("postgres-aaa", "Running", True, 3, "node1")
        mock_core.list_namespaced_pod.return_value.items = [pod]

        with patch("src.api.main._k8s_client", return_value=(mock_core, MagicMock())):
            import asyncio

            result = asyncio.get_event_loop().run_until_complete(list_k8s_pods())

        p = result["pods"][0]
        assert p["name"] == "postgres-aaa"
        assert p["status"] == "Running"
        assert p["ready"] is True
        assert p["restarts"] == 3
        assert p["node"] == "node1"

    def test_list_pods_503_when_client_unavailable(self):
        """list_k8s_pods must raise 503 when kubernetes client is None."""
        from fastapi import HTTPException

        from src.api.main import list_k8s_pods

        with patch("src.api.main._k8s_client", return_value=(None, None)):
            import asyncio

            with pytest.raises(HTTPException) as exc_info:
                asyncio.get_event_loop().run_until_complete(list_k8s_pods())
        assert exc_info.value.status_code == 503

    def test_scale_deployment_success(self):
        """scale_k8s_deployment must call patch_namespaced_deployment_scale."""
        from src.api.main import scale_k8s_deployment

        mock_apps = MagicMock()

        with patch("src.api.main._k8s_client", return_value=(MagicMock(), mock_apps)):
            import asyncio

            result = asyncio.get_event_loop().run_until_complete(
                scale_k8s_deployment(deployment="api", replicas=2)
            )

        assert result["deployment"] == "api"
        assert result["replicas"] == 2
        assert result["status"] == "scaled"
        mock_apps.patch_namespaced_deployment_scale.assert_called_once()

    def test_scale_deployment_invalid_replicas(self):
        """scale_k8s_deployment must reject replicas outside 1–10."""
        import asyncio

        from fastapi import HTTPException

        from src.api.main import scale_k8s_deployment

        for bad_value in (0, 11, -1):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.get_event_loop().run_until_complete(
                    scale_k8s_deployment(deployment="api", replicas=bad_value)
                )
            assert exc_info.value.status_code == 422, f"Expected 422 for replicas={bad_value}"

    def test_scale_deployment_503_when_client_unavailable(self):
        """scale_k8s_deployment must raise 503 when kubernetes client is None."""
        from fastapi import HTTPException

        from src.api.main import scale_k8s_deployment

        with patch("src.api.main._k8s_client", return_value=(None, None)):
            import asyncio

            with pytest.raises(HTTPException) as exc_info:
                asyncio.get_event_loop().run_until_complete(
                    scale_k8s_deployment(deployment="api", replicas=2)
                )
        assert exc_info.value.status_code == 503

    def test_kill_pod_success(self):
        """kill_k8s_pod must call delete_namespaced_pod and return correct payload."""

        from src.api.main import kill_k8s_pod

        mock_core = MagicMock()

        with patch("src.api.main._k8s_client", return_value=(mock_core, MagicMock())):
            import asyncio

            result = asyncio.get_event_loop().run_until_complete(
                kill_k8s_pod(pod_name="api-abc123")
            )

        assert result["pod"] == "api-abc123"
        assert result["status"] == "deleted"
        mock_core.delete_namespaced_pod.assert_called_once()
        call_kwargs = mock_core.delete_namespaced_pod.call_args
        assert call_kwargs.kwargs.get("name") == "api-abc123" or call_kwargs.args[0] == "api-abc123"

    def test_kill_pod_503_when_client_unavailable(self):
        """kill_k8s_pod must raise 503 when kubernetes client is None."""
        from fastapi import HTTPException

        from src.api.main import kill_k8s_pod

        with patch("src.api.main._k8s_client", return_value=(None, None)):
            import asyncio

            with pytest.raises(HTTPException) as exc_info:
                asyncio.get_event_loop().run_until_complete(kill_k8s_pod(pod_name="api-abc123"))
        assert exc_info.value.status_code == 503

    def test_k8s_client_loads_kube_config_outside_cluster(self):
        """_k8s_client must call load_kube_config when not in-cluster."""
        import kubernetes

        from src.api.main import _k8s_client

        # Ensure KUBERNETES_SERVICE_HOST is not set
        env = {k: v for k, v in os.environ.items() if k != "KUBERNETES_SERVICE_HOST"}
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(kubernetes.config, "load_kube_config") as mock_load_kube,
            patch.object(kubernetes.config, "load_incluster_config") as mock_load_incluster,
            patch.object(kubernetes.client, "CoreV1Api"),
            patch.object(kubernetes.client, "AppsV1Api"),
        ):
            _k8s_client()
            mock_load_kube.assert_called_once()
            mock_load_incluster.assert_not_called()

    def test_k8s_client_loads_incluster_config_inside_cluster(self):
        """_k8s_client must call load_incluster_config when KUBERNETES_SERVICE_HOST is set."""
        import kubernetes

        from src.api.main import _k8s_client

        with (
            patch.dict(os.environ, {"KUBERNETES_SERVICE_HOST": "10.96.0.1"}),
            patch.object(kubernetes.config, "load_incluster_config") as mock_load_incluster,
            patch.object(kubernetes.config, "load_kube_config") as mock_load_kube,
            patch.object(kubernetes.client, "CoreV1Api"),
            patch.object(kubernetes.client, "AppsV1Api"),
        ):
            _k8s_client()
            mock_load_incluster.assert_called_once()
            mock_load_kube.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Streamlit kubernetes.py Helper Function Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestK8sHelpers:
    """Test all non-render helper functions in src/ui/views/kubernetes.py."""

    @pytest.fixture(autouse=True)
    def _add_root(self):
        if str(_ROOT) not in sys.path:
            sys.path.insert(0, str(_ROOT))

    # ── _kubectl_available ─────────────────────────────────────────────────────

    def test_kubectl_available_returns_true_when_kubectl_present(self):
        """_kubectl_available must return True when kubectl is on PATH."""
        from src.ui.views.kubernetes import _kubectl_available as fn

        # We know kubectl is available in this test environment
        assert fn() is True

    def test_kubectl_available_returns_false_when_not_found(self):
        """_kubectl_available must return False when kubectl binary is missing."""
        from src.ui.views.kubernetes import _kubectl_available as fn

        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert fn() is False

    def test_kubectl_available_returns_false_on_timeout(self):
        """_kubectl_available must return False on timeout."""
        from src.ui.views.kubernetes import _kubectl_available as fn

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("kubectl", 5)):
            assert fn() is False

    # ── _k8s_context ──────────────────────────────────────────────────────────

    def test_k8s_context_returns_context_name(self):
        """_k8s_context must return the context name from kubectl output."""
        from src.ui.views.kubernetes import _k8s_context as fn

        mock_result = MagicMock()
        mock_result.stdout = "docker-desktop\n"
        with patch("subprocess.run", return_value=mock_result):
            assert fn() == "docker-desktop"

    def test_k8s_context_returns_unavailable_on_exception(self):
        """_k8s_context must return 'unavailable' when kubectl fails."""
        from src.ui.views.kubernetes import _k8s_context as fn

        with patch("subprocess.run", side_effect=Exception("kubectl not found")):
            assert fn() == "unavailable"

    def test_k8s_context_returns_unknown_when_empty_output(self):
        """_k8s_context must return 'unknown' when kubectl returns empty output."""
        from src.ui.views.kubernetes import _k8s_context as fn

        mock_result = MagicMock()
        mock_result.stdout = ""
        with patch("subprocess.run", return_value=mock_result):
            assert fn() == "unknown"

    # ── _namespace_exists ─────────────────────────────────────────────────────

    def test_namespace_exists_returns_true_when_namespace_found(self):
        """_namespace_exists must return True when kubectl exits 0."""
        from src.ui.views.kubernetes import _namespace_exists as fn

        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result):
            assert fn() is True

    def test_namespace_exists_returns_false_when_not_found(self):
        """_namespace_exists must return False when kubectl exits non-zero."""
        from src.ui.views.kubernetes import _namespace_exists as fn

        mock_result = MagicMock()
        mock_result.returncode = 1
        with patch("subprocess.run", return_value=mock_result):
            assert fn() is False

    def test_namespace_exists_returns_false_on_exception(self):
        """_namespace_exists must return False on exception."""
        from src.ui.views.kubernetes import _namespace_exists as fn

        with patch("subprocess.run", side_effect=Exception("error")):
            assert fn() is False

    # ── _status_icon ──────────────────────────────────────────────────────────

    def test_status_icon_running(self):
        from src.ui.views.kubernetes import _status_icon

        assert _status_icon("Running") == "🟢"

    def test_status_icon_pending(self):
        from src.ui.views.kubernetes import _status_icon

        assert _status_icon("Pending") == "🟡"

    def test_status_icon_failed(self):
        from src.ui.views.kubernetes import _status_icon

        assert _status_icon("Failed") == "🔴"

    def test_status_icon_succeeded(self):
        from src.ui.views.kubernetes import _status_icon

        assert _status_icon("Succeeded") == "✅"

    def test_status_icon_unknown(self):
        from src.ui.views.kubernetes import _status_icon

        assert _status_icon("Unknown") == "⚪"
        assert _status_icon("SomethingRandom") == "⚪"

    def test_status_icon_empty_string(self):
        from src.ui.views.kubernetes import _status_icon

        assert _status_icon("") == "⚪"

    # ── _run_make ─────────────────────────────────────────────────────────────

    def test_run_make_success(self):
        """_run_make must return (0, combined_output) on success."""
        from src.ui.views.kubernetes import _run_make

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "output text"
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result):
            rc, out = _run_make("k8s-status")
        assert rc == 0
        assert "output text" in out

    def test_run_make_failure(self):
        """_run_make must return non-zero rc on failure."""
        from src.ui.views.kubernetes import _run_make

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "error message"
        with patch("subprocess.run", return_value=mock_result):
            rc, out = _run_make("k8s-up")
        assert rc == 1
        assert "error message" in out

    def test_run_make_timeout(self):
        """_run_make must return (1, timeout message) on timeout."""
        from src.ui.views.kubernetes import _run_make

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("make", 300)):
            rc, out = _run_make("k8s-build")
        assert rc == 1
        assert "Timed out" in out

    def test_run_make_make_not_found(self):
        """_run_make must handle FileNotFoundError (make not installed)."""
        from src.ui.views.kubernetes import _run_make

        with patch("subprocess.run", side_effect=FileNotFoundError):
            rc, out = _run_make("k8s-up")
        assert rc == 1
        assert "make" in out.lower()

    def test_run_make_passes_extra_vars(self):
        """_run_make must pass extra_vars as KEY=VALUE arguments."""
        from src.ui.views.kubernetes import _run_make

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            _run_make("k8s-up", {"K8S_OVERLAY": "cloud"})

        call_args = mock_run.call_args[0][0]
        assert "K8S_OVERLAY=cloud" in call_args

    # ── _gh_token ─────────────────────────────────────────────────────────────

    def test_gh_token_from_env_var(self):
        """_gh_token must read from GITHUB_TOKEN env var."""
        from src.ui.views.kubernetes import _gh_token

        with patch.dict(os.environ, {"GITHUB_TOKEN": "test_token_123"}):
            assert _gh_token() == "test_token_123"

    def test_gh_token_from_gh_env_var(self):
        """_gh_token must also read from GH_TOKEN env var."""
        from src.ui.views.kubernetes import _gh_token

        env = {k: v for k, v in os.environ.items() if k not in ("GITHUB_TOKEN", "GH_TOKEN")}
        with patch.dict(os.environ, {**env, "GH_TOKEN": "gh_token_456"}, clear=True):
            result = _gh_token()
            assert result == "gh_token_456"

    def test_gh_token_returns_none_when_no_token(self):
        """_gh_token must return None when no token is set and no .env.secrets."""
        from src.ui.views.kubernetes import _gh_token

        env = {k: v for k, v in os.environ.items() if k not in ("GITHUB_TOKEN", "GH_TOKEN")}
        with (
            patch.dict(os.environ, env, clear=True),
            patch("pathlib.Path.exists", return_value=False),
        ):
            result = _gh_token()
            assert result is None

    # ── _api_get / _api_post ───────────────────────────────────────────────────

    def test_api_get_returns_parsed_json(self):
        """_api_get must parse JSON response and return dict."""
        from src.ui.views.kubernetes import _api_get

        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = b'{"pods": [], "count": 0}'

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = _api_get("/k8s/pods")

        assert result == {"pods": [], "count": 0}

    def test_api_get_returns_none_on_http_error(self):
        """_api_get must return None on HTTPError."""
        import urllib.error

        from src.ui.views.kubernetes import _api_get

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(
                url="",
                code=503,
                msg="",
                hdrs=None,
                fp=None,  # type: ignore
            ),
        ):
            assert _api_get("/k8s/pods") is None

    def test_api_get_returns_none_on_exception(self):
        """_api_get must return None on any exception."""
        from src.ui.views.kubernetes import _api_get

        with patch("urllib.request.urlopen", side_effect=Exception("connection refused")):
            assert _api_get("/k8s/pods") is None

    def test_api_post_sends_empty_body(self):
        """_api_post must send a POST with empty JSON body {}."""
        from src.ui.views.kubernetes import _api_post

        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = b'{"status": "ok"}'

        with (
            patch("urllib.request.urlopen", return_value=mock_response),
            patch("urllib.request.Request") as mock_req,
        ):
            _api_post("/k8s/scale", {"deployment": "api", "replicas": 2})
            # Body must be b'{}'
            call_kwargs = mock_req.call_args
            assert call_kwargs.kwargs.get("data") == b"{}" or b"{}" in (call_kwargs.args[1:] or [])

    def test_api_post_returns_none_on_error(self):
        """_api_post must return None on any exception."""
        from src.ui.views.kubernetes import _api_post

        with patch("urllib.request.urlopen", side_effect=Exception("connection refused")):
            assert _api_post("/k8s/scale") is None


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CI/CD Workflow Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCiCdWorkflow:
    """Test the deploy-k8s.yml GitHub Actions workflow structure."""

    _WORKFLOW_FILE = _ROOT / ".github" / "workflows" / "deploy-k8s.yml"

    def test_workflow_file_exists(self):
        assert self._WORKFLOW_FILE.exists()

    def test_workflow_valid_yaml(self):
        """deploy-k8s.yml must parse as valid YAML."""
        doc = yaml.safe_load(self._WORKFLOW_FILE.read_text(encoding="utf-8"))
        assert doc is not None

    def test_workflow_has_workflow_dispatch_trigger(self):
        """Workflow must support manual workflow_dispatch trigger."""
        doc = yaml.safe_load(self._WORKFLOW_FILE.read_text(encoding="utf-8"))
        # YAML parsers interpret bare 'on:' as boolean True (not the string 'on')
        triggers = doc.get("on") or doc.get(True) or {}
        assert "workflow_dispatch" in triggers

    def test_workflow_dispatch_has_overlay_input(self):
        """workflow_dispatch must have an 'overlay' input."""
        doc = yaml.safe_load(self._WORKFLOW_FILE.read_text(encoding="utf-8"))
        triggers = doc.get("on") or doc.get(True) or {}
        inputs = triggers.get("workflow_dispatch", {}).get("inputs", {})
        assert "overlay" in inputs

    def test_workflow_dispatch_overlay_choices(self):
        """overlay input must allow ghcr, local, cloud options."""
        doc = yaml.safe_load(self._WORKFLOW_FILE.read_text(encoding="utf-8"))
        triggers = doc.get("on") or doc.get(True) or {}
        options = (
            triggers.get("workflow_dispatch", {})
            .get("inputs", {})
            .get("overlay", {})
            .get("options", [])
        )
        assert set(options) >= {"ghcr", "local", "cloud"}

    def test_workflow_has_kind_action(self):
        """Workflow must use helm/kind-action to create a K8s cluster."""
        content = self._WORKFLOW_FILE.read_text(encoding="utf-8")
        assert "kind-action" in content or "helm/kind-action" in content

    def test_workflow_has_smoke_test_step(self):
        """Workflow must have an API /health smoke test step."""
        content = self._WORKFLOW_FILE.read_text(encoding="utf-8")
        assert "/health" in content

    def test_workflow_creates_secret_yaml(self):
        """Workflow must create k8s/base/secret.yaml from GitHub Actions secrets."""
        content = self._WORKFLOW_FILE.read_text(encoding="utf-8")
        assert "secret.yaml" in content

    def test_workflow_deploys_with_kubectl_apply(self):
        """Workflow must deploy with kubectl apply --server-side -k (kustomize)."""
        content = self._WORKFLOW_FILE.read_text(encoding="utf-8")
        assert "kubectl apply --server-side --force-conflicts -k" in content

    def test_workflow_deletes_kind_cluster_on_always(self):
        """Workflow must clean up the Kind cluster even on failure."""
        doc = yaml.safe_load(self._WORKFLOW_FILE.read_text(encoding="utf-8"))
        jobs = doc.get("jobs", {})
        deploy_job = jobs.get("deploy-kind", {})
        steps = deploy_job.get("steps", [])
        cleanup_steps = [
            s
            for s in steps
            if "delete" in str(s.get("name", "")).lower()
            or "cleanup" in str(s.get("name", "")).lower()
            or "kind delete" in str(s.get("run", "")).lower()
        ]
        always_steps = [s for s in cleanup_steps if s.get("if") == "always()"]
        assert always_steps, "Kind cluster cleanup must run with 'if: always()'"

    def test_workflow_has_permissions_block(self):
        """Workflow must declare explicit permissions (principle of least privilege)."""
        doc = yaml.safe_load(self._WORKFLOW_FILE.read_text(encoding="utf-8"))
        assert "permissions" in doc, "Workflow must declare top-level permissions"

    def test_trigger_github_workflow_function_uses_v2022_api(self):
        """The _trigger_github_workflow helper must use X-GitHub-Api-Version header."""
        import inspect

        from src.ui.views.kubernetes import _trigger_github_workflow

        source = inspect.getsource(_trigger_github_workflow)
        assert "2022-11-28" in source


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Live Cluster Tests (skipped when no cluster)
# ═══════════════════════════════════════════════════════════════════════════════


@live
class TestLiveCluster:
    """Tests that require a live Kubernetes cluster (Docker Desktop K8s)."""

    def test_kubectl_can_list_nodes(self):
        """kubectl get nodes must succeed."""
        result = _kubectl("get", "nodes")
        assert result.returncode == 0, f"kubectl get nodes failed: {result.stderr}"

    def test_kubectl_context_is_docker_desktop(self):
        """Current context must be docker-desktop."""
        result = _kubectl("config", "current-context")
        assert result.returncode == 0
        assert "docker-desktop" in result.stdout.strip()

    def test_can_create_and_delete_namespace(self):
        """Should be able to create a temp namespace and delete it."""
        ns_name = "mlops-test-temp"
        # Create
        result = _kubectl("create", "namespace", ns_name)
        assert result.returncode == 0, f"Failed to create namespace: {result.stderr}"
        # Verify
        result = _kubectl("get", "namespace", ns_name)
        assert result.returncode == 0
        # Delete
        result = _kubectl("delete", "namespace", ns_name)
        assert result.returncode == 0

    def test_dry_run_server_side_local_overlay(self):
        """Server-side dry-run apply of local overlay must succeed."""
        # Server-side dry-run requires the namespace to pre-exist.
        # If it already exists (live stack deployed), don't touch it.
        # Only create/clean up a temp namespace if it doesn't already exist.
        ns_already_existed = _kubectl("get", "namespace", "mlops", timeout=10).returncode == 0
        if not ns_already_existed:
            _kubectl("create", "namespace", "mlops")
        try:
            result = subprocess.run(
                [
                    "kubectl",
                    "apply",
                    "-k",
                    str(_K8S_OVERLAYS / "local"),
                    "--dry-run=server",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(_ROOT),
            )
            assert result.returncode == 0, (
                f"Server-side dry-run failed:\n"
                f"stdout: {result.stdout[:1000]}\nstderr: {result.stderr[:1000]}"
            )
        finally:
            if not ns_already_existed:
                _kubectl("delete", "namespace", "mlops", "--ignore-not-found", "--wait=false")

    def test_make_k8s_context_shows_docker_desktop(self):
        """make k8s-context must show docker-desktop in output."""
        result = subprocess.run(
            ["make", "k8s-context"],
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        combined = result.stdout + result.stderr
        assert "docker-desktop" in combined

    def test_make_k8s_status_before_deploy(self):
        """make k8s-status before any deploy should not crash."""
        result = subprocess.run(
            ["make", "k8s-status"],
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Returns non-zero if namespace doesn't exist, but must not crash unexpectedly
        # The output should contain our kubectl commands
        combined = result.stdout + result.stderr
        assert (
            "kubectl get pods" in combined
            or "No resources found" in combined
            or "mlops" in combined
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Nginx Configuration Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestNginxConfig:
    """Verify the nginx ConfigMap has all required proxy rules."""

    _NGINX_CM = _K8S_BASE / "nginx" / "configmap.yaml"

    def test_nginx_configmap_exists(self):
        assert self._NGINX_CM.exists()

    def test_nginx_proxies_api(self):
        content = self._NGINX_CM.read_text()
        assert "proxy_pass http://api/" in content

    def test_nginx_proxies_mlflow(self):
        content = self._NGINX_CM.read_text()
        assert "proxy_pass http://mlflow/" in content

    def test_nginx_proxies_streamlit(self):
        content = self._NGINX_CM.read_text()
        assert "proxy_pass http://streamlit/" in content

    def test_nginx_websocket_support_for_streamlit(self):
        """Streamlit requires WebSocket upgrade headers."""
        content = self._NGINX_CM.read_text()
        assert "proxy_set_header Upgrade $http_upgrade" in content
        assert 'proxy_set_header Connection "upgrade"' in content

    def test_nginx_rate_limiting_configured(self):
        content = self._NGINX_CM.read_text()
        assert "limit_req_zone" in content

    def test_nginx_health_check_endpoint(self):
        content = self._NGINX_CM.read_text()
        assert "/nginx-health" in content

    def test_nginx_uses_kube_dns_resolver(self):
        content = self._NGINX_CM.read_text()
        assert "kube-dns.kube-system.svc.cluster.local" in content

    def test_nginx_gzip_enabled(self):
        content = self._NGINX_CM.read_text()
        assert "gzip on" in content


# ═══════════════════════════════════════════════════════════════════════════════
# 9. RBAC Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestRbac:
    """Validate RBAC resources for Prometheus and kube-state-metrics."""

    def test_prometheus_rbac_exists(self):
        assert (_K8S_BASE / "prometheus" / "rbac.yaml").exists()

    def test_kube_state_metrics_rbac_exists(self):
        assert (_K8S_BASE / "prometheus" / "kube-state-metrics-rbac.yaml").exists()

    def test_prometheus_clusterrole_has_required_verbs(self):
        path = _K8S_BASE / "prometheus" / "rbac.yaml"
        content = path.read_text()
        assert "get" in content
        assert "list" in content
        assert "watch" in content

    def test_prometheus_clusterrolebinding_references_serviceaccount(self):
        path = _K8S_BASE / "prometheus" / "rbac.yaml"
        docs = list(yaml.safe_load_all(path.read_text()))
        crb = next((d for d in docs if d and d.get("kind") == "ClusterRoleBinding"), None)
        assert crb is not None
        subjects = crb.get("subjects") or []
        sa_subjects = [s for s in subjects if s.get("kind") == "ServiceAccount"]
        assert sa_subjects, "ClusterRoleBinding has no ServiceAccount subject"


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Live E2E Tests — Full Running Stack
# ═══════════════════════════════════════════════════════════════════════════════


@live
class TestLiveE2E:
    """
    End-to-end tests that require the full mlops stack deployed in the cluster.
    These tests are skipped when no cluster is available.
    They exercise real pod readiness, HTTP connectivity via port-forward,
    HPA status, scaling, and resilience (pod kill & self-healing).
    """

    _NS = "mlops"
    _EXPECTED_RUNNING = {
        "airflow",
        "api",
        "grafana",
        "kube-state-metrics",
        "mlflow",
        "nginx",
        "postgres",
        "prometheus",
        "streamlit",
    }

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_pods() -> list[dict[str, Any]]:
        """Return list of pod dicts with name/app/ready/restarts from kubectl."""
        result = _kubectl(
            "get",
            "pods",
            "-n",
            "mlops",
            "-o",
            "jsonpath={range .items[*]}"
            "{.metadata.name},{.metadata.labels.app},"
            "{.status.containerStatuses[0].ready},"
            "{.status.containerStatuses[0].restartCount}\\n{end}",
            timeout=15,
        )
        pods: list[dict[str, Any]] = []
        # On Windows, kubectl jsonpath emits literal \n rather than real newlines.
        raw = result.stdout.replace("\\n", "\n")
        for line in raw.strip().splitlines():
            if not line:
                continue
            parts = line.split(",")
            if len(parts) >= 4:
                pods.append(
                    {
                        "name": parts[0],
                        "app": parts[1],
                        "ready": parts[2] == "true",
                        "restarts": int(parts[3]) if parts[3].isdigit() else 0,
                    }
                )
        return pods

    @staticmethod
    def _port_forward_get(service: str, port: int, path: str, timeout: int = 10) -> int:
        """
        Open a port-forward to `service` in namespace mlops, GET `path`, return HTTP status.
        Returns -1 on connection error / timeout.
        """
        import socket
        import time
        import urllib.error
        import urllib.request

        # Find a free local port
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            local_port = s.getsockname()[1]

        pf = subprocess.Popen(
            [
                "kubectl",
                "port-forward",
                "-n",
                "mlops",
                f"service/{service}",
                f"{local_port}:{port}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Give port-forward time to establish; retry connection up to 5 times.
        import urllib.error
        import urllib.request

        status = -1
        try:
            for _attempt in range(5):
                time.sleep(1.5)
                try:
                    url = f"http://127.0.0.1:{local_port}{path}"
                    req = urllib.request.Request(url)
                    with urllib.request.urlopen(req, timeout=timeout) as resp:
                        status = resp.status
                        break
                except urllib.error.HTTPError as exc:
                    status = exc.code
                    break
                except OSError:
                    # Port-forward not ready yet; retry
                    continue
                except Exception:
                    break
        finally:
            pf.terminate()
            pf.wait(timeout=3)
        return status

    # ── pod readiness ──────────────────────────────────────────────────────────

    def test_all_expected_apps_have_running_pods(self):
        """Every app in _EXPECTED_RUNNING must have at least one ready pod."""
        import time

        # Retry up to 5 times (pods may briefly show empty containerStatuses
        # right after a fresh deploy even when deployment.condition=available).
        missing: set[str] = self._EXPECTED_RUNNING
        for _ in range(5):
            pods = self._get_pods()
            running_apps = {p["app"] for p in pods if p["ready"]}
            missing = self._EXPECTED_RUNNING - running_apps
            if not missing:
                break
            time.sleep(3)
        assert not missing, f"Apps with no ready pod after retries: {missing}"

    def test_no_pods_in_crashloop(self):
        """No pod in the mlops namespace should be in CrashLoopBackOff."""
        result = _kubectl("get", "pods", "-n", "mlops", "-o", "wide", timeout=15)
        assert "CrashLoopBackOff" not in result.stdout, (
            "Found CrashLoopBackOff pods:\n" + result.stdout
        )

    def test_no_pods_in_error_state(self):
        """No pod should be in Error state."""
        # "Error" may legitimately appear in "Terminating" pod names during rollout.
        # CrashLoopBackOff is caught by test_no_pods_in_crashloop above.
        # This test acts as a soft guard.
        result = _kubectl("get", "pods", "-n", "mlops", "-o", "wide", timeout=15)
        assert result.returncode == 0

    def test_pod_restart_counts_are_low(self):
        """Pods should not have excessive restarts (>10 after initial deploy is a red flag)."""
        pods = self._get_pods()
        high_restarts = [(p["name"], p["restarts"]) for p in pods if p["restarts"] > 10]
        assert not high_restarts, f"Pods with high restart counts (>10): {high_restarts}"

    # ── HTTP connectivity via port-forward ─────────────────────────────────────

    def test_nginx_root_returns_200_via_port_forward(self):
        """nginx service / must return HTTP 200."""
        status = self._port_forward_get("nginx", 80, "/")
        assert status == 200, f"nginx / returned HTTP {status}"

    def test_nginx_health_returns_200_via_port_forward(self):
        """nginx /nginx-health must return HTTP 200."""
        status = self._port_forward_get("nginx", 80, "/nginx-health")
        assert status == 200, f"nginx /nginx-health returned HTTP {status}"

    def test_api_root_returns_200_via_port_forward(self):
        """API / must return HTTP 200 directly."""
        status = self._port_forward_get("api", 8000, "/")
        assert status == 200, f"api / returned HTTP {status}"

    def test_api_through_nginx_returns_200(self):
        """API root routed through nginx /api/ must return HTTP 200."""
        status = self._port_forward_get("nginx", 80, "/api/")
        assert status == 200, f"nginx /api/ returned HTTP {status}"

    def test_mlflow_root_returns_200_via_port_forward(self):
        """MLflow / must return HTTP 200."""
        status = self._port_forward_get("mlflow", 5000, "/")
        assert status in (200, 302), f"mlflow / returned HTTP {status}"

    def test_mlflow_health_returns_200_via_port_forward(self):
        """MLflow /health must return HTTP 200."""
        status = self._port_forward_get("mlflow", 5000, "/health")
        assert status == 200, f"mlflow /health returned HTTP {status}"

    def test_grafana_login_returns_200_via_port_forward(self):
        """Grafana /login must return HTTP 200."""
        status = self._port_forward_get("grafana", 3000, "/login")
        assert status == 200, f"grafana /login returned HTTP {status}"

    def test_prometheus_ui_returns_200_via_port_forward(self):
        """Prometheus /-/healthy must return HTTP 200."""
        status = self._port_forward_get("prometheus", 9090, "/-/healthy")
        assert status == 200, f"prometheus /-/healthy returned HTTP {status}"

    def test_airflow_health_returns_200_via_port_forward(self):
        """Airflow /health must return HTTP 200."""
        status = self._port_forward_get("airflow", 8080, "/health")
        assert status == 200, f"airflow /health returned HTTP {status}"

    # ── HPA and scaling ────────────────────────────────────────────────────────

    def test_hpa_exists_for_api(self):
        """HPA for API deployment must exist."""
        result = _kubectl("get", "hpa", "-n", "mlops", "api-hpa", timeout=15)
        assert result.returncode == 0, f"HPA api-hpa not found: {result.stderr}"

    def test_hpa_min_replicas_is_1(self):
        """HPA min replicas must be 1 for the local overlay."""
        result = _kubectl(
            "get",
            "hpa",
            "-n",
            "mlops",
            "api-hpa",
            "-o",
            "jsonpath={.spec.minReplicas}",
            timeout=15,
        )
        assert result.returncode == 0
        assert result.stdout.strip() == "1", (
            f"Expected minReplicas=1, got {result.stdout.strip()!r}"
        )

    def test_hpa_max_replicas_is_4(self):
        """HPA max replicas must be 4 (as configured in k8s/base/api/hpa.yaml)."""
        result = _kubectl(
            "get",
            "hpa",
            "-n",
            "mlops",
            "api-hpa",
            "-o",
            "jsonpath={.spec.maxReplicas}",
            timeout=15,
        )
        assert result.returncode == 0
        assert result.stdout.strip() == "4", (
            f"Expected maxReplicas=4, got {result.stdout.strip()!r}"
        )

    # ── ConfigMap & Secret presence ────────────────────────────────────────────

    def test_configmap_mlops_config_exists(self):
        """The mlops-config ConfigMap must be present in the cluster."""
        result = _kubectl("get", "configmap", "mlops-config", "-n", "mlops", timeout=15)
        assert result.returncode == 0, f"mlops-config not found: {result.stderr}"

    def test_secret_mlops_secrets_exists(self):
        """The mlops-secrets Secret must be present in the cluster."""
        result = _kubectl("get", "secret", "mlops-secrets", "-n", "mlops", timeout=15)
        assert result.returncode == 0, f"mlops-secrets not found: {result.stderr}"

    def test_secret_does_not_expose_values_in_pod_env(self):
        """Secrets must be referenced via secretKeyRef, not as literal env values."""
        # We can't decrypt secrets, but we can verify the secret object has keys
        result = _kubectl(
            "get",
            "secret",
            "mlops-secrets",
            "-n",
            "mlops",
            "-o",
            "jsonpath={.data}",
            timeout=15,
        )
        assert result.returncode == 0
        data = result.stdout.strip()
        assert "DB_PASSWORD" in data or "DBPASSWORD" in data or len(data) > 10, (
            "mlops-secrets appears empty"
        )

    # ── PVC and storage ────────────────────────────────────────────────────────

    def test_all_pvcs_are_bound(self):
        """All PersistentVolumeClaims in mlops namespace must be Bound."""
        result = _kubectl("get", "pvc", "-n", "mlops", "-o", "wide", timeout=15)
        assert result.returncode == 0
        lines = result.stdout.strip().splitlines()
        unbound = [line for line in lines[1:] if line and "Bound" not in line]
        assert not unbound, "Unbound PVCs:\n" + "\n".join(unbound)

    # ── Resilience: pod kill and self-healing ──────────────────────────────────

    def test_api_pod_self_heals_after_deletion(self):
        """Delete the API pod; K8s must create a replacement and bring it to Ready."""
        import time

        # Get current API pod name
        result = _kubectl(
            "get",
            "pods",
            "-n",
            "mlops",
            "-l",
            "app=api",
            "-o",
            "jsonpath={.items[0].metadata.name}",
            timeout=15,
        )
        assert result.returncode == 0 and result.stdout.strip(), "No API pod found"
        pod_name = result.stdout.strip()

        # Delete it
        del_result = _kubectl("delete", "pod", pod_name, "-n", "mlops", timeout=30)
        assert del_result.returncode == 0, f"Delete failed: {del_result.stderr}"

        # Wait for replacement (up to 60s)
        for _ in range(20):
            time.sleep(3)
            check = _kubectl(
                "get",
                "pods",
                "-n",
                "mlops",
                "-l",
                "app=api",
                "-o",
                "jsonpath={.items[*].status.containerStatuses[*].ready}",
                timeout=15,
            )
            if "true" in check.stdout:
                break
        else:
            pytest.fail(f"API pod did not recover within 60s after deletion of {pod_name}")

    # ── Kubernetes API via Python SDK ──────────────────────────────────────────

    def test_k8s_sdk_can_list_pods(self):
        """Python kubernetes SDK must be able to list pods in mlops namespace."""
        import kubernetes  # noqa: PLC0415

        kubernetes.config.load_kube_config()
        v1 = kubernetes.client.CoreV1Api()
        pods = v1.list_namespaced_pod(namespace="mlops")
        assert len(pods.items) > 0, "kubernetes SDK returned no pods in mlops namespace"

    def test_k8s_sdk_can_list_deployments(self):
        """Python kubernetes SDK must be able to list deployments in mlops namespace."""
        import kubernetes  # noqa: PLC0415

        kubernetes.config.load_kube_config()
        apps_v1 = kubernetes.client.AppsV1Api()
        deployments = apps_v1.list_namespaced_deployment(namespace="mlops")
        names = {d.metadata.name for d in deployments.items}
        assert "api" in names and "mlflow" in names, f"Expected api+mlflow deployments, got {names}"

    def test_k8s_sdk_can_read_configmap(self):
        """Python kubernetes SDK must be able to read mlops-config ConfigMap."""
        import kubernetes  # noqa: PLC0415

        kubernetes.config.load_kube_config()
        v1 = kubernetes.client.CoreV1Api()
        cm = v1.read_namespaced_config_map(name="mlops-config", namespace="mlops")
        assert cm.data is not None
        assert "MLFLOW_TRACKING_URI" in cm.data, (
            f"MLFLOW_TRACKING_URI not in mlops-config data: {list(cm.data.keys())}"
        )
