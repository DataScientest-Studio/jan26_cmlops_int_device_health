"""
K8s test suite conftest — auto-skip all tests when no cluster is reachable.

Checks once per session whether `kubectl get nodes -n mlops` succeeds.
If not (no cluster, no kubeconfig, GitHub Actions CI, etc.) every test in
tests/k8s/ is marked skip so the overall suite still passes.
"""

from __future__ import annotations

import subprocess

import pytest


def _k8s_cluster_available() -> bool:
    """Return True if kubectl can reach a cluster that has the 'mlops' namespace deployed."""
    try:
        result = subprocess.run(
            ["kubectl", "get", "namespace", "mlops"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


_K8S_AVAILABLE: bool = _k8s_cluster_available()

# ── Session-level fixture so individual tests can depend on it ────────────


@pytest.fixture(scope="session", autouse=True)
def require_k8s_cluster() -> None:
    """No-op fixture kept for backward compatibility.

    The actual skip logic lives in ``pytest_collection_modifyitems`` which
    adds a skip marker per-item. This avoids session-wide skips that would
    also affect unrelated test modules collected in the same session
    (e.g., tests/live/).
    """


# ── Collection-time hook — marks every item at collect time ──────────────


def pytest_collection_modifyitems(items: list, config: pytest.Config) -> None:  # type: ignore[type-arg]
    """Add a skip marker to every K8s test when the cluster is not available.

    This runs before the session fixture and ensures the tests appear as
    'skipped' (not 'error') in CI reports, making the summary easier to read.
    """
    if _K8S_AVAILABLE:
        return
    skip = pytest.mark.skip(
        reason="K8s 'mlops' namespace not found — deploy the mlops stack to a cluster first."
    )
    for item in items:
        fspath = str(item.fspath).replace("\\", "/")
        if "tests/k8s" in fspath and "test_00_manifests" not in fspath:
            item.add_marker(skip)
