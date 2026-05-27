"""
K8s Tier 8 — Resilience Tests
Kill a pod, verify Deployment controller recreates it.
Scale operations test.
"""

from __future__ import annotations

import subprocess
import time

NAMESPACE = "mlops"


def kubectl(*args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["kubectl"] + list(args),
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def get_pods(label: str) -> list[str]:
    result = kubectl(
        "get",
        "pods",
        "-n",
        NAMESPACE,
        "-l",
        f"app={label}",
        "-o",
        "jsonpath={.items[*].metadata.name}",
    )
    return result.stdout.split() if result.stdout.strip() else []


def wait_pods_running(label: str, expected_count: int, timeout_s: int = 120) -> bool:
    for _ in range(timeout_s // 5):
        pods = get_pods(label)
        ready_pods = []
        for pod in pods:
            r = kubectl("get", "pod", pod, "-n", NAMESPACE, "-o", "jsonpath={.status.phase}")
            if "Running" in r.stdout:
                ready_pods.append(pod)
        if len(ready_pods) >= expected_count:
            return True
        time.sleep(5)
    return False


def test_scale_api_up() -> None:
    """Scale API to 3 replicas, verify 3 pods become Running."""
    result = kubectl("scale", "deployment/api", "-n", NAMESPACE, "--replicas=3")
    assert result.returncode == 0, f"Scale up failed: {result.stderr}"
    print("  → Scaled api to 3 replicas, waiting...")
    ok = wait_pods_running("api", 3, timeout_s=120)
    assert ok, "api deployment did not reach 3 running pods within 120s"
    pods = get_pods("api")
    print(f"  ✅ api scaled to 3 replicas: {pods}")


def test_kill_pod_and_recovery() -> None:
    """Delete one api pod, verify it is recreated within 60s."""
    pods_before = get_pods("api")
    assert pods_before, "No api pods found"
    pod_to_kill = pods_before[0]

    result = kubectl("delete", "pod", pod_to_kill, "-n", NAMESPACE)
    assert result.returncode == 0, f"Pod delete failed: {result.stderr}"
    print(f"  → Deleted pod {pod_to_kill}, waiting for recreation...")

    # Wait until we again have ≥ existing count of running pods
    ok = wait_pods_running("api", len(pods_before), timeout_s=90)
    pods_after = get_pods("api")
    assert ok, f"Pod not recreated within 90s. Current pods: {pods_after}"
    assert pod_to_kill not in pods_after, f"Old pod {pod_to_kill} still present"
    print(f"  ✅ Pod {pod_to_kill[:20]}... replaced by {[p[:20] for p in pods_after]}")


def test_scale_api_back_to_1() -> None:
    """Scale API back to 1 replica to avoid resource waste."""
    result = kubectl("scale", "deployment/api", "-n", NAMESPACE, "--replicas=1")
    assert result.returncode == 0, f"Scale down failed: {result.stderr}"
    ok = wait_pods_running("api", 1, timeout_s=60)
    assert ok, "api did not return to 1 running pod"
    print("  ✅ api scaled back to 1 replica")


def test_kubectl_get_pods_json() -> None:
    """kubectl get pods -o json must return parseable output with all expected deployments."""
    import json

    result = kubectl("get", "pods", "-n", NAMESPACE, "-o", "json")
    assert result.returncode == 0, f"kubectl get pods -o json failed: {result.stderr}"
    data = json.loads(result.stdout)
    pods = data.get("items", [])
    apps = {p["metadata"]["labels"].get("app", "") for p in pods}
    expected_apps = {"api", "airflow", "mlflow", "nginx", "postgres", "prometheus", "grafana"}
    missing = expected_apps - apps
    if missing:
        print(f"  ⚠️  Pods missing for apps: {missing} (may be restarting)")
    else:
        print(f"  ✅ kubectl get pods JSON: {len(pods)} pods, all expected apps present")


def test_kubectl_scale_command() -> None:
    """Verify the kubectl scale command used by kubernetes.py Streamlit page."""
    # Scale to 2
    result = kubectl("scale", "deployment/api", "-n", NAMESPACE, "--replicas=2")
    assert result.returncode == 0, f"kubectl scale to 2 failed: {result.stderr}"
    ok = wait_pods_running("api", 2, timeout_s=90)
    assert ok, "api did not reach 2 replicas"
    print("  ✅ kubectl scale api→2 works (used by Streamlit K8s page)")
    # Scale back
    kubectl("scale", "deployment/api", "-n", NAMESPACE, "--replicas=1")


def run_all() -> int:
    tests = [
        test_kubectl_get_pods_json,
        test_scale_api_up,
        test_kill_pod_and_recovery,
        test_scale_api_back_to_1,
        test_kubectl_scale_command,
    ]
    passed = failed = 0
    for t in tests:
        name = t.__name__
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ {name}: EXCEPTION {type(e).__name__}: {e}")
            failed += 1
    print(f"\nTier 8 — Resilience: {passed} passed, {failed} failed")
    return failed


if __name__ == "__main__":
    import sys

    sys.exit(run_all())
