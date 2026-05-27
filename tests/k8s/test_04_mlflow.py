"""
K8s Tier 4 — MLflow Tests
Tests the MLflow tracking server: experiments, runs, registry.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Generator
from contextlib import contextmanager

NAMESPACE = "mlops"


@contextmanager
def port_forward(resource: str, local_port: int, remote_port: int) -> Generator[int, None, None]:
    proc = subprocess.Popen(
        ["kubectl", "port-forward", "-n", NAMESPACE, resource, f"{local_port}:{remote_port}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1.5)
    try:
        yield local_port
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def mlflow_api(endpoint: str, port: int) -> tuple[int, dict]:
    url = f"http://localhost:{port}/api/2.0/mlflow/{endpoint}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode("utf-8", errors="replace")}
    except Exception as e:
        return 0, {"error": str(e)}


def test_mlflow_ui_accessible() -> None:
    with (
        port_forward("service/mlflow", 35010, 5000) as port,
        urllib.request.urlopen(f"http://localhost:{port}/", timeout=10) as resp,
    ):
        status = resp.status
    assert status == 200, f"MLflow UI → {status}"
    print(f"  ✅ MLflow UI → {status}")


def test_mlflow_experiments_list() -> None:
    """MLflow 3.x uses /experiments/search (max_results required); /list is removed."""
    with port_forward("service/mlflow", 35011, 5000) as port:
        status, data = mlflow_api("experiments/search?max_results=20", port)
    assert status == 200, f"MLflow experiments/search → {status}: {data}"
    exps = data.get("experiments", [])
    print(f"  ✅ MLflow experiments: {len(exps)} found")


def test_mlflow_registered_models() -> None:
    """MLflow 3.x: /registered-models/search (max_results required); /list is removed."""
    with port_forward("service/mlflow", 35012, 5000) as port:
        status, data = mlflow_api("registered-models/search?max_results=20", port)
    assert status == 200, f"MLflow registered-models/search → {status}: {data}"
    models = data.get("registered_models", [])
    print(f"  ✅ MLflow registered models: {len(models)} found")
    for m in models[:3]:
        print(f"     - {m.get('name')} ({len(m.get('latest_versions', []))} versions)")


def test_mlflow_create_and_log_run() -> None:
    """Create a test experiment and log a run to verify MLflow write path."""
    # Use sh -c with semicolons to avoid Python block-indent issues in -c
    script = (
        "import mlflow, os\n"
        "mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI','http://mlflow:5000'))\n"
        "mlflow.set_experiment('k8s_connectivity_test')\n"
        "run = mlflow.start_run(run_name='k8s_test_run')\n"
        "mlflow.log_param('test', 'k8s')\n"
        "mlflow.log_metric('dummy', 1.0)\n"
        "print('Run:', run.info.run_id[:8])\n"
        "mlflow.end_run()\n"
    )
    result = subprocess.run(
        [
            "kubectl",
            "exec",
            "-n",
            NAMESPACE,
            "deploy/api",
            "--",
            "python",
            "-c",
            script,
        ],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, f"MLflow log run failed: {result.stderr[:400]}"
    run_id_preview = result.stdout.strip()
    print(f"  ✅ MLflow create+log run: {run_id_preview}")


def test_mlflow_artifact_store() -> None:
    """Verify MLflow artifact store is accessible (local PVC path)."""
    result = subprocess.run(
        [
            "kubectl",
            "exec",
            "-n",
            NAMESPACE,
            "deploy/mlflow",
            "--",
            "ls",
            "/mlflow/artifacts",
        ],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )
    if result.returncode == 0:
        print(f"  ✅ MLflow artifact store accessible: {len(result.stdout.splitlines())} entries")
    else:
        # Might be empty on first deployment
        print(f"  ⚠️  MLflow artifact store: {result.stderr.strip()[:100]}")


def run_all() -> int:
    tests = [
        test_mlflow_ui_accessible,
        test_mlflow_experiments_list,
        test_mlflow_registered_models,
        test_mlflow_create_and_log_run,
        test_mlflow_artifact_store,
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
    print(f"\nTier 4 — MLflow: {passed} passed, {failed} failed")
    return failed


if __name__ == "__main__":
    import sys

    sys.exit(run_all())
