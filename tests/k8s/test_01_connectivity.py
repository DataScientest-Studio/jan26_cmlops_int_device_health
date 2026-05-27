"""
K8s Tier 1 — Service Connectivity Tests
Tests all service endpoints via kubectl port-forward.
"""

from __future__ import annotations

import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Generator
from contextlib import contextmanager

NAMESPACE = "mlops"


@contextmanager
def port_forward(resource: str, local_port: int, remote_port: int) -> Generator[int, None, None]:
    """Start kubectl port-forward in background, yield the local port."""
    cmd = [
        "kubectl",
        "port-forward",
        "-n",
        NAMESPACE,
        resource,
        f"{local_port}:{remote_port}",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.5)  # wait for tunnel to establish
    try:
        yield local_port
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def http_get(url: str, timeout: int = 10) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read(4096).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        try:
            body = e.read(4096).decode("utf-8", errors="replace")
        except Exception:
            body = str(e)
        return e.code, body
    except Exception as e:
        return 0, str(e)


# ──────────────────────────────────────────────────────────────────────────────
# Test: nginx routing
# ──────────────────────────────────────────────────────────────────────────────
def test_nginx_root() -> None:
    with port_forward("service/nginx", 39080, 80) as port:
        status, body = http_get(f"http://localhost:{port}/")
    assert status == 200, f"nginx / → expected 200, got {status}"
    print(f"  ✅ nginx /  → {status}")


def test_nginx_health_direct() -> None:
    """nginx /health must route to FastAPI (JSON body with 'status' key).
    Accept 200 OR 503 — 503 is valid when no ML model is loaded yet."""
    with port_forward("service/nginx", 39081, 80) as port:
        status, body = http_get(f"http://localhost:{port}/health")
    assert status in (200, 503), f"nginx /health → unexpected {status}: {body[:200]}"
    assert '"status"' in body, (
        f"nginx /health body is not FastAPI JSON (routing error?): {body[:200]}"
    )
    print(f"  ✅ nginx /health → {status} (FastAPI JSON, routing OK)")


def test_nginx_api_health() -> None:
    """nginx /api/health must route to FastAPI. Accept 200 or 503 (model not loaded)."""
    with port_forward("service/nginx", 39082, 80) as port:
        status, body = http_get(f"http://localhost:{port}/api/health")
    assert status in (200, 503), f"nginx /api/health → unexpected {status}: {body[:200]}"
    assert '"status"' in body, f"nginx /api/health body not JSON: {body[:200]}"
    print(f"  ✅ nginx /api/health → {status} (FastAPI, routing OK)")


def test_nginx_docs() -> None:
    with port_forward("service/nginx", 39083, 80) as port:
        status, body = http_get(f"http://localhost:{port}/docs")
    assert status == 200, f"nginx /docs → expected 200, got {status}"
    assert "swagger" in body.lower() or "openapi" in body.lower(), (
        f"/docs doesn't look like Swagger: {body[:200]}"
    )
    print(f"  ✅ nginx /docs → {status} (Swagger)")


def test_nginx_api_docs() -> None:
    with port_forward("service/nginx", 39084, 80) as port:
        status, body = http_get(f"http://localhost:{port}/api/docs")
    assert status == 200, f"nginx /api/docs → expected 200, got {status}"
    print(f"  ✅ nginx /api/docs → {status}")


# ──────────────────────────────────────────────────────────────────────────────
# Test: MLflow
# ──────────────────────────────────────────────────────────────────────────────
def test_mlflow_ui() -> None:
    with port_forward("service/mlflow", 35001, 5000) as port:
        status, body = http_get(f"http://localhost:{port}/")
    assert status == 200, f"MLflow UI → expected 200, got {status}"
    print(f"  ✅ MLflow UI → {status}")


def test_mlflow_experiments_api() -> None:
    """MLflow 3.x removed /experiments/list; use /experiments/search with max_results."""
    with port_forward("service/mlflow", 35002, 5000) as port:
        status, body = http_get(
            f"http://localhost:{port}/api/2.0/mlflow/experiments/search?max_results=20"
        )
    assert status == 200, f"MLflow experiments/search → {status}: {body[:200]}"
    assert '"experiments"' in body, f"Unexpected response body: {body[:200]}"
    print(f"  ✅ MLflow experiments API → {status}")


# ──────────────────────────────────────────────────────────────────────────────
# Test: Airflow
# ──────────────────────────────────────────────────────────────────────────────
def test_airflow_health() -> None:
    with port_forward("service/airflow", 38080, 8080) as port:
        status, body = http_get(f"http://localhost:{port}/health")
    assert status == 200, f"Airflow /health → expected 200, got {status}"
    assert "healthy" in body.lower() or "metadatabase" in body.lower(), (
        f"Airflow /health body unexpected: {body[:200]}"
    )
    print(f"  ✅ Airflow /health → {status}")


# ──────────────────────────────────────────────────────────────────────────────
# Test: Prometheus
# ──────────────────────────────────────────────────────────────────────────────
def test_prometheus_targets() -> None:
    with port_forward("service/prometheus", 39090, 9090) as port:
        status, body = http_get(f"http://localhost:{port}/api/v1/targets")
    assert status == 200, f"Prometheus /api/v1/targets → {status}"
    assert "activeTargets" in body, "Missing activeTargets in Prometheus response"
    print(f"  ✅ Prometheus targets API → {status}")


# ──────────────────────────────────────────────────────────────────────────────
# Test: Grafana
# ──────────────────────────────────────────────────────────────────────────────
def test_grafana_health() -> None:
    with port_forward("service/grafana", 33000, 3000) as port:
        status, body = http_get(f"http://localhost:{port}/api/health")
    assert status == 200, f"Grafana /api/health → {status}"
    assert "database" in body.lower() or "ok" in body.lower(), (
        f"Grafana /api/health unexpected: {body[:200]}"
    )
    print(f"  ✅ Grafana /api/health → {status}")


# ──────────────────────────────────────────────────────────────────────────────
def run_all() -> int:
    tests = [
        test_nginx_root,
        test_nginx_health_direct,
        test_nginx_api_health,
        test_nginx_docs,
        test_nginx_api_docs,
        test_mlflow_ui,
        test_mlflow_experiments_api,
        test_airflow_health,
        test_prometheus_targets,
        test_grafana_health,
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
    print(f"\nTier 1 — Connectivity: {passed} passed, {failed} failed")
    return failed


if __name__ == "__main__":
    import sys

    sys.exit(run_all())
