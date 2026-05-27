"""
K8s Tier 6 — Monitoring Tests
Verifies Prometheus targets and Grafana dashboards.
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


def http_get_json(url: str, timeout: int = 15) -> tuple[int, dict]:
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode("utf-8", errors="replace")}
    except Exception as e:
        return 0, {"error": str(e)}


def http_get_json_auth(
    url: str, user: str = "admin", pwd: str = "local_dev_password", timeout: int = 15
) -> tuple[int, dict]:
    """HTTP GET with Basic Auth."""
    import base64

    token = base64.b64encode(f"{user}:{pwd}".encode()).decode()
    req = urllib.request.Request(url, headers={"Authorization": f"Basic {token}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode("utf-8", errors="replace")}
    except Exception as e:
        return 0, {"error": str(e)}


def test_prometheus_api_targets() -> None:
    with port_forward("service/prometheus", 39091, 9090) as port:
        status, data = http_get_json(f"http://localhost:{port}/api/v1/targets")
    assert status == 200, f"Prometheus targets → {status}: {data}"
    active = data.get("data", {}).get("activeTargets", [])
    up_count = sum(1 for t in active if t.get("health") == "up")
    print(f"  ✅ Prometheus: {len(active)} targets, {up_count} up")
    for t in active[:5]:
        labels = t.get("labels", {}).get("job", "?")
        print(f"     job={labels} health={t.get('health')}")


def test_prometheus_query_up() -> None:
    with port_forward("service/prometheus", 39092, 9090) as port:
        status, data = http_get_json(f"http://localhost:{port}/api/v1/query?query=up")
    assert status == 200, f"Prometheus query up → {status}: {data}"
    results = data.get("data", {}).get("result", [])
    print(f"  ✅ Prometheus 'up' metric: {len(results)} series")


def test_prometheus_kubernetes_metrics() -> None:
    with port_forward("service/prometheus", 39093, 9090) as port:
        status, data = http_get_json(
            f"http://localhost:{port}/api/v1/query?query=kube_pod_status_phase"
        )
    if status == 200:
        results = data.get("data", {}).get("result", [])
        print(f"  ✅ kube_pod_status_phase: {len(results)} series")
    else:
        print(f"  ⚠️  kube_pod_status_phase not available: {status}")


def test_grafana_health() -> None:
    with port_forward("service/grafana", 33001, 3000) as port:
        status, data = http_get_json(f"http://localhost:{port}/api/health")
    assert status == 200, f"Grafana /api/health → {status}: {data}"
    assert data.get("database") == "ok" or "ok" in str(data).lower(), f"Grafana not healthy: {data}"
    print(f"  ✅ Grafana health: {data}")


def test_grafana_datasources() -> None:
    with port_forward("service/grafana", 33002, 3000) as port:
        status, data = http_get_json_auth(f"http://localhost:{port}/api/datasources")
    assert status == 200, f"Grafana datasources → {status}: {data}"
    sources = data if isinstance(data, list) else []
    prometheus_ds = [s for s in sources if s.get("type") == "prometheus"]
    assert prometheus_ds, f"No Prometheus datasource in Grafana: {[s.get('name') for s in sources]}"
    print(f"  ✅ Grafana datasources: {len(sources)} total, {len(prometheus_ds)} Prometheus")


def test_grafana_dashboards() -> None:
    with port_forward("service/grafana", 33003, 3000) as port:
        status, data = http_get_json_auth(f"http://localhost:{port}/api/search?type=dash-db")
    assert status == 200, f"Grafana dashboards → {status}: {data}"
    boards = data if isinstance(data, list) else []
    print(f"  ✅ Grafana dashboards: {len(boards)} found")
    for b in boards[:5]:
        print(f"     - {b.get('title', '?')}")


def run_all() -> int:
    tests = [
        test_prometheus_api_targets,
        test_prometheus_query_up,
        test_prometheus_kubernetes_metrics,
        test_grafana_health,
        test_grafana_datasources,
        test_grafana_dashboards,
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
    print(f"\nTier 6 — Monitoring: {passed} passed, {failed} failed")
    return failed


if __name__ == "__main__":
    import sys

    sys.exit(run_all())
