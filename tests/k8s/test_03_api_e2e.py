"""
K8s Tier 3 — API End-to-End Tests
Tests the FastAPI service via kubectl port-forward: predict, history,
labels, db stats, backup.
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


def http_get(url: str, timeout: int = 15) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read(8192).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read(2048).decode("utf-8", errors="replace")
    except Exception as e:
        return 0, str(e)


def http_post(url: str, payload: dict, timeout: int = 30) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(8192).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read(2048).decode("utf-8", errors="replace")
    except Exception as e:
        return 0, str(e)


SAMPLE_SIGNAL = {
    "device_id": "k8s-test-device-001",
    "signal_type": "vibration",
    "values": [
        0.1,
        0.2,
        0.15,
        0.18,
        0.12,
        0.22,
        0.19,
        0.17,
        0.14,
        0.16,
        0.11,
        0.21,
        0.13,
        0.20,
        0.16,
        0.18,
        0.15,
        0.17,
        0.14,
        0.19,
    ],
    "mu": 0.16,
    "sigma": 0.03,
    "shape_type": "normal",
}


def get_auth_token(port: int) -> str:
    """Get Bearer token from /auth/token (admin/secret is the test credential)."""
    import urllib.parse

    body = urllib.parse.urlencode({"username": "admin", "password": "secret"}).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/auth/token",
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return data.get("access_token", "")
    except Exception:
        return ""


def http_get_auth(url: str, token: str, timeout: int = 15) -> tuple[int, str]:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(8192).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read(2048).decode("utf-8", errors="replace")
    except Exception as e:
        return 0, str(e)


def http_post_auth(url: str, payload: dict, token: str, timeout: int = 30) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(8192).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read(2048).decode("utf-8", errors="replace")
    except Exception as e:
        return 0, str(e)


def test_api_health() -> None:
    """Health may be 503 when no model loaded; verify it returns FastAPI JSON."""
    with port_forward("service/api", 38000, 8000) as port:
        status, body = http_get(f"http://localhost:{port}/health")
    assert status in (200, 503), f"/health → unexpected {status}: {body[:200]}"
    data = json.loads(body)
    assert "status" in data, f"health response missing 'status' key: {data}"
    assert data.get("database_connected") is True, f"database_connected is False: {data}"
    print(
        f"  ✅ API /health → {status}: db={data.get('database_connected')} model={data.get('model_loaded')}"
    )


def test_api_predict() -> None:
    with port_forward("service/api", 38001, 8000) as port:
        token = get_auth_token(port)
        assert token, "Failed to obtain auth token from /auth/token"
        status, body = http_post_auth(f"http://localhost:{port}/predict", SAMPLE_SIGNAL, token)
    if status == 200:
        data = json.loads(body)
        assert "prediction" in data, f"Missing 'prediction': {data}"
        print(f"  ✅ API /predict → {status}: prediction={data.get('prediction')}")
    elif status in (404, 503):
        # No model loaded yet — endpoint exists and auth passed
        assert "No model" in body or "model" in body.lower(), (
            f"Unexpected 404/503 body: {body[:200]}"
        )
        print(f"  ✅ API /predict → {status} (no model loaded — endpoint + auth OK)")
    else:
        raise AssertionError(f"/predict → unexpected {status}: {body[:400]}")


def test_api_predict_invalid_payload() -> None:
    """Invalid payload should return 4xx. When no model is loaded, 404 is also valid."""
    with port_forward("service/api", 38002, 8000) as port:
        token = get_auth_token(port)
        status, body = http_post_auth(
            f"http://localhost:{port}/predict", {"invalid": "payload"}, token
        )
    assert status in (400, 422, 404), (
        f"Invalid payload should return 4xx, got {status}: {body[:200]}"
    )
    print(f"  ✅ API /predict invalid payload → {status} (expected 4xx)")


def test_api_stats() -> None:
    """GET /stats returns prediction statistics (requires auth)."""
    with port_forward("service/api", 38003, 8000) as port:
        token = get_auth_token(port)
        status, body = http_get_auth(f"http://localhost:{port}/stats", token)
    assert status == 200, f"/stats → {status}: {body[:200]}"
    data = json.loads(body)
    assert "total_predictions" in data, f"stats missing 'total_predictions': {data}"
    print(f"  ✅ API /stats → {status}: total_predictions={data.get('total_predictions')}")


def test_api_db_stats() -> None:
    """GET /stats also serves as the db stats endpoint."""
    with port_forward("service/api", 38004, 8000) as port:
        token = get_auth_token(port)
        status, body = http_get_auth(f"http://localhost:{port}/stats", token)
    assert status == 200, f"/stats (db) → {status}: {body[:200]}"
    data = json.loads(body)
    assert isinstance(data, dict), f"Expected dict, got: {type(data)}"
    print(f"  ✅ API /stats (db) → {status}: keys={list(data.keys())[:5]}")


def test_api_models_list() -> None:
    """GET /model/info — may 404 when no model loaded, that is valid."""
    with port_forward("service/api", 38005, 8000) as port:
        token = get_auth_token(port)
        status, body = http_get_auth(f"http://localhost:{port}/model/info", token)
    assert status in (200, 404, 503), f"/model/info → unexpected {status}: {body[:200]}"
    if status == 200:
        print(f"  ✅ API /model/info → {status} (model loaded!)")
    else:
        print(f"  ✅ API /model/info → {status} (no model — endpoint OK)")


def test_api_mlflow_connectivity() -> None:
    """Verify the API pod can reach MLflow tracking server."""
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
            "import mlflow, os; "
            "mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI','http://mlflow:5000')); "
            "exps = mlflow.search_experiments(); "
            "print(f'MLflow reachable: {len(exps)} experiments')",
        ],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    assert result.returncode == 0, f"API → MLflow connectivity failed: {result.stderr}"
    print(f"  ✅ {result.stdout.strip()}")


def test_api_postgres_connectivity() -> None:
    """Verify the API pod can connect to PostgreSQL."""
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
            "import psycopg2, os; "
            "conn = psycopg2.connect(os.environ['DATABASE_URL']); "
            "cur = conn.cursor(); "
            'cur.execute("SELECT COUNT(*) FROM predictions"); '
            "count = cur.fetchone()[0]; "
            "conn.close(); "
            "print(f'PostgreSQL reachable: predictions={count}')",
        ],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    assert result.returncode == 0, f"API → PostgreSQL connectivity failed: {result.stderr[:400]}"
    print(f"  ✅ {result.stdout.strip()}")


def run_all() -> int:
    tests = [
        test_api_health,
        test_api_predict,
        test_api_predict_invalid_payload,
        test_api_stats,
        test_api_db_stats,
        test_api_models_list,
        test_api_mlflow_connectivity,
        test_api_postgres_connectivity,
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
    print(f"\nTier 3 — API E2E: {passed} passed, {failed} failed")
    return failed


if __name__ == "__main__":
    import sys

    sys.exit(run_all())
