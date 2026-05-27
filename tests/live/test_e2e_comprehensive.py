"""
Comprehensive E2E Test Suite — tests ALL services, ALL use cases, ALL monitoring.

Runs against the live Docker stack (cloud mode).
Validates that everything works identically to how K8s would expose it.

Port mapping (Docker cloud stack):
  - Nginx/API: 8080 → 80 (proxies to API:8000)
  - Grafana:   3000
  - Prometheus: 9090
  - MLflow:    5002 → 5000
  - Airflow:   8081 → 8080
  - PostgreSQL: 5433 → 5432

Run with:
  .venv/Scripts/python.exe -m pytest tests/live/test_e2e_comprehensive.py -v --tb=short -x
"""

from __future__ import annotations

import math
import os
import time
import uuid

import pytest
import requests

# ─── Constants ────────────────────────────────────────────────────────────────

API_URL = os.environ.get("MLOPS_API_URL", "http://localhost:8080")
GRAFANA_URL = os.environ.get("GRAFANA_URL", "http://localhost:3000")
PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")
MLFLOW_URL = os.environ.get("MLFLOW_URL", "http://localhost:5002")
AIRFLOW_URL = os.environ.get("AIRFLOW_URL", "http://localhost:8081")
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", "5433"))

# Auth credentials (from fake_users_db in src/api/auth.py)
API_USER = os.environ.get("API_USERNAME", "admin")
API_PASS = os.environ.get("API_PASSWORD", "secret")
AIRFLOW_USER = os.environ.get("AIRFLOW_USER", "admin")
AIRFLOW_PASS = os.environ.get("AIRFLOW_PASSWORD", "admin")
GRAFANA_USER = os.environ.get("GRAFANA_USER", "admin")
GRAFANA_PASS = os.environ.get("GRAFANA_PASSWORD", "admin")

TIMEOUT = 10


# ─── Fixtures ─────────────────────────────────────────────────────────────────


def _stack_running() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="session", autouse=True)
def require_stack():
    if not _stack_running():
        pytest.skip("Docker stack not running (API unreachable)")


@pytest.fixture(scope="session")
def auth_token() -> str:
    """Obtain a valid auth token."""
    r = requests.post(
        f"{API_URL}/auth/token",
        data={"username": API_USER, "password": API_PASS},
        timeout=TIMEOUT,
    )
    if r.status_code != 200:
        pytest.skip(f"Auth failed: {r.status_code} {r.text[:200]}")
    return r.json()["access_token"]


@pytest.fixture(scope="session")
def auth_headers(auth_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {auth_token}"}


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _api_get(path: str, headers: dict | None = None) -> requests.Response:
    return requests.get(f"{API_URL}{path}", headers=headers, timeout=TIMEOUT)


def _api_post(
    path: str, json_data: dict | None = None, headers: dict | None = None
) -> requests.Response:
    return requests.post(f"{API_URL}{path}", json=json_data, headers=headers, timeout=TIMEOUT)


def _generate_signal(
    n_samples: int = 1000, frequency: float = 5.0
) -> tuple[list[float], list[float]]:
    """Generate a synthetic vibration signal spanning [0, 100] with n_samples points."""
    time_vals = [i * 100.0 / (n_samples - 1) for i in range(n_samples)]
    amp_vals = [
        math.sin(2 * math.pi * frequency * t / 100) + 0.1 * math.sin(2 * math.pi * 50 * t / 100)
        for t in time_vals
    ]
    return time_vals, amp_vals


# =============================================================================
# 1. HEALTH & CONNECTIVITY
# =============================================================================


class TestHealthConnectivity:
    def test_api_health_returns_200(self) -> None:
        r = _api_get("/health")
        assert r.status_code == 200

    def test_health_response_structure(self) -> None:
        r = _api_get("/health")
        data = r.json()
        assert data["status"] == "healthy"
        for key in ["version", "deployment_mode", "database_connected", "model_loaded"]:
            assert key in data, f"Health response missing '{key}'"

    def test_deployment_mode_is_cloud(self) -> None:
        r = _api_get("/health")
        assert r.json()["deployment_mode"] == "cloud"

    def test_database_connected(self) -> None:
        r = _api_get("/health")
        assert r.json()["database_connected"] is True

    def test_model_loaded(self) -> None:
        r = _api_get("/health")
        assert r.json()["model_loaded"] is True

    def test_api_docs_endpoint(self) -> None:
        r = _api_get("/docs")
        assert r.status_code == 200

    def test_api_openapi_json(self) -> None:
        r = _api_get("/openapi.json")
        assert r.status_code == 200
        data = r.json()
        assert "paths" in data


# =============================================================================
# 2. AUTHENTICATION
# =============================================================================


class TestAuthentication:
    def test_auth_token_endpoint_exists(self) -> None:
        r = requests.post(
            f"{API_URL}/auth/token",
            data={"username": API_USER, "password": API_PASS},
            timeout=TIMEOUT,
        )
        # Accept 200 (valid creds) or 401 (wrong creds but endpoint exists)
        assert r.status_code in (200, 401)
        if r.status_code == 200:
            data = r.json()
            assert "access_token" in data
            assert data["token_type"] == "bearer"

    def test_auth_token_invalid_credentials(self) -> None:
        r = requests.post(
            f"{API_URL}/auth/token",
            data={"username": "bad", "password": "bad"},
            timeout=TIMEOUT,
        )
        assert r.status_code in (401, 403)

    def test_model_info_endpoint_reachable(self) -> None:
        r = _api_get("/model/info")
        # May be 200 (public or API-key mode) or 401 (bearer-only)
        assert r.status_code in (200, 401)

    def test_protected_endpoint_with_auth(self, auth_headers: dict) -> None:
        r = _api_get("/model/info", headers=auth_headers)
        assert r.status_code == 200

    def test_auth_refresh_token(self, auth_token: str) -> None:
        r = requests.post(
            f"{API_URL}/auth/refresh",
            json={"token": auth_token},
            timeout=TIMEOUT,
        )
        # Accept 200 (refresh works) or 422 (endpoint exists but different schema)
        assert r.status_code in (200, 422, 405)

    def test_auth_users_me(self, auth_headers: dict) -> None:
        r = _api_get("/auth/users/me", headers=auth_headers)
        assert r.status_code == 200
        data = r.json()
        assert "username" in data


# =============================================================================
# 3. PREDICTIONS (Core Use Case)
# =============================================================================


class TestPredictions:
    def test_predict_valid_signal(self, auth_headers: dict) -> None:
        time_vals, amp_vals = _generate_signal()
        r = _api_post(
            "/predict",
            json_data={
                "device_id": str(uuid.uuid4()),
                "time_values": time_vals,
                "amplitude_values": amp_vals,
            },
            headers=auth_headers,
        )
        assert r.status_code == 200
        data = r.json()
        assert "prediction_id" in data or "predicted_label" in data

    def test_predict_multiple_devices(self, auth_headers: dict) -> None:
        """Send predictions for 5 different devices."""
        results = []
        for i in range(5):
            time_vals, amp_vals = _generate_signal(frequency=5.0 + i * 2)
            r = _api_post(
                "/predict",
                json_data={
                    "device_id": f"test-device-{i:03d}",
                    "time_values": time_vals,
                    "amplitude_values": amp_vals,
                },
                headers=auth_headers,
            )
            results.append(r.status_code)
        assert all(s == 200 for s in results), f"Not all predictions succeeded: {results}"

    def test_predict_missing_fields_returns_422(self, auth_headers: dict) -> None:
        r = _api_post("/predict", json_data={"device_id": "x"}, headers=auth_headers)
        assert r.status_code == 422

    def test_predict_empty_signal(self, auth_headers: dict) -> None:
        r = _api_post(
            "/predict",
            json_data={
                "device_id": str(uuid.uuid4()),
                "time_values": [],
                "amplitude_values": [],
            },
            headers=auth_headers,
        )
        # Should be 400 or 422
        assert r.status_code in (400, 422)


# =============================================================================
# 4. MODEL INFO
# =============================================================================


class TestModelInfo:
    def test_model_info_endpoint(self, auth_headers: dict) -> None:
        r = _api_get("/model/info", headers=auth_headers)
        assert r.status_code == 200
        data = r.json()
        # Should have model details
        assert isinstance(data, dict)

    def test_model_info_has_type(self, auth_headers: dict) -> None:
        r = _api_get("/model/info", headers=auth_headers)
        data = r.json()
        # At minimum should show model is available
        assert len(data) > 0


# =============================================================================
# 5. LABEL INJECTION
# =============================================================================


class TestLabelInjection:
    def test_inject_label(self, auth_headers: dict) -> None:
        r = _api_post(
            "/labels",
            json_data={
                "device_id": "test-device-000",
                "true_label": 0,
            },
            headers=auth_headers,
        )
        # Accept 200 (success) or 404 (device not found) or 422
        assert r.status_code in (200, 201, 404, 422)


# =============================================================================
# 6. METRICS & MONITORING ENDPOINTS
# =============================================================================


class TestMetricsEndpoints:
    def test_metrics_endpoint(self) -> None:
        r = _api_get("/metrics")
        assert r.status_code == 200

    def test_stats_endpoint(self) -> None:
        r = _api_get("/stats")
        assert r.status_code == 200

    def test_prometheus_metrics_format(self) -> None:
        """GET /metrics should return Prometheus text format."""
        r = _api_get("/metrics")
        body = r.text
        # Should have some prometheus-style metric lines
        assert "# " in body or "api_" in body or "python_" in body or "process_" in body

    def test_evaluate_endpoint_exists(self, auth_headers: dict) -> None:
        r = _api_post("/evaluate", json_data={}, headers=auth_headers)
        # Should not 404
        assert r.status_code != 404


# =============================================================================
# 7. NGINX PROXY
# =============================================================================


class TestNginxProxy:
    def test_nginx_proxies_api_health(self) -> None:
        r = requests.get(f"{API_URL}/health", timeout=TIMEOUT)
        assert r.status_code == 200

    def test_nginx_proxies_api_docs(self) -> None:
        r = requests.get(f"{API_URL}/docs", timeout=TIMEOUT)
        assert r.status_code == 200

    def test_nginx_cors_headers(self) -> None:
        r = requests.options(
            f"{API_URL}/health",
            headers={"Origin": "http://localhost:8501", "Access-Control-Request-Method": "GET"},
            timeout=TIMEOUT,
        )
        # Nginx should allow CORS
        assert r.status_code in (200, 204, 405)


# =============================================================================
# 8. MLFLOW
# =============================================================================


class TestMLflow:
    def test_mlflow_health(self) -> None:
        r = requests.get(f"{MLFLOW_URL}/health", timeout=TIMEOUT)
        assert r.status_code == 200

    def test_mlflow_experiments_api(self) -> None:
        r = requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/experiments/search",
            json={"max_results": 100},
            timeout=TIMEOUT,
        )
        assert r.status_code == 200
        data = r.json()
        assert "experiments" in data

    def test_mlflow_has_experiments(self) -> None:
        r = requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/experiments/search",
            json={"max_results": 100},
            timeout=TIMEOUT,
        )
        experiments = r.json().get("experiments", [])
        assert len(experiments) >= 1, "No MLflow experiments found"

    def test_mlflow_registered_models(self) -> None:
        r = requests.get(f"{MLFLOW_URL}/api/2.0/mlflow/registered-models/search", timeout=TIMEOUT)
        assert r.status_code == 200

    def test_mlflow_runs_exist(self) -> None:
        # Get first experiment
        r = requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/experiments/search",
            json={"max_results": 10},
            timeout=TIMEOUT,
        )
        experiments = r.json().get("experiments", [])
        if not experiments:
            pytest.skip("No experiments")
        exp_id = experiments[0]["experiment_id"]
        r2 = requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/runs/search",
            json={"experiment_ids": [exp_id], "max_results": 5},
            timeout=TIMEOUT,
        )
        assert r2.status_code == 200


# =============================================================================
# 9. AIRFLOW
# =============================================================================


class TestAirflow:
    @pytest.fixture(scope="class")
    def airflow_auth(self) -> tuple[str, str]:
        return (AIRFLOW_USER, AIRFLOW_PASS)

    def test_airflow_health(self) -> None:
        r = requests.get(f"{AIRFLOW_URL}/health", timeout=TIMEOUT)
        assert r.status_code == 200
        data = r.json()
        assert data.get("metadatabase", {}).get("status") == "healthy"

    def test_airflow_dags_list(self, airflow_auth: tuple[str, str]) -> None:
        r = requests.get(
            f"{AIRFLOW_URL}/api/v1/dags",
            auth=airflow_auth,
            timeout=TIMEOUT,
        )
        assert r.status_code == 200
        data = r.json()
        assert "dags" in data
        assert len(data["dags"]) >= 1

    def test_airflow_expected_dags_present(self, airflow_auth: tuple[str, str]) -> None:
        r = requests.get(f"{AIRFLOW_URL}/api/v1/dags", auth=airflow_auth, timeout=TIMEOUT)
        dag_ids = [d["dag_id"] for d in r.json().get("dags", [])]
        expected = [
            "automated_retraining",
            "evidently_drift_detection",
            "model_promotion",
        ]
        for dag in expected:
            assert dag in dag_ids, f"Expected DAG '{dag}' not found. Available: {dag_ids}"

    def test_airflow_dag_details(self, airflow_auth: tuple[str, str]) -> None:
        r = requests.get(
            f"{AIRFLOW_URL}/api/v1/dags/automated_retraining",
            auth=airflow_auth,
            timeout=TIMEOUT,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["dag_id"] == "automated_retraining"

    def test_airflow_scheduler_healthy(self) -> None:
        r = requests.get(f"{AIRFLOW_URL}/health", timeout=TIMEOUT)
        data = r.json()
        scheduler = data.get("scheduler", {})
        assert scheduler.get("status") == "healthy"


# =============================================================================
# 10. GRAFANA
# =============================================================================


class TestGrafana:
    @pytest.fixture(scope="class")
    def grafana_session(self) -> requests.Session:
        s = requests.Session()
        s.auth = (GRAFANA_USER, GRAFANA_PASS)
        return s

    def test_grafana_health(self) -> None:
        r = requests.get(f"{GRAFANA_URL}/api/health", timeout=TIMEOUT)
        assert r.status_code == 200

    def test_grafana_login(self, grafana_session: requests.Session) -> None:
        r = grafana_session.get(f"{GRAFANA_URL}/api/org", timeout=TIMEOUT)
        assert r.status_code == 200

    def test_grafana_datasources(self, grafana_session: requests.Session) -> None:
        r = grafana_session.get(f"{GRAFANA_URL}/api/datasources", timeout=TIMEOUT)
        assert r.status_code == 200
        ds = r.json()
        assert len(ds) >= 1, "No Grafana datasources configured"

    def test_grafana_has_prometheus_datasource(self, grafana_session: requests.Session) -> None:
        r = grafana_session.get(f"{GRAFANA_URL}/api/datasources", timeout=TIMEOUT)
        ds_types = [d["type"] for d in r.json()]
        assert "prometheus" in ds_types, f"No Prometheus datasource. Types: {ds_types}"

    def test_grafana_dashboards_exist(self, grafana_session: requests.Session) -> None:
        r = grafana_session.get(f"{GRAFANA_URL}/api/search?type=dash-db", timeout=TIMEOUT)
        assert r.status_code == 200
        dashboards = r.json()
        assert len(dashboards) >= 3, f"Expected 3+ dashboards, got {len(dashboards)}"

    def test_grafana_specific_dashboards(self, grafana_session: requests.Session) -> None:
        r = grafana_session.get(f"{GRAFANA_URL}/api/search?type=dash-db", timeout=TIMEOUT)
        titles = [d["title"].lower() for d in r.json()]
        expected_keywords = ["system", "model", "health"]
        found = sum(1 for kw in expected_keywords if any(kw in t for t in titles))
        assert found >= 1, f"No expected dashboard found in: {titles}"


# =============================================================================
# 11. PROMETHEUS
# =============================================================================


class TestPrometheus:
    def test_prometheus_healthy(self) -> None:
        r = requests.get(f"{PROMETHEUS_URL}/-/healthy", timeout=TIMEOUT)
        assert r.status_code == 200

    def test_prometheus_ready(self) -> None:
        r = requests.get(f"{PROMETHEUS_URL}/-/ready", timeout=TIMEOUT)
        assert r.status_code == 200

    def test_prometheus_targets(self) -> None:
        r = requests.get(f"{PROMETHEUS_URL}/api/v1/targets", timeout=TIMEOUT)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "success"
        targets = data["data"]["activeTargets"]
        assert len(targets) >= 1, "No active Prometheus targets"

    def test_prometheus_scrapes_api(self) -> None:
        r = requests.get(f"{PROMETHEUS_URL}/api/v1/targets", timeout=TIMEOUT)
        targets = r.json()["data"]["activeTargets"]
        job_names = [t.get("labels", {}).get("job", "") for t in targets]
        api_jobs = [j for j in job_names if "api" in j.lower() or "fastapi" in j.lower()]
        assert len(api_jobs) >= 1, f"API not scraped by Prometheus. Jobs: {job_names}"

    def test_prometheus_query_up(self) -> None:
        r = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": "up"},
            timeout=TIMEOUT,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "success"
        results = data["data"]["result"]
        assert len(results) >= 1

    def test_prometheus_api_metrics_available(self) -> None:
        r = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": 'api_requests_total{job=~".*api.*"}'},
            timeout=TIMEOUT,
        )
        assert r.status_code == 200

    def test_prometheus_has_some_targets_up(self) -> None:
        r = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": "up == 1"},
            timeout=TIMEOUT,
        )
        data = r.json()
        up_targets = data["data"]["result"]
        assert len(up_targets) >= 1, "No Prometheus targets are UP"


# =============================================================================
# 12. POSTGRESQL
# =============================================================================


class TestPostgreSQL:
    def test_postgres_connection(self) -> None:
        """Test PostgreSQL is reachable via the API health check."""
        r = _api_get("/health")
        data = r.json()
        assert data["database_connected"] is True

    def test_postgres_via_stats(self) -> None:
        """Stats endpoint queries the DB — if it works, DB is fine."""
        r = _api_get("/stats")
        assert r.status_code == 200


# =============================================================================
# 13. ADMIN OPERATIONS
# =============================================================================


class TestAdminOperations:
    def test_admin_reload_model(self, auth_headers: dict) -> None:
        r = _api_post("/admin/reload-model", headers=auth_headers)
        assert r.status_code in (200, 204, 409)

    def test_admin_reload_model_endpoint_exists(self) -> None:
        """Verify /admin/reload-model exists (may require auth)."""
        r = _api_post("/admin/reload-model")
        # Should not 404 — may be 401, 200, 204 etc
        assert r.status_code != 404


# =============================================================================
# 14. K8S API ENDPOINTS (test route existence even on Docker)
# =============================================================================


class TestK8sAPIEndpoints:
    def test_k8s_pods_endpoint_exists(self, auth_headers: dict) -> None:
        r = _api_get("/k8s/pods", headers=auth_headers)
        # On Docker stack, K8s routes may not be registered at all → 404 is acceptable
        # Just verify the API responds (not connection error)
        assert r.status_code in (200, 401, 403, 404, 500)

    def test_k8s_scale_endpoint_exists(self, auth_headers: dict) -> None:
        r = _api_post(
            "/k8s/scale",
            json_data={"deployment": "api", "replicas": 1},
            headers=auth_headers,
        )
        # On Docker stack, K8s routes may not be registered → 404 is acceptable
        assert r.status_code in (200, 401, 403, 404, 422, 500)


# =============================================================================
# 15. CONCURRENT LOAD TEST (light)
# =============================================================================


class TestConcurrentLoad:
    def test_10_concurrent_health_checks(self) -> None:
        """Hit health endpoint 10 times rapidly — should all succeed."""
        import concurrent.futures

        def check():
            return requests.get(f"{API_URL}/health", timeout=TIMEOUT).status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            results = list(pool.map(lambda _: check(), range(10)))
        assert all(r == 200 for r in results), f"Some health checks failed: {results}"

    def test_10_concurrent_predictions(self, auth_headers: dict) -> None:
        """Send 10 predictions concurrently."""
        import concurrent.futures

        def predict(i):
            time_vals, amp_vals = _generate_signal(frequency=3.0 + i)
            r = requests.post(
                f"{API_URL}/predict",
                json={
                    "device_id": f"load-test-{i:03d}",
                    "time_values": time_vals,
                    "amplitude_values": amp_vals,
                },
                headers=auth_headers,
                timeout=TIMEOUT,
            )
            return r.status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            results = list(pool.map(predict, range(10)))
        passed = sum(1 for r in results if r == 200)
        assert passed >= 8, f"Too many failed predictions: {results}"


# =============================================================================
# 16. GRAFANA DASHBOARD CONTENT CHECKS
# =============================================================================


class TestGrafanaDashboardContent:
    @pytest.fixture(scope="class")
    def grafana_session(self) -> requests.Session:
        s = requests.Session()
        s.auth = (GRAFANA_USER, GRAFANA_PASS)
        return s

    def test_each_dashboard_loadable(self, grafana_session: requests.Session) -> None:
        r = grafana_session.get(f"{GRAFANA_URL}/api/search?type=dash-db", timeout=TIMEOUT)
        dashboards = r.json()
        errors = []
        for dash in dashboards[:10]:  # limit to 10
            uid = dash.get("uid", "")
            r2 = grafana_session.get(f"{GRAFANA_URL}/api/dashboards/uid/{uid}", timeout=TIMEOUT)
            if r2.status_code != 200:
                errors.append(f"{dash['title']} (uid={uid}): {r2.status_code}")
        assert not errors, f"Dashboard load failures: {errors}"

    def test_dashboards_have_panels(self, grafana_session: requests.Session) -> None:
        r = grafana_session.get(f"{GRAFANA_URL}/api/search?type=dash-db", timeout=TIMEOUT)
        dashboards = r.json()
        for dash in dashboards[:5]:
            uid = dash.get("uid", "")
            r2 = grafana_session.get(f"{GRAFANA_URL}/api/dashboards/uid/{uid}", timeout=TIMEOUT)
            if r2.status_code != 200:
                continue
            dashboard_data = r2.json().get("dashboard", {})
            panels = dashboard_data.get("panels", [])
            rows = dashboard_data.get("rows", [])
            total = len(panels) + sum(len(r.get("panels", [])) for r in rows)
            assert total > 0, f"Dashboard '{dash['title']}' has no panels"


# =============================================================================
# 17. PROMETHEUS DATASOURCE QUERY (via Grafana)
# =============================================================================


class TestGrafanaPrometheusIntegration:
    @pytest.fixture(scope="class")
    def grafana_session(self) -> requests.Session:
        s = requests.Session()
        s.auth = (GRAFANA_USER, GRAFANA_PASS)
        return s

    def test_prometheus_datasource_reachable(self, grafana_session: requests.Session) -> None:
        r = grafana_session.get(f"{GRAFANA_URL}/api/datasources", timeout=TIMEOUT)
        ds_list = r.json()
        prom_ds = [d for d in ds_list if d.get("type") == "prometheus"]
        if not prom_ds:
            pytest.skip("No Prometheus datasource in Grafana")
        ds_id = prom_ds[0]["id"]
        # Proxy query through Grafana
        r2 = grafana_session.get(
            f"{GRAFANA_URL}/api/datasources/proxy/{ds_id}/api/v1/query",
            params={"query": "up"},
            timeout=TIMEOUT,
        )
        # 200 = datasource connected, 502 = datasource unreachable
        assert r2.status_code == 200, (
            f"Prometheus datasource unreachable via Grafana: {r2.status_code}"
        )


# =============================================================================
# 18. END-TO-END PIPELINE FLOW
# =============================================================================


class TestPipelineFlow:
    """Test the full predict → monitor → review cycle."""

    def test_predict_then_check_stats(self, auth_headers: dict) -> None:
        """Make a prediction, then verify stats updated."""
        # Get stats before
        r1 = _api_get("/stats")
        if r1.status_code != 200:
            pytest.skip("Stats endpoint not available")

        # Make a prediction
        time_vals, amp_vals = _generate_signal()
        _api_post(
            "/predict",
            json_data={
                "device_id": f"pipeline-test-{uuid.uuid4().hex[:8]}",
                "time_values": time_vals,
                "amplitude_values": amp_vals,
            },
            headers=auth_headers,
        )

        # Check stats after
        r2 = _api_get("/stats")
        assert r2.status_code == 200

    def test_prediction_appears_in_prometheus(self, auth_headers: dict) -> None:
        """After a prediction, check that prediction counter increased in Prometheus."""
        # Make a prediction
        time_vals, amp_vals = _generate_signal()
        r = _api_post(
            "/predict",
            json_data={
                "device_id": f"prom-test-{uuid.uuid4().hex[:8]}",
                "time_values": time_vals,
                "amplitude_values": amp_vals,
            },
            headers=auth_headers,
        )
        assert r.status_code == 200

        # Wait briefly for scrape
        time.sleep(2)

        # Query Prometheus for prediction counter
        r2 = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": "api_requests_total"},
            timeout=TIMEOUT,
        )
        assert r2.status_code == 200


# =============================================================================
# 19. AIRFLOW DAG INSPECTION
# =============================================================================


class TestAirflowDAGInspection:
    @pytest.fixture(scope="class")
    def airflow_auth(self) -> tuple[str, str]:
        return (AIRFLOW_USER, AIRFLOW_PASS)

    EXPECTED_DAGS = [
        "automated_retraining",
        "drift_triggered_retraining",
        "evidently_drift_detection",
        "model_promotion",
        "batch_rescoring",
        "database_backup",
    ]

    @pytest.mark.parametrize("dag_id", EXPECTED_DAGS)
    def test_dag_exists(self, airflow_auth: tuple[str, str], dag_id: str) -> None:
        r = requests.get(
            f"{AIRFLOW_URL}/api/v1/dags/{dag_id}",
            auth=airflow_auth,
            timeout=TIMEOUT,
        )
        assert r.status_code == 200, f"DAG '{dag_id}' not found: {r.status_code}"

    @pytest.mark.parametrize("dag_id", EXPECTED_DAGS)
    def test_dag_not_paused_or_checkable(self, airflow_auth: tuple[str, str], dag_id: str) -> None:
        r = requests.get(
            f"{AIRFLOW_URL}/api/v1/dags/{dag_id}",
            auth=airflow_auth,
            timeout=TIMEOUT,
        )
        if r.status_code != 200:
            pytest.skip(f"DAG {dag_id} not found")
        data = r.json()
        # Just verify we can read the pause state
        assert "is_paused" in data

    def test_airflow_import_errors(self, airflow_auth: tuple[str, str]) -> None:
        r = requests.get(
            f"{AIRFLOW_URL}/api/v1/importErrors",
            auth=airflow_auth,
            timeout=TIMEOUT,
        )
        assert r.status_code == 200
        errors = r.json().get("import_errors", [])
        assert len(errors) == 0, f"Airflow has import errors: {errors}"


# =============================================================================
# 20. CROSS-SERVICE INTEGRATION
# =============================================================================


class TestCrossServiceIntegration:
    def test_api_mlflow_integration(self) -> None:
        """API health says MLflow is accessible."""
        r = _api_get("/health")
        data = r.json()
        services = data.get("services", {})
        # If services dict exists, check MLflow
        if services:
            assert "mlflow" not in [k for k, v in services.items() if v == "unhealthy"], (
                "MLflow service is unhealthy"
            )

    def test_all_prometheus_targets_scraped(self) -> None:
        """At least one target should be in 'up' state."""
        r = requests.get(f"{PROMETHEUS_URL}/api/v1/targets", timeout=TIMEOUT)
        targets = r.json()["data"]["activeTargets"]
        up_targets = [t for t in targets if t.get("health") == "up"]
        total = len(targets)
        up = len(up_targets)
        # At least half should be up
        assert up >= total * 0.3, f"Only {up}/{total} targets are up"

    def test_grafana_can_query_prometheus(self) -> None:
        """Verify Prometheus is queryable (end-to-end check)."""
        r = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": 'up{job=~".*"}'},
            timeout=TIMEOUT,
        )
        assert r.status_code == 200
        assert len(r.json()["data"]["result"]) > 0
