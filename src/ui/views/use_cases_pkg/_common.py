"""Shared helpers for use-case tabs: mode detection, MLflow client, API helpers, CSS."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]

# ---------------------------------------------------------------------------
# Custom CSS for coloured expander-like sections
# ---------------------------------------------------------------------------

SECTION_CSS = """
<style>
/* Section headers inside cards — always readable */
.signal-section-healthy {
    background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
    border-left: 4px solid #4caf50;
    border-radius: 8px;
    padding: 0.65rem 1rem;
    margin-bottom: 0.75rem;
    color: #e8f5e9 !important;
}
.signal-section-healthy strong { color: #e8f5e9 !important; }

.signal-section-unhealthy {
    background: linear-gradient(135deg, #b71c1c 0%, #c62828 100%);
    border-left: 4px solid #e53935;
    border-radius: 8px;
    padding: 0.65rem 1rem;
    margin-bottom: 0.75rem;
    color: #ffebee !important;
}
.signal-section-unhealthy strong { color: #ffebee !important; }

.signal-section-general {
    background: linear-gradient(135deg, #0d47a1 0%, #1565c0 100%);
    border-left: 4px solid #1e88e5;
    border-radius: 8px;
    padding: 0.65rem 1rem;
    margin-bottom: 0.75rem;
    color: #e3f2fd !important;
}
.signal-section-general strong { color: #e3f2fd !important; }
</style>
"""

# Lorentzian γ (HWHM) = √(2 ln 2) × σ ≈ 1.1775 × σ
GAMMA_SIGMA_FACTOR = 1.1775

MODEL_NAME_BASE = "device_health_classifier"


def get_model_name(mode: str | None = None) -> str:
    """Return the model registry name.

    Priority:
    1. ``MODEL_REGISTRY_NAME`` env var — set explicitly by the ``.env.*`` files,
       which contain the canonical name for each deployment mode.
    2. Constructed from the base name + mode suffix as a fallback.
    """
    env_name = os.environ.get("MODEL_REGISTRY_NAME", "").strip()
    if env_name:
        return env_name
    m = mode or detect_mode()
    return f"{MODEL_NAME_BASE}_{m}"


def get_experiment_name(mode: str | None = None) -> str:
    """Return the MLflow experiment name with a mode suffix.

    Uses ``MODEL_REGISTRY_NAME`` env var when set (keeps experiment and model
    names aligned for the K8s deployment which uses un-suffixed names).
    """
    env_name = os.environ.get("MODEL_REGISTRY_NAME", "").strip()
    if env_name:
        return env_name
    m = mode or detect_mode()
    return f"{MODEL_NAME_BASE}_{m}"


# Kept for backward-compat imports; callers that need mode-awareness should
# prefer ``get_model_name()`` instead.
MODEL_NAME = MODEL_NAME_BASE


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------


def detect_mode() -> str:
    """Return 'local' or 'cloud'.

    Priority:
    1. ``DEPLOYMENT_MODE`` env var — set explicitly by ``make ui`` at launch time.
       This is the most reliable source because ``make ui`` reads ``.current_mode``
       and forwards it as an env var before starting Streamlit.
    2. ``.current_mode`` file — written by ``compose_up()`` / ``make local`` /
       ``make cloud``.
    3. Default to ``'local'``.
    """
    # 1. .current_mode file — always updated when mode changes (authoritative).
    #    DEPLOYMENT_MODE env var is baked in at ``make ui`` startup and becomes
    #    stale when the user switches from local → cloud (or GHCR) inside the UI.
    mode_file = PROJECT_ROOT / ".current_mode"
    if mode_file.exists():
        val = mode_file.read_text().strip()
        if val in ("local", "cloud"):
            return val
        # K8s mode has all cloud services (Airflow, MLflow, full registry).
        # Map to 'cloud' so cloud-mode features are enabled in use-cases.
        if val == "k8s":
            return "cloud"

    # 2. Env var fallback (useful when .current_mode is absent on first run / CI)
    env_mode = os.environ.get("DEPLOYMENT_MODE", "").strip()
    if env_mode in ("local", "cloud"):
        return env_mode
    if env_mode == "k8s":
        return "cloud"

    return "local"


def get_host_db_url() -> str:
    """Return a PostgreSQL URL accessible from the HOST (Streamlit process).

    Streamlit runs directly on the host — not inside Docker — so it cannot
    reach the ``postgres`` Docker hostname.

    Tries candidates in order, returning the first that connects successfully:

    1. ``POSTGRES_HOST:POSTGRES_PORT`` — explicit env var (``make ui`` sets
       this to ``localhost`` by default; override in ``.env.windows.local``
       or your shell for non-standard setups).
    2. ``localhost:{docker_port}`` — ``docker port`` discovery of the actual
       host-mapped port.  Works on all platforms (Windows, macOS, Linux)
       regardless of Docker runtime (Docker Desktop, OrbStack, Colima).
    3. ``mlops_postgres.orb.local:5432`` — OrbStack direct-connect DNS.
       Bypasses host port mapping so it works even when a local PostgreSQL
       already occupies port 5432 on the macOS host.

    Returns an empty string when no candidate connects successfully.
    """
    # Read credentials: env vars first (set by Makefile/shell), then defaults
    # that match the docker-compose.yml fallbacks.
    db_user = os.environ.get("DB_USER", "mlops_user")
    db_pass = os.environ.get("DB_PASSWORD", "changeme")
    db_name = os.environ.get("DB_NAME", "mlops_db")

    # Also try reading from .env.secrets if env vars are at defaults
    if db_pass == "changeme":
        secrets_file = PROJECT_ROOT / ".env.secrets"
        if secrets_file.exists():
            for line in secrets_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("DB_PASSWORD="):
                    db_pass = line.split("=", 1)[1].strip()
                elif line.startswith("DB_USER="):
                    db_user = line.split("=", 1)[1].strip()
                elif line.startswith("DB_NAME="):
                    db_name = line.split("=", 1)[1].strip()

    # Build an ordered candidate list and return the first URL that connects.
    # This works on all platforms with zero machine-specific config:
    #   1. POSTGRES_HOST env var (set by make ui; default: localhost)
    #   2. docker port discovery — actual host-mapped port, works everywhere
    #   3. mlops_postgres.orb.local — OrbStack DNS fallback for macOS users
    #      whose local PostgreSQL occupies port 5432 (OrbStack bypasses host
    #      port mapping and reaches the container directly)
    from src.database.database import Database
    from src.ui.components.docker_utils import get_host_port

    postgres_host = os.environ.get("POSTGRES_HOST", "").strip()
    postgres_port = os.environ.get("POSTGRES_PORT", "5432")

    seen: set[str] = set()
    candidates: list[str] = []

    def _add_url(url: str) -> None:
        if url not in seen:
            seen.add(url)
            candidates.append(url)

    if postgres_host:
        _add_url(f"postgresql://{db_user}:{db_pass}@{postgres_host}:{postgres_port}/{db_name}")

    # Docker port discovery: returns the actual host-mapped port regardless of OS.
    docker_port = get_host_port("mlops_postgres", 5432)
    if docker_port:
        _add_url(f"postgresql://{db_user}:{db_pass}@localhost:{docker_port}/{db_name}")

    # OrbStack-only fallback: direct container DNS, bypasses host port mapping.
    _add_url(f"postgresql://{db_user}:{db_pass}@mlops_postgres.orb.local:5432/{db_name}")

    for url in candidates:
        try:
            _db = Database(db_url=url)
            _db.count_all_signals()
            _db.close()
            return url
        except Exception:
            continue

    return ""


# ---------------------------------------------------------------------------
# FastAPI helpers
# ---------------------------------------------------------------------------


def stop_api() -> None:
    """Stop the FastAPI container."""
    from src.ui.components.docker_utils import compose_stop_service

    compose_stop_service("api")


def reload_api_model() -> tuple[bool, str, dict]:
    """Ask the running API to reload its model from the registry via /admin/reload-model.

    This is preferred over a full container restart: no downtime, instant feedback,
    and the API keeps serving the old model while the new one loads.

    Returns:
        (success, human-readable message, full JSON response dict)
    """
    import json
    import os
    import time as _time
    import urllib.error
    import urllib.request

    # Prefer Nginx-proxied URL (always reachable from host, even on Windows/Mac
    # where port 8000 is not published). Fall back to direct port when Nginx port
    # is also absent (local dev without Docker).
    nginx_port = os.environ.get("NGINX_HTTP_PORT", os.environ.get("NGINX_PORT", ""))
    api_port = os.environ.get("API_PORT", "8000")
    if nginx_port:
        url = f"http://localhost:{nginx_port}/admin/reload-model"
    else:
        url = f"http://localhost:{api_port}/admin/reload-model"

    # Retry up to 3 times with a short delay.  When the model is loading Nginx
    # may briefly return 502; a quick retry avoids a full container restart.
    last_exc: str = ""
    for _attempt in range(3):
        if _attempt > 0:
            _time.sleep(3)
        try:
            req = urllib.request.Request(
                url,
                method="POST",
                headers={"Accept": "application/json", "Content-Length": "0"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data: dict = json.loads(resp.read().decode())
            return True, data.get("message", "Model reloaded"), data
        except urllib.error.HTTPError as exc:
            body = exc.read().decode()[:300]
            last_exc = f"API returned HTTP {exc.code}: {body}"
        except Exception as exc:
            last_exc = str(exc)
    return False, last_exc, {}


def restart_api() -> tuple[str, int]:
    """Restart the FastAPI container (hard restart via Docker).

    Prefer :func:`reload_api_model` for model hot-reload without downtime.
    Use this only when a full container restart is required.
    """
    from src.ui.components.docker_utils import compose_restart

    return compose_restart("api")


# ---------------------------------------------------------------------------
# MLflow client
# ---------------------------------------------------------------------------


def _mlflow_client_factory(tracking_uri: str, _user: str, _token: str):  # noqa: ANN001, ANN202
    """Create an MlflowClient for the given URI (called by the cache wrapper below).

    Kept as a plain function so ``reset_mlflow_client_cache()`` can call
    ``_cached_mlflow_client.clear()`` from anywhere in the codebase.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


def _get_cache_decorator():  # noqa: ANN202
    """Return the st.cache_resource-wrapped factory (imported lazily to avoid Streamlit boot issues)."""
    import streamlit as st

    return st.cache_resource(show_spinner=False)(_mlflow_client_factory)


# Module-level cached factory — one MlflowClient per (tracking_uri, user, token) tuple.
# Using a mutable container so reset_mlflow_client_cache() can swap it out.
_cached_mlflow_client_holder: list = []


def _get_cached_factory():  # noqa: ANN202
    """Return (and lazily initialise) the @st.cache_resource wrapped factory."""
    if not _cached_mlflow_client_holder:
        _cached_mlflow_client_holder.append(_get_cache_decorator())
    return _cached_mlflow_client_holder[0]


def reset_mlflow_client_cache() -> None:
    """Clear the MLflow client resource cache.

    Call this when the MLflow container is known to have restarted mid-session
    so the next ``get_mlflow_client()`` call creates a fresh ``MlflowClient``
    instead of reusing a stale cached one.
    """
    if _cached_mlflow_client_holder:
        with contextlib.suppress(Exception):
            _cached_mlflow_client_holder[0].clear()
        _cached_mlflow_client_holder.clear()


def get_mlflow_client():  # noqa: ANN202
    """Return a configured (MlflowClient, tracking_uri) tuple.

    The underlying MlflowClient instance is cached per-URI via
    ``st.cache_resource`` so buffer connections are reused across Streamlit
    reruns instead of reconnected on every render.

    If the MLflow container restarts mid-session the caller should call
    ``reset_mlflow_client_cache()`` before calling this function again so a
    fresh connection is established.
    """
    from src.ui.views.mlflow_explorer import _tracking_uri

    uri = _tracking_uri()
    os.environ["MLFLOW_TRACKING_URI"] = uri

    # In cloud mode the buffer is local — no DagsHub credentials needed here.
    # (DagsHub credentials are only used by the sync/restore buttons in the
    # MLflow Explorer DagsHub View / Sync tabs.)
    mode = detect_mode()
    user = ""
    token = ""
    if mode == "cloud":
        from src.ui.views.mlflow_explorer import _env_or_file

        user = _env_or_file("DAGSHUB_USER") or ""
        token = _env_or_file("DAGSHUB_TOKEN") or ""
        if user:
            os.environ["MLFLOW_TRACKING_USERNAME"] = user
        if token:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    cached_factory = _get_cached_factory()
    return cached_factory(uri, user, token), uri


def fetch_champion_info(client, mode: str | None = None):  # noqa: ANN001, ANN202
    """Return (model_version, run) for the Production champion, or (None, None).

    Uses alias-first lookup (MLflow 3.x ``champion`` alias), then falls
    back to legacy ``current_stage == 'Production'`` for older servers.
    """
    model_name = get_model_name(mode)
    # Strategy 1: alias lookup (MLflow >= 2.9 / 3.x)
    try:
        mv = client.get_model_version_by_alias(model_name, "champion")
        try:
            run = client.get_run(mv.run_id)
            return mv, run
        except Exception:
            return mv, None
    except Exception:
        pass  # alias not set or server doesn't support aliases

    # Strategy 2: scan all versions for champion alias
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception:
        return None, None
    for v in sorted(versions, key=lambda x: int(x.version), reverse=True):
        v_aliases = getattr(v, "aliases", []) or []
        if "champion" in v_aliases:
            try:
                run = client.get_run(v.run_id)
                return v, run
            except Exception:
                return v, None
    return None, None
