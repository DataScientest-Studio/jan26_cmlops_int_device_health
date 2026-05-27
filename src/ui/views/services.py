"""Service Dashboard — links + embedded iframes for every running service.

URLs are resolved **dynamically** via ``docker port`` so that the actual
host-port mappings (which may differ from the default internal ones) are
always used.  See :func:`src.ui.components.docker_utils.get_service_url`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from src.ui.components.docker_utils import (
    get_container_statuses,
    get_host,
    get_k8s_pod_statuses,
    get_service_url,
)
from src.ui.logging_ui import get_ui_logger
from src.ui.styles import get_global_css, hero_section

_logger = get_ui_logger(__name__)

# Static service definitions — URLs are built at render-time.
# ORDER: Core Networking → Core Application → Databases → ML Platform → Monitoring
# (matches the canonical Docker container ordering used throughout the UI)
#
# Tuple: (display_name, container_name, internal_port, path, description, icon, iframe_ok)
# iframe_ok=False → service has no web UI; shown in list but non-clickable and disabled in viewer
_SERVICES_STATIC: list[tuple[str, str, int, str, str, str, bool]] = [
    # ── Core Networking ──────────────────────────────────────────────────────
    ("Nginx Status", "mlops_nginx", 80, "/health", "Reverse proxy health check.", "🌐", True),
    # ── Core Application ─────────────────────────────────────────────────────
    (
        "FastAPI Docs (Swagger)",
        "mlops_nginx",
        80,
        "/docs",
        "Interactive API documentation with try-it-out support.",
        "⚡",
        True,
    ),
    (
        "FastAPI Docs (ReDoc)",
        "mlops_nginx",
        80,
        "/redoc",
        "Alternate API documentation with search.",
        "📄",
        True,
    ),
    # ── Databases ────────────────────────────────────────────────────────────
    (
        "PostgreSQL (App)",
        "mlops_postgres",
        5432,
        "",
        "Application database (device signals, predictions, labels). "
        "TCP-only \u2014 no web UI available.",
        "🗄️",
        False,  # No web UI — TCP wire protocol only
    ),
    # ── ML Platform ──────────────────────────────────────────────────────────
    (
        "MLflow Tracking",
        "mlops_mlflow",
        5000,
        "",
        "Experiment tracking, model registry, and artefact browser. (Local mode)",
        "🔬",
        False,  # MLflow sets X-Frame-Options: SAMEORIGIN — open in new tab
    ),
    (
        "Airflow Webserver",
        "mlops_airflow",
        8081,
        "",
        "DAG management, task logs, and scheduling interface.",
        "🔄",
        True,
    ),
    # ── Monitoring ───────────────────────────────────────────────────────────
    (
        "Grafana Dashboards",
        "mlops_grafana",
        3000,
        "",
        "Pre-provisioned dashboards for predictions, model performance, and system health.",
        "📈",
        True,
    ),
    (
        "Prometheus",
        "mlops_prometheus",
        9090,
        "",
        "PromQL query interface and metric exploration.",
        "📊",
        True,
    ),
    (
        "Alertmanager",
        "mlops_alertmanager",
        9093,
        "",
        "Active alerts, silences, and notification routing.",
        "🔔",
        True,
    ),
    (
        "cAdvisor",
        "mlops_cadvisor",
        8080,
        "",
        "Real-time container resource usage and performance stats.",
        "📦",
        True,
    ),
    (
        "Node Exporter",
        "mlops_node_exporter",
        9100,
        "/metrics",
        "Raw host metrics in Prometheus format.",
        "💻",
        True,
    ),
    (
        "PostgreSQL Exporter (App DB)",
        "mlops_postgres_exporter",
        9187,
        "/metrics",
        "Exports PostgreSQL app-DB metrics (connections, cache hit ratio, query duration) to Prometheus.",
        "🐘",
        True,
    ),
    (
        "Blackbox Exporter",
        "mlops_blackbox_exporter",
        9115,
        "",
        "HTTP/HTTPS endpoint prober. Exposes probe_success metric for NginxDown and AirflowDown alerts.",
        "🔍",
        True,
    ),
]

# Cloud-mode-only services (not present in local mode)
_SERVICES_CLOUD_EXTRA: list[tuple[str, str, int, str, str, str, bool]] = [
    (
        "MLflow Buffer (Cloud)",
        "mlops_mlflow_buffer",
        5000,
        "",
        "Local-first MLflow buffer container (primary MLflow in cloud mode). Syncs to DagsHub on schedule.",
        "🔬",
        True,  # MLflow UI is embeddable in Streamlit srcdoc iframes (SAMEORIGIN check not triggered)
    ),
    (
        "PostgreSQL (MLflow)",
        "mlops_postgres_mlflow",
        5432,
        "",
        "Dedicated PostgreSQL backend for the MLflow buffer container (cloud mode only). "
        "TCP-only \u2014 no web UI available.",
        "🗄️",
        False,  # No web UI — TCP wire protocol only
    ),
    (
        "PostgreSQL Exporter (MLflow DB)",
        "mlops_postgres_mlflow_exporter",
        9187,
        "/metrics",
        "Exports PostgreSQL MLflow-DB metrics (connections, cache hit ratio, query duration) to Prometheus.",
        "🐘",
        True,
    ),
]


def _build_services_meta_k8s() -> list[tuple[str, str, str, str, bool]]:
    """Build K8s-specific service list.

    Only includes services that are available via ``make k8s-ports`` port-forwards.
    MLflow is exposed through the K8s nginx proxy at ``/mlflow/`` so it can be
    embedded in iframes (nginx strips ``X-Frame-Options``).

    Services NOT included (not present in K8s cluster):
    - Alertmanager, cAdvisor, Node Exporter, PG Exporter, Blackbox Exporter,
      MLflow Buffer (Docker-cloud-only), PG MLflow, PG MLflow Exporter.
    """
    nginx_port = os.environ.get("NGINX_HTTP_PORT", "8888")
    airflow_port = os.environ.get("AIRFLOW_PORT", "8080")
    grafana_port = os.environ.get("GRAFANA_PORT", "3000")
    prometheus_port = os.environ.get("PROMETHEUS_PORT", "9090")
    db_port = os.environ.get("DB_PORT", "5434")
    host = get_host()

    nginx_base = f"http://{host}:{nginx_port}"
    return [
        (
            "Nginx Status",
            f"{nginx_base}/health",
            "K8s nginx reverse-proxy health check.",
            "🌐",
            True,
        ),
        (
            "FastAPI Docs (Swagger)",
            f"{nginx_base}/docs",
            "Interactive API documentation with try-it-out support.",
            "⚡",
            True,
        ),
        (
            "FastAPI Docs (ReDoc)",
            f"{nginx_base}/redoc",
            "Alternate API documentation with search.",
            "📄",
            True,
        ),
        (
            "PostgreSQL (App)",
            f"postgresql://{host}:{db_port}",
            "Application database (device signals, predictions, labels). TCP-only — no web UI.",
            "🗄️",
            False,
        ),
        (
            "MLflow Tracking",
            f"{nginx_base}/mlflow/",
            "Experiment tracking and model registry — embedded via K8s nginx proxy.",
            "🔬",
            True,  # K8s nginx strips X-Frame-Options via the /mlflow/ proxy route
        ),
        (
            "Airflow Webserver",
            f"http://{host}:{airflow_port}",
            "DAG management, task logs, and scheduling interface.",
            "🔄",
            True,
        ),
        (
            "Grafana Dashboards",
            f"http://{host}:{grafana_port}",
            "Pre-provisioned dashboards for predictions, model performance, and system health.",
            "📈",
            True,
        ),
        (
            "Prometheus",
            f"http://{host}:{prometheus_port}",
            "PromQL query interface and metric exploration.",
            "📊",
            True,
        ),
    ]


def _build_services_meta() -> list[tuple[str, str, str, str, bool]]:
    """Build (display_name, url, description, icon, iframe_ok) with dynamic ports.

    K8s mode uses a separate, minimal service list (only port-forwarded services).
    Cloud mode changes:
    - The static ``MLflow Tracking`` entry (mlops_mlflow, local mode) is hidden;
      instead ``MLflow Buffer (Cloud)`` (mlops_mlflow_buffer) is shown.
    - ``mlops_postgres_mlflow`` is inserted right after ``mlops_postgres`` (App).
    - ``mlops_postgres_mlflow_exporter`` is appended at end.
    - Airflow is skipped in local mode (no container running).
    """
    mode = _detect_mode()

    # K8s mode: use a dedicated service list with correct port-forward ports
    if mode == "k8s":
        return _build_services_meta_k8s()

    result: list[tuple[str, str, str, str, bool]] = []
    for name, container, port, path, desc, icon, iframe_ok in _SERVICES_STATIC:
        # Skip Airflow in local mode — container is disabled (scale: 0)
        if name == "Airflow Webserver" and mode == "local":
            continue
        # In cloud mode or k8s, skip the local MLflow container entry; direct MLflow used.
        if name == "MLflow Tracking" and mode == "cloud":
            continue
        url = get_service_url(container, port, path)
        result.append((name, url, desc, icon, iframe_ok))
        # In cloud mode, insert PostgreSQL (MLflow) right after PostgreSQL (App)
        if mode == "cloud" and name == "PostgreSQL (App)":
            pg_mlflow = _SERVICES_CLOUD_EXTRA[1]  # PostgreSQL (MLflow)
            pg_url = get_service_url(pg_mlflow[1], pg_mlflow[2], pg_mlflow[3])
            result.append((pg_mlflow[0], pg_url, pg_mlflow[4], pg_mlflow[5], pg_mlflow[6]))

    # Cloud mode: add MLflow buffer before monitoring section (insert at correct position)
    if mode == "cloud":
        # MLflow Buffer goes after PostgreSQL (MLflow), before Grafana
        mlflow_buf = _SERVICES_CLOUD_EXTRA[0]
        # MLflow Buffer: serve through nginx at /mlflow-buffer/ so the browser
        # can embed it in the Streamlit iframe viewer (avoids SAMEORIGIN block).
        # nginx proxies to mlflow_buffer:5000 and strips X-Frame-Options.
        # MLflow is configured with --static-prefix /mlflow-buffer.
        import os as _os

        _nginx_http_port = _os.environ.get("NGINX_HTTP_PORT", "8080")
        buf_url = f"http://{get_host()}:{_nginx_http_port}/mlflow-buffer/"
        # Find Grafana position and insert before it
        grafana_idx = next(
            (i for i, r in enumerate(result) if r[0] == "Grafana Dashboards"), len(result)
        )
        result.insert(
            grafana_idx, (mlflow_buf[0], buf_url, mlflow_buf[4], mlflow_buf[5], mlflow_buf[6])
        )
        # PG MLflow exporter appended at end
        pg_exp = _SERVICES_CLOUD_EXTRA[2]
        pg_exp_url = get_service_url(pg_exp[1], pg_exp[2], pg_exp[3])
        result.append((pg_exp[0], pg_exp_url, pg_exp[4], pg_exp[5], pg_exp[6]))
    return result


def _detect_mode() -> str:
    """Detect deployment mode (local, cloud, or k8s).

    Priority: .current_mode file → DEPLOYMENT_MODE env var → default local.
    The file is always up-to-date (written on every stack start/stop), whereas
    DEPLOYMENT_MODE is baked into the process environment by ``make ui`` at
    startup and stays stale if the user switches mode from within Streamlit.
    Returns 'k8s' when running against the Kubernetes cluster so that the
    services page can display K8s pod health instead of Docker containers.
    """
    mode_file = Path(_PROJECT_ROOT) / ".current_mode"
    if mode_file.exists():
        val = mode_file.read_text().strip()
        if val in ("local", "cloud", "k8s"):
            return val
    env_mode = os.environ.get("DEPLOYMENT_MODE", "").strip()
    if env_mode in ("local", "cloud", "k8s"):
        return env_mode
    return "local"


def _load_env_file(name: str) -> dict[str, str]:
    """Parse a simple KEY=VALUE env file."""
    fpath = Path(_PROJECT_ROOT) / name
    env: dict[str, str] = {}
    if not fpath.exists():
        return env
    for line in fpath.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if value and not value.startswith("$"):
                env[key] = value
    return env


def _env_or_file(key: str) -> str:
    """Get env var, falling back to .env.secrets."""
    val = os.environ.get(key, "")
    if val:
        return val
    secrets = _load_env_file(".env.secrets")
    return secrets.get(key, "")


def _cloud_mlflow_url() -> str:
    """Return the DagsHub MLflow URL for cloud mode."""
    env_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if env_uri and "dagshub.com" in env_uri:
        return env_uri
    user = _env_or_file("DAGSHUB_USER")
    repo = _env_or_file("DAGSHUB_REPO")
    if user and repo:
        return f"https://dagshub.com/{user}/{repo}.mlflow"
    return get_service_url("mlops_mlflow", 5000)


def render() -> None:
    """Render the Services Dashboard page."""
    try:
        _render_content()
    except Exception:
        _logger.exception("Unhandled exception in services.render()")
        raise


def _render_content() -> None:
    """Inner render — separated so exceptions are caught and logged by render()."""
    st.markdown(get_global_css(), unsafe_allow_html=True)
    st.markdown(
        hero_section(
            "Service Dashboard",
            "Quick access to all running services — open in a new tab or view embedded below.",
        ),
        unsafe_allow_html=True,
    )

    # ── Quick-link cards ────────────────────────────────────────
    st.markdown(
        '<div class="section-header">🔗 Quick Links</div>'
        '<div class="section-subheader">Click to open any service in a new browser tab</div>',
        unsafe_allow_html=True,
    )

    # Build URL list dynamically from running Docker containers
    services_meta = _build_services_meta()

    # Check which containers / pods are up
    _current_mode = _detect_mode()
    if _current_mode == "k8s":
        container_statuses = {s.display_name: s for s in get_k8s_pod_statuses()}
    else:
        container_statuses = {s.display_name: s for s in get_container_statuses()}

    # Map from service display name to the container whose status should be
    # shown.  FastAPI Swagger/ReDoc are proxied via Nginx, but the relevant
    # health indicator is the *API* container, not Nginx.
    status_container_override: dict[str, str] = {
        "FastAPI Docs (Swagger)": "FastAPI Prediction",
        "FastAPI Docs (ReDoc)": "FastAPI Prediction",
    }

    cols = st.columns(3)
    for i, (name, url, desc, icon, iframe_ok) in enumerate(services_meta):
        with cols[i % 3]:
            # Find matching container status
            dot = "🟢"
            lookup_name = status_container_override.get(name, name)
            for cs in container_statuses.values():
                if (
                    cs.display_name.lower() in lookup_name.lower()
                    or lookup_name.lower() in cs.display_name.lower()
                ):
                    dot = cs.status_emoji
                    break

            # PostgreSQL containers have no web UI regardless of url
            no_web_ui = not iframe_ok and "PostgreSQL" in name

            if no_web_ui:
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:0.75rem;'
                    f"padding:0.5rem;border-radius:8px;opacity:0.55;"
                    f'border:1px dashed #334155">'
                    f'<span style="font-size:1.5rem">{icon}</span>'
                    f"<div>"
                    f'<div style="font-weight:600;color:#94a3b8">{dot} {name}</div>'
                    f'<div style="font-size:0.8rem;color:#64748b">{desc}</div>'
                    f'<div style="font-size:0.75rem;color:#475569;font-style:italic">'
                    f"TCP only \u2014 no web UI</div>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<a href="{url}" target="_blank" '
                    f'style="display:flex;align-items:center;gap:0.75rem;'
                    f"text-decoration:none;color:inherit;padding:0.5rem;"
                    f'border-radius:8px;transition:background 0.2s">'
                    f'<span style="font-size:1.5rem">{icon}</span>'
                    f"<div>"
                    f'<div style="font-weight:600">{dot} {name}</div>'
                    f'<div style="font-size:0.8rem;color:#94a3b8">{desc}</div>'
                    f"</div></a>",
                    unsafe_allow_html=True,
                )
            st.markdown("")

    # ── Embedded service viewer ─────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="section-header">🖥️ Embedded Service Viewer</div>'
        '<div class="section-subheader">Select a service to view its interface directly within this dashboard</div>',
        unsafe_allow_html=True,
    )

    # ALL services in the selectbox — PostgreSQL ones shown with "(no web UI)" suffix
    all_service_names = []
    url_map = {}
    iframe_ok_map = {}
    no_web_ui_set = set()
    for name, url, _, _, has_iframe in services_meta:
        is_no_ui = not has_iframe and "PostgreSQL" in name
        display = f"{name} (no web UI)" if is_no_ui else name
        all_service_names.append(display)
        url_map[display] = url
        iframe_ok_map[display] = has_iframe and not is_no_ui
        if is_no_ui:
            no_web_ui_set.add(display)

    selected = st.selectbox("Select service to embed", all_service_names, key="embed_service")
    embed_url = url_map.get(selected, "")
    can_iframe = iframe_ok_map.get(selected, True)
    is_no_web_ui = selected in no_web_ui_set

    col_h, col_open = st.columns([4, 1])
    with col_h:
        height = st.slider("Frame height (px)", 400, 1200, 700, 50)
    with col_open:
        if not is_no_web_ui:
            st.markdown(
                f'<a href="{embed_url}" target="_blank" '
                f'style="display:inline-block;margin-top:1.5rem;padding:0.4rem 1rem;'
                f"background:#6366f1;color:white;border-radius:8px;"
                f'text-decoration:none;font-weight:600;font-size:0.85rem">'
                f"Open in new tab \u2197</a>",
                unsafe_allow_html=True,
            )

    if is_no_web_ui:
        st.info(
            f"**{selected.replace(' (no web UI)', '')}** is a database service that communicates "
            f"via the PostgreSQL TCP wire protocol. There is no HTTP web interface available.  \n\n"
            f"Use a tool like **pgAdmin**, **DBeaver**, or `psql` to connect directly on the published port."
        )
    elif can_iframe:
        st.markdown(
            f'<div class="iframe-container">'
            f'<iframe src="{embed_url}" width="100%" height="{height}px" '
            f'style="border:none;border-radius:12px;background:#1e293b" '
            f'loading="lazy" sandbox="allow-same-origin allow-scripts allow-popups allow-forms">'
            f"</iframe></div>",
            unsafe_allow_html=True,
        )
    else:
        st.info(
            f"**{selected}** is hosted externally and does not support iframe embedding "
            f"(the provider blocks it via X-Frame-Options / CSP headers).  \n\n"
            f"**[Open {selected} in a new tab \u2197]({embed_url})**"
        )

    st.caption(
        "\u26a0\ufe0f Some services may block iframe embedding due to X-Frame-Options headers. "
        "Use the 'Open in new tab' link as a fallback."
    )
