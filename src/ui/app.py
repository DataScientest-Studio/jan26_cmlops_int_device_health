"""
MLOps Device Health — Streamlit Dashboard
==========================================

Main entry-point.  Run with:

    streamlit run src/ui/app.py

Or via the project Makefile:

    make ui
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path (Streamlit runs scripts directly)
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
import streamlit.components.v1 as _stc

# ── Configure UI logging (loguru) ───────────────────────────────
# Must be called BEFORE any page module is imported so that loggers
# obtained via get_ui_logger() in views are already wired to the file sink.
from src.ui.logging_ui import configure_ui_logging  # noqa: E402

configure_ui_logging()

# ── Page config (must be first Streamlit call) ──────────────────
st.set_page_config(
    page_title="MLOps Device Health",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import page modules ─────────────────────────────────────────
from src.ui.views import (  # noqa: E402
    about,
    airflow_control,
    app_console,
    architecture,
    dags,
    data_pipeline,
    database,
    docker_control,
    dvc_dagshub,
    github_dashboard,
    home,
    kubernetes,
    mlflow_explorer,
    monitoring,
    predictions,
    services,
    use_cases,
)

# ── Navigation definition ───────────────────────────────────────
PAGES: dict[str, dict] = {
    "🏠 Home": {"module": home, "icon": "🏠"},
    "🏗️ Architecture": {"module": architecture, "icon": "🏗️"},
    "📦 DVC & DagsHub": {"module": dvc_dagshub, "icon": "📦"},
    "📡 Data & Signals": {"module": data_pipeline, "icon": "📡"},
    "🔄 DAGs & Pipelines": {"module": dags, "icon": "🔄"},
    "🐳 Docker Control": {"module": docker_control, "icon": "🐳"},
    "🔗 Services": {"module": services, "icon": "🔗"},
    "🧪 Use Cases": {"module": use_cases, "icon": "🧪"},
    "🎯 Predictions": {"module": predictions, "icon": "🎯"},
    "🗄️ PostgreSQL Database": {"module": database, "icon": "🗄️"},
    "🔬 MLflow Explorer": {"module": mlflow_explorer, "icon": "🔬"},
    "✈️ Airflow Control": {"module": airflow_control, "icon": "✈️"},
    "📊 Monitoring": {"module": monitoring, "icon": "📊"},
    "🐙 GitHub CI/CD": {"module": github_dashboard, "icon": "🐙"},
    "☸️ Kubernetes": {"module": kubernetes, "icon": "☸️"},
    "🖥️ App Console": {"module": app_console, "icon": "🖥️"},
    "ℹ️ About": {"module": about, "icon": "ℹ️"},
}


def main() -> None:
    """Application entry-point."""

    # Use Streamlit's built-in sidebar for fixed navigation
    st.sidebar.markdown(
        """
        <div style="text-align:center;padding:0.5rem 0 0.3rem 0">
            <div style="font-size:2rem">🏥</div>
            <div style="font-size:0.9rem;font-weight:700;
                 background:linear-gradient(135deg,#6366f1,#06b6d4);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                 background-clip:text">
                MLOps Device Health
            </div>
            <div style="font-size:0.65rem;color:#64748b;margin-top:0.15rem">
                IoT Health Monitoring
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    selected = st.sidebar.radio(
        "Navigation",
        options=list(PAGES.keys()),
        index=0,
        key="_nav_radio",
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")
    import os
    from pathlib import Path as _Path

    from src.ui.components.docker_utils import get_service_url

    def _detect_sidebar_mode() -> str:
        mode_file = _Path(__file__).resolve().parents[2] / ".current_mode"
        if mode_file.exists():
            return mode_file.read_text().strip()
        return os.environ.get("DEPLOYMENT_MODE", "unknown")

    @st.cache_data(ttl=30, show_spinner=False)
    def _sidebar_links(mode: str) -> dict[str, str]:
        """Build sidebar quick-links. Mode is passed explicitly so cache
        invalidates when the deployment mode changes."""
        # MLflow port: use env vars (set by make ui from the correct .env file)
        if mode == "cloud":
            mlflow_port = os.environ.get("MLFLOW_BUFFER_PORT", "5002")
            mlflow_url = os.environ.get("MLFLOW_TRACKING_URI", f"http://localhost:{mlflow_port}")
        elif mode == "k8s":
            # K8s MLflow via port-forward; MLFLOW_TRACKING_URI=http://localhost:5000
            mlflow_url = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        else:
            mlflow_port = os.environ.get("MLFLOW_PORT", "5001")
            mlflow_url = os.environ.get("MLFLOW_TRACKING_URI", f"http://localhost:{mlflow_port}")
        if mode == "k8s":
            nginx_port = os.environ.get("NGINX_HTTP_PORT", "8888")
            return {
                "grafana": get_service_url("mlops_grafana", 3000),
                "mlflow": mlflow_url,
                "airflow": get_service_url("mlops_airflow", 8080),
                "prometheus": get_service_url("mlops_prometheus", 9090),
                "api_docs": f"http://localhost:{nginx_port}/docs",
            }
        return {
            "grafana": get_service_url("mlops_grafana", 3000),
            "mlflow": mlflow_url,
            "airflow": get_service_url("mlops_airflow", 8080),
            "prometheus": get_service_url("mlops_prometheus", 9090),
            "api_docs": get_service_url("mlops_nginx", 80, "/docs"),
        }

    _sidebar_mode = _detect_sidebar_mode()
    _links = _sidebar_links(_sidebar_mode)
    _grafana = _links["grafana"]
    _mlflow = _links["mlflow"]
    _airflow = _links["airflow"]
    _prometheus = _links["prometheus"]
    _api_docs = _links["api_docs"]

    _airflow_line = (
        f'<a href="{_airflow}" target="_blank" style="color:#818cf8">🔄 Airflow</a><br>'
        if _sidebar_mode in ("cloud", "k8s")
        else '<span style="color:#475569">🔄 Airflow (disabled)</span><br>'
    )
    st.sidebar.markdown(
        f"""
        <div style="font-size:0.75rem;color:#64748b">
            <b>Quick Links</b><br>
            <a href="{_grafana}" target="_blank" style="color:#818cf8">📈 Grafana</a><br>
            <a href="{_mlflow}" target="_blank" style="color:#818cf8">🔬 MLflow</a><br>
            {_airflow_line}
            <a href="{_prometheus}" target="_blank" style="color:#818cf8">📊 Prometheus</a><br>
            <a href="{_api_docs}" target="_blank" style="color:#818cf8">⚡ API Docs</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Main content area — show a loading spinner when navigating to a new page
    # so the user gets immediate feedback and doesn't see stale content from
    # the previous page while the new page is being rendered.
    page = PAGES[selected]
    _prev_page = st.session_state.get("_active_page")
    _is_new_page = _prev_page != selected
    st.session_state["_active_page"] = selected

    if _is_new_page:
        # Scroll the Streamlit main pane to the top whenever the user navigates
        # to a different page.  The iframe approach via components.v1.html is
        # the only reliable way to run JavaScript in a Streamlit app.
        _stc.html(
            "<script>window.parent.document.querySelector('.main').scrollTop = 0;</script>",
            height=0,
        )
        with st.spinner("🔄 Loading page…"):
            page["module"].render()
    else:
        page["module"].render()


if __name__ == "__main__":
    main()
