"""About / Impressum page — credits, tech stack, acknowledgements."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from src.ui.logging_ui import get_ui_logger
from src.ui.styles import get_global_css, hero_section

_logger = get_ui_logger(__name__)


def render() -> None:
    """Render the About / Impressum page."""
    try:
        _render_content()
    except Exception:
        _logger.exception("Unhandled exception in about.render()")
        raise


def _render_content() -> None:
    """Inner render — separated so exceptions are caught and logged by render()."""
    st.markdown(get_global_css(), unsafe_allow_html=True)
    st.markdown(
        hero_section(
            "About This Project",
            "Impressum, technology acknowledgements, and project credits.",
        ),
        unsafe_allow_html=True,
    )

    # ── Project info ────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">📋 Project Information</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="info-card">'
        "<h3>MLOps Device Health Monitoring System</h3>"
        "<p>A production-grade end-to-end MLOps platform for IoT device health prediction, "
        "developed as a capstone project to demonstrate modern machine learning operations "
        "practices.</p>"
        "<ul>"
        "<li><b>Domain:</b> IoT device health classification using signal analysis</li>"
        "<li><b>ML Approach:</b> Semi-supervised learning with synthetic signal data</li>"
        "<li><b>Architecture:</b> Containerised microservices with full observability</li>"
        "<li><b>Timeline:</b> ~15 weeks of implementation (Feb 6 – May 26, 2026)</li>"
        "</ul></div>",
        unsafe_allow_html=True,
    )

    # ── Author ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="section-header">👤 Author</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="info-card">'
        "<h3>Fred Richter</h3>"
        "<p>Project Lead — Design, Architecture, Implementation</p>"
        "<p>Responsible for the complete system design and full-stack implementation: "
        "signal processing, ML pipeline, FastAPI microservice, CI/CD, Docker and "
        "Kubernetes deployment, MLflow experiment tracking, Airflow orchestration, "
        "Prometheus/Grafana monitoring, and DagsHub integration.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Technology stack ────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="section-header">🛠️ Technology Stack</div>',
        unsafe_allow_html=True,
    )

    tech_categories = {
        "Core ML & Data": [
            ("Python 3.12", "Core language with uv package manager"),
            ("scikit-learn", "Logistic Regression, model evaluation"),
            ("XGBoost", "Gradient boosting classifier"),
            ("NumPy / Pandas / Polars", "Data manipulation and analysis"),
            ("SciPy", "Signal processing and curve fitting"),
        ],
        "API & Serving": [
            ("FastAPI", "Async REST API with type-safe endpoints"),
            ("Pydantic v2", "Request/response validation"),
            ("Uvicorn", "ASGI server"),
            ("Nginx", "Reverse proxy, TLS, rate limiting, security headers"),
        ],
        "MLOps Infrastructure": [
            ("MLflow", "Experiment tracking and model registry"),
            ("DVC", "Data version control and pipeline management"),
            ("DagsHub", "Cloud remote for MLflow + DVC"),
            ("Airflow", "DAG orchestration for ML workflows"),
        ],
        "Monitoring & Observability": [
            ("Prometheus", "Metrics collection and alerting rules"),
            ("Grafana", "Dashboard visualisation"),
            ("Alertmanager", "Alert routing and notification"),
            ("EvidentlyAI", "Data drift and model performance reports"),
            ("cAdvisor", "Container resource monitoring"),
            ("Node Exporter", "Host system metrics"),
        ],
        "DevOps & Deployment": [
            ("Docker Compose", "Multi-container orchestration (13 services)"),
            ("Kubernetes", "K8s deployment with Kustomize overlays (local + cloud)"),
            ("GitHub Actions", "CI/CD pipelines (8 workflows)"),
            ("GHCR", "Container image registry"),
            ("PostgreSQL", "Persistent storage for predictions and metadata"),
        ],
        "Testing & Quality": [
            ("Pytest", "808 CI + 43 live integration tests (851 total)"),
            ("Coverage.py", "~82% code coverage"),
            ("Ruff", "Linting and formatting"),
            ("mypy", "Static type checking"),
        ],
        "Frontend": [
            ("Streamlit", "Interactive dashboard and control panel"),
            ("Plotly", "Interactive data visualisations"),
            ("Mermaid", "Architecture and pipeline diagrams"),
        ],
    }

    for category, tools in tech_categories.items():
        st.markdown(f"#### {category}")
        cols = st.columns(3)
        for i, (tool, desc) in enumerate(tools):
            with cols[i % 3]:
                st.markdown(
                    f'<div class="tech-badge" style="display:block;margin-bottom:0.5rem">'
                    f"<b>{tool}</b><br>"
                    f'<span style="font-size:0.75rem;color:var(--text-muted)">{desc}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
        st.markdown("")

    # ── Project structure ───────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="section-header">📁 Project Structure</div>',
        unsafe_allow_html=True,
    )
    st.code(
        """
mlops_device_health/
├── src/
│   ├── api/                # FastAPI endpoints (predict, train, health)
│   ├── database/           # SQLAlchemy models, CRUD operations
│   ├── monitoring/         # Prometheus metrics, Grafana integration
│   ├── signal_processing/  # Feature extraction, validators
│   ├── training/           # Training pipeline, evaluation
│   └── ui/                 # Streamlit dashboard (this app)
├── airflow/
│   └── dags/               # 9 Airflow DAGs
├── docker/                 # Dockerfiles, Nginx, Grafana, Prometheus configs
├── k8s/                    # Kubernetes manifests + Kustomize overlays
│   ├── base/               # Base K8s resources
│   └── overlays/           # local / cloud overlays
├── scripts/                # CLI tools (generate, simulate, promote, sync)
├── tests/                  # 851 tests across 12 directories
│   ├── k8s/ kubernetes/    # K8s manifest validation (274 tests)
│   ├── core/ ml/ database/ # Domain logic, ML pipeline (259 tests)
│   ├── config/             # Config & workflow tests (74 tests)
│   └── live/               # Live integration tests (43, require Docker)
├── .github/workflows/      # CI/CD pipelines (8 workflows)
├── doc/                    # Architecture docs, guides, specs
├── data/                   # Raw, processed, gold standard data
├── dvc.yaml                # DVC pipeline (5 stages)
├── docker-compose.yml      # 13-service container stack
├── pyproject.toml          # Project metadata and dependencies
├── README.md               # Project overview and documentation
└── Makefile                # Build, test, run targets
    """.strip(),
        language="text",
    )

    # ── Acknowledgements ────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="section-header">🙏 Acknowledgements</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="info-card">'
        "<h3>Honoured Mentions</h3>"
        "<ul>"
        "<li><b>Nicolas Fradin</b> — MLOps project mentor. Thank you for your guidance, "
        "feedback, and expertise throughout this project.</li>"
        "<li><b>MLflow team</b> — for the excellent experiment tracking and model registry platform</li>"
        "<li><b>DVC / Iterative</b> — for making data versioning accessible and reproducible</li>"
        "<li><b>DagsHub</b> — for providing integrated cloud hosting for MLflow + DVC</li>"
        "<li><b>EvidentlyAI</b> — for the powerful drift detection and monitoring framework</li>"
        "<li><b>Apache Airflow</b> — for robust workflow orchestration</li>"
        "<li><b>Streamlit</b> — for rapid, beautiful data app development</li>"
        "<li><b>FastAPI</b> — for the modern, high-performance API framework</li>"
        "<li><b>Docker</b> — for containerisation that makes deployment reproducible</li>"
        "<li><b>Prometheus + Grafana</b> — for the industry-standard observability stack</li>"
        "</ul></div>",
        unsafe_allow_html=True,
    )

    # ── Footer ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;color:var(--text-muted);padding:1rem 0">'
        "<p>MLOps Device Health Monitoring System</p>"
        "<p style='font-size:0.8rem'>Built with ❤️ using Python, FastAPI, MLflow, "
        "Docker, Kubernetes, Airflow, Prometheus, Grafana, and Streamlit</p>"
        "</div>",
        unsafe_allow_html=True,
    )
