"""Home / Project Overview page."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from src.ui.logging_ui import get_ui_logger
from src.ui.styles import get_global_css, hero_section, metric_card

_logger = get_ui_logger(__name__)


def render() -> None:
    """Render the Home page."""
    try:
        _render_content()
    except Exception:
        _logger.exception("Unhandled exception in home.render()")
        raise


def _render_content() -> None:
    """Inner render — separated so exceptions are caught and logged by render()."""
    st.markdown(get_global_css(), unsafe_allow_html=True)

    # ── Hero ────────────────────────────────────────────────────
    st.markdown(
        hero_section(
            "MLOps Device Health Monitoring",
            "Production-grade end-to-end MLOps platform for IoT device health prediction — "
            "from synthetic signal generation to automated retraining with full observability.",
        ),
        unsafe_allow_html=True,
    )

    # ── Key metrics row ─────────────────────────────────────────
    cols = st.columns(6)
    metrics = [
        ("🐳", "14", "Docker Services"),
        ("🧪", "808", "Default Tests (CI)"),
        ("🔬", "43", "Live Integration Tests"),
        ("⚙️", "8", "GitHub Workflows"),
        ("🎯", "18", "MLOps Use Cases"),
        ("🔄", "9", "Airflow DAGs"),
    ]
    for col, (icon, value, label) in zip(cols, metrics, strict=False):
        col.markdown(metric_card(icon, value, label), unsafe_allow_html=True)

    st.markdown("---")

    # ── Test suite statistics ────────────────────────────────────
    st.markdown(
        '<div class="section-header">🧪 Test Suite & Code Coverage</div>',
        unsafe_allow_html=True,
    )

    test_cols = st.columns(4)
    test_stats = [
        (
            "🤖 Default CI",
            "808",
            "tests",
            "Run on every push. All pass, 0 errors. "
            "Includes K8s, config, API, ML, database, security, monitoring tests. "
            "Excludes live Docker integration tests.",
        ),
        (
            "🐳 Live Integration",
            "43",
            "tests",
            "Require a running Docker stack. "
            "Exercise real API endpoints, database round-trips, and service health checks.",
        ),
        (
            "📊 Total",
            "851",
            "tests",
            "Grand total across all markers. Markers: <code>live</code> (43), CI-default (808).",
        ),
        (
            "⚙️ CI Workflows",
            "8",
            "workflows",
            "lint, test, build, code-quality, live-tests, "
            "model-quality-gate, deploy, deploy-k8s. Covering quality, testing, and deployment.",
        ),
    ]
    for col, (title, value, unit, detail) in zip(test_cols, test_stats, strict=False):
        with col:
            st.markdown(
                f'<div class="info-card" style="text-align:center">'
                f"<h3>{title}</h3>"
                f'<div style="font-size:2.2rem;font-weight:700;'
                f"background:linear-gradient(135deg,#6366f1,#06b6d4);"
                f"-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
                f'background-clip:text">{value}</div>'
                f'<div style="font-size:0.8rem;color:#94a3b8;margin-bottom:0.5rem">{unit}</div>'
                f'<div style="font-size:0.75rem;color:#64748b">{detail}</div>'
                "</div>",
                unsafe_allow_html=True,
            )

    # ── Coverage bars ────────────────────────────────────────────
    cov_core = 61  # api, database, signal_processing, monitoring (measured via pytest-cov)
    cov_full = 19  # entire src/ including UI (Streamlit UI not exercised by unit tests)
    st.markdown(
        f"""
        <div style="margin:1.2rem 0 0.4rem 0">
            <span style="font-size:0.85rem;color:#94a3b8;font-weight:600">
                📋 Core Module Coverage — Default CI run
            </span>
            <span style="float:right;font-size:1.1rem;font-weight:700;color:#10b981">
                {cov_core}%
            </span>
        </div>
        <div style="background:#1e293b;border-radius:8px;height:14px;overflow:hidden">
            <div style="background:linear-gradient(90deg,#6366f1,#10b981);
                        width:{cov_core}%;height:100%;border-radius:8px"></div>
        </div>
        <div style="margin:0.8rem 0 0.4rem 0">
            <span style="font-size:0.85rem;color:#94a3b8;font-weight:600">
                📋 Full <code style="color:#6366f1">src/</code> Coverage (incl. UI)
            </span>
            <span style="float:right;font-size:1.1rem;font-weight:700;color:#f59e0b">
                {cov_full}%
            </span>
        </div>
        <div style="background:#1e293b;border-radius:8px;height:14px;overflow:hidden">
            <div style="background:linear-gradient(90deg,#6366f1,#f59e0b);
                        width:{cov_full}%;height:100%;border-radius:8px"></div>
        </div>
        <div style="font-size:0.7rem;color:#475569;margin-top:0.3rem">
            Core = api, database, signal_processing, monitoring modules.
            Full includes Streamlit UI code (not exercised by unit tests).
            Measured via <code>pytest-cov</code> against <code>src/</code>.
            HTML report: <code>htmlcov/index.html</code>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Per-category breakdown ───────────────────────────────────
    st.markdown(
        '<div class="info-card" style="margin-top:1rem">'
        "<h4>🗂️ Test Category Breakdown</h4>"
        '<table style="width:100%;border-collapse:collapse;font-size:0.83rem">'
        "<thead><tr>"
        '<th style="text-align:left;padding:0.4rem 0.6rem;border-bottom:1px solid #334155;color:#94a3b8">'
        "Category</th>"
        '<th style="text-align:right;padding:0.4rem 0.6rem;border-bottom:1px solid #334155;color:#94a3b8">'
        "Tests</th>"
        '<th style="text-align:left;padding:0.4rem 0.6rem;border-bottom:1px solid #334155;color:#94a3b8">'
        "What is covered</th>"
        "</tr></thead><tbody>"
        # core
        '<tr><td style="padding:0.35rem 0.6rem;color:#e2e8f0"><code>tests/core/</code></td>'
        '<td style="text-align:right;padding:0.35rem 0.6rem;font-weight:700;color:#6366f1">103</td>'
        '<td style="padding:0.35rem 0.6rem;color:#94a3b8">Signal generation, feature extraction, '
        "validators, Pydantic models, e2e pipeline, health classification</td></tr>"
        # config
        '<tr><td style="padding:0.35rem 0.6rem;color:#e2e8f0"><code>tests/config/</code></td>'
        '<td style="text-align:right;padding:0.35rem 0.6rem;font-weight:700;color:#6366f1">74</td>'
        '<td style="padding:0.35rem 0.6rem;color:#94a3b8">Airflow DAGs, DVC pipeline, GitHub workflows, '
        "Grafana dashboards, params.yaml, Prometheus config</td></tr>"
        # database
        '<tr><td style="padding:0.35rem 0.6rem;color:#e2e8f0"><code>tests/database/</code></td>'
        '<td style="text-align:right;padding:0.35rem 0.6rem;font-weight:700;color:#6366f1">103</td>'
        '<td style="padding:0.35rem 0.6rem;color:#94a3b8">Label injection logic, prediction '
        "persistence (in-memory SQLite), database migrations</td></tr>"
        # ml
        '<tr><td style="padding:0.35rem 0.6rem;color:#e2e8f0"><code>tests/ml/</code></td>'
        '<td style="text-align:right;padding:0.35rem 0.6rem;font-weight:700;color:#6366f1">53</td>'
        '<td style="padding:0.35rem 0.6rem;color:#94a3b8">Drift detection, prediction, model promotion, '
        "semi-supervised learning, training pipeline</td></tr>"
        # k8s
        '<tr><td style="padding:0.35rem 0.6rem;color:#e2e8f0"><code>tests/k8s/ + tests/kubernetes/</code></td>'
        '<td style="text-align:right;padding:0.35rem 0.6rem;font-weight:700;color:#6366f1">274</td>'
        '<td style="padding:0.35rem 0.6rem;color:#94a3b8">K8s manifests, kustomize overlays, deployment '
        "correctness, RBAC, HPA, service configs (skipped without cluster)</td></tr>"
        # api
        '<tr><td style="padding:0.35rem 0.6rem;color:#e2e8f0"><code>tests/api/</code></td>'
        '<td style="text-align:right;padding:0.35rem 0.6rem;font-weight:700;color:#6366f1">29</td>'
        '<td style="padding:0.35rem 0.6rem;color:#94a3b8">Health endpoint, predict endpoint '
        "(unit-level, no real DB)</td></tr>"
        # monitoring
        '<tr><td style="padding:0.35rem 0.6rem;color:#e2e8f0"><code>tests/monitoring/</code></td>'
        '<td style="text-align:right;padding:0.35rem 0.6rem;font-weight:700;color:#6366f1">18</td>'
        '<td style="padding:0.35rem 0.6rem;color:#94a3b8">Prometheus metric registration, '
        "counter/gauge/histogram correctness</td></tr>"
        # security
        '<tr><td style="padding:0.35rem 0.6rem;color:#e2e8f0"><code>tests/security/</code></td>'
        '<td style="text-align:right;padding:0.35rem 0.6rem;font-weight:700;color:#6366f1">13</td>'
        '<td style="padding:0.35rem 0.6rem;color:#94a3b8">Input validation, JWT auth, '
        "XSS / oversized-payload hardening</td></tr>"
        # performance
        '<tr><td style="padding:0.35rem 0.6rem;color:#e2e8f0"><code>tests/performance/</code></td>'
        '<td style="text-align:right;padding:0.35rem 0.6rem;font-weight:700;color:#6366f1">4</td>'
        '<td style="padding:0.35rem 0.6rem;color:#94a3b8">Latency thresholds for feature '
        "extraction and signal generation</td></tr>"
        # reproducibility
        '<tr><td style="padding:0.35rem 0.6rem;color:#e2e8f0"><code>tests/reproducibility/</code></td>'
        '<td style="text-align:right;padding:0.35rem 0.6rem;font-weight:700;color:#6366f1">4</td>'
        '<td style="padding:0.35rem 0.6rem;color:#94a3b8">E2E reproducibility: identical predictions '
        "and F1 across two independent training runs</td></tr>"
        # live
        '<tr><td style="padding:0.35rem 0.6rem;color:#e2e8f0"><code>tests/live/</code> '
        '<span style="font-size:0.7rem;background:#92400e;color:#fef3c7;border-radius:4px;'
        'padding:1px 5px">requires Docker</span></td>'
        '<td style="text-align:right;padding:0.35rem 0.6rem;font-weight:700;color:#f59e0b">43</td>'
        '<td style="padding:0.35rem 0.6rem;color:#94a3b8">Live API endpoints, DB round-trips, '
        "service health — run with <code>pytest -m live</code></td></tr>"
        # total
        '<tr style="border-top:2px solid #334155">'
        '<td style="padding:0.35rem 0.6rem;color:#e2e8f0;font-weight:700">Total</td>'
        '<td style="text-align:right;padding:0.35rem 0.6rem;font-weight:700;color:#10b981">851</td>'
        '<td style="padding:0.35rem 0.6rem;color:#64748b">808 CI-default + 43 live · '
        "0 fails · skipped-when-cluster/docker-absent</td></tr>"
        "</tbody></table>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Project motivation ──────────────────────────────────────
    st.markdown(
        '<div class="section-header">🎯 Project Motivation</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        Modern IoT deployments generate millions of telemetry signals per day.
        Manual inspection is impossible —  **machine learning** can classify device
        health automatically, but only if the ML system itself is **production-ready**,
        **observable**, and **self-healing**.

        This project demonstrates a **complete MLOps lifecycle**: from synthetic data
        generation and semi-supervised learning through automated drift detection,
        retraining, and blue-green deployment — all containerised, version-controlled,
        and monitored.
        """
    )

    # ── Two-column: Synthetic Data + Model ──────────────────────
    left, right = st.columns(2)

    with left:
        st.markdown(
            '<div class="info-card">'
            "<h3>📡 Synthetic Signal Generation</h3>"
            "<p>Signals are synthesised with controlled parameters to simulate IoT "
            "device telemetry:</p>"
            "<ul>"
            "<li><b>Healthy devices</b> — Gaussian peaks: σ ∈ [2.0, 5.0], low noise</li>"
            "<li><b>Unhealthy devices</b> — Lorentzian peaks: γ = 1.1775 σ, high noise</li>"
            "<li><b>Drift scenarios</b> — gradual parameter shifts simulate real-world degradation</li>"
            "</ul>"
            "<p>By controlling distribution parameters, we can reproducibly generate "
            "data drift, concept drift, and out-of-distribution samples.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            '<div class="info-card">'
            "<h3>🤖 Semi-Supervised Learning</h3>"
            "<p>In production, labels arrive <b>sparsely and asynchronously</b>. "
            "Our training pipeline accounts for this:</p>"
            "<ul>"
            "<li><b>Bootstrap</b> — Greenfield training uses sparse labels + <b>K-means clustering</b> "
            "to pseudo-label unlabelled data (semi-supervised)</li>"
            "<li><b>Classifier</b> — Logistic Regression trained on the pseudo-labelled set; "
            "a separate champion/challenger evaluation gate controls promotion</li>"
            "<li><b>Retraining</b> — Automated retraining triggers when EvidentlyAI detects drift "
            "or model accuracy drops below threshold</li>"
            "</ul>"
            "<p>MLflow tracks every experiment: hyperparameters, metrics, feature importance, "
            "and model artefacts — all linked to the DVC data version.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Pipeline overview ───────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="section-header">🔄 MLOps Lifecycle at a Glance</div>',
        unsafe_allow_html=True,
    )

    pipeline_steps = [
        (
            "1️⃣",
            "Generate & Version Data",
            "Signals are generated into PostgreSQL (real-time storage) and exported to Parquet files. "
            "DVC versions exported datasets. Generation scripts cover normal, feature-drift, concept-drift, and covariate-shift scenarios.",
        ),
        (
            "2️⃣",
            "Extract Features",
            "Peak height, FWHM, SNR, noise level, peak area, peak center — 6 features extracted per signal.",
        ),
        (
            "3️⃣",
            "Train & Evaluate",
            "Models trained and evaluated with cross-validation. MLflow logs all runs with metrics and artefacts.",
        ),
        (
            "4️⃣",
            "Register & Promote",
            "Best model registered in MLflow Model Registry. Champion/Challenger pattern with automated promotion.",
        ),
        (
            "5️⃣",
            "Serve & Monitor",
            "FastAPI behind Nginx reverse proxy. Prometheus scrapes metrics. Grafana visualises. EvidentlyAI detects drift.",
        ),
        (
            "6️⃣",
            "Detect & Retrain",
            "Airflow DAGs monitor for drift, trigger automated retraining, promote new models — full closed-loop MLOps.",
        ),
    ]

    cols = st.columns(3)
    for i, (num, title, desc) in enumerate(pipeline_steps):
        with cols[i % 3]:
            st.markdown(
                f'<div class="info-card"><h3>{num} {title}</h3><p>{desc}</p></div>',
                unsafe_allow_html=True,
            )

    # ── Tech stack badges ───────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="section-header">🛠️ Technology Stack</div>',
        unsafe_allow_html=True,
    )

    tech = [
        "🐍 Python 3.12",
        "⚡ FastAPI",
        "🐘 PostgreSQL",
        "🔬 MLflow",
        "📦 DVC",
        "🐳 Docker Compose",
        "☸️ Kubernetes",
        "🌐 Nginx",
        "📊 Prometheus",
        "📈 Grafana",
        "🔔 Alertmanager",
        "🔄 Airflow",
        "🔍 EvidentlyAI",
        "🧪 Pytest",
        "☁️ DagsHub",
        "🔁 GitHub Actions",
        "📦 GHCR",
        "🎨 Streamlit",
        "📊 Plotly",
        "🔷 Pydantic",
        "🔧 Kustomize",
    ]
    badge_html = " ".join(f'<span class="tech-badge">{t}</span>' for t in tech)
    st.markdown(
        f"<div style='display:flex;flex-wrap:wrap;gap:0.5rem'>{badge_html}</div>",
        unsafe_allow_html=True,
    )
