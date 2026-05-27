"""Architecture visualisation page with Mermaid diagrams."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from src.ui.logging_ui import get_ui_logger
from src.ui.styles import get_global_css, hero_section

_logger = get_ui_logger(__name__)

# ── Detect current mode ─────────────────────────────────────────


def _detect_mode() -> str:
    """Check .current_mode file first (authoritative); env var is stale after mode switch."""
    mode_file = Path(_PROJECT_ROOT) / ".current_mode"
    if mode_file.exists():
        val = mode_file.read_text().strip()
        if val in ("local", "cloud", "k8s"):
            return val
    env_mode = os.environ.get("DEPLOYMENT_MODE", "").strip()
    if env_mode in ("local", "cloud", "k8s"):
        return env_mode
    return "local"


# ── Mermaid diagram definitions ─────────────────────────────────

ARCHITECTURE_DIAGRAM_CLOUD = r"""
graph TB
    subgraph CLOUD["Cloud Services"]
        dagshub["DagsHub\nMLflow + DVC Remote"]
        github["GitHub\nSource + CI/CD"]
        ghcr["GHCR\nContainer Registry"]
    end

    subgraph DOCKER["Docker Network - mlops_network"]
        subgraph PROXY["Ingress Layer"]
            nginx["Nginx\nReverse Proxy :80"]
        end

        subgraph APP["Application Layer"]
            api["FastAPI\nPrediction API :8000"]
        end

        subgraph DATA_LAYER["Databases"]
            postgres_app["PostgreSQL App\nmlops_db :5432"]
            postgres_mlflow["PostgreSQL MLflow\n:5432 internal"]
        end

        subgraph ML_LAYER["ML Platform"]
            mlflow_buf["MLflow Buffer\nLocal-first :5000"]
            airflow["Airflow\nDAG Scheduler :8081"]
        end

        subgraph MONITOR["Monitoring Stack"]
            prometheus["Prometheus\nMetrics :9090"]
            grafana["Grafana\nDashboards :3000"]
            alertmanager["Alertmanager\nAlerts :9093"]
            cadvisor["cAdvisor\nContainer Metrics :8080"]
            pg_exp["PG Exporter App\n:9187"]
            pg_mlflow_exp["PG Exporter MLflow\n:9187"]
        end
    end

    nginx --> api
    api --> postgres_app
    api --> mlflow_buf
    mlflow_buf --> postgres_mlflow
    airflow --> api
    airflow -.->|sync DAG| dagshub
    mlflow_buf -.->|scheduled push| dagshub

    prometheus --> api
    prometheus --> cadvisor
    prometheus --> pg_exp
    prometheus --> pg_mlflow_exp
    grafana --> prometheus
    alertmanager --> prometheus
    pg_exp --> postgres_app
    pg_mlflow_exp --> postgres_mlflow

    github -.->|CI/CD| ghcr
    ghcr -.->|pull images| DOCKER

    classDef cloud fill:#1e1b4b,stroke:#818cf8,color:#e2e8f0
    classDef proxy fill:#0c4a6e,stroke:#06b6d4,color:#e2e8f0
    classDef app fill:#064e3b,stroke:#10b981,color:#e2e8f0
    classDef ml fill:#713f12,stroke:#f59e0b,color:#e2e8f0
    classDef monitor fill:#4c1d95,stroke:#a78bfa,color:#e2e8f0
    classDef data fill:#1e3a5f,stroke:#60a5fa,color:#e2e8f0
    classDef orch fill:#7c2d12,stroke:#fb923c,color:#e2e8f0

    class dagshub,github,ghcr cloud
    class nginx proxy
    class api app
    class mlflow_buf ml
    class airflow orch
    class prometheus,grafana,alertmanager,cadvisor,pg_exp,pg_mlflow_exp monitor
    class postgres_app,postgres_mlflow data
"""

ARCHITECTURE_DIAGRAM_LOCAL = r"""
graph TB
    subgraph DOCKER["Docker Network - mlops_network (local mode)"]
        subgraph PROXY["Ingress"]
            nginx["Nginx\nReverse Proxy :80"]
        end

        subgraph APP["Application"]
            api["FastAPI\nPrediction API :8000"]
        end

        subgraph DATA_LAYER["Databases"]
            postgres_app["PostgreSQL App\nmlops_db :5432"]
        end

        subgraph ML_LAYER["ML Platform"]
            mlflow["MLflow\nLocal Tracking :5000"]
        end

        subgraph MONITOR["Monitoring"]
            prometheus["Prometheus :9090"]
            grafana["Grafana :3000"]
            alertmanager["Alertmanager :9093"]
            cadvisor["cAdvisor :8080"]
            pg_exp["PG Exporter :9187"]
        end
    end

    nginx --> api
    api --> postgres_app
    api --> mlflow
    mlflow --> postgres_app

    prometheus --> api
    prometheus --> cadvisor
    prometheus --> pg_exp
    grafana --> prometheus
    alertmanager --> prometheus

    classDef proxy fill:#0c4a6e,stroke:#06b6d4,color:#e2e8f0
    classDef app fill:#064e3b,stroke:#10b981,color:#e2e8f0
    classDef ml fill:#713f12,stroke:#f59e0b,color:#e2e8f0
    classDef monitor fill:#4c1d95,stroke:#a78bfa,color:#e2e8f0
    classDef data fill:#1e3a5f,stroke:#60a5fa,color:#e2e8f0

    class nginx proxy
    class api app
    class mlflow ml
    class prometheus,grafana,alertmanager,cadvisor,pg_exp monitor
    class postgres_app data
"""

DATAFLOW_DIAGRAM_SIGNALS = r"""
flowchart LR
    subgraph GEN["Signal Generation"]
        gen["generate_signal()\nGaussian or Lorentzian\nseed-controlled params"]
    end

    subgraph FEAT["Feature Extraction"]
        extract["extract_features()\nfwhm, peak_height\npeak_area, snr\nnoise_level, peak_center"]
    end

    subgraph INFER["Inference"]
        api_pred["FastAPI /predict\nLogisticRegression\nor champion model"]
        db_store["PostgreSQL\ndevices, signals\npredictions, labels"]
    end

    subgraph TRACK["ML Tracking"]
        mlflow_buf["MLflow Buffer\nruns, experiments\nmodel registry"]
    end

    gen --> extract --> api_pred
    api_pred --> db_store
    api_pred --> mlflow_buf

    classDef step fill:#1e293b,stroke:#6366f1,color:#e2e8f0
    class gen,extract,api_pred,db_store,mlflow_buf step
"""

DATAFLOW_DIAGRAM_TRAINING = r"""
flowchart TD
    subgraph DATA["Training Data"]
        pg["PostgreSQL\nsignals + labels"]
        ci_ref["data/ci/\nCI reference signals\ngit-tracked CSV"]
    end

    subgraph TRAIN["Training Pipeline"]
        greenfield["Greenfield Use Case\nBootstrapConfig defaults\nLogisticRegression"]
        retrain["Retraining DAG\nautomated_retraining\nAirflow"]
    end

    subgraph EVAL["Evaluation"]
        quality["CI Quality Gate\nmodel-quality-gate.yml\naccuracy >= 0.80\nF1 >= 0.75"]
        mlflow_reg["MLflow Registry\nchampion alias\nstaging / archived"]
    end

    subgraph SERVE["Serving"]
        api_v["FastAPI\nloads champion\nmodel at startup"]
    end

    pg --> retrain
    ci_ref --> quality
    greenfield --> mlflow_reg
    retrain --> mlflow_reg
    quality -->|pass| mlflow_reg
    quality -->|fail| x["block PR"]
    mlflow_reg --> api_v

    classDef step fill:#1e293b,stroke:#6366f1,color:#e2e8f0
    classDef gate fill:#7c2d12,stroke:#fb923c,color:#e2e8f0
    classDef ok fill:#064e3b,stroke:#10b981,color:#e2e8f0
    classDef bad fill:#4c0519,stroke:#ef4444,color:#e2e8f0
    class greenfield,retrain,api_v step
    class pg,ci_ref,mlflow_reg step
    class quality gate
    class x bad
"""

DATAFLOW_DIAGRAM_SYNC = r"""
flowchart LR
    subgraph LOCAL["Local Stack"]
        pg_local["PostgreSQL\nproduction data"]
        mlflow_buf["MLflow Buffer\nexperiments + models"]
    end

    subgraph DAGSHUB["DagsHub (Cloud Remote)"]
        dvc_remote["DVC Remote\nS3-compatible\ndata versioning"]
        mlflow_dag["DagsHub MLflow\nexperiment archive"]
    end

    subgraph AIRFLOW["Airflow DAGs"]
        sync_dag["sync_production_data\ndaily schedule"]
        mlflow_sync["mlflow_sync\nscheduled push"]
    end

    pg_local -->|export CSV/JSON| sync_dag
    sync_dag -->|dvc add + push| dvc_remote
    mlflow_buf -->|incremental push| mlflow_sync
    mlflow_sync --> mlflow_dag

    classDef step fill:#1e293b,stroke:#6366f1,color:#e2e8f0
    class pg_local,mlflow_buf,sync_dag,mlflow_sync step
    class dvc_remote,mlflow_dag step
"""

CI_CD_DIAGRAM_CORE = r"""
flowchart LR
    push["git push"] --> gha["GitHub Actions"]

    gha --> lint["lint.yml\nruff + mypy"]
    gha --> test["test.yml\npytest not-live"]
    gha --> build["build.yml\nDocker build\nTrivy scan"]
    gha --> quality["code-quality.yml\nbandit + pip-audit"]

    lint --> gate{"all pass?"}
    test --> gate
    build --> gate
    quality --> gate

    gate -->|yes| ghcr["push to GHCR\nmain/tag only"]
    gate -->|no| fail["CI failed\nPR blocked"]

    classDef action fill:#1e293b,stroke:#6366f1,color:#e2e8f0
    classDef decision fill:#2d1b69,stroke:#a78bfa,color:#e2e8f0
    classDef ok fill:#064e3b,stroke:#10b981,color:#e2e8f0
    classDef bad fill:#4c0519,stroke:#ef4444,color:#e2e8f0

    class push,gha,lint,test,build,quality,ghcr action
    class gate decision
    class fail bad
"""

CI_CD_DIAGRAM_ML = r"""
flowchart LR
    pr["PR to main\ntouching training code"] --> qgate["model-quality-gate.yml\nCI quality gate"]

    qgate --> load["load\ndata/ci/quality_gate_signals.csv"]
    load --> feat["extract_features()"]
    feat --> train["train model\n80/20 split"]
    train --> eval["accuracy >= 0.80\nF1 >= 0.75"]

    eval -->|pass| ok["PR can merge\nreport artifact uploaded"]
    eval -->|fail| block["PR blocked\nregression detected"]

    classDef step fill:#1e293b,stroke:#6366f1,color:#e2e8f0
    classDef ok fill:#064e3b,stroke:#10b981,color:#e2e8f0
    classDef bad fill:#4c0519,stroke:#ef4444,color:#e2e8f0

    class pr,qgate,load,feat,train,eval step
    class ok ok
    class block bad
"""

CI_CD_DIAGRAM_DEPLOY = r"""
flowchart LR
    trigger["workflow_dispatch\nor v* tag push"] --> deploy["deploy.yml"]

    deploy --> pull["pull image from GHCR"]
    pull --> smoke["smoke test\nstart API + postgres\nGET /health"]

    smoke -->|healthy| run["deploy step\nplaceholder\nno real SSH target"]
    smoke -->|unhealthy| rollback["smoke test failed\nskip deploy"]

    run --> verify["print deployment\nsummary + image SHA"]

    classDef step fill:#1e293b,stroke:#6366f1,color:#e2e8f0
    classDef bad fill:#4c0519,stroke:#ef4444,color:#e2e8f0

    class trigger,deploy,pull,smoke,run,verify step
    class rollback bad
"""


def _render_mermaid(diagram: str, height: int = 600) -> None:
    """Render a Mermaid diagram via embedded Mermaid.js with auto-sizing."""
    import streamlit.components.v1 as components

    html_content = f"""
    <html><head>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    </head><body style="background:transparent;margin:0;overflow:hidden">
    <div id="diagram" class="mermaid" style="background:#0f172a;border-radius:12px;padding:1rem;
         min-height:200px;max-height:{height}px;overflow:auto;margin-bottom:1rem;">
{diagram.strip()}
    </div>
    <script>
      (function tryRender(n) {{
        if (typeof mermaid !== 'undefined') {{
          mermaid.initialize({{startOnLoad:false,theme:'dark',securityLevel:'loose'}});
          mermaid.run({{nodes:[document.getElementById('diagram')]}});
          var _a = 0;
          var _p = setInterval(function() {{
            _a++;
            var svg = document.querySelector('#diagram svg');
            if (svg || _a > 100) {{
              clearInterval(_p);
              var el = document.getElementById('diagram');
              if (el && window.frameElement) {{
                window.frameElement.style.height = Math.min(el.scrollHeight + 32, {height}) + 'px';
              }}
            }}
          }}, 100);
        }} else if (n < 100) {{
          setTimeout(function() {{ tryRender(n + 1); }}, 100);
        }}
      }})(0);
    </script>
    </body></html>
    """
    components.html(html_content, height=height, scrolling=True)


def render() -> None:
    """Render the Architecture page."""
    try:
        _render_content()
    except Exception:
        _logger.exception("Unhandled exception in architecture.render()")
        raise


def _render_content() -> None:
    """Inner render — separated so exceptions are caught and logged by render()."""
    st.markdown(get_global_css(), unsafe_allow_html=True)
    mode = _detect_mode()
    st.markdown(
        hero_section(
            "System Architecture",
            f"Microservice architecture with Docker containers, cloud integrations, "
            f"and a full CI/CD pipeline. Current mode: **{mode.upper()}**",
        ),
        unsafe_allow_html=True,
    )

    # Use st.radio (keyed) instead of st.tabs() \u2014 st.tabs() resets to tab 0
    # on every full Streamlit rerun, causing the "jump to System Architecture" bug.
    _ARCH_TABS = [
        "\U0001f3d7\ufe0f System Architecture",
        "\U0001f504 Data Flow",
        "\U0001f680 CI/CD Pipeline",
        "\u2638\ufe0f Kubernetes Network",
    ]
    active_arch = st.radio(
        "Architecture view",
        _ARCH_TABS,
        horizontal=True,
        key="_arch_tab",
        label_visibility="collapsed",
    )
    st.markdown(
        "<hr style='margin:0 0 1rem 0;border-color:#334155;'>",
        unsafe_allow_html=True,
    )

    if active_arch == _ARCH_TABS[0]:
        _render_system_architecture(mode)
    elif active_arch == _ARCH_TABS[1]:
        _render_data_flow()
    elif active_arch == _ARCH_TABS[2]:
        _render_cicd_pipeline()
    else:
        _render_kubernetes_architecture()


def _render_system_architecture(mode: str) -> None:
    """System architecture tab with mode-appropriate diagram."""
    st.markdown(
        '<div class="section-header">\U0001f3d7\ufe0f Microservice Architecture</div>',
        unsafe_allow_html=True,
    )

    if mode == "cloud":
        st.info(
            "\U0001f4a1 **Cloud mode** — showing cloud stack with MLflow Buffer + dedicated PostgreSQL (MLflow). "
            "13 containers total."
        )
        _render_mermaid(ARCHITECTURE_DIAGRAM_CLOUD, height=700)
    elif mode == "k8s":
        st.info(
            "\U0001f4a1 **K8s mode** — showing Kubernetes cluster deployment. "
            "Pods: api, airflow, grafana, mlflow, nginx, postgres, prometheus, streamlit."
        )
        _render_mermaid(ARCHITECTURE_DIAGRAM_CLOUD, height=700)
    else:
        st.info(
            "\U0001f4a1 **Local mode** — showing local stack (no Airflow, no cloud sync). "
            "11 containers total."
        )
        _render_mermaid(ARCHITECTURE_DIAGRAM_LOCAL, height=550)

    _architecture_details(mode)


def _render_data_flow() -> None:
    """Data flow tab with radio navigation for different flow types."""
    st.markdown(
        '<div class="section-header">\U0001f504 Data Flow & ML Pipeline</div>',
        unsafe_allow_html=True,
    )

    _DF_TABS = [
        "\U0001f4e1 Signal Inference Flow",
        "\U0001f916 Training & Promotion Flow",
        "\U0001f4e4 Sync & Versioning Flow",
    ]
    sel = st.radio(
        "Data flow view",
        _DF_TABS,
        horizontal=True,
        key="_arch_df_sel",
        label_visibility="collapsed",
    )
    st.markdown("<hr style='margin:0 0 1rem 0;border-color:#334155;'>", unsafe_allow_html=True)

    if sel == _DF_TABS[0]:
        st.markdown(
            "**Signal Inference Flow** — from signal generation through feature extraction "
            "to prediction and storage in PostgreSQL."
        )
        _render_mermaid(DATAFLOW_DIAGRAM_SIGNALS, height=420)
    elif sel == _DF_TABS[1]:
        st.markdown(
            "**Training & Promotion Flow** — from training data through model training, "
            "CI quality gate, and promotion to champion."
        )
        _render_mermaid(DATAFLOW_DIAGRAM_TRAINING, height=600)
    else:
        st.markdown(
            "**Sync & Versioning Flow** — from local PostgreSQL and MLflow Buffer "
            "to DagsHub (DVC + MLflow archive) via Airflow DAGs. Cloud mode only."
        )
        _render_mermaid(DATAFLOW_DIAGRAM_SYNC, height=420)

    _dataflow_details()


def _render_cicd_pipeline() -> None:
    """CI/CD pipeline tab with radio navigation for each workflow group."""
    st.markdown(
        '<div class="section-header">\U0001f680 CI/CD Pipeline</div>',
        unsafe_allow_html=True,
    )

    _CD_TABS = [
        "\u2699\ufe0f Core CI Workflows",
        "\U0001f916 ML Quality Gate",
        "\U0001f6a7 Deploy Workflow",
        "\u2638\ufe0f K8s CI/CD Deploy",
    ]
    sel = st.radio(
        "CI/CD view",
        _CD_TABS,
        horizontal=True,
        key="_arch_cd_sel",
        label_visibility="collapsed",
    )
    st.markdown("<hr style='margin:0 0 1rem 0;border-color:#334155;'>", unsafe_allow_html=True)

    if sel == _CD_TABS[0]:
        st.markdown(
            "**Core CI** \u2014 lint, unit tests, Docker build & scan, security audit. "
            "Runs on every push and PR."
        )
        _render_mermaid(CI_CD_DIAGRAM_CORE, height=450)
    elif sel == _CD_TABS[1]:
        st.markdown(
            "**ML Quality Gate** (`model-quality-gate.yml`) \u2014 trains a fresh model on the "
            "git-committed reference signals and checks accuracy/F1 thresholds. "
            "Runs on PRs touching training code."
        )
        _render_mermaid(CI_CD_DIAGRAM_ML, height=450)
    elif sel == _CD_TABS[2]:
        st.markdown(
            "**Deploy** (`deploy.yml`) \u2014 smoke tests the Docker image and runs the deploy step. "
            "Manually triggered or on version tag pushes. Deploy step is currently a placeholder."
        )
        _render_mermaid(CI_CD_DIAGRAM_DEPLOY, height=420)
    else:
        st.markdown(
            "**K8s CI/CD Deploy** (`deploy-k8s.yml`) \u2014 spins up a Kind cluster in GitHub Actions, "
            "pulls the latest GHCR images, applies Kustomize manifests, and smoke-tests the API. "
            "Auto-triggered after every successful `build.yml` run on `main`, or triggered manually."
        )
        _render_mermaid(
            """
flowchart LR
    A["Push to main"] --> B["build.yml succeeds"]
    B --> C["deploy-k8s.yml triggered"]
    C --> D["Create Kind cluster"]
    D --> E["Pull GHCR images"]
    E --> F["kubectl apply -k overlays/ghcr"]
    F --> G["Wait for pods Ready"]
    G --> H{"GET /health"}
    H -->|"200 OK"| I["✅ Smoke test passed"]
    H -->|"Fail"| J["❌ Workflow fails"]
    I --> K["Teardown cluster"]
""",
            height=420,
        )


def _render_kubernetes_architecture() -> None:
    """Kubernetes cluster network topology tab."""
    st.markdown(
        '<div class="section-header">\u2638\ufe0f Kubernetes Cluster Network</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "\U0001f4a1 **Kubernetes mode** \u2014 all Docker Compose services are re-deployed as K8s "
        "Deployments inside namespace `mlops`. "
        "Use `make k8s-up` to start and `make k8s-status` to inspect. "
        "The Streamlit **Kubernetes** page provides a GUI for cluster control."
    )
    _render_mermaid(
        """
flowchart LR
    subgraph EXT["🌐 External Access"]
        U["User Browser / Streamlit"]
    end
    subgraph NS["☸️ K8s namespace: mlops"]
        subgraph NP["🔌 NodePort Services"]
            SVC_NGINX["mlops-nginx\\nNodePort :30080"]
            SVC_AF["mlops-airflow\\nNodePort :30081"]
            SVC_MLFLOW["mlops-mlflow\\nNodePort :30502"]
            SVC_GF["mlops-grafana\\nNodePort :30300"]
            SVC_PROM["mlops-prometheus\\nNodePort :30900"]
        end
        subgraph PODS["🖥️ Pods"]
            P_NGINX["nginx\\n(1 replica)"]
            P_API["api\\n(1-3 replicas)\\nHPA enabled"]
            P_AF["airflow\\n(1 replica)"]
            P_MLFLOW["mlflow / buffer\\n(1 replica)"]
            P_GF["grafana\\n(1 replica)"]
            P_PROM["prometheus\\n(1 replica)"]
            P_PG["postgres\\n(1 replica)"]
        end
        subgraph PVC["💾 PersistentVolumeClaims"]
            V_PG["postgres-pvc"]
            V_MLFLOW["mlflow-pvc"]
            V_GF["grafana-pvc"]
        end
    end
    subgraph GHCR["📦 GHCR Registry"]
        IMG["ghcr.io/<GITHUB_OWNER>/\nmlops-* images"]
    end

    U -->|"http://localhost:30080"| SVC_NGINX
    SVC_NGINX --> P_NGINX
    P_NGINX -->|"proxy /predict"| P_API
    P_API --> P_PG
    P_API --> P_MLFLOW
    SVC_AF --> P_AF
    P_AF --> P_PG
    SVC_MLFLOW --> P_MLFLOW
    SVC_GF --> P_GF
    SVC_PROM --> P_PROM
    P_GF -->|"scrape"| P_PROM
    P_PG --- V_PG
    P_MLFLOW --- V_MLFLOW
    P_GF --- V_GF
    IMG -.->|"imagePull"| PODS

    style EXT fill:#1e3a5f,stroke:#3b82f6,color:#e0f2fe
    style NP fill:#1a3a2a,stroke:#22c55e,color:#dcfce7
    style PODS fill:#2d1b4e,stroke:#a855f7,color:#f3e8ff
    style PVC fill:#3b2000,stroke:#f59e0b,color:#fef3c7
    style GHCR fill:#1e1b4b,stroke:#818cf8,color:#e0e7ff
    style NS fill:#0f172a,stroke:#475569,color:#cbd5e1
""",
        height=720,
    )
    st.markdown("#### Kubernetes Component Summary")
    st.markdown(
        """
| Resource | Kind | Replicas | NodePort | Purpose |
|:---------|:-----|:---------|:---------|:--------|
| `mlops-nginx` | Deployment | 1 | 30080 | Reverse proxy — entry point for all traffic |
| `mlops-api` | Deployment | 1–3 (HPA) | — | FastAPI prediction service (behind nginx) |
| `mlops-postgres` | Deployment | 1 | — | Application PostgreSQL database |
| `mlops-mlflow` | Deployment | 1 | 30502 | MLflow tracking server / buffer |
| `mlops-airflow` | Deployment | 1 | 30081 | Airflow webserver + scheduler |
| `mlops-grafana` | Deployment | 1 | 30300 | Grafana dashboards |
| `mlops-prometheus` | Deployment | 1 | 30900 | Prometheus metrics + alerts |
| `postgres-pvc` | PVC | — | — | PostgreSQL data persistence |
| `mlflow-pvc` | PVC | — | — | MLflow artifact persistence |
| `grafana-pvc` | PVC | — | — | Grafana dashboard persistence |
| `api-hpa` | HorizontalPodAutoscaler | 1–3 | — | Auto-scales API on CPU > 70% |

> **Note:** In cloud overlay (`make k8s-up K8S_OVERLAY=cloud`) the API defaults to 3 replicas.
> In GHCR overlay (`make k8s-ghcr-up`) images are pulled from `ghcr.io` instead of built locally.
"""
    )


def _architecture_details(mode: str) -> None:
    """Show component detail cards beneath the architecture diagram."""
    st.markdown("#### Component Details")
    components_local = [
        (
            "🌐",
            "Nginx Reverse Proxy",
            "TLS termination, CORS headers, rate limiting (100 req/s), CSP security headers.",
        ),
        (
            "⚡",
            "FastAPI Prediction Service",
            "REST API at /predict, /train, /health endpoints. Pydantic validation, Prometheus metrics.",
        ),
        (
            "🗄️",
            "PostgreSQL (App)",
            "Application database: device records, signals, predictions, features, sparse labels.",
        ),
        (
            "🔬",
            "MLflow Tracking (local)",
            "Experiment tracking, model registry (Staging → champion), artefact storage. Local container.",
        ),
        (
            "📊",
            "Prometheus",
            "Scrapes /metrics from API every 15 s. Alert rules for latency, error rate.",
        ),
        (
            "📈",
            "Grafana",
            "Pre-provisioned dashboards: predictions, model performance, system health, container metrics.",
        ),
    ]
    components_cloud_extra = [
        (
            "🔬",
            "MLflow Buffer (cloud)",
            "Local-first MLflow container. All training/promotion uses this. Syncs to DagsHub on schedule.",
        ),
        (
            "🗄️",
            "PostgreSQL (MLflow)",
            "Dedicated PostgreSQL backend for the MLflow Buffer container.",
        ),
        (
            "🔄",
            "Airflow Webserver",
            "5 DAGs: automated retraining, drift detection, model promotion, production data sync.",
        ),
        (
            "🐘",
            "PG Exporter (App + MLflow)",
            "Exports PostgreSQL metrics (connections, cache hit ratio, query duration) to Prometheus.",
        ),
    ]
    component_list = (
        components_local
        if mode not in ("cloud", "k8s")
        else components_local + components_cloud_extra
    )
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(component_list):
        with cols[i % 3]:
            st.markdown(
                f'<div class="info-card"><h3>{icon} {title}</h3><p>{desc}</p></div>',
                unsafe_allow_html=True,
            )


def _dataflow_details() -> None:
    """Show pipeline stage details table."""
    st.markdown("#### Pipeline Stages (dvc.yaml)")
    st.markdown(
        """
| Stage | Script | Input | Output |
|:------|:-------|:------|:-------|
| **generate_data** | `scripts/generate_data.py` | Config params | `data/raw/*.json` |
| **extract_features** | `src/signal_processing/preprocess.py` | Raw signals | `data/processed/features_*.csv` + `labels_*.csv` |
| **train** | `src/training/train.py` | Features + labels | `models/champion_model.pkl` + metrics |
| **evaluate** | `src/training/evaluate.py` | Model + test data | `metrics/eval_metrics.json` |
| **sync_production_data** | Airflow DAG | PostgreSQL prod data | DVC-tracked exports → DagsHub (cloud only) |

> **Note:** `dvc repro` reproduces the first four stages for local development.
> In production the pipeline is driven by the FastAPI endpoints and Airflow DAGs, not `dvc repro`.
> The `sync_production_data` stage is not run via `dvc repro` — it runs via the Airflow DAG on a schedule.
"""
    )
