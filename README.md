<div align="center">

# 🏥 MLOps Device Health

**Production-Grade MLOps Platform for IoT Signal Classification**

[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](pyproject.toml)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](src/api/)
[![MLflow 3.9](https://img.shields.io/badge/MLflow-3.9-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Streamlit 1.54](https://img.shields.io/badge/Streamlit-1.54-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](src/ui/)
[![Docker](https://img.shields.io/badge/Docker-14_containers-2496ED?style=for-the-badge&logo=docker&logoColor=white)](docker-compose.yml)
[![Airflow 2.8](https://img.shields.io/badge/Airflow-2.8-017CEE?style=for-the-badge&logo=apacheairflow&logoColor=white)](airflow/)
[![DVC + DagsHub](https://img.shields.io/badge/DVC-DagsHub-13ADC7?style=for-the-badge&logo=dvc&logoColor=white)](https://dagshub.com/)
[![Tests](https://img.shields.io/badge/tests-851_passing-brightgreen?style=for-the-badge&logo=pytest)](tests/)
[![GitHub Actions](https://img.shields.io/badge/CI%2FCD-8_workflows-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)](.github/workflows/)
[![License MIT](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](LICENSE)

*Semi-supervised learning · Champion/Challenger · Drift detection · Full observability · 3 deployment modes · Kubernetes*

[Quick Start](#-quick-start) · [Architecture](#-architecture) · [Dashboard](#-streamlit-dashboard) · [Documentation](#-reference-manuals)

</div>

---

## 🎯 Overview

A complete MLOps system that classifies IoT device health from raw time-series signals. The platform covers the **full ML lifecycle** — from data generation and semi-supervised training through automated drift detection and retraining to production deployment with full observability.

**14 Docker containers** · **16 Streamlit pages** · **9 interactive use cases** · **9 Airflow DAGs** · **8 CI/CD workflows** · **3 deployment modes + Kubernetes**

> 📋 This project is based on the [MLOps Project Proposal](doc/MLOps_Project_Proposal_Fred_Richter.md) by Fred Richter.

### What Makes This Project Stand Out

| Aspect | Implementation |
|:-------|:---------------|
| **Signal Processing** | Raw amplitude signals → 6 engineered features (peak_height, peak_center, FWHM, peak_area, SNR, noise_level) |
| **Semi-Supervised Learning** | Handles sparse labels using K-Means clustering + label propagation + domain heuristics |
| **Automated Retraining** | Drift-triggered via EvidentlyAI → Prometheus alerts → Airflow DAG → Champion/Challenger evaluation |
| **Full Observability** | 20+ Prometheus metrics, Grafana dashboards, 17 alert rules, Alertmanager routing |
| **Three Deployment Modes** | Local sandbox · Cloud (DagsHub + Airflow) · GHCR (pre-built images from registry) |
| **Kubernetes Support** | Full K8s manifests (Kustomize), HPA autoscaling, one-command deploy with `make k8s-up`, GHCR overlay for CI/CD |
| **Kubernetes CI/CD** | `deploy-k8s.yml` — Kind cluster in GitHub Actions: pulls GHCR images → deploys K8s manifests → smoke-tests API → auto-triggered after every `build.yml` on `main` |
| **Interactive Control Plane** | 16-page Streamlit dashboard with live Docker management, Kubernetes cluster control, predictions, and model lineage |
| **Cross-Platform** | Windows, Linux, macOS — with platform-aware Makefile and environment detection |

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🤖 ML Pipeline
- Semi-supervised training (K-Means + label propagation)
- 6 signal features: peak_height, peak_center, FWHM, peak_area, SNR, noise_level
- LogisticRegression / SVC / XGBoost classifiers
- Champion/Challenger with **fair comparison** (champion re-evaluated on challenger's test signals)
- Training data lineage: `model_training_data` table records every train/test split
- Quality gate (F1 ≥ 0.75, accuracy ≥ 0.80)

</td>
<td width="50%">

### 📊 Experiment Tracking
- MLflow 3.9 experiment tracking & model registry
- Champion / Challenger / Archived model aliases
- DVC data versioning with DagsHub remote
- MLflow buffer architecture (cloud mode)
- Bi-directional DagsHub sync

</td>
</tr>
<tr>
<td width="50%">

### 🚀 Production API
- FastAPI with OAuth2 + API key authentication
- Nginx reverse proxy with rate limiting + SSL/TLS
- Full prediction lineage (model_version, git_sha, dvc_data_hash, airflow_run_id)
- Prometheus metrics instrumentation
- Hot model reload (`POST /admin/reload-model`)

</td>
<td width="50%">

### 🔍 Drift Detection
- 4 drift types: data, concept, feature, prior probability
- EvidentlyAI + KS-test per-feature analysis
- Automated drift-triggered retraining DAG
- Drift provocation with configurable parameters
- Reference vs drifted distribution histograms

</td>
</tr>
<tr>
<td width="50%">

### 📈 Monitoring & Alerting
- Prometheus + Grafana + Alertmanager stack
- cAdvisor (container), Node Exporter (host), PG Exporter (database)
- Blackbox Exporter (endpoint probing)
- 17 alert rules across 5 severity groups
- Real-time container health in Streamlit UI

</td>
<td width="50%">

### 🔄 Orchestration & CI/CD
- 7 Airflow DAGs (retraining, drift, sync, backup, promotion)
- 8 GitHub Actions workflows (lint, test, build, quality, deploy, K8s deploy)
- 3 Docker images auto-built and pushed to GHCR
- Trivy security scanning on every build
- Model quality gate in CI pipeline

</td>
</tr>
</table>

---

## 🏗 Architecture

```
                           ┌────────────────────────────────┐
                           │       Nginx (:80/:443)         │
                           │  Reverse proxy · Rate limiting │
                           └──────────┬─────────────────────┘
                                      │
                    ┌─────────────────┼──────────────────┐
                    ▼                 ▼                  ▼
           ┌──────────────┐  ┌─────────────┐   ┌──────────────┐
           │   FastAPI    │  │  Streamlit  │   │   Grafana    │
           │   API :8000  │  │  UI :8501   │   │   :3000      │
           └───────┬──────┘  └─────────────┘   └───────┬──────┘
                   │                                   │
          ┌────────┼─────────────────┐                 │
          ▼        ▼                 ▼                 ▼
   ┌────────────┐ ┌──────────┐ ┌────────────┐  ┌──────────────┐
   │ PostgreSQL │ │  MLflow  │ │ Prometheus │  │ Alertmanager │
   │ :5432      │ │  :5001   │ │  :9090     │  │ :9093        │
   │            │ │ (local)  │ │            │  │              │
   │ 7 tables   │ │  :5002   │ │ 17 alert   │  │ Email/Slack/ │
   │            │ │ (cloud   │ │ rules      │  │ Webhook      │
   │            │ │  buffer) │ │            │  │              │
   └────────────┘ └──────────┘ └───────┬────┘  └──────────────┘
                                       │
                              ┌────────┼────────┐
                              ▼        ▼        ▼
                       ┌──────────┐ ┌────────┐ ┌──────────────┐
                       │ Node     │ │cAdvisor│ │ Blackbox     │
                       │ Exporter │ │ :8080  │ │ Exporter     │
                       │ :9100    │ │        │ │ :9115        │
                       └──────────┘ └────────┘ └──────────────┘

   ┌──────────────────────────────────────────────────────────────┐
   │                    Airflow :8081 (Cloud Mode)                │
   │                                                              │
   │  automated_retraining        (Weekly Sun 02:00 UTC)          │
   │  drift_triggered_retraining  (On drift alert)                │
   │  evidently_drift_detection   (Daily 06:00 UTC)               │
   │  sync_production_data        (Daily 04:00 UTC)               │
   │  database_backup             (Daily 02:00 UTC)               │
   │  model_promotion             (Manual)                        │
   │  sync_mlflow_to_dagshub      (Manual)                        │
   └──────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Docker** 24+ with Compose V2
- **Python** 3.12+
- **Git** and **GNU Make**

### 1. Clone & Install

```bash
git clone https://github.com/DataScientest-Studio/jan26_cmlops_int_device_health.git
cd jan26_cmlops_int_device_health

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh    # Linux/macOS
# or: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Install dependencies
uv sync --dev
```

### 2. Configure Credentials

```bash
cp .env.secrets.example .env.secrets
# Edit .env.secrets — minimum required: DB_PASSWORD, GRAFANA_PASSWORD, AIRFLOW_PASSWORD
```

### 3. Start the Stack

```bash
make local           # Local sandbox (no Airflow, no DagsHub)
# or: make cloud     # Full cloud mode (Airflow + DagsHub)
# or: make ghcr      # Pre-built images from GHCR
```

### 4. Bootstrap & Verify

```bash
# Start the Streamlit dashboard
make ui
# Open http://localhost:8501 → Use Cases → Greenfield Bootstrap

# Or verify via CLI:
curl http://localhost/health
curl -X POST http://localhost/predict \
  -H "X-API-Key: dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{"device_id":"test","time_values":[0,1,2,3,4],"amplitude_values":[0.1,0.5,-0.3,0.7,-0.2]}'
```

### 5. Run Tests

```bash
make test            # Fast unit + integration (no Docker needed)
make test-ui         # Streamlit UI tests
make test-live       # Docker stack integration tests
make test-all        # Everything
```

---

## 🎨 Streamlit Dashboard

Access at **http://localhost:8501** — start with `make ui`

### 16 Pages

| Page | Purpose |
|:-----|:--------|
| 🏠 **Home** | Project overview, metrics grid, coverage stats |
| 🏗️ **Architecture** | 7 interactive Mermaid diagrams (mode-aware) |
| 📡 **Data & Signals** | Signal comparison, live feature extraction, drift scenarios |
| 🔄 **DAGs & Pipelines** | Airflow DAG documentation with Mermaid flowcharts |
| 🐳 **Docker Control** | Start/stop stack, container status, streaming console |
| 🔗 **Services** | Service catalog with embedded web UIs |
| 🧪 **Use Cases** | 9 interactive MLOps workflows (see below) |
| 🎯 **Predictions** | Single/batch predictions, history, label injection |
| 🗄️ **PostgreSQL** | Schema viewer, data browser, backup/restore |
| 🔬 **MLflow Explorer** | Experiments, runs, model registry, DagsHub sync |
| ✈️ **Airflow Control** | DAG management — pause/unpause, trigger, monitor |
| 📊 **Monitoring** | Prometheus metrics, container health, active alerts |
| 🐙 **GitHub CI/CD** | Workflow status, triggers, GHCR image browser, K8s CI/CD trigger |
| ☸️ **Kubernetes** | Cluster control (build/up/down/nuke), pod list, scaling, resilience |
| 🖥️ **App Console** | Live log viewer with filtering |
| ℹ️ **About** | Credits, technology stack |

### 9 Interactive Use Cases

| Use Case | What It Does |
|:---------|:-------------|
| 🚀 **Greenfield Bootstrap** | Wipe all → generate data → train model → register → verify |
| 🔄 **Retraining Pipeline** | Generate predictions → inject labels → trigger Airflow DAG |
| 📊 **Drift Provocation** | Inject data/concept/feature/prior drift → KS-tests → trigger retraining |
| 🥊 **Champion/Challenger** | Train challenger → side-by-side comparison → promote if better |
| 🔀 **A/B Testing** | Dual API containers → send to both → compare predictions |
| ⚖️ **Nginx Traffic Split** | Infrastructure-level traffic split via Nginx `split_clients` (configurable %) |
| 🏆 **Model Promotion** | Promote / rollback / archive model versions in the MLflow registry |
| 🔍 **Model Lineage Audit** | Trace prediction → model → MLflow run → git SHA → DVC hash |
| ⏮️ **Batch Re-Scoring** | Re-score historical predictions with current champion → Airflow DAG |

---

## 🌐 Deployment Modes

| Mode | Command | Services | MLflow | Airflow | DVC |
|:-----|:--------|:---------|:-------|:--------|:----|
| **Local** | `make local` | 12 containers | Docker (:5001) | Disabled | Disabled |
| **Local (rebuild)** | `make local-rebuild` | 12 containers | Docker (:5001) | Disabled | Disabled |
| **Cloud** | `make cloud` | 16 containers | Buffer (:5002) + DagsHub | Enabled (:8081) | DagsHub remote |
| **Cloud (rebuild)** | `make cloud-rebuild` | 16 containers | Buffer (:5002) + DagsHub | Enabled (:8081) | DagsHub remote |
| **GHCR** | `make ghcr` | Pre-built images | Buffer (:5002) + DagsHub | Enabled | DagsHub remote |
| **Kubernetes** | `make k8s-up` | All services as K8s Pods | In-cluster | Enabled | DagsHub remote |

```bash
make local              # Start local sandbox
make local-rebuild      # Rebuild images, then start local
make cloud              # Start cloud mode (requires DagsHub credentials)
make cloud-rebuild      # Rebuild images, then start cloud
make ghcr               # Pull pre-built images from GHCR
make down               # Stop all containers
make safe-down          # Sync MLflow to DagsHub, then stop
make status             # Show current mode + container status
make restart            # Stop + restart in same mode
make nuke               # Nuclear teardown (all data lost)
```

**Kubernetes quick reference:**

```bash
make k8s-build          # Build Docker images for K8s
make k8s-up             # Deploy full stack to Kubernetes (local overlay)
make k8s-up K8S_OVERLAY=cloud  # Deploy with 3 API replicas + HPA
make k8s-ghcr-up        # Deploy with GHCR images (CI/CD overlay — pulls from ghcr.io)
make k8s-status         # Show pods, deployments, services, HPA
make k8s-logs           # Stream API pod logs
make k8s-scale REPLICAS=2      # Scale API to 2 replicas
make k8s-down           # Teardown stack (keep PVC data)
make k8s-nuke           # Full teardown including all PVC data
make k8s-context        # Show current kubectl context
```

> **Windows users:** Ports 80/443 may conflict. Create `.env.windows.local` with `NGINX_HTTP_PORT=8080`.

---

## 📡 API Reference

**Base URL:** `http://localhost` (Nginx) or `http://localhost:8000` (direct)
**Interactive docs:** [Swagger](http://localhost/docs) · [ReDoc](http://localhost/redoc)

### Key Endpoints

| Method | Path | Auth | Purpose |
|:-------|:-----|:-----|:--------|
| `GET` | `/health` | Public | Health check (DB, model, services) |
| `POST` | `/predict` | API key / OAuth2 | Classify a device signal |
| `POST` | `/evaluate` | API key / OAuth2 | Batch classify + return predictions list |
| `GET` | `/predictions` | API key / OAuth2 | List prediction history |
| `GET` | `/predictions/{id}/lineage` | API key / OAuth2 | Full lineage for one prediction |
| `PUT` | `/predictions/{id}/labels` | API key / OAuth2 | Inject ground truth label |
| `POST` | `/labels` | API key / OAuth2 | Inject label by prediction ID |
| `GET` | `/model/info` | API key / OAuth2 | Current model metadata + version |
| `GET` | `/stats` | API key / OAuth2 | Aggregated prediction statistics |
| `POST` | `/auth/token` | Public | Get OAuth2 JWT token |
| `POST` | `/auth/refresh` | Public | Refresh JWT token |
| `POST` | `/admin/reload-model` | Admin | Hot-reload model from registry |
| `GET` | `/metrics` | Public | Prometheus metrics scrape |
| `GET` | `/k8s/pods` | Admin | List K8s pods (K8s mode) |
| `POST` | `/k8s/scale` | Admin | Scale K8s deployment (K8s mode) |
| `POST` | `/k8s/kill-pod` | Admin | Delete a pod for resilience testing |

### Authentication

| Method | Header | Credentials |
|:-------|:-------|:-----------|
| **API Key** (simplest) | `X-API-Key: dev-key-12345` | Full access (dev) |
| **API Key** (read-only) | `X-API-Key: monitoring-key-67890` | Read-only |
| **OAuth2 JWT** | `Authorization: Bearer <token>` | `admin`/`secret` or `user`/`secret` |

```bash
# Quick prediction with API key
curl -X POST http://localhost/predict \
  -H "X-API-Key: dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{"device_id":"sensor_001","time_values":[0,1,2,3,4],"amplitude_values":[0.1,0.5,-0.3,0.7,-0.2]}'
```

---

## 📈 Monitoring Stack

| Component | Port | Purpose |
|:----------|:-----|:--------|
| **Prometheus** | 9090 | Metrics collection + 17 alert rules |
| **Grafana** | 3000 | Dashboards (System Health, Model Performance, Data Quality, Business KPIs) |
| **Alertmanager** | 9093 | Alert routing (email, Slack, webhook) |
| **cAdvisor** | 8080 | Container resource metrics |
| **Node Exporter** | 9100 | Host metrics (Linux/macOS only) |
| **PG Exporter** | 9187 | PostgreSQL metrics |
| **Blackbox Exporter** | 9115 | HTTP endpoint probing |

---

## 🔄 Airflow DAGs (Cloud Mode)

| DAG | Schedule | Purpose |
|:----|:---------|:--------|
| `automated_retraining` | Weekly Sun 02:00 UTC | DB-backed challenger training, fair champion re-evaluation, promote/cleanup |
| `drift_triggered_retraining` | On drift alert | Emergency retrain on detected drift |
| `evidently_drift_detection` | Daily 06:00 UTC | Run EvidentlyAI drift reports |
| `batch_rescoring` | Manual / scheduled | Re-score past predictions with current champion model |
| `sync_production_data` | Daily 04:00 UTC | Export DB → DVC → DagsHub |
| `database_backup` | Daily 02:00 UTC | pg_dump + rotation |
| `model_promotion` | Manual | Promote/archive model versions |
| `sync_mlflow_to_dagshub` | Manual | Push MLflow buffer → DagsHub |

---

## 🔧 GitHub CI/CD

8 workflows in `.github/workflows/`:

| Workflow | Trigger | Purpose |
|:---------|:--------|:--------|
| **lint.yml** | Push, PR | ruff check/format + mypy type checking |
| **test.yml** | Push, PR | Unit/integration tests + Codecov upload |
| **build.yml** | Push (main), tags | Build 3 Docker images + Trivy scan → GHCR push |
| **code-quality.yml** | PR, weekly | Bandit SAST + pip-audit CVE check |
| **live-tests.yml** | Nightly, PR (infra) | Full Docker stack integration tests |
| **model-quality-gate.yml** | PR (training code) | Model regression testing (F1 ≥ 0.75, accuracy ≥ 0.80) |
| **deploy.yml** | Tags (v*), manual | Pull GHCR image → smoke test → deploy |
| **deploy-k8s.yml** | After build (main), manual | Kind cluster → apply K8s manifests → smoke-test API |

---

## 🧪 Testing

```bash
make test              # Fast unit + integration (no Docker)
make test-ui           # Streamlit UI tests
make test-live         # Docker stack integration tests
make test-all          # Everything
```

| Category | Count | Description |
|:---------|:------|:-----------|
| Core + API + Database + ML | ~288 | Signal logic, API, database, training pipeline |
| K8s + Config | ~348 | Kubernetes manifests, Kustomize overlays, workflow configs |
| Security + Monitoring + Perf | ~35 | Auth, metrics, latency thresholds |
| Live (requires Docker) | ~43 | Docker stack end-to-end |
| Reproducibility | ~4 | End-to-end model reproducibility |
| **Total** | **~851** | 808 CI-default · 43 live |

---

## 📁 Project Structure

```
mlops-device-health/
├── src/
│   ├── api/                      # FastAPI service (routes, auth, security)
│   ├── training/                 # ML pipeline (train, evaluate, promote, sync)
│   ├── signal_processing/        # Signal generation + feature extraction
│   ├── database/                 # SQLAlchemy ORM (7 tables)
│   ├── monitoring/               # Prometheus metrics instrumentation
│   ├── ui/                       # Streamlit dashboard (16 pages, 9 use cases)
│   │   ├── app.py                # Entry point + page dispatch
│   │   ├── views/                # 15 page modules
│   │   ├── views/use_cases_pkg/  # 9 interactive use case modules
│   │   └── components/           # Docker utils, signal viz, styles
│   └── config.py                 # Central configuration
├── airflow/dags/                 # 9 Airflow DAGs
├── tests/                        # ~851 tests (808 CI + 43 live)
├── scripts/                      # 28+ utility scripts
├── data/                         # DVC-tracked datasets
├── models/                       # Trained model artifacts
├── docker/                       # Dockerfiles + service configs
│   ├── api.Dockerfile            # FastAPI container
│   ├── airflow.Dockerfile        # Airflow container
│   ├── streamlit.Dockerfile      # Streamlit container
│   ├── nginx/                    # Reverse proxy + SSL
│   ├── prometheus/               # Scrape config + alert rules
│   ├── grafana/                  # Dashboard provisioning
│   └── alertmanager/             # Alert routing
├── k8s/                          # Kubernetes manifests (Kustomize)
│   ├── base/                     # Base manifests (Deployment, Service, PVC)
│   └── overlays/                 # cloud, ghcr, local overlays
├── doc/                          # Project proposal
├── .github/workflows/            # 8 CI/CD workflows
├── docker-compose.yml            # Base (14 services)
├── docker-compose.local.yml      # Local overlay
├── docker-compose.cloud.yml      # Cloud overlay
├── docker-compose.ghcr.yml       # GHCR overlay
├── Makefile                      # 50+ deployment commands
├── pyproject.toml                # Python project metadata
├── params.yaml                   # DVC pipeline parameters
└── dvc.yaml                      # DVC pipeline stages
```

---

##  Make Commands (Quick Reference)

| Command | Purpose |
|:--------|:--------|
| `make local` | Start local sandbox |
| `make local-rebuild` | Rebuild images, start local |
| `make cloud` | Start cloud mode |
| `make cloud-rebuild` | Rebuild images, start cloud |
| `make ghcr` | Pull GHCR images + start |
| `make down` | Stop all containers |
| `make safe-down` | Sync MLflow, then stop |
| `make status` | Show mode + container status |
| `make logs` | Tail container logs |
| `make restart` | Stop + restart same mode |
| `make ui` | Start Streamlit dashboard |
| `make test` | Fast test suite |
| `make test-live` | Docker integration tests |
| `make nuke` | Nuclear reset (all data lost) |
| `make fix-db-password` | Sync DB password |
| `make dvc-status` | Check DVC sync |
| `make mlflow-sync` | Push buffer → DagsHub |
| `make wipe` | Remove test data (dry-run) |
| `make k8s-up` | Deploy Kubernetes stack |
| `make k8s-down` | Teardown Kubernetes stack |
| `make k8s-status` | Show K8s pods and services |
| `make k8s-scale REPLICAS=2` | Scale API pods |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built by Fred Richter** · MSc in Applied Mathematics · Technical University Dresden

*MLOps Device Health — 2026*

</div>
