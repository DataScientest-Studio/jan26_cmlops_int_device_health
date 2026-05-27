# =============================================================================
# Makefile - MLOps Device Health Platform
# =============================================================================
# Single-command deployment mode switching.
# Requires: Docker, docker compose v2, dvc
#
# Quick reference:
#   make local    -> Start all services in local mode (MLflow in Docker)
#   make cloud    -> Start all services in cloud mode (MLflow on DagsHub)
#   make down     -> Stop all services
#   make status   -> Show current mode + container states
#   make logs     -> Tail all service logs
#   make restart  -> Stop and restart in the same mode
#   make test     -> Run full pytest suite
# =============================================================================

# ── Internal helpers ──────────────────────────────────────────────────────────

# Detect current mode from running containers or env files
CURRENT_MODE := $(shell \
  if [ -f .current_mode ]; then cat .current_mode; \
  elif docker inspect --format '{{.State.Status}}' mlops_mlflow 2>/dev/null | grep -q 'running'; then echo "local"; \
  elif docker inspect --format '{{.State.Status}}' mlops_mlflow_buffer 2>/dev/null | grep -q 'running'; then echo "cloud"; \
  elif docker inspect --format '{{.State.Status}}' mlops_api 2>/dev/null | grep -q 'running'; then echo "cloud"; \
  else echo "unknown"; fi)

# Base compose command (docker compose v2)
COMPOSE := docker compose

# ── OS / platform detection ───────────────────────────────────────────────────
# $(OS) is "Windows_NT" on Windows; uname -s returns "Linux" or "Darwin" on Unix.
ifeq ($(OS),Windows_NT)
_PLATFORM     := windows
# .env.windows.local overrides ports 80/443 → 8080/8443 on Windows (git-ignored).
_PLATFORM_ENV := $(if $(wildcard .env.windows.local),--env-file .env.windows.local,)
# node_exporter requires Linux host mounts — unavailable on Windows Docker Desktop.
_LINUX_PROFILES :=
# Virtual-environment binaries on Windows live in Scripts/ not bin/
_STREAMLIT    := .venv/Scripts/streamlit.exe
_PYTHON       := .venv/Scripts/python.exe
else
_UNAME_S        := $(shell uname -s)
ifeq ($(_UNAME_S),Linux)
_PLATFORM       := linux
_PLATFORM_ENV   :=
# On a real Linux host, enable Linux-only services (e.g. node_exporter) in cloud mode.
_LINUX_PROFILES := --profile linux
else
# macOS (Darwin) — Docker Desktop uses a Linux VM; /proc, /sys, / bind-mounts
# are available to containers (they target the VM's filesystem, not macOS host).
# So node_exporter works fine on macOS, same as native Linux.
_PLATFORM       := macos
_PLATFORM_ENV   :=
_LINUX_PROFILES := --profile linux
endif
_STREAMLIT := .venv/bin/streamlit
_PYTHON    := .venv/bin/python
endif

# Env file combos
LOCAL_ENV  := --env-file .env.local --env-file .env.secrets $(_PLATFORM_ENV)
CLOUD_ENV  := --env-file .env.cloud --env-file .env.secrets $(_PLATFORM_ENV)

# Compose file combos
LOCAL_FILES  := -f docker-compose.yml -f docker-compose.local.yml
CLOUD_FILES  := -f docker-compose.yml -f docker-compose.cloud.yml
GHCR_FILES   := -f docker-compose.yml -f docker-compose.cloud.yml -f docker-compose.ghcr.yml

# GHCR configuration
# GITHUB_OWNER: registry namespace — override in .env.secrets or on the command line.
# GHCR_TAG:     image tag to pull — defaults to "main" (the branch tag on main).
#               For a feature branch build set e.g.: make GHCR_TAG=feature-mlops-cross-platform ghcr
GITHUB_OWNER ?= fredrichter80
GHCR_TAG     ?= main
# Export so docker compose inherits them for ${GITHUB_OWNER} / ${GHCR_TAG} substitution
# in docker-compose.ghcr.yml without needing them in any .env file.
export GITHUB_OWNER
export GHCR_TAG

# Export the current git commit SHA so Airflow/API containers receive it as GIT_SHA.
# This enables get_git_commit_hash() to work inside Docker where .git is not mounted.
GIT_SHA := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
export GIT_SHA

# Color output — printf in recipes interprets \033 as ESC (echo does not in POSIX sh)
GREEN  := \033[0;32m
YELLOW := \033[0;33m
RED    := \033[0;31m
BLUE   := \033[0;34m
NC     := \033[0m

# Access URLs.  On Windows, Docker Desktop WSL2 IPv6 relay is broken so
# localhost resolves to ::1 which fails.  Use 127.0.0.1 (IPv4) instead.
# Windows also uses port 8080 for nginx (NGINX_HTTP_PORT in .env.windows.local).
ifeq ($(OS),Windows_NT)
_BASE_URL  := http://127.0.0.1
_NGINX_URL := http://127.0.0.1:8080
else
_BASE_URL  := http://localhost
_NGINX_URL := http://localhost
endif

.PHONY: local cloud ghcr ghcr-rebuild ghcr-login down status logs restart test test-ui test-live test-all check-secrets wipe reset-db reset-local-db fix-db-password local-rebuild cloud-rebuild cloud-with-ui cloud-rebuild-with-ui local-wipe local-pull dvc-status dvc-clean data-push nuke nuke-cloud nuke-local help ui safe-down mlflow-restore mlflow-sync k8s-build k8s-up k8s-down k8s-nuke k8s-status k8s-logs k8s-scale k8s-ghcr-up k8s-context k8s-secret k8s-sync-dags k8s-ports k8s-ports-stop k8s-test k8s-setup k8s-full

# ── Default target ────────────────────────────────────────────────────────────
help:
	@echo ""
	@printf "  $(BLUE)MLOps Device Health Platform$(NC)\n"
	@echo ""
	@printf "  $(BLUE)-- Docker Compose Modes ---------------------------------------------------$(NC)\n"
	@printf "  $(GREEN)make local$(NC)              Start all services in local sandbox mode\n"
	@printf "  $(GREEN)make local-wipe$(NC)         Wipe local data, then start in local mode\n"
	@printf "  $(GREEN)make local-pull$(NC)         Wipe local + pull from DagsHub cloud, then start\n"
	@printf "  $(GREEN)make local-rebuild$(NC)      Full rebuild (no Docker cache) in local mode\n"
	@printf "  $(GREEN)make cloud$(NC)              Start all services in cloud mode (MLflow on DagsHub)\n"
	@printf "  $(GREEN)make cloud-rebuild$(NC)      Full rebuild (no Docker cache) in cloud mode\n"
	@printf "  $(GREEN)make cloud-with-ui$(NC)      Cloud stack + Streamlit container (all in Docker)\n"
	@printf "  $(GREEN)make cloud-rebuild-with-ui$(NC) Full rebuild: cloud stack + Streamlit container\n"
	@printf "  $(GREEN)make ghcr$(NC)               Pull pre-built images from GHCR and start\n"
	@printf "  $(GREEN)make ghcr-rebuild$(NC)       Force pull latest GHCR images and restart\n"
	@echo ""
	@printf "  $(BLUE)-- Kubernetes -------------------------------------------------------------$(NC)\n"
	@printf "  $(GREEN)make k8s-full$(NC)           Full K8s workflow: secret -> sync-dags -> build -> up -> ports -> test\n"
	@printf "  $(GREEN)make k8s-setup$(NC)          Setup only: secret -> sync-dags -> build -> up\n"
	@printf "  $(GREEN)make k8s-build$(NC)          Build Docker images for Kubernetes\n"
	@printf "  $(GREEN)make k8s-up$(NC)             Deploy full stack to Kubernetes (local overlay)\n"
	@printf "  $(GREEN)make k8s-up K8S_OVERLAY=cloud$(NC)  Deploy with 3 API replicas + HPA\n"
	@printf "  $(GREEN)make k8s-ghcr-up$(NC)        Deploy with GHCR images (CI/CD overlay)\n"
	@printf "  $(GREEN)make k8s-secret$(NC)         Generate k8s/base/secret.yaml from .env.secrets\n"
	@printf "  $(GREEN)make k8s-sync-dags$(NC)      Sync airflow/dags/*.py -> dags-configmap.yaml\n"
	@printf "  $(GREEN)make k8s-ports$(NC)          Start port-forwards (all services on localhost)\n"
	@printf "  $(GREEN)make k8s-ports-stop$(NC)     Stop all port-forward jobs\n"
	@printf "  $(GREEN)make k8s-test$(NC)           Smoke-test all services via health endpoints\n"
	@printf "  $(YELLOW)make k8s-down$(NC)           Tear down K8s stack (keeps PVC data)\n"
	@printf "  $(RED)make k8s-nuke$(NC)            Tear down K8s stack + delete all PVCs (DATA LOSS)\n"
	@printf "  $(YELLOW)make k8s-status$(NC)         Show pods / deployments / services / HPA\n"
	@printf "  $(YELLOW)make k8s-logs$(NC)           Tail logs from API pods\n"
	@printf "  $(YELLOW)make k8s-scale REPLICAS=2$(NC) Scale API deployment to N replicas\n"
	@printf "  $(YELLOW)make k8s-context$(NC)        Show / list kubectl contexts\n"
	@echo ""
	@printf "  $(BLUE)-- UI & Services ----------------------------------------------------------$(NC)\n"
	@printf "  $(GREEN)make ui$(NC)                 Start Streamlit dashboard on host (requires stack)\n"
	@printf "  $(YELLOW)make down$(NC)               Stop all running services\n"
	@printf "  $(YELLOW)make status$(NC)             Show current deployment mode + container states\n"
	@printf "  $(YELLOW)make logs$(NC)               Tail logs from all services\n"
	@printf "  $(YELLOW)make restart$(NC)            Stop and restart in the current mode\n"
	@printf "  $(YELLOW)make safe-down$(NC)          Sync MLflow to DagsHub, then stop (cloud mode)\n"
	@echo ""
	@printf "  $(BLUE)-- MLflow Sync ------------------------------------------------------------$(NC)\n"
	@printf "  $(YELLOW)make mlflow-restore$(NC)     Restore MLflow buffer from DagsHub (disaster recovery)\n"
	@printf "  $(YELLOW)make mlflow-sync$(NC)        Push MLflow buffer runs to DagsHub\n"
	@echo ""
	@printf "  $(BLUE)-- Database ---------------------------------------------------------------$(NC)\n"
	@printf "  $(YELLOW)make reset-db$(NC)           Wipe entire postgres volume (all data lost)\n"
	@printf "  $(YELLOW)make reset-local-db$(NC)     Drop + recreate mlops_local only\n"
	@printf "  $(YELLOW)make fix-db-password$(NC)    Sync DB password without wiping data\n"
	@echo ""
	@printf "  $(BLUE)-- DVC / Data -------------------------------------------------------------$(NC)\n"
	@printf "  $(YELLOW)make dvc-status$(NC)         Show DVC cache/remote sync status\n"
	@printf "  $(YELLOW)make dvc-clean$(NC)          Remove stale DVC lock file\n"
	@printf "  $(YELLOW)make data-push$(NC)          Push DVC pointer files to GitHub\n"
	@printf "  $(YELLOW)make wipe$(NC)               Remove all test/demo data (dry-run; add EXEC=1)\n"
	@echo ""
	@printf "  $(BLUE)-- Testing ----------------------------------------------------------------$(NC)\n"
	@printf "  $(YELLOW)make test$(NC)               Run fast tests (unit + integration, no Docker)\n"
	@printf "  $(YELLOW)make test-ui$(NC)            Run Streamlit UI tests\n"
	@printf "  $(YELLOW)make test-live$(NC)          Run live Docker stack integration tests\n"
	@printf "  $(YELLOW)make test-all$(NC)           Run all tests (fast + live)\n"
	@echo ""
	@printf "  $(BLUE)-- Nuclear Reset ----------------------------------------------------------$(NC)\n"
	@printf "  $(RED)make nuke$(NC)               Remove ALL containers+networks+volumes\n"
	@printf "  $(RED)make nuke-cloud$(NC)         Nuclear reset + full cloud rebuild\n"
	@printf "  $(RED)make nuke-local$(NC)         Nuclear reset + full local rebuild\n"
	@echo ""
ifeq ($(OS),Windows_NT)
	@echo "  Windows: use 127.0.0.1 (not localhost) - Docker Desktop WSL2 IPv6 bug"
	@echo "  Access:  $(_NGINX_URL)/docs  (API via nginx)"
else
	@echo "  Access:  $(_NGINX_URL)/docs  (API via nginx)"
endif
	@echo ""
	@echo "  First time? Run:  cp .env.secrets.example .env.secrets"
	@echo "             Then:  make local"
	@echo ""
	@echo "  Mac/Linux fresh checkout? Run:  make nuke-cloud   (or nuke-local)"
	@echo ""

# ── Prerequisite check ────────────────────────────────────────────────────────
check-secrets:
	@if [ ! -f .env.secrets ]; then \
		printf "$(RED)ERROR: .env.secrets not found.$(NC)\n"; \
		echo "       Run: cp .env.secrets.example .env.secrets"; \
		echo "       Then fill in your values and try again."; \
		exit 1; \
	fi

# ── LOCAL MODE (sandbox — no DVC, no cloud push) ─────────────────────────────
local: check-secrets
	@printf "$(GREEN)>>  Starting in LOCAL sandbox mode$(NC)\n"
	@echo "   MLflow  -> local Docker container ($(_BASE_URL):5001)"
	@echo "   DVC     -> disabled (local sandbox)"
	@$(COMPOSE) $(LOCAL_ENV) $(LOCAL_FILES) $(_LINUX_PROFILES) up -d --build
	@docker exec mlops_postgres bash /scripts/pg_ensure_passwords.sh 2>/dev/null || true
	@echo "local" > .current_mode
	@echo ""
	@printf "$(GREEN)[OK]  Local sandbox started$(NC)\n"
	@echo "   API        -> $(_NGINX_URL)/docs"
	@echo "   MLflow     -> $(_BASE_URL):5001"
	@echo "   Grafana    -> $(_BASE_URL):3000"
	@echo "   Prometheus -> $(_BASE_URL):9090"
	@echo "   (Airflow is disabled in local mode)"
	@echo ""

# -- LOCAL WIPE (wipe local data, then start) ─────────────────────────────────
local-wipe: check-secrets
	@printf "$(YELLOW)>>  Wiping local sandbox data, then starting...$(NC)\n"
	@$(COMPOSE) $(LOCAL_ENV) $(LOCAL_FILES) $(_LINUX_PROFILES) up -d --build
	@docker exec mlops_postgres bash /scripts/pg_ensure_passwords.sh 2>/dev/null || true
	@echo "local" > .current_mode
	@python scripts/local_start.py wipe
	@echo ""
	@printf "$(GREEN)[OK]  Local sandbox started (data wiped)$(NC)\n"
	@echo ""

# ── LOCAL PULL (wipe + pull from cloud, then start) ──────────────────────────
local-pull: check-secrets
	@printf "$(BLUE)>>  Wiping local data + pulling from DagsHub, then starting...$(NC)\n"
	@$(COMPOSE) $(LOCAL_ENV) $(LOCAL_FILES) $(_LINUX_PROFILES) up -d --build
	@docker exec mlops_postgres bash /scripts/pg_ensure_passwords.sh 2>/dev/null || true
	@echo "local" > .current_mode
	@python scripts/local_start.py pull
	@echo ""
	@printf "$(GREEN)[OK]  Local sandbox started (refreshed from cloud)$(NC)\n"
	@echo ""

# ── CLOUD MODE ────────────────────────────────────────────────────────────────
cloud: check-secrets
	@printf "$(BLUE)>>  Starting in CLOUD mode$(NC)\n"
	@echo "   MLflow  -> DagsHub remote tracking"
	@echo "   DVC     -> DagsHub S3-compatible storage"
	@$(COMPOSE) $(CLOUD_ENV) $(CLOUD_FILES) $(_LINUX_PROFILES) up -d --build
	@docker exec mlops_postgres bash /scripts/pg_ensure_passwords.sh 2>/dev/null || true
	@echo "cloud" > .current_mode
	@echo ""
	@printf "$(BLUE)[OK]  Cloud mode started$(NC)\n"
	@echo "   API        -> $(_NGINX_URL)/docs"
	@echo "   MLflow     -> DagsHub (check your repo)"
	@echo "   Airflow    -> $(_BASE_URL):8081"
	@echo "   Grafana    -> $(_BASE_URL):3000"
	@echo "   Prometheus -> $(_BASE_URL):9090"
	@echo ""
	@$(MAKE) --no-print-directory _set_dvc_remote_cloud

# -- FULL REBUILD (no cache) ──────────────────────────────────────────────────
# These targets tear down, purge all Docker build cache for this project, and
# rebuild every image from scratch.  Use when you suspect stale layers.

local-rebuild: check-secrets
	@printf "$(GREEN)>>  Full rebuild in LOCAL sandbox mode (no Docker cache)$(NC)\n"
	@$(MAKE) --no-print-directory down
	@printf "$(YELLOW)   Clearing Grafana data volume so updated dashboards are provisioned fresh...$(NC)\n"
	@docker volume rm mlops-device-health_grafana_data 2>/dev/null || true
	@$(COMPOSE) $(LOCAL_ENV) $(LOCAL_FILES) $(_LINUX_PROFILES) build --no-cache
	@$(COMPOSE) $(LOCAL_ENV) $(LOCAL_FILES) $(_LINUX_PROFILES) up -d --force-recreate
	@docker exec mlops_postgres bash /scripts/pg_ensure_passwords.sh 2>/dev/null || true
	@echo "local" > .current_mode
	@echo ""
	@printf "$(GREEN)[OK]  Local sandbox started (clean rebuild)$(NC)\n"
	@echo "   API        -> $(_NGINX_URL)/docs"
	@echo "   MLflow     -> $(_BASE_URL):5001"
	@echo "   Grafana    -> $(_BASE_URL):3000"
	@echo "   Prometheus -> $(_BASE_URL):9090"
	@echo "   (Airflow is disabled in local mode)"
	@echo ""

cloud-rebuild: check-secrets
	@printf "$(BLUE)>>  Full rebuild in CLOUD mode (no Docker cache)$(NC)\n"
	@$(MAKE) --no-print-directory down
	@printf "$(YELLOW)   Clearing Grafana data volume so updated dashboards are provisioned fresh...$(NC)\n"
	@docker volume rm mlops-device-health_grafana_data 2>/dev/null || true
	@$(COMPOSE) $(CLOUD_ENV) $(CLOUD_FILES) build --no-cache
	@$(COMPOSE) $(CLOUD_ENV) $(CLOUD_FILES) $(_LINUX_PROFILES) up -d --force-recreate
	@docker exec mlops_postgres bash /scripts/pg_ensure_passwords.sh 2>/dev/null || true
	@echo "cloud" > .current_mode
	@echo ""
	@printf "$(BLUE)[OK]  Cloud mode started (clean rebuild)$(NC)\n"
	@echo "   API        -> $(_NGINX_URL)/docs"
	@echo "   MLflow     -> DagsHub (check your repo)"
	@echo "   Airflow    -> $(_BASE_URL):8081"
	@echo "   Grafana    -> $(_BASE_URL):3000"
	@echo "   Prometheus -> $(_BASE_URL):9090"
	@echo ""
	@$(MAKE) --no-print-directory _set_dvc_remote_cloud

# ── GHCR MODE (pull pre-built images from GitHub Container Registry) ──────────
# ghcr-login reads GITHUB_TOKEN from .env.secrets and authenticates with ghcr.io.
# A GitHub PAT with "read:packages" scope is required to pull private packages.
# If the packages have been made public in GitHub, login is not required.
.PHONY: ghcr-login
ghcr-login: check-secrets
	@GHCR_OWNER_VAL=$$(grep '^GITHUB_OWNER=' .env.secrets 2>/dev/null | cut -d= -f2- | tr -d '"' | tr -d "'" | tr -d ' '); \
	GHCR_USER=$${GHCR_OWNER_VAL:-$(GITHUB_OWNER)}; \
	GHCR_TOKEN=$$(grep '^GITHUB_TOKEN=' .env.secrets 2>/dev/null | cut -d= -f2- | tr -d '"' | tr -d "'" | tr -d ' '); \
	if [ -n "$$GHCR_TOKEN" ]; then \
		echo "$$GHCR_TOKEN" | docker login ghcr.io -u $$GHCR_USER --password-stdin && \
		printf "$(GREEN)   Logged in to ghcr.io as $$GHCR_USER$(NC)\n"; \
	else \
		printf "$(YELLOW)   GITHUB_TOKEN not set in .env.secrets -- assuming already logged in$(NC)\n"; \
		printf "$(YELLOW)   To log in manually: docker login ghcr.io -u $$GHCR_USER$(NC)\n"; \
	fi

ghcr: check-secrets ghcr-login
	@printf "$(BLUE)>>  Starting in GHCR mode (pulling images from ghcr.io)$(NC)\n"
	@echo "   Owner:  $(GITHUB_OWNER)   Tag: $(GHCR_TAG)"
	@echo "   Images: mlops-device-health-api, -airflow, -streamlit"
	@$(COMPOSE) $(CLOUD_ENV) $(GHCR_FILES) pull
	@$(COMPOSE) $(CLOUD_ENV) $(GHCR_FILES) $(_LINUX_PROFILES) up -d --force-recreate
	@docker exec mlops_postgres bash /scripts/pg_ensure_passwords.sh 2>/dev/null || true
	@echo "cloud" > .current_mode
	@echo ""
	@printf "$(BLUE)[OK]  GHCR mode started$(NC)\n"
	@echo "   API        -> $(_NGINX_URL)/docs"
	@echo "   Streamlit  -> $(_BASE_URL):8501"
	@echo "   Airflow    -> $(_BASE_URL):8081"
	@echo "   Grafana    -> $(_BASE_URL):3000"
	@echo "   Prometheus -> $(_BASE_URL):9090"
	@echo ""
	@$(MAKE) --no-print-directory _set_dvc_remote_cloud

ghcr-rebuild: check-secrets ghcr-login
	@printf "$(BLUE)>>  GHCR mode -- force pull latest images and restart$(NC)\n"
	@echo "   Owner:  $(GITHUB_OWNER)   Tag: $(GHCR_TAG)"
	@$(MAKE) --no-print-directory down
	@printf "$(YELLOW)   Clearing Grafana data volume so updated dashboards are provisioned fresh...$(NC)\n"
	@docker volume rm mlops-device-health_grafana_data 2>/dev/null || true
	@$(COMPOSE) $(CLOUD_ENV) $(GHCR_FILES) pull
	@$(COMPOSE) $(CLOUD_ENV) $(GHCR_FILES) $(_LINUX_PROFILES) up -d --force-recreate
	@docker exec mlops_postgres bash /scripts/pg_ensure_passwords.sh 2>/dev/null || true
	@echo "cloud" > .current_mode
	@echo ""
	@printf "$(BLUE)[OK]  GHCR mode started (fresh pull)$(NC)\n"
	@echo "   API        -> $(_NGINX_URL)/docs"
	@echo "   Streamlit  -> $(_BASE_URL):8501"
	@echo "   Airflow    -> $(_BASE_URL):8081"
	@echo ""
	@$(MAKE) --no-print-directory _set_dvc_remote_cloud

# ── CLOUD + STREAMLIT CONTAINER (optional all-in-Docker mode) ────────────────
# Starts the full cloud stack AND the Streamlit dashboard as a Docker container
# (built locally from docker/streamlit.Dockerfile).  Useful when you want every
# service — including the UI — running inside Docker, without requiring a host
# Python environment for  make ui .
#
# Access the UI:  http://localhost:8501  in any host browser (Chrome, Edge, …)
#
# The Streamlit container runs on the mlops_network alongside the other
# services, so it can reach the API, MLflow buffer, Airflow, etc. via their
# Docker service names.  It cannot execute  make  commands or control Docker
# itself, so the Docker Control page shows a read-only notice.
cloud-with-ui: check-secrets
	@printf "$(BLUE)>>  Starting CLOUD mode + Streamlit container$(NC)\n"
	@$(COMPOSE) $(CLOUD_ENV) $(CLOUD_FILES) -f docker-compose.streamlit.yml $(_LINUX_PROFILES) up -d --build
	@docker exec mlops_postgres bash /scripts/pg_ensure_passwords.sh 2>/dev/null || true
	@echo "cloud" > .current_mode
	@echo ""
	@printf "$(BLUE)[OK]  Cloud mode + Streamlit container started$(NC)\n"
	@echo "   API        -> $(_NGINX_URL)/docs"
	@echo "   Streamlit  -> $(_BASE_URL):8501"
	@echo "   Airflow    -> $(_BASE_URL):8081"
	@echo "   Grafana    -> $(_BASE_URL):3000"
	@echo "   Prometheus -> $(_BASE_URL):9090"
	@echo ""
	@$(MAKE) --no-print-directory _set_dvc_remote_cloud

cloud-rebuild-with-ui: check-secrets
	@printf "$(BLUE)>>  Full rebuild in CLOUD mode + Streamlit container (no Docker cache)$(NC)\n"
	@$(MAKE) --no-print-directory down
	@printf "$(YELLOW)   Clearing Grafana data volume so updated dashboards are provisioned fresh...$(NC)\n"
	@docker volume rm mlops-device-health_grafana_data 2>/dev/null || true
	@$(COMPOSE) $(CLOUD_ENV) $(CLOUD_FILES) -f docker-compose.streamlit.yml build --no-cache
	@$(COMPOSE) $(CLOUD_ENV) $(CLOUD_FILES) -f docker-compose.streamlit.yml $(_LINUX_PROFILES) up -d --force-recreate
	@docker exec mlops_postgres bash /scripts/pg_ensure_passwords.sh 2>/dev/null || true
	@echo "cloud" > .current_mode
	@echo ""
	@printf "$(BLUE)[OK]  Cloud mode + Streamlit container started (clean rebuild)$(NC)\n"
	@echo "   API        -> $(_NGINX_URL)/docs"
	@echo "   Streamlit  -> $(_BASE_URL):8501"
	@echo "   Airflow    -> $(_BASE_URL):8081"
	@echo "   Grafana    -> $(_BASE_URL):3000"
	@echo "   Prometheus -> $(_BASE_URL):9090"
	@echo ""
	@$(MAKE) --no-print-directory _set_dvc_remote_cloud

# ── DVC remote switching (internal targets, cloud mode only) ──────────────────
_set_dvc_remote_cloud:
	@if command -v dvc > /dev/null 2>&1; then \
		if dvc remote list 2>/dev/null | grep -q "^dagshub"; then \
			dvc remote default dagshub 2>/dev/null && \
			echo "   DVC remote -> dagshub"; \
		else \
			printf "$(YELLOW)   DVC remote 'dagshub' not configured - skipping$(NC)\n"; \
			printf "$(YELLOW)   To configure: dvc remote add dagshub <url>$(NC)\n"; \
		fi \
	else \
		printf "$(YELLOW)   dvc not installed - skipping DVC remote switch$(NC)\n"; \
	fi

# ── DVC STATUS ────────────────────────────────────────────────────────────────
# Check local cache vs DagsHub remote (cloud mode only)
dvc-status:
	@printf "$(BLUE)>>  DVC status (local ↔ remote)$(NC)\n"
	@if command -v dvc > /dev/null 2>&1; then \
		printf "$(BLUE)-- Local cache status --$(NC)\n"; \
		dvc status 2>/dev/null || true; \
		printf "\n$(BLUE)-- Remote (DagsHub) status --$(NC)\n"; \
		dvc status --cloud 2>/dev/null || printf "$(YELLOW)  (no remote configured or not reachable)$(NC)\n"; \
	else \
		printf "$(YELLOW)  dvc not installed$(NC)\n"; \
	fi

# ── DVC LOCK CLEANUP ──────────────────────────────────────────────────────────
# Remove stale DVC lock file (prevents "lock file exists" errors after crashes)
dvc-clean:
	@printf "$(YELLOW)>>  Removing stale DVC lock file...$(NC)\n"
	@rm -f .dvc/tmp/lock
	@printf "$(GREEN)[OK]  DVC lock cleared$(NC)\n"

# ── DATA PUSH (DVC pointer files → GitHub) ────────────────────────────────────
# Stage only DVC pointer/lock files (not raw data — those go to DagsHub via DVC push).
# Commits with [skip ci] so lint/test/build pipelines are NOT triggered.
# Use this from the host machine after the sync_production_data Airflow DAG runs.
data-push:
	@printf "$(YELLOW)>>  Staging DVC pointer files...$(NC)\n"
	@git add --ignore-missing \
		"data/**/*.dvc" \
		"dvc.lock" \
		"data/.gitignore" \
		".gitignore" 2>/dev/null || true
	@if git diff --cached --quiet; then \
		printf "$(YELLOW)  Nothing to push — DVC pointers are up to date$(NC)\n"; \
	else \
		git diff --cached --name-only | while read f; do printf "  + $$f\n"; done; \
		git commit -m "[skip ci] chore(data): update DVC snapshots" && \
		git push && \
		printf "$(GREEN)[OK]  DVC pointer files pushed to GitHub$(NC)\n"; \
	fi

# ── NUCLEAR RESET ─────────────────────────────────────────────────────────────
# Completely removes all project containers, networks, AND volumes so the next
# 'make cloud' / 'make local' initialises every database and service from zero.
#
# Use cases:
#   - Fresh checkout on a new machine (Mac, CI, colleague's laptop)
#   - Persistent DB password mismatch after env changes
#   - Corrupted Airflow/MLflow metadata that a restart cannot fix
#   - Switching between branches with incompatible schema changes
#
# 'make nuke'         — tears everything down, then prints next steps
# 'make nuke-cloud'   — nuke + full cloud rebuild (no Docker cache)
# 'make nuke-local'   — nuke + full local rebuild (no Docker cache)
#
# Note: MLflow data stored only in the local buffer volume is lost.
#       Run 'make safe-down' first if you want to sync it to DagsHub.
nuke: check-secrets
	@printf "$(RED)⚠️   NUCLEAR RESET — removing ALL project containers, networks and volumes$(NC)\n"
	@printf "$(RED)     All database data, MLflow buffer, Grafana state will be erased.$(NC)\n"
	@printf "$(YELLOW)     Press Ctrl-C within 5 seconds to abort...$(NC)\n"
	@sleep 5
	@printf "$(RED)>>  Stopping all services (both modes)...$(NC)\n"
	@$(COMPOSE) $(LOCAL_ENV) $(LOCAL_FILES) --profile linux down --volumes --remove-orphans 2>/dev/null || true
	@$(COMPOSE) $(CLOUD_ENV) $(CLOUD_FILES) --profile linux down --volumes --remove-orphans 2>/dev/null || true
	@printf "$(RED)>>  Removing any leftover named volumes...$(NC)\n"
	@for vol in \
		mlops-device-health_postgres_data \
		mlops-device-health_prometheus_data \
		mlops-device-health_alertmanager_data \
		mlops-device-health_grafana_data \
		mlops-device-health_mlflow_buffer_pg_data \
		mlops-device-health_mlflow_artifacts \
		mlops-device-health_mlflow_db; do \
		docker volume rm $$vol 2>/dev/null && printf "  removed $$vol\n" || true; \
	done
	@printf "$(RED)>>  Removing orphan project networks...$(NC)\n"
	@docker network prune -f --filter "label=com.docker.compose.project=mlops-device-health" 2>/dev/null || true
	@rm -f .current_mode
	@rm -f .dvc/tmp/lock
	@printf "$(GREEN)[OK]  Nuclear reset complete — all volumes and networks removed.$(NC)\n"
	@echo ""
	@echo "  Next steps (pick one):"
	@printf "    $(BLUE)make cloud$(NC)         — start fresh in cloud mode\n"
	@printf "    $(GREEN)make local$(NC)         — start fresh in local mode\n"
	@printf "    $(BLUE)make nuke-cloud$(NC)     — rebuild images + start cloud (no Docker cache)\n"
	@printf "    $(GREEN)make nuke-local$(NC)    — rebuild images + start local (no Docker cache)\n"
	@echo ""

nuke-cloud: check-secrets
	@printf "$(BLUE)>>  Nuclear reset + full cloud rebuild (no Docker cache)$(NC)\n"
	@$(MAKE) --no-print-directory nuke
	@$(COMPOSE) $(CLOUD_ENV) $(CLOUD_FILES) build --no-cache
	@$(COMPOSE) $(CLOUD_ENV) $(CLOUD_FILES) $(_LINUX_PROFILES) up -d --force-recreate
	@docker exec mlops_postgres bash /scripts/pg_ensure_passwords.sh 2>/dev/null || true
	@echo "cloud" > .current_mode
	@echo ""
	@printf "$(BLUE)[OK]  Cloud mode started (full fresh rebuild)$(NC)\n"
	@echo "   API        -> $(_NGINX_URL)/docs"
	@echo "   MLflow     -> DagsHub (check your repo)"
	@echo "   Airflow    -> $(_BASE_URL):8081"
	@echo "   Grafana    -> $(_BASE_URL):3000"
	@echo "   Prometheus -> $(_BASE_URL):9090"
	@echo ""
	@$(MAKE) --no-print-directory _set_dvc_remote_cloud

nuke-local: check-secrets
	@printf "$(GREEN)>>  Nuclear reset + full local rebuild (no Docker cache)$(NC)\n"
	@$(MAKE) --no-print-directory nuke
	@$(COMPOSE) $(LOCAL_ENV) $(LOCAL_FILES) $(_LINUX_PROFILES) build --no-cache
	@$(COMPOSE) $(LOCAL_ENV) $(LOCAL_FILES) $(_LINUX_PROFILES) up -d --force-recreate
	@docker exec mlops_postgres bash /scripts/pg_ensure_passwords.sh 2>/dev/null || true
	@echo "local" > .current_mode
	@echo ""
	@printf "$(GREEN)[OK]  Local mode started (full fresh rebuild)$(NC)\n"
	@echo "   API        -> $(_NGINX_URL)/docs"
	@echo "   MLflow     -> $(_BASE_URL):5001"
	@echo "   Grafana    -> $(_BASE_URL):3000"
	@echo "   Prometheus -> $(_BASE_URL):9090"
	@echo "   (Airflow disabled in local mode)"
	@echo ""

# ── STOP ──────────────────────────────────────────────────────────────────────
down:
	@printf "$(YELLOW)>>  Stopping all services...$(NC)\n"
	@# Also stop the mlops_streamlit container if it is running (it is not managed by any
	@# compose file but may have been started by cloud-with-ui / ghcr-with-ui).
	@docker stop mlops_streamlit 2>/dev/null && docker rm mlops_streamlit 2>/dev/null || true
	@# Pass --profile linux so profile-gated services (node_exporter) are included in scope.
	@$(COMPOSE) $(LOCAL_ENV) $(LOCAL_FILES) --profile linux down 2>/dev/null || true
	@$(COMPOSE) $(CLOUD_ENV) $(CLOUD_FILES) --profile linux down 2>/dev/null || true
	@rm -f .current_mode
	@rm -f .dvc/tmp/lock
	@printf "$(YELLOW)[OK]  All services stopped$(NC)\n"

# ── RESET DATABASE (removes postgres volume — data will be lost) ───────────────
reset-db: check-secrets
	@printf "$(RED)>>  Resetting postgres volume (all data will be lost)...$(NC)\n"
	@$(MAKE) --no-print-directory down
	@$(COMPOSE) $(LOCAL_ENV) $(LOCAL_FILES) rm -f -v postgres 2>/dev/null || true
	@docker volume rm mlops-device-health_postgres_data 2>/dev/null || true
	@printf "$(GREEN)[OK]  Postgres volume removed$(NC)\n"
	@echo "   Run 'make local' or 'make cloud' to restart with a fresh database."

# ── RESET LOCAL DATABASE (drops + recreates mlops_local only — mlops_prod intact) ─
reset-local-db:
	@printf "$(YELLOW)>>  Resetting local sandbox database (mlops_local) ...$(NC)\n"
	@if ! docker ps --format '{{.Names}}' | grep -q '^mlops_postgres$$'; then \
		printf "$(RED)ERROR: mlops_postgres is not running.$(NC)\n"; \
		echo "       Start the stack first: make local"; \
		exit 1; \
	fi
	@docker exec mlops_postgres psql -U $${DB_USER:-mlops_user} -d postgres \
		-c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = 'mlops_local' AND pid <> pg_backend_pid();" \
		>/dev/null 2>&1 || true
	@docker exec mlops_postgres psql -U $${DB_USER:-mlops_user} -d postgres \
		-c "DROP DATABASE IF EXISTS mlops_local"
	@docker exec mlops_postgres psql -U $${DB_USER:-mlops_user} -d postgres \
		-c "CREATE DATABASE mlops_local OWNER \"$${DB_USER:-mlops_user}\""
	@docker exec mlops_postgres psql -U $${DB_USER:-mlops_user} -d postgres \
		-c "GRANT ALL PRIVILEGES ON DATABASE mlops_local TO \"$${DB_USER:-mlops_user}\""
	@printf "$(GREEN)[OK]  mlops_local reset — all local sandbox data cleared$(NC)\n"
	@echo "   Cloud data in mlops_prod is untouched."
	@echo "   Restart API to reconnect: docker compose restart api"

# ── FIX DB PASSWORD (sync password without wiping data) ───────────────────────
# Reads POSTGRES_PASSWORD from the running postgres container's own environment
# (set by .env.secrets when the stack started) and runs ALTER USER to re-sync.
# Requires: the postgres container must already be running.
fix-db-password:
	@printf "$(YELLOW)>>  Syncing postgres password to match .env.secrets ...$(NC)\n"
	@if ! docker ps --format '{{.Names}}' | grep -q '^mlops_postgres$$'; then \
		printf "$(RED)ERROR: mlops_postgres container is not running.$(NC)\n"; \
		echo "       Start it first: make local  (or: docker compose ... up -d postgres)"; \
		exit 1; \
	fi
	@docker exec mlops_postgres bash -c \
		"psql -U \$${POSTGRES_USER:-mlops_user} -d postgres -c \"ALTER USER \$${POSTGRES_USER:-mlops_user} PASSWORD '\$${POSTGRES_PASSWORD:-changeme}';\""
	@printf "$(GREEN)[OK]  Password synced - restart the API to pick up the change:$(NC)\n"
	@echo "   docker compose ... restart api"

# ── STATUS ────────────────────────────────────────────────────────────────────
status:
	@echo ""
	@printf "$(BLUE)-- Deployment Mode -------------------------------------------$(NC)\n"
	@if [ -f .current_mode ]; then \
		MODE=$$(cat .current_mode); \
		if [ "$$MODE" = "local" ]; then \
			printf "  Mode:    $(GREEN)LOCAL$(NC) (MLflow: local Docker, DVC: filesystem)\n"; \
		elif [ "$$MODE" = "cloud" ]; then \
			printf "  Mode:    $(BLUE)CLOUD$(NC) (MLflow: local buffer -> DagsHub, DVC: DagsHub)\n"; \
		else \
			printf "  Mode:    $(YELLOW)$$MODE$(NC)\n"; \
		fi; \
	elif docker ps --format '{{.Names}}' 2>/dev/null | grep -q 'mlops_api'; then \
		if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'mlops_mlflow'; then \
			printf "  Mode:    $(GREEN)LOCAL$(NC) (detected from running containers)\n"; \
			echo "local" > .current_mode; \
		else \
			printf "  Mode:    $(BLUE)CLOUD$(NC) (detected from running containers)\n"; \
			echo "cloud" > .current_mode; \
		fi; \
	else \
		printf "  Mode:    $(YELLOW)unknown$(NC) (no .current_mode - run 'make local' or 'make cloud')\n"; \
	fi
	@echo ""
	@printf "$(BLUE)-- Running Containers ----------------------------------------$(NC)\n"
	@docker ps --format "{{.Names}}\t{{.Status}}" --filter "name=mlops_" 2>/dev/null | awk 'BEGIN{FS="\t"}{printf "  %-42s %s\n", $$1, $$2}' || echo "  Docker not accessible"
	@echo ""
	@printf "$(BLUE)-- Access URLs -----------------------------------------------$(NC)\n"
	@echo "   API / Swagger  -> $(_NGINX_URL)/docs"
	@echo "   Airflow        -> $(_BASE_URL):8081"
	@echo "   Grafana        -> $(_BASE_URL):3000"
	@echo "   Prometheus     -> $(_BASE_URL):9090"
	@if docker ps --format '{{.Names}}' 2>/dev/null | grep -q 'mlops_mlflow$$'; then \
		echo "   MLflow (local) -> $(_BASE_URL):5001"; \
	fi
	@if docker ps --format '{{.Names}}' 2>/dev/null | grep -q 'mlops_mlflow_buffer'; then \
		echo "   MLflow buffer  -> $(_BASE_URL):5002"; \
	fi
	@echo ""
	@printf "$(BLUE)-- DVC Remote ------------------------------------------------$(NC)\n"
	@if command -v dvc > /dev/null 2>&1; then \
		dvc remote list 2>/dev/null || echo "  No DVC remotes configured"; \
		echo "  Default: $$(dvc remote default 2>/dev/null || echo 'none')"; \
	else \
		echo "  dvc not installed"; \
	fi
	@echo ""

# ── LOGS ──────────────────────────────────────────────────────────────────────
logs:
	@if [ -f .current_mode ] && [ "$$(cat .current_mode)" = "cloud" ]; then \
		$(COMPOSE) $(CLOUD_ENV) $(CLOUD_FILES) $(_LINUX_PROFILES) logs -f --tail=50; \
	else \
		$(COMPOSE) $(LOCAL_ENV) $(LOCAL_FILES) logs -f --tail=50; \
	fi

# ── RESTART ───────────────────────────────────────────────────────────────────
restart:
	@if [ -f .current_mode ]; then \
		MODE=$$(cat .current_mode); \
		printf "$(YELLOW)>>  Restarting in $$MODE mode...$(NC)\n"; \
		$(MAKE) --no-print-directory down; \
		$(MAKE) --no-print-directory $$MODE; \
	else \
		printf "$(RED)ERROR: No current mode set. Run 'make local' or 'make cloud' first.$(NC)\n"; \
		exit 1; \
	fi

# ── UI ────────────────────────────────────────────────────────────────────────
ui:
	@printf "$(BLUE)>>  Starting Streamlit dashboard...$(NC)\n"
	@_MODE=$$(cat .current_mode 2>/dev/null || echo "unknown"); \
	  if [ "$$_MODE" = "cloud" ]; then \
	    _ENVFILE=.env.cloud; \
	  elif [ "$$_MODE" = "k8s" ]; then \
	    _ENVFILE=.env.k8s; \
	  else \
	    _ENVFILE=.env.local; \
	  fi; \
	  echo "   Mode: $$_MODE  (env: $$_ENVFILE)"; \
	  set -a; \
	  . ./$$_ENVFILE; \
	  [ -f .env.secrets ] && . ./.env.secrets; \
	  [ -f .env.windows.local ] && . ./.env.windows.local; \
	  [ "$$_MODE" = "k8s" ] && . ./.env.k8s; \
	  set +a; \
	  _PG_HOST=$${POSTGRES_HOST:-localhost}; \
	  _PG_PORT=$${DB_PORT:-5432}; \
	  _PG_USER=$${DB_USER:-mlops_user}; \
	  _PG_PASS=$${DB_PASSWORD:-changeme}; \
	  _PG_DB=$${DB_NAME:-mlops_db}; \
	  PYTHONPATH=$(PWD) \
	  DEPLOYMENT_MODE=$$_MODE \
	  POSTGRES_HOST=$$_PG_HOST \
	  POSTGRES_PORT=$$_PG_PORT \
	  POSTGRES_USER=$$_PG_USER \
	  POSTGRES_PASSWORD=$$_PG_PASS \
	  POSTGRES_DB=$$_PG_DB \
	  DATABASE_URL="postgresql://$$_PG_USER:$$_PG_PASS@$$_PG_HOST:$$_PG_PORT/$$_PG_DB" \
	  UI_LOG_FILE=$(PWD)/logs/ui_app.log \
	  PYTHONUTF8=1 \
	  PYTHONIOENCODING=utf-8 \
	  $(_STREAMLIT) run src/ui/app.py --server.port 8501 --server.headless true --server.address 127.0.0.1; \
	  _EXIT=$$?; \
	  if [ $$_EXIT -eq 130 ] || [ $$_EXIT -eq 2 ]; then \
	    exit 0; \
	  else \
	    exit $$_EXIT; \
	  fi

# ── MLFLOW BUFFER MANAGEMENT ──────────────────────────────────────────────────
# Local-first MLflow architecture: buffer is the live system; DagsHub is backup.
# See Section 27 of Data_Versioning_And_Storage_Analysis.md for full details.

# Restore the local MLflow buffer from DagsHub (download DagsHub → buffer).
# Use when: buffer is empty after docker volume rm, fresh clone, or disaster recovery.
# WARNING: Runs on the HOST (not inside a container). Requires make cloud to be running.
mlflow-restore:
	@printf "$(BLUE)>>  Restoring MLflow buffer from DagsHub...$(NC)\n"
	@set -a; \
	  . ./.env.cloud; \
	  [ -f .env.secrets ] && . ./.env.secrets; \
	  set +a; \
	  PYTHONPATH=$(PWD) python -c "\
from src.training.mlflow_sync import pull_from_dagshub, build_dagshub_uri, get_dagshub_credentials; \
import os; \
user, token, repo = get_dagshub_credentials(); \
dh_uri = build_dagshub_uri(user, repo); \
buf_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5002'); \
print(f'Restoring: {dh_uri} -> {buf_uri}'); \
result = pull_from_dagshub(local_mlflow_uri=buf_uri, dagshub_uri=dh_uri, dagshub_user=user, dagshub_token=token, sync_artifacts=True); \
print(f'Done: {result}'); \
"
	@printf "$(GREEN)[OK]  MLflow buffer restore complete.$(NC)\n"

# Push local MLflow buffer runs to DagsHub (upload buffer → DagsHub, incremental).
# Use when: you want to sync before shutting down without triggering the Airflow DAG.
# WARNING: DagsHub rate-limits apply; if you get 429, wait and retry.
mlflow-sync:
	@printf "$(BLUE)>>  Syncing MLflow buffer -> DagsHub...$(NC)\n"
	@set -a; \
	  . ./.env.cloud; \
	  [ -f .env.secrets ] && . ./.env.secrets; \
	  set +a; \
	  PYTHONPATH=$(PWD) python -c "\
from src.training.mlflow_sync import push_to_dagshub, build_dagshub_uri, get_dagshub_credentials; \
import os; \
user, token, repo = get_dagshub_credentials(); \
dh_uri = build_dagshub_uri(user, repo); \
buf_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5002'); \
print(f'Pushing: {buf_uri} -> {dh_uri}'); \
result = push_to_dagshub(local_mlflow_uri=buf_uri, dagshub_uri=dh_uri, dagshub_user=user, dagshub_token=token, sync_artifacts=True); \
print(f'Done: {result}'); \
"
	@printf "$(GREEN)[OK]  MLflow sync complete.$(NC)\n"

# Safe shutdown: sync buffer to DagsHub BEFORE stopping containers.
# Use instead of 'make down' when you have unsynced MLflow data.
# DANGER: 'make down' without syncing first may lose unsynced MLflow runs.
safe-down: mlflow-sync down
	@printf "$(GREEN)[OK]  Safe shutdown complete (buffer synced + containers stopped).$(NC)\n"

# ── WIPE ──────────────────────────────────────────────────────────────────────
# Remove all test / demo data.  Default: dry-run.  Set EXEC=1 to apply.
wipe:
	@if [ "$(EXEC)" = "1" ]; then \
		python scripts/wipe_test_data.py --execute; \
	else \
		python scripts/wipe_test_data.py; \
	fi

# ── TEST ──────────────────────────────────────────────────────────────────────
# Fast suite — unit, integration, performance (no Docker required, ~30s)
test:
	@printf "$(BLUE)>>  Running fast test suite (no Docker, no UI)...$(NC)\n"
	pytest tests/ -q --tb=short -m "not live and not ui"
	@printf "$(GREEN)[OK]  Fast test suite complete$(NC)\n"

# UI suite — Streamlit dashboard tests (no Docker required)
test-ui:
	@printf "$(BLUE)>>  Running UI test suite...$(NC)\n"
	pytest tests/ -q --tb=short -m ui
	@printf "$(GREEN)[OK]  UI test suite complete$(NC)\n"

# Live suite — requires local Docker stack (auto-started by conftest if not running)
# Set DOCKER_KEEP_UP=1 to keep containers up after the run.
test-live:
	@printf "$(BLUE)>>  Running live Docker stack tests...$(NC)\n"
	pytest tests/ -q --tb=short -m live
	@printf "$(GREEN)[OK]  Live test suite complete$(NC)\n"

# Full suite — fast + live (Docker must be running or will be auto-started)
test-all:
	@printf "$(BLUE)>>  Running full test suite (fast + live)...$(NC)\n"
	pytest tests/ -q --tb=short
	@printf "$(GREEN)[OK]  Full test suite complete$(NC)\n"

# ── KUBERNETES ────────────────────────────────────────────────────────────────
# All K8s targets use the local overlay by default (Docker Desktop K8s).
# Override with: make k8s-up K8S_OVERLAY=cloud
K8S_OVERLAY ?= local
K8S_DIR     := k8s/overlays/$(K8S_OVERLAY)
REPLICAS    ?= 3

## Build container images for Kubernetes (tags: mlops-device-health/*)
k8s-build:
	@printf "$(BLUE)>>  Building K8s images (overlay=$(K8S_OVERLAY))...$(NC)\n"
	$(eval _GIT_SHA := $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown))
	@printf "$(BLUE)    GIT_SHA=$(_GIT_SHA)$(NC)\n"
	docker build --build-arg GIT_SHA=$(_GIT_SHA) \
		-t mlops-device-health/api:latest \
		-t mlops-device-health/api:$(_GIT_SHA) \
		-f docker/api.Dockerfile .
	docker build \
		-t mlops-device-health/streamlit:latest \
		-t mlops-device-health/streamlit:$(_GIT_SHA) \
		-f docker/streamlit.Dockerfile .
	docker build --build-arg GIT_SHA=$(_GIT_SHA) \
		-t mlops-device-health/airflow:latest \
		-t mlops-device-health/airflow:$(_GIT_SHA) \
		-f docker/airflow_mlops.Dockerfile .
	@printf "$(GREEN)[OK]  K8s images built (GIT_SHA=$(_GIT_SHA))$(NC)\n"
	@# If the cluster is running, force pods to pick up the new image by updating
	@# the deployment image spec to the SHA-specific tag. This changes the spec →
	@# K8s triggers a rolling update → new pods use the freshly built image.
	@# (imagePullPolicy:Never ensures the local SHA-tagged image is used, not pulled)
	@printf "$(BLUE)>>  Forcing K8s deployments to use new image (SHA tag)...$(NC)\n"
	@kubectl set image deployment/api \
		api=mlops-device-health/api:$(_GIT_SHA) -n mlops 2>/dev/null || true
	@kubectl set image deployment/streamlit \
		streamlit=mlops-device-health/streamlit:$(_GIT_SHA) -n mlops 2>/dev/null || true
	@kubectl set image deployment/airflow \
		airflow=mlops-device-health/airflow:$(_GIT_SHA) -n mlops 2>/dev/null || true
	@printf "$(GREEN)[OK]  Image spec updated (cluster will roll out if running)$(NC)\n"

## Apply all manifests and start the full stack on Kubernetes
k8s-up:
	@printf "$(BLUE)>>  Deploying to Kubernetes (overlay=$(K8S_OVERLAY))...$(NC)\n"
	@if [ ! -f k8s/base/secret.yaml ]; then \
		printf "$(RED)  ERROR: k8s/base/secret.yaml not found — run 'make k8s-secret' first!$(NC)\n"; \
		exit 1; \
	fi
	kubectl apply --server-side --force-conflicts -k $(K8S_DIR)
	@printf "$(BLUE)>>  Applying mlops-secrets (k8s/base/secret.yaml)...$(NC)\n"
	kubectl apply -f k8s/base/secret.yaml
	@printf "$(BLUE)>>  Forcing pod restart so new images are picked up...$(NC)\n"
	@# imagePullPolicy:IfNotPresent uses the current local Docker image for the tag.
	@# kubectl apply only restarts pods when the spec CHANGES; rollout restart
	@# guarantees pods are cycled and pick up any freshly-built 'latest' image.
	@kubectl rollout restart deployment/api       -n mlops 2>/dev/null || true
	@kubectl rollout restart deployment/streamlit -n mlops 2>/dev/null || true
	@kubectl rollout restart deployment/airflow   -n mlops 2>/dev/null || true
	kubectl rollout status deployment/api        -n mlops --timeout=240s || true
	kubectl rollout status deployment/streamlit  -n mlops --timeout=120s || true
	kubectl rollout status deployment/nginx      -n mlops --timeout=60s  || true
	@# Airflow takes 2-3 min to start (DB init + scheduler).  Wait up to 360s
	@# so port-forwards connect to a ready pod instead of an empty service.
	kubectl rollout status deployment/airflow    -n mlops --timeout=360s || true
	@printf 'k8s' > .current_mode
	@printf "$(GREEN)[OK]  K8s stack is up. Run 'make k8s-ports' then open http://localhost:8888$(NC)\n"

## Tear down the K8s stack (keeps PVCs / data)
k8s-down:
	@printf "$(YELLOW)>>  Tearing down K8s stack (PVCs preserved)...$(NC)\n"
	kubectl delete -k $(K8S_DIR) --ignore-not-found
	@rm -f .current_mode
	@printf "$(GREEN)[OK]  K8s stack removed$(NC)\n"

## Tear down the K8s stack AND delete all PersistentVolumeClaims (DATA LOSS)
k8s-nuke:
	@printf "$(RED)>>  Nuking K8s stack including all PVCs (DATA LOSS)...$(NC)\n"
	kubectl delete -k $(K8S_DIR) --ignore-not-found
	kubectl delete pvc --all -n mlops --ignore-not-found
	@rm -f .current_mode
	@printf "$(GREEN)[OK]  K8s stack and all PVCs deleted$(NC)\n"

## Show pod / deployment / service status
k8s-status:
	@printf "$(BLUE)>>  K8s status (namespace=mlops)$(NC)\n"
	@echo "--- Pods ---"
	kubectl get pods -n mlops
	@echo "--- Deployments ---"
	kubectl get deployments -n mlops
	@echo "--- Services ---"
	kubectl get services -n mlops
	@echo "--- HPA ---"
	kubectl get hpa -n mlops 2>/dev/null || true

## Tail logs from the API pods
k8s-logs:
	kubectl logs -n mlops -l app=api --tail=100 --follow

## Scale the API deployment (default: REPLICAS=3)
k8s-scale:
	@printf "$(BLUE)>>  Scaling api deployment to $(REPLICAS) replicas...$(NC)\n"
	kubectl scale deployment/api -n mlops --replicas=$(REPLICAS)
	@printf "$(GREEN)[OK]  Scaled api to $(REPLICAS) replicas$(NC)\n"

## Deploy using GHCR images (CI/CD mode — pulls from ghcr.io, no local build needed)
k8s-ghcr-up:
	@printf "$(BLUE)>>  Deploying to Kubernetes using GHCR images (overlay=ghcr)...$(NC)\n"
	@if [ -f k8s/base/secret.yaml ]; then \
		printf "$(BLUE)>>  Applying mlops-secrets (k8s/base/secret.yaml)...$(NC)\n"; \
		kubectl apply -f k8s/base/secret.yaml; \
	else \
		printf "$(RED)  ERROR: k8s/base/secret.yaml not found — run 'make k8s-secret' first!$(NC)\n"; \
		exit 1; \
	fi
	kubectl apply --server-side --force-conflicts -k k8s/overlays/ghcr
	@printf "$(BLUE)>>  Waiting for deployments to become available...$(NC)\n"
	kubectl rollout status deployment/postgres   -n mlops --timeout=120s || true
	kubectl rollout status deployment/mlflow     -n mlops --timeout=120s || true
	kubectl rollout status deployment/api        -n mlops --timeout=180s || true
	kubectl rollout status deployment/streamlit  -n mlops --timeout=120s || true
	kubectl rollout status deployment/nginx      -n mlops --timeout=60s  || true
	@printf 'k8s' > .current_mode
	@printf "$(GREEN)[OK]  K8s stack running with GHCR images. Run 'make k8s-ports' then open http://localhost:8888$(NC)\n"

## Show / switch kubectl context
k8s-context:
	@printf "$(BLUE)>>  Current kubectl context:$(NC)\n"
	kubectl config current-context
	@printf "$(BLUE)>>  Available contexts:$(NC)\n"
	kubectl config get-contexts

## Generate k8s/base/secret.yaml from .env.secrets (run before first k8s-up)
k8s-secret:
	@printf "$(BLUE)>>  Generating k8s/base/secret.yaml from .env.secrets...$(NC)\n"
ifeq ($(OS),Windows_NT)
	powershell -ExecutionPolicy Bypass -File scripts/k8s_setup_secret.ps1
else
	bash scripts/k8s_setup_secret.sh
endif

## Sync airflow/dags/*.py -> k8s/base/airflow/dags-configmap.yaml
k8s-sync-dags:
	@printf "$(BLUE)>>  Syncing Airflow DAGs to Kubernetes ConfigMap...$(NC)\n"
ifeq ($(OS),Windows_NT)
	powershell -ExecutionPolicy Bypass -File scripts/k8s_sync_dags.ps1
else
	@printf "$(YELLOW)  [INFO] DAG sync on Linux/macOS: use kubectl directly:$(NC)\n"
	@printf "  kubectl create configmap airflow-dags --from-file=airflow/dags/ -n mlops --dry-run=client -o yaml > k8s/base/airflow/dags-configmap.yaml\n"
endif

## Start background port-forwards so all services are on localhost
k8s-ports:
	@printf "$(BLUE)>>  Starting port-forward jobs for all K8s services...$(NC)\n"
	bash scripts/k8s_port_forward.sh

## Stop all port-forward background jobs
k8s-ports-stop:
	@printf "$(YELLOW)>>  Stopping port-forward jobs...$(NC)\n"
	bash scripts/k8s_port_forward.sh --stop

## Smoke-test all services through their health endpoints (requires k8s-ports)
k8s-test:
	@printf "$(BLUE)>>  Running K8s smoke tests...$(NC)\n"
ifeq ($(OS),Windows_NT)
	powershell -ExecutionPolicy Bypass -File scripts/k8s_smoke_test.ps1
else
	@printf "$(YELLOW)  [INFO] Smoke test on Linux/macOS: use curl manually$(NC)\n"
	@curl -sf http://localhost:8000/ && printf "$(GREEN)[OK] api$(NC)\n"   || printf "$(RED)[FAIL] api$(NC)\n"
	@curl -sf http://localhost:5000/ && printf "$(GREEN)[OK] mlflow$(NC)\n" || printf "$(RED)[FAIL] mlflow$(NC)\n"
	@curl -sf http://localhost:8080/health && printf "$(GREEN)[OK] airflow$(NC)\n" || printf "$(RED)[FAIL] airflow$(NC)\n"
	@curl -sf http://localhost:3000/api/health && printf "$(GREEN)[OK] grafana$(NC)\n" || printf "$(RED)[FAIL] grafana$(NC)\n"
endif

## Full K8s setup: secret -> sync-dags -> build -> up (idempotent)
k8s-setup: k8s-secret k8s-sync-dags k8s-build k8s-up

## Full K8s workflow: setup -> ports -> smoke test
k8s-full: k8s-setup k8s-ports k8s-test
