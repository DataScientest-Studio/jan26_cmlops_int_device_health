# =============================================================================
# Airflow MLOps Dockerfile — extends apache/airflow:2.8.1-python3.10
# =============================================================================
# Bakes all project-specific Python packages into the image at BUILD time.
# This guarantees they are available in every task subprocess regardless of
# network availability or pip errors at container startup.
#
# Packages added on top of the base Apache Airflow 2.8.1 image:
#   • scipy / scikit-learn  — signal feature extraction (sklearn 1.7.2 on Python 3.10)
#   • pydantic >= 2.0       — SignalData model (uses field_validator from v2)
#   • mlflow                — experiment / model-registry tracking
#   • evidently==0.7.20     — drift detection report (run_drift_detection task)
#   • dvc[s3]               — data versioning + DagsHub S3-compatible storage
#   • dagshub               — DagsHub authentication + MLflow auto-login
# =============================================================================

FROM apache/airflow:2.8.1-python3.10

# Install git (required by DVC for tracking .dvc files and pushing commits)
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Switch to airflow user (required by the base image security policy)
USER airflow

# Install all project-required packages.
# --no-cache-dir keeps the image lean.
# Using apache/airflow:2.8.1-python3.10 (Python 3.10) to satisfy:
#   mlflow==3.9.0    — requires Python>=3.10
#   evidently==0.7.20 — requires Python>=3.9 (matches requirements.txt)
# Note: scikit-learn is left unpinned — Python 3.10 resolves to 1.7.2 (latest for 3.10).
#   Models pickled on the host with sklearn 1.8.0 are patched at load time in
#   batch_rescoring.py (multi_class compatibility shim).
RUN pip install --no-cache-dir \
    "dvc[s3]" \
    dagshub \
    scipy \
    scikit-learn \
    "pydantic>=2.0,<3" \
    "mlflow==3.9.0" \
    "evidently==0.7.20"

# Copy project source package so DAG task callables can import from src.*
# (mirrors docker-compose.yml: ./src:/opt/airflow/src + PYTHONPATH=/opt/airflow)
USER root
COPY src/ /opt/airflow/src/
USER airflow

# K8s: make src importable in every task subprocess
ENV PYTHONPATH=/opt/airflow

# Inject git SHA at build time so DAG tasks can tag MLflow runs with the
# correct commit even without a .git directory inside the container.
ARG GIT_SHA=unknown
ENV GIT_SHA=$GIT_SHA
