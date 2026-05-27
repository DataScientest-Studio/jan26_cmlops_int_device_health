# API Container - FastAPI inference service
# Base: Python 3.12 slim for minimal footprint
FROM python:3.12-slim

# Metadata
LABEL maintainer="Fred Richter"
LABEL description="MLOps Device Health - API Container"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv using pip (faster package installer)
RUN pip install --no-cache-dir uv

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies using uv (much faster than pip)
RUN uv pip install --system --no-cache -r requirements.txt

# Inject git SHA before COPY src/ so that Docker cache is invalidated
# whenever the git SHA changes (= code committed).  This guarantees that
# 'make k8s-build' always bakes the latest source into the image even if
# requirements.txt is unchanged (which would otherwise keep the COPY cached).
ARG GIT_SHA=unknown
ENV GIT_SHA=$GIT_SHA

# Copy source code (layer is invalidated when GIT_SHA changes, ensuring
# fresh code on every k8s-build after a commit)
COPY src/ ./src/
COPY pyproject.toml ./
COPY README.md ./

# Install package in development mode using uv
RUN uv pip install --system --no-cache -e .

# Create models and logs directories.  The API can start in a fully greenfield
# state with no pre-installed model; the /health endpoint will return 503
# (no model) until the first Greenfield Bootstrap or Airflow retraining run
# promotes a model to the MLflow registry.
RUN mkdir -p /app/models /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI with uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
