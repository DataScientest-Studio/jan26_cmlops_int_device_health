# Airflow Container - Workflow orchestration
# Base: Python 3.12 slim for minimal footprint
FROM python:3.12-slim

# Metadata
LABEL maintainer="Fred Richter"
LABEL description="MLOps Device Health - Airflow Container"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv using pip (faster package installer)
RUN pip install --no-cache-dir uv

# Copy requirements first for layer caching
COPY requirements.txt .

# Install dependencies using uv (much faster than pip)
RUN uv pip install --system --no-cache -r requirements.txt

# evidently==0.7.20 in requirements.txt conflicts with scikit-learn==1.8.0
# (evidently 0.7.x declares scikit-learn<1.8.0).  Downgrade to 0.4.x which
# supports sklearn 1.8.0 and whose API is handled by the try/except in
# src/monitoring/drift_detection.py.
RUN uv pip install --system --no-cache "evidently>=0.4.30,<0.5.0"

# Copy source code
COPY src/ ./src/
COPY pyproject.toml ./
COPY README.md ./
COPY airflow/ ./airflow/

# Install package in development mode
RUN uv pip install --system --no-cache -e .

# Create directories
RUN mkdir -p /app/logs /app/airflow/dags /app/airflow/logs /app/airflow/plugins

# Set environment variables
ENV PYTHONPATH=/app
ENV AIRFLOW_HOME=/app/airflow
ENV AIRFLOW__CORE__DAGS_FOLDER=/app/airflow/dags
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV AIRFLOW__CORE__EXECUTOR=LocalExecutor

# Expose Airflow webserver port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Initialize Airflow DB and run webserver
CMD ["bash", "-c", "airflow db init && \
     airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com || true && \
     airflow webserver & airflow scheduler"]
