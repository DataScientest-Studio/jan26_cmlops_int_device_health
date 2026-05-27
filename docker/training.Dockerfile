# Training Container - Isolated environment for model training
# Base: Python 3.12 slim for minimal footprint
FROM python:3.12-slim

# Metadata
LABEL maintainer="Fred Richter"
LABEL description="MLOps Device Health - Training Container"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv using pip (faster package installer)
RUN pip install --no-cache-dir uv

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies using uv (much faster than pip)
RUN uv pip install --system --no-cache -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY pyproject.toml ./
COPY README.md ./

# Install package in development mode using uv
RUN uv pip install --system --no-cache -e .

# Create directories for data and models
RUN mkdir -p /app/data /app/models /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV DVC_NO_ANALYTICS=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command: run training script
CMD ["python", "-m", "src.training.train"]
