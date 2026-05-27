# =============================================================================
# Streamlit Dashboard Dockerfile
# =============================================================================
# Builds the Streamlit UI container for the MLOps Device Health platform.
# Used by docker-compose.local.yml (local mode only).
# =============================================================================

FROM python:3.12-slim

LABEL maintainer="Fred Richter"
LABEL description="MLOps Device Health - Streamlit Dashboard"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN pip install --no-cache-dir uv

# Copy dependency files first (better layer caching)
COPY requirements.txt .
COPY pyproject.toml ./
# Install all Python dependencies (requirements.txt = core packages compiled from pyproject.toml,
# plus dev group which contains streamlit — required for this container to function).
RUN uv pip install --system --no-cache -r requirements.txt
RUN uv pip install --system --no-cache "streamlit>=1.54.0" plotly

# Copy source code
COPY src/ ./src/
COPY README.md ./

# Install the project in editable mode (no extras — streamlit already installed above)
RUN uv pip install --system --no-cache -e .

# Create required directories
RUN mkdir -p /app/models /app/logs

# Set environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/ui/app.py", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--server.address=0.0.0.0", \
     "--browser.gatherUsageStats=false"]
