"""
API module for MLOps device health monitoring.

FastAPI application with endpoints for:
- Health predictions
- Sparse label injection
- Performance monitoring
- Model information
"""

from .main import app

__all__ = ["app"]
