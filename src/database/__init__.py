"""
Database package for MLOps device health monitoring.

Provides:
- create_tables: Initialize database schema
- Database: CRUD operations for predictions, signals, features, labels
- generate_device_id: UUID generation for device identifiers
"""

from .database import Database
from .init_db import create_tables, generate_device_id, verify_schema

__all__ = [
    "Database",
    "create_tables",
    "generate_device_id",
    "verify_schema",
]
