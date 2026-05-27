#!/usr/bin/env python
"""
UC-17: Database Schema Migration (SQLite → PostgreSQL) — demonstration.

Shows that the Database adapter works identically in both SQLite (dev/CI)
and PostgreSQL (Docker/production) modes with zero code changes.

Usage:
    # With SQLite (always works, no setup required)
    python scripts/demo_schema_migration.py

    # With PostgreSQL (requires Docker stack running)
    POSTGRES_HOST=localhost python scripts/demo_schema_migration.py
    # OR
    DATABASE_URL=postgresql://mlops_user:changeme@localhost:5432/mlops_db \
        python scripts/demo_schema_migration.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _demo_db(label: str, **kwargs):  # type: ignore[no-untyped-def]
    """Run a small write + read cycle and report results."""
    from src.database.database import Database

    print(f"\n{'=' * 60}")
    print(f"  Backend: {label}")
    print("=" * 60)

    try:
        db = Database(**kwargs)
        backend = getattr(db, "_backend", "unknown")
        print(f"  _backend attribute : {backend}")

        # Register a test device
        device_id = db.register_device(
            device_name="migration-demo-device",
            device_type="Sensor-Demo",
            location="Demo-Lab",
        )
        print(f"  register_device()  : device_id = {device_id}")

        # Retrieve it
        device = db.get_device(device_id)
        assert device is not None, "Device not found after insert!"
        print(
            f"  get_device()       : name = {device.get('device_name')}, "
            f"type = {device.get('device_type')}"
        )

        # Count rows
        from contextlib import suppress

        with suppress(Exception):
            cursor = db.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM devices")
            row = cursor.fetchone()
            count = row[0] if row else "?"
            print(f"  devices row count  : {count}")

        db.close()
        print(f"  ✅ {label} backend works correctly — same API, zero code changes.")
        return True

    except Exception as exc:
        print(f"  ❌ Error with {label}: {exc}")
        return False


def main() -> int:
    results: list[bool] = []

    # ── 1. SQLite (always available) ─────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "demo.db"
        ok = _demo_db("SQLite (dev / CI)", db_path=str(db_path))
        results.append(ok)

    # ── 2. PostgreSQL (if configured) ────────────────────────────────────────
    database_url = os.environ.get("DATABASE_URL", "")
    postgres_host = os.environ.get("POSTGRES_HOST", "")

    if database_url and database_url.startswith("postgresql"):
        ok = _demo_db("PostgreSQL (DATABASE_URL)", db_url=database_url)
        results.append(ok)
    elif postgres_host:
        user = os.environ.get("POSTGRES_USER", "mlops_user")
        password = os.environ.get("POSTGRES_PASSWORD", "changeme")
        port = os.environ.get("POSTGRES_PORT", "5432")
        db = os.environ.get("POSTGRES_DB", "mlops_db")
        pg_url = f"postgresql://{user}:{password}@{postgres_host}:{port}/{db}"
        ok = _demo_db("PostgreSQL (POSTGRES_HOST)", db_url=pg_url)
        results.append(ok)
    else:
        print(
            "\n[INFO] POSTGRES_HOST / DATABASE_URL not set — skipping PostgreSQL demo.\n"
            "       Start the Docker stack and set POSTGRES_HOST=localhost to test\n"
            "       the PostgreSQL backend with the same code path."
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    passed = sum(results)
    total = len(results)
    print(f"  Result: {passed}/{total} backends verified ✅")
    print("=" * 60)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
