#!/usr/bin/env python
"""
UC-16: PostgreSQL Backup & Recovery — ad-hoc pg_dump trigger.

Reads POSTGRES_HOST / DATABASE_URL from the environment (same logic as the
Streamlit PostgreSQL page) and calls pg_dump via src.database.backup.

Usage:
    python scripts/run_pg_backup.py [--keep-n 7] [--output-dir data/backups]

Requires PostgreSQL to be reachable.  In dev / CI, uses SQLite and skips the
actual pg_dump, reporting the current SQLite backend instead.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ── Project root on sys.path ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _resolve_pg_url() -> str | None:
    """Return a PostgreSQL connection URL, or None if not configured."""
    url = os.environ.get("DATABASE_URL", "")
    if url and url.startswith("postgresql"):
        return url

    host = os.environ.get("POSTGRES_HOST", "")
    if host:
        user = os.environ.get("POSTGRES_USER", os.environ.get("DB_USER", "mlops_user"))
        password = os.environ.get(
            "DB_PASSWORD", os.environ.get("POSTGRES_PASSWORD", "local_dev_password")
        )
        port = os.environ.get("POSTGRES_PORT", os.environ.get("DB_PORT", "5432"))
        db = os.environ.get("POSTGRES_DB", os.environ.get("DB_NAME", "mlops_db"))
        return f"postgresql://{user}:{password}@{host}:{port}/{db}"

    # Auto-detect: try a real credential-based connection.  OrbStack DNS is
    # tried first.  We resolve hostnames to IPv4 explicitly to avoid the
    # multi-second latency caused by psycopg2 probing IPv6 first.
    # A plain socket probe (or trusting POSTGRES_HOST=localhost blindly) can
    # hit a macOS-local postgres that has no mlops_user role.
    import socket

    user = os.environ.get("DB_USER", "mlops_user")
    password = os.environ.get("DB_PASSWORD", "local_dev_password")
    db = os.environ.get("DB_NAME", "mlops_db")

    seen: set[str] = set()
    candidates: list[str] = []
    pg_host = os.environ.get("POSTGRES_HOST", "")
    if pg_host:
        candidates.append(pg_host)
        seen.add(pg_host)
    for h in ["mlops_postgres.orb.local", "localhost"]:
        if h not in seen:
            candidates.append(h)

    for try_host in candidates:
        # Resolve to IPv4 to avoid slow IPv6-first probing by psycopg2
        resolved = try_host
        if try_host != "localhost":
            try:
                for info in socket.getaddrinfo(try_host, 5432, socket.AF_INET, socket.SOCK_STREAM):
                    resolved = str(info[4][0])  # AF_INET sockaddr is (host, port); host is str
                    break
            except OSError:
                pass
        try:
            import psycopg2

            conn = psycopg2.connect(
                host=resolved,
                port=5432,
                user=user,
                password=password,
                dbname=db,
                connect_timeout=10,
            )
            conn.close()
            return f"postgresql://{user}:{password}@{resolved}:5432/{db}"
        except Exception:
            continue

    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a pg_dump backup immediately.")
    parser.add_argument(
        "--keep-n",
        type=int,
        default=7,
        help="Number of most-recent backups to keep (default: 7).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/backups",
        help="Directory for .dump files (default: data/backups).",
    )
    args = parser.parse_args()

    db_url = _resolve_pg_url()
    if db_url is None:
        print(
            "[INFO] POSTGRES_HOST / DATABASE_URL not set — running in SQLite mode.\n"
            "       pg_dump is only available when connected to PostgreSQL.\n"
            "       Start the Docker stack (docker compose up) and set\n"
            "       POSTGRES_HOST=localhost (or DATABASE_URL) to enable backups."
        )
        return 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from src.database.backup import (
            backup_postgres,
            cleanup_old_backups,
            get_backup_filename,
            list_backups,
        )

        filename = get_backup_filename()
        out_path = output_dir / filename

        print(f"[INFO] Running pg_dump → {out_path} …")
        backup_postgres(db_url=db_url, output_path=out_path)
        print(f"[OK]   Backup written: {out_path}  ({out_path.stat().st_size:,} bytes)")

        removed = cleanup_old_backups(backup_dir=output_dir, keep_n=args.keep_n)
        if removed:
            print(f"[INFO] Removed {removed} old backup(s) (keeping {args.keep_n}).")

        backups = list_backups(backup_dir=output_dir)
        print(f"\nBackups in {output_dir} ({len(backups)} total):")
        for b in backups:
            size_kb = b.stat().st_size // 1024
            print(f"  {b.name}  ({size_kb} KB)")

        return 0

    except Exception as exc:
        print(f"[ERROR] Backup failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
