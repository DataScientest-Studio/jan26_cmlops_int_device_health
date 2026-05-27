"""
PostgreSQL backup and restore utilities for the MLOps device-health system.

Public surface
--------------
backup_postgres(db_url, output_path, *, pg_dump_cmd) -> Path
    Run pg_dump and write a binary custom-format dump to *output_path*.

restore_postgres(db_url, backup_path, *, pg_restore_cmd) -> None
    Restore a custom-format dump via pg_restore.

get_backup_filename(prefix) -> str
    Generate a timestamped filename, e.g. ``mlops_db_backup_20260305_143000.dump``.

list_backups(backup_dir, *, pattern) -> list[Path]
    Return all ``.dump`` files in *backup_dir*, sorted newest first.

cleanup_old_backups(backup_dir, *, keep_n, pattern) -> int
    Delete all but the *keep_n* newest backups; return number deleted.

All subprocess calls are thin wrappers so they can be mocked in tests — no
real PostgreSQL instance is required for the unit-test suite.
"""

from __future__ import annotations

import logging
import subprocess
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "backup_postgres",
    "restore_postgres",
    "get_backup_filename",
    "list_backups",
    "cleanup_old_backups",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_db_url(db_url: str) -> dict[str, str]:
    """
    Parse a PostgreSQL connection URL into its component parts.

    Returns a dict with keys: host, port, user, password, dbname.
    """
    parsed = urllib.parse.urlparse(db_url)
    return {
        "host": parsed.hostname or "localhost",
        "port": str(parsed.port or 5432),
        "user": parsed.username or "",
        "password": parsed.password or "",
        "dbname": parsed.path.lstrip("/"),
    }


def _build_env(db_url: str) -> dict[str, str]:
    """
    Build a ``subprocess`` environment dict that sets PGPASSWORD so that
    pg_dump / pg_restore do not prompt for a password.
    """
    import os

    parts = _parse_db_url(db_url)
    env = os.environ.copy()
    env["PGPASSWORD"] = parts["password"]
    return env


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def backup_postgres(
    db_url: str,
    output_path: str | Path,
    *,
    pg_dump_cmd: str = "pg_dump",
) -> Path:
    """
    Run ``pg_dump`` and write a binary custom-format (``.dump``) backup.

    Parameters
    ----------
    db_url:
        PostgreSQL connection URL,
        e.g. ``"postgresql://user:pass@localhost:5432/mlops_db"``.
    output_path:
        Destination file path.  Parent directories are created automatically.
    pg_dump_cmd:
        Executable name / path for pg_dump (override in tests).

    Returns
    -------
    Path
        Resolved path of the created backup file.

    Raises
    ------
    subprocess.CalledProcessError
        If pg_dump exits with a non-zero return code.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parts = _parse_db_url(db_url)
    cmd = [
        pg_dump_cmd,
        "--no-password",
        "--format=custom",
        "--no-owner",  # omit ownership commands so restore works as any superuser
        "--no-acl",  # omit GRANT/REVOKE commands for portability
        f"--host={parts['host']}",
        f"--port={parts['port']}",
        f"--username={parts['user']}",
        f"--dbname={parts['dbname']}",
        f"--file={output_path}",
    ]

    env = _build_env(db_url)
    logger.info("Starting pg_dump → %s", output_path)

    result = subprocess.run(  # noqa: S603
        cmd,
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error("pg_dump failed (rc=%d): %s", result.returncode, result.stderr)
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=result.stdout,
            stderr=result.stderr,
        )

    logger.info(
        "pg_dump completed successfully → %s  (%d bytes)", output_path, output_path.stat().st_size
    )
    return output_path


def restore_postgres(
    db_url: str,
    backup_path: str | Path,
    *,
    pg_restore_cmd: str = "pg_restore",
) -> None:
    """
    Restore a custom-format pg_dump file via ``pg_restore``.

    Parameters
    ----------
    db_url:
        PostgreSQL connection URL for the target database.
    backup_path:
        Path to the ``.dump`` file produced by :func:`backup_postgres`.
    pg_restore_cmd:
        Executable name / path for pg_restore (override in tests).

    Raises
    ------
    FileNotFoundError
        If *backup_path* does not exist.
    subprocess.CalledProcessError
        If pg_restore exits with a non-zero return code.
    """
    backup_path = Path(backup_path).resolve()
    if not backup_path.exists():
        raise FileNotFoundError(f"Backup file not found: {backup_path}")

    parts = _parse_db_url(db_url)
    cmd = [
        pg_restore_cmd,
        "--no-password",
        "--clean",
        "--if-exists",
        f"--host={parts['host']}",
        f"--port={parts['port']}",
        f"--username={parts['user']}",
        f"--dbname={parts['dbname']}",
        str(backup_path),
    ]

    env = _build_env(db_url)
    logger.info("Starting pg_restore ← %s", backup_path)

    result = subprocess.run(  # noqa: S603
        cmd,
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error("pg_restore failed (rc=%d): %s", result.returncode, result.stderr)
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=result.stdout,
            stderr=result.stderr,
        )

    logger.info("pg_restore completed successfully ← %s", backup_path)


def get_backup_filename(prefix: str = "mlops_db_backup") -> str:
    """
    Return a timestamped backup filename.

    Example: ``"mlops_db_backup_20260305_143000.dump"``

    Parameters
    ----------
    prefix:
        Filename prefix (no extension).
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}.dump"


def list_backups(
    backup_dir: str | Path,
    *,
    pattern: str = "*.dump",
) -> list[Path]:
    """
    Return all backup files matching *pattern* in *backup_dir*, newest first.

    Parameters
    ----------
    backup_dir:
        Directory that contains backup files.
    pattern:
        Glob pattern – defaults to ``"*.dump"``.

    Returns
    -------
    list[Path]
        Sorted so that the newest file is at index 0.  Returns an empty list
        when the directory does not exist or contains no matching files.
    """
    backup_dir = Path(backup_dir)
    if not backup_dir.exists():
        logger.debug("Backup directory does not exist: %s", backup_dir)
        return []
    files = sorted(backup_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def cleanup_old_backups(
    backup_dir: str | Path,
    *,
    keep_n: int = 7,
    pattern: str = "*.dump",
) -> int:
    """
    Delete all but the *keep_n* newest backups in *backup_dir*.

    Parameters
    ----------
    backup_dir:
        Directory that contains backup files.
    keep_n:
        How many recent backups to retain (must be >= 1).
    pattern:
        Glob pattern – defaults to ``"*.dump"``.

    Returns
    -------
    int
        Number of files deleted.

    Raises
    ------
    ValueError
        If *keep_n* < 1.
    """
    if keep_n < 1:
        raise ValueError(f"keep_n must be >= 1, got {keep_n}")

    backups = list_backups(backup_dir, pattern=pattern)
    to_delete = backups[keep_n:]

    deleted = 0
    for path in to_delete:
        try:
            path.unlink()
            logger.info("Deleted old backup: %s", path)
            deleted += 1
        except OSError as exc:
            logger.warning("Could not delete %s: %s", path, exc)

    return deleted
