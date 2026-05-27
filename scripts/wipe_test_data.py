#!/usr/bin/env python3
"""
wipe_test_data.py — Remove test / demo data from all storage layers
====================================================================

This script provides a **single command** to clean up experiment data
generated during development, demos, or testing.  It is idempotent and
safe to re-run.

Usage
-----
    # Preview what would be deleted (dry-run — default):
    python scripts/wipe_test_data.py

    # Actually delete everything:
    python scripts/wipe_test_data.py --execute

    # Only wipe local artifacts:
    python scripts/wipe_test_data.py --execute --local-only

    # Include cloud cleanup (DagsHub, remote MLflow):
    python scripts/wipe_test_data.py --execute --include-cloud

Cleaned locations
-----------------
Local (always):
  - Docker volumes           (mlflow_artifacts, mlflow_db, postgres_data, …)
  - DVC cache                (.dvc/cache, .dvc/tmp)
  - Local data artifacts     (data/processed/*, data/raw/*, logs/*)
  - .current_mode file
  - __pycache__ trees

Cloud (--include-cloud):
  - MLflow experiments on DagsHub  (via MLflow Tracking API)
  - DVC remote cache on DagsHub    (dvc push --remove)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directories to clear (relative to PROJECT_ROOT).
# These are emptied but the directory itself is kept.
LOCAL_DATA_DIRS = [
    "data/processed",
    "data/raw",
    "logs",
    "logs/airflow",
    "logs/nginx",
]

# Files and globs to remove.
LOCAL_FILES = [
    ".current_mode",
]

# Docker volumes to remove (named volumes created by docker-compose)
DOCKER_VOLUMES = [
    "mlops_device_health_postgres_data",
    "mlops_device_health_mlflow_artifacts",
    "mlops_device_health_mlflow_db",
    "mlops_device_health_prometheus_data",
    "mlops_device_health_alertmanager_data",
    "mlops_device_health_grafana_data",
]

# Colour helpers
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
RED = "\033[0;31m"
BLUE = "\033[0;34m"
NC = "\033[0m"  # No colour


def _info(msg: str) -> None:
    print(f"  {BLUE}ℹ{NC}  {msg}")


def _ok(msg: str) -> None:
    print(f"  {GREEN}✓{NC}  {msg}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{NC}  {msg}")


def _err(msg: str) -> None:
    print(f"  {RED}✗{NC}  {msg}")


def _run(cmd: list[str], *, check: bool = False, timeout: int = 60) -> tuple[str, int]:
    """Run a command and return (stdout, returncode)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT),
        )
        return result.stdout.strip(), result.returncode
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return str(exc), 1


# ── Local cleanup ───────────────────────────────────────────────


def stop_docker_stack(*, dry_run: bool) -> None:
    """Stop all running containers before cleaning volumes."""
    if dry_run:
        _info("Would stop Docker stack (make down)")
        return
    _info("Stopping Docker stack …")
    out, rc = _run(["make", "down"], timeout=120)
    if rc == 0:
        _ok("Docker stack stopped")
    else:
        _warn(f"make down returned rc={rc} — continuing anyway")


def remove_docker_volumes(*, dry_run: bool) -> None:
    """Remove named Docker volumes."""
    for vol in DOCKER_VOLUMES:
        if dry_run:
            _info(f"Would remove Docker volume: {vol}")
            continue
        out, rc = _run(["docker", "volume", "rm", "-f", vol])
        if rc == 0:
            _ok(f"Removed volume: {vol}")
        else:
            _warn(f"Could not remove volume {vol} (may not exist)")


def clear_local_data_dirs(*, dry_run: bool) -> None:
    """Empty local data directories (keep the directory itself)."""
    for rel in LOCAL_DATA_DIRS:
        dirpath = PROJECT_ROOT / rel
        if not dirpath.exists():
            continue
        files = list(dirpath.iterdir())
        # Skip .gitkeep
        files = [f for f in files if f.name != ".gitkeep"]
        if not files:
            continue
        if dry_run:
            _info(f"Would clear {len(files)} items from {rel}/")
            continue
        for f in files:
            if f.is_dir():
                shutil.rmtree(f)
            else:
                f.unlink()
        _ok(f"Cleared {len(files)} items from {rel}/")


def remove_local_files(*, dry_run: bool) -> None:
    """Remove individual files."""
    for rel in LOCAL_FILES:
        fpath = PROJECT_ROOT / rel
        if not fpath.exists():
            continue
        if dry_run:
            _info(f"Would remove: {rel}")
            continue
        fpath.unlink()
        _ok(f"Removed: {rel}")


def clear_dvc_cache(*, dry_run: bool) -> None:
    """Remove the local DVC cache."""
    cache_dir = PROJECT_ROOT / ".dvc" / "cache"
    tmp_dir = PROJECT_ROOT / ".dvc" / "tmp"
    for d in (cache_dir, tmp_dir):
        if not d.exists():
            continue
        if dry_run:
            _info(f"Would remove: {d.relative_to(PROJECT_ROOT)}")
            continue
        shutil.rmtree(d)
        _ok(f"Removed: {d.relative_to(PROJECT_ROOT)}")


def clear_pycache(*, dry_run: bool) -> None:
    """Remove all __pycache__ directories."""
    caches = list(PROJECT_ROOT.rglob("__pycache__"))
    if not caches:
        return
    if dry_run:
        _info(f"Would remove {len(caches)} __pycache__ directories")
        return
    for d in caches:
        shutil.rmtree(d)
    _ok(f"Removed {len(caches)} __pycache__ directories")


# ── Cloud cleanup ───────────────────────────────────────────────


def wipe_mlflow_remote(*, dry_run: bool) -> None:
    """Delete all experiments on the remote MLflow server (DagsHub).

    Requires MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, and
    MLFLOW_TRACKING_PASSWORD to be set (typically via .env.cloud +
    .env.secrets).
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if "localhost" in tracking_uri or not tracking_uri:
        _warn("MLFLOW_TRACKING_URI not set or points to localhost — skipping remote MLflow wipe")
        return

    if dry_run:
        _info(f"Would delete all experiments on {tracking_uri}")
        return

    try:
        import mlflow  # noqa: E402

        mlflow.set_tracking_uri(tracking_uri)
        experiments = mlflow.search_experiments()
        deleted = 0
        for exp in experiments:
            if exp.name == "Default":
                continue
            try:
                mlflow.delete_experiment(exp.experiment_id)
                deleted += 1
            except Exception as exc:
                _warn(f"Could not delete experiment '{exp.name}': {exc}")
        _ok(f"Deleted {deleted} experiment(s) on {tracking_uri}")
    except ImportError:
        _warn("mlflow not installed — skipping remote MLflow wipe")
    except Exception as exc:
        _err(f"MLflow remote wipe failed: {exc}")


def wipe_dvc_remote(*, dry_run: bool) -> None:
    """Remove data from the DagsHub DVC remote cache."""
    if dry_run:
        _info("Would run: dvc gc --cloud --all-branches --all-tags --force")
        return
    out, rc = _run(
        ["dvc", "gc", "--cloud", "--all-branches", "--all-tags", "--force"],
        timeout=120,
    )
    if rc == 0:
        _ok("DVC remote cache garbage-collected")
    else:
        _warn(f"dvc gc --cloud returned rc={rc}: {out[:200]}")


# ── Main ────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wipe test / demo data from all storage layers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete data. Without this flag the script runs in dry-run mode.",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Only clean local artifacts (Docker volumes, DVC cache, data dirs). "
        "Skip cloud cleanup even if --include-cloud is set.",
    )
    parser.add_argument(
        "--include-cloud",
        action="store_true",
        help="Also wipe remote data (DagsHub MLflow experiments, DVC remote cache). "
        "Requires credentials in environment.",
    )
    args = parser.parse_args()
    dry_run = not args.execute

    if dry_run:
        print(f"\n{YELLOW}══ DRY RUN — no changes will be made ══{NC}\n")
    else:
        print(f"\n{RED}══ EXECUTING — data will be permanently deleted ══{NC}\n")

    # Local cleanup
    print(f"{BLUE}── Local Cleanup ──{NC}")
    stop_docker_stack(dry_run=dry_run)
    remove_docker_volumes(dry_run=dry_run)
    clear_local_data_dirs(dry_run=dry_run)
    remove_local_files(dry_run=dry_run)
    clear_dvc_cache(dry_run=dry_run)
    clear_pycache(dry_run=dry_run)

    # Cloud cleanup
    if args.include_cloud and not args.local_only:
        print(f"\n{BLUE}── Cloud Cleanup ──{NC}")
        wipe_mlflow_remote(dry_run=dry_run)
        wipe_dvc_remote(dry_run=dry_run)
    elif not args.local_only:
        print(f"\n{YELLOW}  Tip: add --include-cloud to also wipe DagsHub data{NC}")

    if dry_run:
        print(f"\n{YELLOW}  Re-run with --execute to apply changes.{NC}\n")
    else:
        print(f"\n{GREEN}══ Cleanup complete ══{NC}\n")


if __name__ == "__main__":
    main()
