#!/usr/bin/env python3
"""
Local-mode start helper with three options:

1. **proceed**  – Keep existing local data and models; start normally.
2. **wipe**     – Delete ALL data from every table (greenfield reset) and
                  wipe local MLflow experiments, then start fresh.
3. **pull**     – Delete only locally-generated data (deployment_mode='local')
                  from the database, wipe local MLflow experiments, then pull
                  fresh data and MLflow experiments from DagsHub cloud.

Usage
-----
    python scripts/local_start.py proceed
    python scripts/local_start.py wipe
    python scripts/local_start.py pull
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import typer
from rich.console import Console

console = Console()
app = typer.Typer(help="Local sandbox start helper (proceed / wipe / pull)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open_db():
    """Return a Database connected to PostgreSQL (Docker) or SQLite (fallback)."""
    from src.database.database import Database

    db_url = os.environ.get("DATABASE_URL", "")
    if db_url.startswith("postgresql"):
        return Database(db_url=db_url)
    return Database("data/database/mlops.db")


def _wipe_local_db_data() -> dict[str, int]:
    """Delete all rows with deployment_mode='local' from the database."""
    db = _open_db()
    counts = db.delete_local_data()
    console.print(f"  Deleted local-mode rows: {counts}")
    return counts


def _wipe_all_db_data() -> dict[str, int]:
    """Delete ALL rows from every data table (greenfield reset)."""
    db = _open_db()
    counts = db.wipe_all_data()
    console.print(f"  Deleted all rows: {counts}")
    return counts


def _wipe_local_mlflow_experiments() -> None:
    """Delete local MLflow experiments directory (artifact store).

    The local Docker MLflow stores artifacts under ``mlflow_data/``.
    Wiping this directory forces a clean slate when pulling from cloud.
    """
    mlflow_data = PROJECT_ROOT / "mlflow_data"
    if mlflow_data.exists():
        import shutil

        shutil.rmtree(mlflow_data)
        console.print("  Wiped mlflow_data/ directory")
    else:
        console.print("  [dim]mlflow_data/ not found — nothing to wipe[/dim]")


def _pull_from_dagshub() -> None:
    """Pull MLflow experiments/runs from DagsHub into local MLflow."""
    user = os.environ.get("DAGSHUB_USER", "")
    token = os.environ.get("DAGSHUB_TOKEN", "")
    repo = os.environ.get("DAGSHUB_REPO", "")

    if not user or not token or not repo:
        console.print(
            "[red]Cannot pull from DagsHub: DAGSHUB_USER, DAGSHUB_TOKEN, "
            "and DAGSHUB_REPO must be set in .env.secrets[/red]"
        )
        raise SystemExit(1)

    from src.training.mlflow_sync import pull_from_dagshub

    console.print("  Pulling MLflow experiments from DagsHub …")
    summary = pull_from_dagshub(sync_artifacts=True)
    console.print(
        f"  Pulled {summary.get('runs_synced', 0)} runs from "
        f"{summary.get('experiments_synced', 0)} experiments"
    )


def _pull_dvc_data() -> None:
    """Pull DVC-tracked data (CSVs, artifacts) from DagsHub remote."""
    import subprocess

    console.print("  Pulling DVC-tracked data from DagsHub …")
    result = subprocess.run(
        ["dvc", "pull", "-r", "dagshub"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        console.print(f"[red]DVC pull failed:[/red]\n{result.stderr}")
        raise SystemExit(1)
    console.print(f"  DVC pull complete: {result.stdout.strip()}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def proceed() -> None:
    """Keep existing local data and start normally."""
    console.print("[bold green]▶  Proceeding with existing local data[/bold green]")
    console.print("  Nothing to do — local sandbox is ready.")


@app.command()
def wipe() -> None:
    """Delete ALL data from every table and wipe MLflow — greenfield reset."""
    console.print("[bold yellow]▶  Greenfield wipe: deleting ALL data[/bold yellow]")
    _wipe_all_db_data()
    _wipe_local_mlflow_experiments()
    console.print("[bold green]✓  All data wiped — greenfield sandbox ready[/bold green]")


@app.command()
def pull() -> None:
    """Wipe local-mode data, then pull fresh data and MLflow experiments from DagsHub."""
    console.print("[bold cyan]▶  Wipe local data + pull from DagsHub cloud[/bold cyan]")

    console.print("\n[bold]Step 1:[/bold] Wipe local-mode data …")
    _wipe_local_db_data()
    _wipe_local_mlflow_experiments()

    console.print("\n[bold]Step 2:[/bold] Pull MLflow from DagsHub …")
    _pull_from_dagshub()

    console.print("\n[bold]Step 3:[/bold] Pull DVC-tracked data from DagsHub …")
    _pull_dvc_data()

    console.print("\n[bold green]✓  Local sandbox refreshed from cloud[/bold green]")


if __name__ == "__main__":
    app()
