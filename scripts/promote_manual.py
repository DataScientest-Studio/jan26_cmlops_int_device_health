#!/usr/bin/env python3
"""
Manual model promotion – select a specific model version for production.

The operator takes full responsibility for the promotion. The script records
that it was a manual promotion via a ``promoted_by`` tag and logs the
promotion event in MLflow.

Usage
-----
    # Promote version 3 to Production:
    python scripts/promote_manual.py --version 3

    # Dry-run (show what would happen, don't change anything):
    python scripts/promote_manual.py --version 3 --dry-run

    # Promote and specify a reason (recorded as MLflow tag):
    python scripts/promote_manual.py --version 3 --reason "hotfix for drift regression"
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(help="Manually promote a model version to Production")


def _mlflow_uri() -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if uri:
        return uri
    mode_file = PROJECT_ROOT / ".current_mode"
    mode = mode_file.read_text().strip() if mode_file.exists() else "local"
    if mode == "cloud":
        user = os.environ.get("DAGSHUB_USER", "")
        repo = os.environ.get("DAGSHUB_REPO", "")
        if user and repo:
            return f"https://dagshub.com/{user}/{repo}.mlflow"
    return "http://localhost:5001"


def _setup_mlflow(uri: str) -> None:
    import mlflow

    mlflow.set_tracking_uri(uri)
    mode_file = PROJECT_ROOT / ".current_mode"
    mode = mode_file.read_text().strip() if mode_file.exists() else "local"
    if mode == "cloud":
        user = os.environ.get("DAGSHUB_USER", "")
        token = os.environ.get("DAGSHUB_TOKEN", "")
        if user:
            os.environ["MLFLOW_TRACKING_USERNAME"] = user
        if token:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = token


def _show_version_info(client, model_name: str, version: str) -> dict | None:
    """Display details of a model version and return its run metrics."""
    try:
        mv = client.get_model_version(model_name, version)
    except Exception:
        console.print(f"[red]Version {version} not found in {model_name}[/red]")
        return None

    run = client.get_run(mv.run_id)
    metrics = run.data.metrics
    tags = run.data.tags

    table = Table(title=f"Model {model_name} v{version}")
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("Run ID", mv.run_id)
    table.add_row("Current Stage", getattr(mv, "current_stage", "None"))
    table.add_row("Created", str(mv.creation_timestamp))

    # Lineage
    table.add_row("git_commit", tags.get("git_commit", "N/A"))
    table.add_row("dvc_data_version", tags.get("dvc_data_version", "N/A"))
    table.add_row("airflow_run_id", tags.get("airflow_run_id", "N/A"))

    # Metrics
    for k, v in sorted(metrics.items()):
        if k.startswith("test_"):
            table.add_row(k, f"{v:.4f}")

    console.print(table)
    return {"metrics": metrics, "tags": tags, "run_id": mv.run_id}


@app.command()
def promote(
    version: int = typer.Option(..., help="Model version number to promote"),
    model_name: str = typer.Option("device_health_classifier", help="Registered model name"),
    reason: str = typer.Option("", help="Reason for manual promotion (recorded as tag)"),
    dry_run: bool = typer.Option(False, help="Preview only, don't change anything"),
) -> None:
    """Promote a specific model version to Production."""
    uri = _mlflow_uri()
    _setup_mlflow(uri)

    from mlflow.tracking import MlflowClient

    client = MlflowClient()

    # Show what we're promoting
    info = _show_version_info(client, model_name, str(version))
    if info is None:
        raise typer.Exit(1)

    # Check lineage completeness
    tags = info["tags"]
    lineage_keys = ["git_commit", "dvc_data_version", "airflow_run_id"]
    missing = [k for k in lineage_keys if not tags.get(k)]
    if missing:
        console.print(f"\n[yellow]Warning: Missing lineage tags: {', '.join(missing)}[/yellow]")
        console.print("[yellow]Promotion will proceed but traceability is incomplete.[/yellow]")

    if dry_run:
        console.print("\n[yellow]DRY RUN — no changes made[/yellow]")
        return

    # Perform promotion
    from src.training.registry import promote_model

    promote_model(
        model_name=model_name,
        version=version,
        stage="Production",
        archive_existing_production=True,
    )

    # Tag the promotion event
    now = datetime.now(tz=timezone.utc).isoformat()
    client.set_model_version_tag(model_name, str(version), "promoted_by", "manual_promotion_script")
    client.set_model_version_tag(model_name, str(version), "promoted_at", now)
    if reason:
        client.set_model_version_tag(model_name, str(version), "promotion_reason", reason)

    console.print(f"\n[bold green]v{version} promoted to Production[/bold green]")
    console.print("  promoted_by:  manual_promotion_script")
    console.print(f"  promoted_at:  {now}")
    if reason:
        console.print(f"  reason:       {reason}")


if __name__ == "__main__":
    app()
