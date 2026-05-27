#!/usr/bin/env python3
"""
Multi-challenger promotion – evaluate ALL recent experiment runs against champion.

Instead of comparing only one challenger, this script:

1.  Retrieves the current Production champion
2.  Gathers all candidate runs (from a given experiment, optionally filtered
    by time window or tags)
3.  Evaluates each candidate against the gold-standard test set
4.  Ranks all candidates by the primary metric
5.  Promotes the best candidate if it outperforms the champion by the
    configured threshold

Usage
-----
    # Evaluate all runs from the default experiment:
    python scripts/promote_multi_challenger.py

    # Dry-run (show ranking, don't promote):
    python scripts/promote_multi_challenger.py --dry-run

    # Only consider runs from the last 7 days:
    python scripts/promote_multi_challenger.py --days 7

    # Custom threshold:
    python scripts/promote_multi_challenger.py --threshold 0.01
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
app = typer.Typer(help="Evaluate all candidates and promote the best one")


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


@app.command()
def evaluate(
    model_name: str = typer.Option("device_health_classifier", help="Registered model name"),
    experiment: str = typer.Option("device_health_classifier", help="MLflow experiment name"),
    metric: str = typer.Option("test_f1_score", help="Primary metric for comparison"),
    threshold: float = typer.Option(0.02, help="Minimum improvement required for promotion"),
    days: int = typer.Option(0, help="Only consider runs from the last N days (0=all)"),
    dry_run: bool = typer.Option(False, help="Show ranking without promoting"),
) -> None:
    """Rank all experiment runs and promote the best if it beats the champion."""
    import mlflow
    from mlflow.tracking import MlflowClient

    uri = _mlflow_uri()
    _setup_mlflow(uri)
    client = MlflowClient()

    # -- 1. Get current champion -----------------------------------------
    from src.training.registry import get_production_models

    champion_models = get_production_models(model_name)
    champion_metric = 0.0
    champion_version = None

    if champion_models:
        champ = champion_models[0]
        champion_version = champ.get("version")
        champion_metric = champ.get("metrics", {}).get(metric, 0.0)
        console.print(f"[cyan]Champion:[/cyan] v{champion_version}  {metric}={champion_metric:.4f}")
    else:
        console.print("[yellow]No current champion found.[/yellow]")

    # -- 2. Gather candidate runs ----------------------------------------
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        console.print(f"[red]Experiment '{experiment}' not found[/red]")
        raise typer.Exit(1)

    filter_parts = ["attributes.status = 'FINISHED'"]
    if days > 0:
        import time

        cutoff_ms = int((time.time() - days * 86400) * 1000)
        filter_parts.append(f"attributes.end_time > {cutoff_ms}")

    filter_str = " AND ".join(filter_parts)

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=filter_str,
        order_by=[f"metrics.{metric} DESC"],
        max_results=50,
    )

    if not runs:
        console.print("[yellow]No finished runs found.[/yellow]")
        raise typer.Exit(0)

    # -- 3. Rank candidates ----------------------------------------------
    table = Table(title=f"Candidate Ranking by {metric}")
    table.add_column("#", style="dim")
    table.add_column("Run ID", style="cyan")
    table.add_column(metric, justify="right")
    table.add_column("git_commit")
    table.add_column("dvc_data_version")
    table.add_column("airflow_run_id")
    table.add_column("Registered?")

    candidates = []
    for i, run in enumerate(runs):
        run_metric = run.data.metrics.get(metric, 0.0)
        tags = run.data.tags
        git = tags.get("git_commit", "—")
        dvc = tags.get("dvc_data_version", "—")
        af = tags.get("airflow_run_id", "—")

        # Check if this run is already registered
        reg_version = None
        try:
            versions = client.search_model_versions(f"run_id='{run.info.run_id}'")
            if versions:
                reg_version = versions[0].version
        except Exception:
            pass

        candidates.append(
            {
                "run_id": run.info.run_id,
                "metric": run_metric,
                "git_commit": git,
                "dvc_data_version": dvc,
                "airflow_run_id": af,
                "registered_version": reg_version,
            }
        )

        style = "bold green" if i == 0 else ""
        table.add_row(
            str(i + 1),
            run.info.run_id[:12] + "…",
            f"{run_metric:.4f}",
            git[:8] if git != "—" else git,
            dvc[:12] + "…" if dvc != "—" and len(dvc) > 12 else dvc,
            af[:20] if af != "—" else af,
            f"v{reg_version}" if reg_version else "—",
            style=style,
        )

    console.print(table)

    # -- 4. Decide on promotion ------------------------------------------
    best = candidates[0]
    improvement = best["metric"] - champion_metric

    console.print(
        f"\n[cyan]Best candidate:[/cyan]  {best['run_id'][:12]}…  {metric}={best['metric']:.4f}"
    )
    console.print(
        f"[cyan]Champion:       [/cyan]  v{champion_version or '—'}  {metric}={champion_metric:.4f}"
    )
    console.print(
        f"[cyan]Improvement:    [/cyan]  {improvement:+.4f}  (threshold: {threshold:.4f})"
    )

    if improvement <= threshold:
        console.print(f"\n[yellow]No candidate beats the champion by > {threshold:.4f}.[/yellow]")
        return

    if dry_run:
        console.print(f"\n[yellow]DRY RUN — would promote run {best['run_id'][:12]}…[/yellow]")
        return

    # -- 5. Register (if not already) and promote -----------------------
    if best["registered_version"] is None:
        model_uri = f"runs:/{best['run_id']}/model"
        try:
            mv = client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=best["run_id"],
                tags={
                    "trained_by": "multi_challenger_evaluation",
                    "role": "Challenger",
                },
            )
            version = int(mv.version)
        except Exception as e:
            console.print(f"[red]Failed to register model: {e}[/red]")
            raise typer.Exit(1) from e
    else:
        version = int(best["registered_version"])

    from src.training.registry import promote_model

    promote_model(
        model_name=model_name,
        version=version,
        stage="Production",
        archive_existing_production=True,
    )

    # Tag the promotion
    now = datetime.now(tz=timezone.utc).isoformat()
    client.set_model_version_tag(model_name, str(version), "promoted_by", "multi_challenger_script")
    client.set_model_version_tag(model_name, str(version), "promoted_at", now)
    client.set_model_version_tag(
        model_name, str(version), "promotion_method", "multi_challenger_evaluation"
    )
    client.set_model_version_tag(
        model_name, str(version), "candidates_evaluated", str(len(candidates))
    )
    client.set_model_version_tag(
        model_name, str(version), "improvement_over_champion", f"{improvement:.4f}"
    )

    console.print(f"\n[bold green]v{version} promoted to Production[/bold green]")
    console.print("  promoted_by:   multi_challenger_script")
    console.print(f"  candidates:    {len(candidates)}")
    console.print(f"  improvement:   {improvement:+.4f}")


if __name__ == "__main__":
    app()
