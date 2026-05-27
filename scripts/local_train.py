#!/usr/bin/env python3
"""
Local-mode training script (no Airflow, no model promotion).

Runs the same ``train_model()`` function used by the Airflow
``train_challenger_model`` DAG, but:

- Forces ``DEPLOYMENT_MODE=local`` environment variable.
- Logs to the local MLflow Docker container (http://mlflow:5000 or
  the value of ``MLFLOW_TRACKING_URI``).
- Does **not** promote the model.  Use ``scripts/promote_model.py``
  from cloud mode for promotion.

Usage
-----
    python scripts/local_train.py                 # default paths
    python scripts/local_train.py --data data/train.json --model models/model.pkl
    python scripts/local_train.py --no-mlflow     # skip MLflow logging
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
app = typer.Typer(help="Train a challenger model locally (no promotion)")


@app.command()
def train(
    data: Path = typer.Option(
        Path("data/train.json"),
        "--data",
        "-d",
        help="Path to training data (JSON with 'signals' array)",
    ),
    model: Path = typer.Option(
        Path("models/challenger_model.pkl"),
        "--model",
        "-m",
        help="Path to save trained model",
    ),
    version: str = typer.Option(
        "v1.0_local",
        "--version",
        "-v",
        help="Model version identifier",
    ),
    experiment: str = typer.Option(
        "device_health_classifier",
        "--experiment",
        "-e",
        help="MLflow experiment name",
    ),
    mlflow: bool = typer.Option(
        True,
        "--mlflow/--no-mlflow",
        help="Enable MLflow experiment tracking",
    ),
) -> None:
    """Train a challenger model in local sandbox mode."""
    # Enforce local mode
    os.environ["DEPLOYMENT_MODE"] = "local"

    console.print("[bold green]▶  Local training (no promotion)[/bold green]")
    console.print(f"  Data:       {data}")
    console.print(f"  Model:      {model}")
    console.print(f"  Version:    {version}")
    console.print(f"  Experiment: {experiment}")
    console.print(f"  MLflow:     {mlflow}")

    if not data.exists():
        console.print(f"[red]Training data not found: {data}[/red]")
        raise SystemExit(1)

    model.parent.mkdir(parents=True, exist_ok=True)

    from src.training.train import train_model

    result = train_model(
        train_data_path=data,
        model_output_path=model,
        model_version=version,
        use_mlflow=mlflow,
        mlflow_experiment_name=experiment,
        dvc_pull=False,  # never pull DVC in local mode
    )

    console.print("\n[bold]Training result:[/bold]")
    for key in ("model_version", "primary_metric", "primary_metric_value", "total_samples"):
        if key in result:
            console.print(f"  {key}: {result[key]}")

    if mlflow and "mlflow_run_id" in result:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
        console.print(f"  MLflow run: {result['mlflow_run_id']}")
        console.print(f"  MLflow UI:  {tracking_uri}")

    console.print(
        "\n[bold yellow]Note:[/bold yellow] Model promotion is disabled in local mode. "
        "Use cloud mode + scripts/promote_model.py to promote."
    )


if __name__ == "__main__":
    app()
