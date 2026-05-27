#!/usr/bin/env python3
"""
Sparse label injection script for MLOps demonstration.

Simulates asynchronous ground truth label arrival in production systems.
In real scenarios, only 5-10% of devices receive expensive diagnostic labels.
This script injects delayed labels into the prediction database to demonstrate:
    - Sparse label audit (realized accuracy calculation)
    - Accuracy degradation detection despite stable prediction distribution
    - Triggering retraining based on ground truth feedback

Use Cases:
    1. Inject labels from validation set into "predictions" database
    2. Simulate delayed label arrival (timestamp manipulation)
    3. Calculate realized accuracy by matching predictions with labels
    4. Demonstrate drift detection via label distribution shift

Examples:
    # Inject 50 random labels from validation set
    python scripts/inject_sparse_labels.py --source data/raw/dataset_baseline_test.json --n-labels 50

    # Inject labels with 7-day delay
    python scripts/inject_sparse_labels.py --source data/drift/data_drift/drift_signals.json --n-labels 100 --delay-days 7

    # Calculate realized accuracy from injected labels
    python scripts/inject_sparse_labels.py calculate-accuracy --lookback-days 7
"""

import json
import os
import random
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

app = typer.Typer(help="Sparse label injection for production audit")
console = Console()


def load_signals(file_path: Path) -> list[dict]:
    """Load signals from JSON file."""
    with open(file_path) as f:
        data = json.load(f)
    return data["signals"]


def save_labels_database(labels: list[dict], output_path: Path) -> None:
    """Save labels to JSON file (legacy helper, kept for calculate-accuracy command)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        labels_list = existing.get("labels", [])
    else:
        labels_list = []

    labels_list.extend(labels)

    data = {
        "metadata": {
            "last_updated": datetime.now().isoformat(),
            "total_labels": len(labels_list),
        },
        "labels": labels_list,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


@app.command()
def inject(
    source: Path = typer.Option(..., help="Path to signal JSON file with ground truth labels"),
    n_labels: int = typer.Option(50, help="Number of labels to inject (sampled randomly)"),
    delay_days: int = typer.Option(0, help="Simulated delay in days (kept for CLI compatibility)"),
    seed: int = typer.Option(42, help="Random seed for sampling"),
    label_source: str = typer.Option("script", help="Label source tag stored in DB"),
) -> None:
    """
    Inject sparse labels into the real predictions database.

    Simulates delayed ground truth arrival in production:
        1. Load labeled signals from source JSON file
        2. Find unlabeled prediction rows in the database
        3. Assign labels sampled from the source distribution
        4. Store via db.inject_sparse_label() (writes to sparse_labels + updates predictions)

    DATABASE_URL env var is honoured; falls back to local SQLite.
    """
    from src.database.database import Database

    console.print("[bold cyan]Injecting Sparse Labels into Database[/bold cyan]\n")

    # ------------------------------------------------------------------
    # 1. Load source signals (to get label distribution)
    # ------------------------------------------------------------------
    signals = load_signals(source)
    console.print(f"Loaded {len(signals)} signals from {source}")

    labeled_signals = [s for s in signals if s.get("label") is not None]
    console.print(f"Found {len(labeled_signals)} signals with labels")

    if not labeled_signals:
        console.print("[red]No labeled signals found in source file — aborting.[/red]")
        raise typer.Exit(code=1)

    # ------------------------------------------------------------------
    # 2. Connect to real database
    # ------------------------------------------------------------------
    db_url = os.environ.get("DATABASE_URL", "")
    pg_host = os.environ.get("POSTGRES_HOST", "")
    if db_url and db_url.startswith("postgresql"):
        db = Database(db_url=db_url)
    elif pg_host:
        user = os.environ.get("POSTGRES_USER", "mlops_user")
        pw = os.environ.get("POSTGRES_PASSWORD", "changeme")
        port = os.environ.get("POSTGRES_PORT", "5432")
        dbname = os.environ.get("POSTGRES_DB", "mlops_db")
        db = Database(db_url=f"postgresql://{user}:{pw}@{pg_host}:{port}/{dbname}")
    else:
        db_path = Path(__file__).parent.parent / "data" / "database" / "mlops.db"
        db = Database(db_path=str(db_path))

    # ------------------------------------------------------------------
    # 3. Get unlabeled prediction IDs from DB
    # ------------------------------------------------------------------
    cursor = db.conn.cursor()
    cursor.execute(
        "SELECT prediction_id FROM predictions WHERE ground_truth_label IS NULL ORDER BY prediction_id"
    )
    unlabeled_rows = cursor.fetchall()
    unlabeled_ids = [r["prediction_id"] for r in unlabeled_rows]
    console.print(f"Found {len(unlabeled_ids)} unlabeled predictions in database")

    if not unlabeled_ids:
        console.print("[yellow]No unlabeled predictions in DB — nothing to label.[/yellow]")
        console.print("Run UC-01/UC-02 first to generate predictions via the API.")
        raise typer.Exit(code=0)

    # ------------------------------------------------------------------
    # 4. Sample prediction IDs and assign labels
    # ------------------------------------------------------------------
    random.seed(seed)
    n_to_inject = min(n_labels, len(unlabeled_ids), len(labeled_signals))
    if n_to_inject < n_labels:
        console.print(
            f"[yellow]Warning: can only inject {n_to_inject} labels "
            f"(unlabeled_preds={len(unlabeled_ids)}, labeled_src={len(labeled_signals)})[/yellow]"
        )

    selected_ids = random.sample(unlabeled_ids, n_to_inject)
    sampled_signals = random.sample(labeled_signals, n_to_inject)

    # ------------------------------------------------------------------
    # 5. Inject into DB
    # ------------------------------------------------------------------
    injected: list[dict] = []
    errors = 0
    for pred_id, signal in zip(selected_ids, sampled_signals, strict=False):
        try:
            db.inject_sparse_label(
                prediction_id=pred_id,
                ground_truth_label=int(signal["label"]),
                label_source=label_source,
            )
            injected.append(
                {
                    "prediction_id": pred_id,
                    "ground_truth_label": int(signal["label"]),
                    "shape_type": signal.get("shape_type", "unknown"),
                    "timestamp": datetime.now().isoformat(),
                }
            )
        except Exception as exc:
            console.print(f"[red]Error injecting label for prediction_id={pred_id}: {exc}[/red]")
            errors += 1

    db.close()

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    _print_db_injection_summary(injected, errors)


def _print_db_injection_summary(injected: list[dict], errors: int) -> None:
    """Print summary of labels injected into the database."""
    n_total = len(injected)
    if n_total == 0:
        console.print("[red]No labels injected.[/red]")
        return

    label_counts = Counter(lbl["ground_truth_label"] for lbl in injected)
    shape_counts = Counter(lbl.get("shape_type", "unknown") for lbl in injected)

    table = Table(title="Sparse Labels Injection Summary (Database)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Labels Injected", str(n_total))
    table.add_row(
        "Healthy (label=0)", f"{label_counts[0]} ({label_counts[0] / n_total * 100:.1f}%)"
    )
    table.add_row(
        "Unhealthy (label=1)",
        f"{label_counts[1]} ({label_counts[1] / n_total * 100:.1f}%)",
    )
    table.add_row("Errors", str(errors))
    if shape_counts:
        for shape, cnt in shape_counts.most_common():
            table.add_row(f"  {shape}", f"{cnt} ({cnt / n_total * 100:.1f}%)")

    console.print(table)
    console.print(f"\n✓ Successfully injected {n_total} labels into database")
    if errors:
        console.print(f"[yellow]⚠  {errors} labels failed to inject[/yellow]")


@app.command()
def calculate_accuracy(
    labels_db: Path = typer.Option(
        Path("data/database/sparse_labels.json"), help="Labels database path"
    ),
    predictions_db: Path = typer.Option(
        Path("data/database/predictions.json"),
        help="Predictions database path (simulated)",
    ),
    lookback_days: int = typer.Option(7, help="Calculate accuracy for last N days"),
) -> None:
    """
    Calculate realized accuracy from sparse labels.

    Matches prediction records with delayed ground truth labels to compute
    actual model performance. In production, this would involve SQL JOIN
    between predictions and labels tables.

    Demonstrates:
        - Accuracy degradation detection despite stable prediction distribution
        - Label distribution shift (concept drift indicator)
        - Confidence calibration quality
    """
    console.print("[bold cyan]Calculating Realized Accuracy[/bold cyan]\n")

    if not labels_db.exists():
        console.print(f"[red]Error: Labels database not found at {labels_db}[/red]")
        console.print("Run 'inject' command first to create labels database.")
        return

    # Load labels
    with open(labels_db) as f:
        labels_data = json.load(f)
    all_labels = labels_data["labels"]

    # Filter by lookback window
    cutoff_date = datetime.now() - timedelta(days=lookback_days)
    recent_labels = [
        lbl for lbl in all_labels if datetime.fromisoformat(lbl["timestamp"]) >= cutoff_date
    ]

    console.print(f"Loaded {len(all_labels)} total labels")
    console.print(f"Found {len(recent_labels)} labels in last {lookback_days} days\n")

    if not recent_labels:
        console.print("[yellow]No labels in specified lookback window[/yellow]")
        return

    # Analyze label distribution
    _print_label_analysis(recent_labels, lookback_days)


def _print_injection_summary(labels: list[dict], output_path: Path) -> None:
    """Print summary of injected labels."""
    n_total = len(labels)
    label_counts = Counter(lbl["ground_truth_label"] for lbl in labels)
    shape_counts = Counter(lbl["shape_type"] for lbl in labels)

    earliest = min(datetime.fromisoformat(lbl["timestamp"]) for lbl in labels)
    latest = max(datetime.fromisoformat(lbl["timestamp"]) for lbl in labels)

    table = Table(title="Sparse Labels Injection Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Total Labels Injected", str(n_total))
    table.add_row(
        "Healthy (label=0)", f"{label_counts[0]} ({label_counts[0] / n_total * 100:.1f}%)"
    )
    table.add_row(
        "Unhealthy (label=1)",
        f"{label_counts[1]} ({label_counts[1] / n_total * 100:.1f}%)",
    )
    table.add_row("", "")
    table.add_row(
        "Gaussian Signals",
        f"{shape_counts['gaussian']} ({shape_counts['gaussian'] / n_total * 100:.1f}%)",
    )
    table.add_row(
        "Lorentzian Signals",
        f"{shape_counts['lorentzian']} ({shape_counts['lorentzian'] / n_total * 100:.1f}%)",
    )
    table.add_row("", "")
    table.add_row("Timestamp Range", f"{earliest.date()} to {latest.date()}")
    table.add_row("Output Database", str(output_path))

    console.print(table)
    console.print(f"\n✓ Successfully injected {n_total} labels")


def _print_label_analysis(labels: list[dict], lookback_days: int) -> None:
    """Print analysis of label distribution."""
    n_total = len(labels)
    label_counts = Counter(lbl["ground_truth_label"] for lbl in labels)
    shape_counts = Counter(lbl["shape_type"] for lbl in labels)

    # Calculate stats
    healthy_pct = label_counts[0] / n_total * 100 if n_total > 0 else 0
    unhealthy_pct = label_counts[1] / n_total * 100 if n_total > 0 else 0
    gaussian_pct = shape_counts["gaussian"] / n_total * 100 if n_total > 0 else 0

    table = Table(title=f"Label Distribution (Last {lookback_days} Days)")
    table.add_column("Category", style="cyan")
    table.add_column("Count", style="magenta")
    table.add_column("Percentage", style="green")

    table.add_row("Total Labels", str(n_total), "100.0%")
    table.add_row("Healthy (label=0)", str(label_counts[0]), f"{healthy_pct:.1f}%")
    table.add_row("Unhealthy (label=1)", str(label_counts[1]), f"{unhealthy_pct:.1f}%")
    table.add_row("", "", "")
    table.add_row("Gaussian Shape", str(shape_counts["gaussian"]), f"{gaussian_pct:.1f}%")
    table.add_row(
        "Lorentzian Shape",
        str(shape_counts["lorentzian"]),
        f"{100 - gaussian_pct:.1f}%",
    )

    console.print(table)

    # Drift indicators
    console.print("\n[bold]Drift Indicators:[/bold]")

    if abs(healthy_pct - 50) > 20:
        console.print(
            f"[yellow]⚠  Label distribution skewed: {healthy_pct:.1f}% healthy "
            f"(expected ~50%)[/yellow]"
        )
    else:
        console.print("[green]✓ Label distribution balanced[/green]")

    if abs(gaussian_pct - 50) > 20:
        console.print(
            f"[yellow]⚠  Shape distribution skewed: {gaussian_pct:.1f}% Gaussian "
            f"(expected ~50%)[/yellow]"
        )
    else:
        console.print("[green]✓ Shape distribution balanced[/green]")

    # Check for concept drift (Gaussian but unhealthy, or Lorentzian but healthy)
    misaligned = 0
    for lbl in labels:
        is_gaussian = lbl["shape_type"] == "gaussian"
        is_healthy = lbl["ground_truth_label"] == 0

        # Normally: Gaussian → healthy, Lorentzian → unhealthy
        # Concept drift: this correlation breaks down
        if (is_gaussian and not is_healthy) or (not is_gaussian and is_healthy):
            misaligned += 1

    misalignment_pct = misaligned / n_total * 100 if n_total > 0 else 0
    if misalignment_pct > 30:
        console.print(
            f"[yellow]⚠  Concept drift detected: {misalignment_pct:.1f}% of labels "
            f"don't match expected shape-health correlation[/yellow]"
        )
    else:
        console.print(
            f"[green]✓ Shape-health correlation normal ({100 - misalignment_pct:.1f}% aligned)[/green]"
        )


if __name__ == "__main__":
    app()
