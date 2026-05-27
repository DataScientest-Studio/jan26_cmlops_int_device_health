#!/usr/bin/env python3
"""
Dataset generation script for device health signal data.

Creates synthetic datasets with configurable drift scenarios for MLOps pipeline testing.
"""

import json
import sys
from pathlib import Path
from typing import Any, Literal

# Ensure repo root is on sys.path so 'src' is importable when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console
from rich.progress import Progress

from src.signal_processing.signal_generator import generate_dataset
from src.signal_processing.signal_models import LabeledSignal

app = typer.Typer(help="Generate synthetic device health signal datasets")
console = Console()


def save_dataset(
    dataset: list[LabeledSignal], output_path: Path, include_labels: bool = True
) -> None:
    """
    Save dataset to JSON file.

    Args:
        dataset: List of LabeledSignal instances
        output_path: Output file path
        include_labels: Whether to include ground truth labels
    """
    dataset_dict: dict[str, Any] = {
        "n_samples": len(dataset),
        "signals": [],
    }

    for idx, labeled_signal in enumerate(dataset):
        signal_entry = {
            "id": idx,
            "time": labeled_signal.signal.time,
            "amplitude": labeled_signal.signal.amplitude,
            "shape_type": labeled_signal.signal.shape_type,
            "metadata": labeled_signal.metadata,
        }

        if include_labels:
            signal_entry["label"] = labeled_signal.label

        dataset_dict["signals"].append(signal_entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(dataset_dict, f, indent=2)


@app.command()
def generate(
    n_samples: int = typer.Option(100, help="Number of signals to generate"),
    gaussian_fraction: float = typer.Option(0.7, help="Fraction of Gaussian peaks (0.0-1.0)"),
    drift_scenario: Literal["baseline", "data_drift", "concept_drift"] = typer.Option(
        "baseline", help="Drift scenario for MLOps testing"
    ),
    output_dir: Path = typer.Option(Path("data/raw"), help="Output directory for dataset files"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    split: bool = typer.Option(True, help="Split into train/test sets"),
    test_fraction: float = typer.Option(0.2, help="Fraction for test set"),
) -> None:
    """
    Generate a synthetic dataset of device health signals.

    Examples:
        # Generate baseline dataset (100 samples, 50/50 Gaussian/Lorentzian)
        python scripts/generate_data.py generate

        # Generate data drift scenario with 200 samples
        python scripts/generate_data.py generate --n-samples 200 --drift-scenario data_drift

        # Generate concept drift with 70% Gaussian signals
        python scripts/generate_data.py generate --gaussian-fraction 0.7 --drift-scenario concept_drift
    """
    console.print(f"\n[bold blue]Generating {n_samples} signals...[/bold blue]")
    console.print(f"Drift scenario: [cyan]{drift_scenario}[/cyan]")
    console.print(f"Gaussian fraction: [cyan]{gaussian_fraction:.1%}[/cyan]")
    console.print(f"Random seed: [cyan]{seed}[/cyan]\n")

    with Progress() as progress:
        task = progress.add_task("[green]Generating signals...", total=1)

        # Generate full dataset
        dataset = generate_dataset(
            n_samples=n_samples,
            gaussian_fraction=gaussian_fraction,
            drift_scenario=drift_scenario,
            seed=seed,
        )

        progress.update(task, completed=1)

    # Save full dataset with labels
    full_path = output_dir / f"dataset_{drift_scenario}_full.json"
    save_dataset(dataset, full_path, include_labels=True)
    console.print(f"✓ Saved full dataset: [green]{full_path}[/green]")

    # Optionally split into train/test
    if split:
        n_test = int(n_samples * test_fraction)
        n_train = n_samples - n_test

        train_dataset = dataset[:n_train]
        test_dataset = dataset[n_train:]

        train_path = output_dir / f"dataset_{drift_scenario}_train.json"
        test_path = output_dir / f"dataset_{drift_scenario}_test.json"

        save_dataset(train_dataset, train_path, include_labels=True)
        save_dataset(test_dataset, test_path, include_labels=True)

        console.print(f"✓ Saved train set ({n_train} samples): [green]{train_path}[/green]")
        console.print(f"✓ Saved test set ({n_test} samples): [green]{test_path}[/green]")

    # Statistics
    n_healthy = sum(1 for s in dataset if s.label == 0)
    n_unhealthy = sum(1 for s in dataset if s.label == 1)

    console.print("\n[bold]Dataset Statistics:[/bold]")
    console.print(f"  Total samples: {len(dataset)}")
    console.print(f"  Healthy (label=0): {n_healthy} ({n_healthy / len(dataset):.1%})")
    console.print(f"  Unhealthy (label=1): {n_unhealthy} ({n_unhealthy / len(dataset):.1%})")


@app.command()
def generate_bootstrap(
    n_samples: int = typer.Option(100, help="Total number of samples"),
    labeled_fraction: float = typer.Option(0.2, help="Fraction with known labels (default 20%)"),
    gaussian_fraction: float = typer.Option(0.7, help="Fraction of Gaussian peaks"),
    output_dir: Path = typer.Option(Path("data/bootstrap"), help="Output directory"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """
    Generate dataset for Bootstrap Scenario B (semi-supervised learning).

    Creates:
        - Full dataset with all ground truth labels
        - Labeled subset (20% by default)
        - Unlabeled subset (80% with labels hidden)

    Example:
        python scripts/generate_data.py generate-bootstrap
    """
    console.print("\n[bold blue]Generating Bootstrap Dataset[/bold blue]")
    console.print(f"Total samples: [cyan]{n_samples}[/cyan]")
    console.print(f"Labeled fraction: [cyan]{labeled_fraction:.1%}[/cyan]\n")

    # Generate full dataset
    dataset = generate_dataset(
        n_samples=n_samples,
        gaussian_fraction=gaussian_fraction,
        drift_scenario="baseline",
        seed=seed,
    )

    # Split into labeled/unlabeled
    n_labeled = int(n_samples * labeled_fraction)
    labeled_dataset = dataset[:n_labeled]
    unlabeled_dataset = dataset[n_labeled:]

    # Save files
    output_dir.mkdir(parents=True, exist_ok=True)

    full_path = output_dir / "bootstrap_full.json"
    labeled_path = output_dir / "bootstrap_labeled.json"
    unlabeled_path = output_dir / "bootstrap_unlabeled.json"

    save_dataset(dataset, full_path, include_labels=True)
    save_dataset(labeled_dataset, labeled_path, include_labels=True)
    save_dataset(unlabeled_dataset, unlabeled_path, include_labels=False)

    console.print(f"✓ Full dataset ({n_samples} samples): [green]{full_path}[/green]")
    console.print(f"✓ Labeled subset ({n_labeled} samples): [green]{labeled_path}[/green]")
    console.print(
        f"✓ Unlabeled subset ({len(unlabeled_dataset)} samples): [green]{unlabeled_path}[/green]"
    )

    console.print(
        "\n[bold green]Bootstrap dataset ready for semi-supervised training![/bold green]"
    )


if __name__ == "__main__":
    app()
