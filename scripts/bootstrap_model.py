#!/usr/bin/env python3
"""
Bootstrap model training script using semi-supervised learning.

Implements Bootstrap Scenario B:
1. Load 20% labeled + 80% unlabeled samples
2. Combine into single dataset for semi-supervised training
3. Call production training script (src/training/train.py) with allow_unlabeled=True
4. Evaluate on held-out test set using production evaluation

REFACTORED: Now calls train_model() from src/training/train.py to eliminate code duplication.
"""

import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path so 'src' is importable when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))
import typer
from rich.console import Console
from rich.table import Table

from src.training.train import evaluate_model, train_model

app = typer.Typer(help="Train bootstrap model using semi-supervised learning")
console = Console()


def combine_json_datasets(
    labeled_path: Path,
    unlabeled_path: Path,
    output_path: Path,
) -> None:
    """
    Combine labeled and unlabeled JSON datasets into single file.

    Args:
        labeled_path: Path to labeled dataset JSON
        unlabeled_path: Path to unlabeled dataset JSON (can be missing)
        output_path: Path to save combined dataset

    The combined dataset will have:
    - Labeled signals with their ground truth labels
    - Unlabeled signals with label=null (will be converted to -1 during loading)
    """
    # Load labeled data
    with open(labeled_path) as f:
        labeled_data = json.load(f)

    # Load unlabeled data (if exists)
    unlabeled_signals = []
    if unlabeled_path.exists():
        with open(unlabeled_path) as f:
            unlabeled_data = json.load(f)
            unlabeled_signals = unlabeled_data.get("signals", [])

    # Combine signals
    combined_signals = []

    # Add labeled signals (keep labels)
    for signal in labeled_data["signals"]:
        combined_signals.append(signal)

    # Add unlabeled signals (set label to null for semi-supervised)
    for signal in unlabeled_signals:
        signal_copy = signal.copy()
        signal_copy["label"] = None  # Mark as unlabeled
        combined_signals.append(signal_copy)

    # Create combined dataset
    combined_data = {
        "n_samples": len(combined_signals),
        "signals": combined_signals,
    }

    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(combined_data, f, indent=2)

    console.print(f"✓ Combined dataset saved to {output_path}")
    console.print(f"  Total signals: {len(combined_signals)}")
    console.print(f"  Labeled: {len(labeled_data['signals'])}")
    console.print(f"  Unlabeled: {len(unlabeled_signals)}")


@app.command()
def train(
    labeled_data: Path = typer.Option(
        Path("data/raw/bootstrap_labeled.json"), help="Path to labeled dataset JSON"
    ),
    unlabeled_data: Path = typer.Option(
        Path("data/raw/bootstrap_unlabeled.json"), help="Path to unlabeled dataset JSON"
    ),
    test_data: Path = typer.Option(
        Path("data/raw/dataset_baseline_test.json"), help="Path to test dataset JSON"
    ),
    model_output: Path = typer.Option(
        Path("models/bootstrap_model.pkl"), help="Output path for trained model"
    ),
    k_range_min: int = typer.Option(2, help="Minimum K for optimal K search"),
    k_range_max: int = typer.Option(8, help="Maximum K for optimal K search"),
    k_method: str = typer.Option(
        "silhouette", help="K optimization method: silhouette, elbow, or calinski"
    ),
    seed: int = typer.Option(42, help="Random seed"),
    mlflow_tracking: bool = typer.Option(True, help="Enable MLflow tracking"),
) -> None:
    """
    Train bootstrap model using production-grade semi-supervised learning.

    This script now calls train_model() from src/training/train.py to eliminate code duplication.

    Bootstrap Scenario B workflow:
        1. Combine labeled (20%) and unlabeled (80%) datasets
        2. Call train_model() with allow_unlabeled=True and filter_unlabeled=False
        3. train_model() handles:
            - Feature extraction
            - Optimal K selection via silhouette/elbow/calinski
            - K-Means clustering with label propagation
            - Unlabeled cluster handling with intelligent heuristics
            - Model training with propagated labels
            - Gold standard test set evaluation
        4. Evaluate on separate test set

    Example:
        python scripts/bootstrap_model.py train
        python scripts/bootstrap_model.py train --k-range-max 10 --k-method elbow
    """
    console.print("\n[bold blue]Bootstrap Model Training - Scenario B (Refactored)[/bold blue]\n")

    # Step 1: Combine labeled and unlabeled datasets
    console.print("[cyan]Step 1:[/cyan] Combining datasets...")
    combined_data_path = Path("data/processed/bootstrap_combined.json")
    combine_json_datasets(labeled_data, unlabeled_data, combined_data_path)
    console.print()

    # Step 2: Train model using production training script
    console.print("[cyan]Step 2:[/cyan] Training model with semi-supervised learning...")
    console.print("  Calling train_model() from src/training/train.py")
    console.print(f"  K range: [{k_range_min}, {k_range_max}]")
    console.print(f"  K method: {k_method}")
    console.print()

    results = train_model(
        train_data_path=combined_data_path,
        model_output_path=model_output,
        model_version="bootstrap_v1.0",
        use_mlflow=mlflow_tracking,
        mlflow_experiment_name="bootstrap_training",
        # Semi-supervised parameters
        allow_unlabeled=True,  # Allow loading unlabeled samples
        filter_unlabeled=False,  # Keep unlabeled for clustering
        k_range=(k_range_min, k_range_max),
        k_method=k_method,
        distance_threshold=2.0,
        knn_neighbors=5,
        use_domain_heuristics=True,
        # No sliding window - use all data for bootstrap
        window_size=None,
        window_days=None,
        # Test split
        test_size=0.2,
        stratify=True,
        primary_metric="f1_score",
        # Model parameters
        max_iter=1000,
        random_state=seed,
    )

    # Step 3: Display results
    console.print("\n[bold green]Training Complete![/bold green]\n")

    console.print("[bold]Training Results:[/bold]")
    console.print(f"  Model version: {results['model_version']}")
    console.print(f"  Model saved to: {results['model_path']}")
    console.print(f"  Optimal K: {results.get('optimal_k', 'N/A')}")
    console.print(f"  Training samples: {results['train_samples']}")
    console.print(f"  Test samples: {results['test_samples']}")
    console.print()

    console.print("[bold]Performance Metrics:[/bold]")
    console.print(f"  Train Accuracy: {results['train_accuracy']:.2%}")
    console.print(f"  Train F1 Score: {results['train_f1_score']:.4f}")
    console.print(f"  [bold cyan]Test Accuracy: {results['test_accuracy']:.2%}[/bold cyan]")
    console.print(f"  [bold cyan]Test F1 Score: {results['test_f1_score']:.4f}[/bold cyan]")
    console.print()

    # Display confusion matrix if available
    if "confusion_matrix" in results:
        cm = results["confusion_matrix"]
        console.print("[bold]Confusion Matrix:[/bold]")
        table = Table(title="Actual vs Predicted")
        table.add_column("", style="cyan")
        table.add_column("Pred: Healthy (0)", style="green")
        table.add_column("Pred: Unhealthy (1)", style="red")

        table.add_row("Actual: Healthy (0)", str(cm[0][0]), str(cm[0][1]))
        table.add_row("Actual: Unhealthy (1)", str(cm[1][0]), str(cm[1][1]))

        console.print(table)
        console.print()

    # Display classification report if available
    if "classification_report" in results:
        console.print("[bold]Classification Report:[/bold]")
        console.print(results["classification_report"])
        console.print()

    # Step 4: Evaluate on external test set (optional)
    if test_data.exists():
        console.print("[cyan]Step 3:[/cyan] Evaluating on external test set...")
        test_results = evaluate_model(model_output, test_data)

        console.print("\n[bold]External Test Set Results:[/bold]")
        console.print(f"  Test samples: {test_results['test_samples']}")
        console.print(f"  Test accuracy: {test_results['test_accuracy']:.2%}")
        console.print()

    console.print("[bold]Bootstrap training complete! 🎉[/bold]\n")


if __name__ == "__main__":
    app()
