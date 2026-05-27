"""
MLflow Experiment Tracking Demonstration.

This script demonstrates Phase 2.1 functionality:
- Training multiple models with different hyperparameters
- Automatic experiment tracking with MLflow
- Querying best runs
- Comparing Champion vs Challenger models
- Launching MLflow UI

Usage:
    python scripts/demo_mlflow_tracking.py

Requirements:
    - Training data must exist (run dvc repro first)
    - MLflow installed

View results:
    mlflow ui
    # Then open http://localhost:5000
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from src.training.mlflow_utils import compare_runs, get_best_run
from src.training.train import train_model

console = Console()


def main():
    """Run MLflow tracking demonstration."""
    console.print("\n[bold blue]═══ MLflow Experiment Tracking Demo ═══[/bold blue]\n")

    # Define hyperparameter configurations to test
    configs = [
        {"name": "Baseline", "C": 1.0, "max_iter": 1000},
        {"name": "High Regularization", "C": 0.1, "max_iter": 1000},
        {"name": "Low Regularization", "C": 10.0, "max_iter": 1000},
        {"name": "Extended Training", "C": 1.0, "max_iter": 2000},
        {"name": "Aggressive", "C": 5.0, "max_iter": 1500},
    ]

    # Check if training data exists
    train_data_path = Path("data/processed/train.json")
    test_data_path = Path("data/processed/test.json")

    if not train_data_path.exists():
        console.print(
            "[red]✗ Training data not found. Run 'dvc repro' first.[/red]",
            style="bold",
        )
        print("\nExpected file:", train_data_path)
        return

    console.print(f"[green]✓ Training data found:[/green] {train_data_path}")
    console.print(
        f"[green]✓ Test data found:[/green] {test_data_path}\n" if test_data_path.exists() else ""
    )

    # Train models with different configurations
    run_ids = []
    results_list = []

    console.print("[bold]Training models with different hyperparameters...[/bold]\n")

    for i, config in enumerate(configs, 1):
        console.print(f"[cyan]({i}/{len(configs)})[/cyan] Training: {config['name']}")
        console.print(f"  → C={config['C']}, max_iter={config['max_iter']}")

        model_path = Path(f"models/demo_model_{i}.pkl")

        try:
            results = train_model(
                train_data_path=train_data_path,
                test_data_path=test_data_path if test_data_path.exists() else None,
                model_output_path=model_path,
                model_version=f"demo_v{i}",
                algorithm="logistic_regression",
                use_mlflow=True,
                mlflow_experiment_name="device_health_demo",
                C=config["C"],
                max_iter=config["max_iter"],
            )

            run_ids.append(results["mlflow_run_id"])
            results_list.append(
                {
                    "config": config["name"],
                    "train_acc": results["train_accuracy"],
                    "train_f1": results["train_f1_score"],
                    "test_acc": results.get("test_accuracy", None),
                    "test_f1": results.get("test_f1_score", None),
                    "run_id": results["mlflow_run_id"],
                }
            )

            console.print(f"  [green]✓[/green] Train Accuracy: {results['train_accuracy']:.2%}")
            if "test_accuracy" in results:
                console.print(f"  [green]✓[/green] Test Accuracy: {results['test_accuracy']:.2%}")
            console.print(f"  [dim]Run ID: {results['mlflow_run_id'][:8]}...[/dim]\n")

        except Exception as e:
            console.print(f"  [red]✗ Training failed: {e}[/red]\n")
            continue

    if not results_list:
        console.print("[red]No models trained successfully. Exiting.[/red]")
        return

    # Display results table
    console.print("\n[bold blue]═══ Training Results ═══[/bold blue]\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Configuration", style="white", width=20)
    table.add_column("Train Acc", justify="right", style="green")
    table.add_column("Train F1", justify="right", style="green")
    table.add_column("Test Acc", justify="right", style="yellow")
    table.add_column("Test F1", justify="right", style="yellow")
    table.add_column("Run ID", style="dim")

    for result in results_list:
        table.add_row(
            result["config"],
            f"{result['train_acc']:.2%}",
            f"{result['train_f1']:.4f}",
            f"{result['test_acc']:.2%}" if result["test_acc"] else "N/A",
            f"{result['test_f1']:.4f}" if result["test_f1"] else "N/A",
            result["run_id"][:12] + "...",
        )

    console.print(table)

    # Query best run
    console.print("\n[bold blue]═══ Best Run Query ═══[/bold blue]\n")

    metric_name = "test_accuracy" if test_data_path.exists() else "train_accuracy"
    best_run = get_best_run("device_health_demo", metric_name=metric_name)

    if best_run:
        console.print(f"[green]✓ Best run found[/green] (by {metric_name}):")
        console.print(f"  Run ID: {best_run['run_id']}")
        console.print(f"  Accuracy: {best_run['metrics'].get(metric_name, 0):.2%}")
        console.print(
            f"  Parameters: C={best_run['params'].get('C')}, "
            f"max_iter={best_run['params'].get('max_iter')}"
        )
    else:
        console.print("[yellow]No runs found in experiment.[/yellow]")

    # Compare Champion vs Challenger
    if len(run_ids) >= 2:
        console.print("\n[bold blue]═══ Champion vs Challenger Comparison ═══[/bold blue]\n")

        champion_id = run_ids[0]  # Baseline
        challenger_id = run_ids[2] if len(run_ids) > 2 else run_ids[1]  # Low reg or High reg

        console.print(f"[cyan]Champion:[/cyan] {results_list[0]['config']}")
        console.print(
            f"[cyan]Challenger:[/cyan] {results_list[2 if len(run_ids) > 2 else 1]['config']}"
        )

        comparison = compare_runs(champion_id, challenger_id)

        # Display metric differences
        console.print("\n[bold]Metric Differences[/bold] (Challenger - Champion):")

        comparison_table = Table(show_header=True, header_style="bold")
        comparison_table.add_column("Metric", style="white")
        comparison_table.add_column("Champion", justify="right", style="cyan")
        comparison_table.add_column("Challenger", justify="right", style="magenta")
        comparison_table.add_column("Difference", justify="right")

        for metric, diff in comparison["metric_diff"].items():
            if metric in ["train_accuracy", "test_accuracy", "train_f1_score", "test_f1_score"]:
                champion_val = comparison["run_1"]["metrics"].get(metric, 0)
                challenger_val = comparison["run_2"]["metrics"].get(metric, 0)

                # Color code the difference
                if diff > 0.01:
                    diff_style = "green"
                    diff_symbol = "↑"
                elif diff < -0.01:
                    diff_style = "red"
                    diff_symbol = "↓"
                else:
                    diff_style = "white"
                    diff_symbol = "→"

                comparison_table.add_row(
                    metric,
                    f"{champion_val:.4f}",
                    f"{challenger_val:.4f}",
                    f"[{diff_style}]{diff_symbol} {diff:+.4f}[/{diff_style}]",
                )

        console.print(comparison_table)

        # Parameter differences
        if comparison["param_diff"]:
            console.print("\n[bold]Parameter Differences:[/bold]")
            for param, (val1, val2) in comparison["param_diff"].items():
                console.print(f"  {param}: {val1} → {val2}")

    # Instructions for viewing MLflow UI
    console.print("\n[bold blue]═══ View Results in MLflow UI ═══[/bold blue]\n")
    console.print("Launch the MLflow UI to explore experiments interactively:\n")
    console.print("  [bold cyan]mlflow ui[/bold cyan]")
    console.print("\nThen open: [link]http://localhost:5000[/link]\n")
    console.print("[dim]Filter by experiment name: 'device_health_demo'[/dim]\n")

    console.print("[green]✓ Demo complete![/green]")
    console.print(f"\nTrained {len(run_ids)} models with MLflow tracking.")
    console.print("Experiment: device_health_demo")
    console.print(f"Tracking URI: {Path('mlruns').absolute()}\n")


if __name__ == "__main__":
    main()
