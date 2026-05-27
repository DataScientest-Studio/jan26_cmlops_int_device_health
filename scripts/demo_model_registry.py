"""
MLflow Model Registry & Promotion Demonstration.

This script demonstrates Phase 2.2 functionality:
- Training multiple models
- Registering models in MLflow Model Registry
- Promoting models through stages (None → Staging → Production)
- Comparing Champion (Production) vs Challenger (Staging)
- Archiving old models

Workflow:
1. Train baseline model → Register → Promote to Production (Champion)
2. Train challenger models with different hyperparameters
3. Register challengers → Promote to Staging
4. Compare metrics: Champion vs Challengers
5. Promote best challenger to Production
6. Archive old champion

Usage:
    python scripts/demo_model_registry.py

Requirements:
    - Training data must exist (run dvc repro first)
    - MLflow installed

View results:
    mlflow ui
    # Then open http://localhost:5000 and click "Models" tab
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from src.training.registry import (
    get_latest_model_version,
    get_production_models,
    get_staging_models,
    list_registered_models,
    promote_model,
    register_model,
)
from src.training.train import train_model

console = Console()


def main():
    """Run model registry demonstration."""
    console.print("\n[bold blue]═══ MLflow Model Registry Demo ═══[/bold blue]\n")

    model_name = "device_health_classifier"

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
    if test_data_path.exists():
        console.print(f"[green]✓ Test data found:[/green] {test_data_path}\n")
    else:
        console.print("[yellow]⚠ Test data not found (continuing with train only)[/yellow]\n")

    # Step 1: Train and register baseline (Champion) model
    console.print("[bold cyan]Step 1: Train Baseline Model (Champion)[/bold cyan]\n")

    baseline_model_path = Path("models/baseline_champion.pkl")
    console.print("Training baseline model (C=1.0, max_iter=1000)...")

    baseline_results = train_model(
        train_data_path=train_data_path,
        test_data_path=test_data_path if test_data_path.exists() else None,
        model_output_path=baseline_model_path,
        model_version="baseline_v1",
        algorithm="logistic_regression",
        use_mlflow=True,
        mlflow_experiment_name="model_registry_demo",
        C=1.0,
        max_iter=1000,
    )

    console.print(f"[green]✓ Trained:[/green] Accuracy = {baseline_results['train_accuracy']:.2%}")
    if "test_accuracy" in baseline_results:
        console.print(f"[green]  Test Accuracy:[/green] {baseline_results['test_accuracy']:.2%}")

    # Register baseline model
    console.print("\nRegistering baseline model...")
    baseline_version = register_model(
        run_id=baseline_results["mlflow_run_id"],
        model_name=model_name,
        description="Baseline model with C=1.0",
    )
    console.print(f"[green]✓ Registered:[/green] {model_name} v{baseline_version}")

    # Promote to Production (Champion)
    console.print("\nPromoting baseline to Production...")
    promote_model(model_name, baseline_version, stage="Production")
    console.print(f"[green]✓ Promoted:[/green] v{baseline_version} → Production (Champion)\n")

    # Step 2: Train challenger models
    console.print("[bold cyan]Step 2: Train Challenger Models[/bold cyan]\n")

    challengers = [
        {"name": "High Regularization", "C": 0.1, "max_iter": 1000},
        {"name": "Low Regularization", "C": 10.0, "max_iter": 1000},
        {"name": "Extended Training", "C": 1.0, "max_iter": 2000},
    ]

    challenger_versions = []

    for i, config in enumerate(challengers, 1):
        console.print(f"[cyan]({i}/{len(challengers)})[/cyan] Training: {config['name']}")
        console.print(f"  Parameters: C={config['C']}, max_iter={config['max_iter']}")

        model_path = Path(f"models/challenger_{i}.pkl")

        results = train_model(
            train_data_path=train_data_path,
            test_data_path=test_data_path if test_data_path.exists() else None,
            model_output_path=model_path,
            model_version=f"challenger_v{i}",
            algorithm="logistic_regression",
            use_mlflow=True,
            mlflow_experiment_name="model_registry_demo",
            C=config["C"],
            max_iter=config["max_iter"],
        )

        console.print(f"  [green]✓ Trained:[/green] Accuracy = {results['train_accuracy']:.2%}")
        if "test_accuracy" in results:
            console.print(f"    Test Accuracy: {results['test_accuracy']:.2%}")

        # Register and promote to Staging
        version = register_model(
            run_id=results["mlflow_run_id"],
            model_name=model_name,
            description=f"{config['name']} (C={config['C']})",
        )
        promote_model(model_name, version, stage="Staging")

        console.print(f"  [green]✓ Registered:[/green] v{version} → Staging")
        challenger_versions.append(version)
        console.print()

    # Step 3: Compare Champion vs Challengers
    console.print("[bold cyan]Step 3: Champion vs Challenger Comparison[/bold cyan]\n")

    # Get champion
    production_models = get_production_models(model_name)
    if not production_models:
        console.print("[red]✗ No production model found[/red]")
        return

    champion = production_models[0]
    console.print(f"[bold green]Champion:[/bold green] v{champion['version']} (Production)")

    # Get challengers
    staging_models = get_staging_models(model_name)
    console.print(
        f"[bold yellow]Challengers:[/bold yellow] {len(staging_models)} models in Staging\n"
    )

    # Create comparison table
    metric_key = "test_accuracy" if test_data_path.exists() else "train_accuracy"

    table = Table(show_header=True, header_style="bold cyan", title="Model Comparison")
    table.add_column("Version", style="white", width=8)
    table.add_column("Stage", style="white", width=12)
    table.add_column("Accuracy", justify="right", style="green")
    table.add_column("F1 Score", justify="right", style="yellow")
    table.add_column("C", justify="right")
    table.add_column("Max Iter", justify="right")

    # Champion row
    champion_accuracy = champion["metrics"].get(metric_key, 0)
    champion_f1 = champion["metrics"].get(f"{metric_key.replace('accuracy', 'f1_score')}", 0)
    table.add_row(
        f"v{champion['version']}",
        "🏆 Production",
        f"{champion_accuracy:.2%}",
        f"{champion_f1:.4f}",
        champion["params"].get("C", "N/A"),
        champion["params"].get("max_iter", "N/A"),
    )

    # Challenger rows
    best_challenger = None
    best_challenger_accuracy = 0

    for challenger in staging_models:
        challenger_accuracy = challenger["metrics"].get(metric_key, 0)
        challenger_f1 = challenger["metrics"].get(
            f"{metric_key.replace('accuracy', 'f1_score')}", 0
        )

        # Track best challenger
        if challenger_accuracy > best_challenger_accuracy:
            best_challenger = challenger
            best_challenger_accuracy = challenger_accuracy

        # Highlight if better than champion
        emoji = "🔥" if challenger_accuracy > champion_accuracy else "🔷"

        table.add_row(
            f"v{challenger['version']}",
            f"{emoji} Staging",
            f"{challenger_accuracy:.2%}",
            f"{challenger_f1:.4f}",
            challenger["params"].get("C", "N/A"),
            challenger["params"].get("max_iter", "N/A"),
        )

    console.print(table)

    # Step 4: Promote best challenger if better than champion
    console.print("\n[bold cyan]Step 4: Model Promotion Decision[/bold cyan]\n")

    if best_challenger and best_challenger_accuracy > champion_accuracy:
        improvement = best_challenger_accuracy - champion_accuracy
        console.print(
            f"[green]✓ Best challenger (v{best_challenger['version']}) "
            f"outperforms champion by {improvement:.2%}[/green]"
        )
        console.print("\nPromoting challenger to Production and archiving old champion...")

        promote_model(
            model_name,
            best_challenger["version"],
            stage="Production",
            archive_existing_production=True,
        )

        console.print(f"[green]✓ Promoted:[/green] v{best_challenger['version']} → Production")
        console.print(f"[yellow]✓ Archived:[/yellow] v{champion['version']} (old champion)\n")

    else:
        console.print(f"[yellow]→ Champion (v{champion['version']}) remains best model[/yellow]")
        console.print("  No promotion needed.\n")

    # Step 5: Display final registry state
    console.print("[bold cyan]Step 5: Final Model Registry State[/bold cyan]\n")

    models = list_registered_models()
    registry_table = Table(show_header=True, header_style="bold")
    registry_table.add_column("Model Name", style="white")
    registry_table.add_column("Latest Version", justify="center")
    registry_table.add_column("Production", justify="center", style="green")
    registry_table.add_column("Staging", justify="center", style="yellow")

    for model in models:
        if model["name"] == model_name:
            registry_table.add_row(
                model["name"],
                str(model["latest_version"]),
                f"v{model['production_versions']}" if model["production_versions"] else "-",
                f"v{model['staging_versions']}" if model["staging_versions"] else "-",
            )

    console.print(registry_table)

    # Final instructions
    console.print("\n[bold blue]═══ View in MLflow UI ═══[/bold blue]\n")
    console.print("Launch the MLflow UI to explore the Model Registry:\n")
    console.print("  [bold cyan]mlflow ui[/bold cyan]")
    console.print("\nThen open: [link]http://localhost:5000[/link]")
    console.print("Click: [bold]Models[/bold] tab → [bold]device_health_classifier[/bold]\n")

    console.print("[green]✓ Demo complete![/green]")
    console.print(f"\nRegistered {get_latest_model_version(model_name)} model versions")
    console.print(f"Model name: {model_name}\n")


if __name__ == "__main__":
    main()
