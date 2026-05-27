#!/usr/bin/env python3
"""
Automated Champion/Challenger Model Promotion Script.

This script automates the model promotion decision process:
1. Compares challenger models (Staging) against champion (Production)
2. Evaluates performance improvements with statistical rigor
3. Promotes best challenger if it meets criteria
4. Archives old champion model

Usage:
    # Evaluate and promote (if criteria met)
    python scripts/promote_model.py --model-name device_health_classifier

    # Dry run (evaluate only, don't promote)
    python scripts/promote_model.py --model-name device_health_classifier --dry-run

    # Custom thresholds
    python scripts/promote_model.py \\
        --model-name device_health_classifier \\
        --metric test_accuracy \\
        --min-improvement 0.01 \\
        --no-archive

    # Verbose mode
    python scripts/promote_model.py --model-name device_health_classifier --verbose

Configuration:
    Set environment variables to customize behavior:
    - MLFLOW_TRACKING_URI: MLflow server URI (default: file:./mlruns)
    - PROMOTION_MIN_IMPROVEMENT: Minimum improvement threshold (default: 0.005)
    - PROMOTION_METRIC: Metric to optimize (default: test_accuracy)

Exit Codes:
    0: Success (model promoted or no promotion needed)
    1: Error (missing models, invalid arguments, etc.)
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
from rich.console import Console
from rich.table import Table

from src.training.promotion import auto_promote_model
from src.training.registry import (
    get_production_models,
    get_staging_models,
)

console = Console()


def display_model_comparison(
    champion: dict,
    challengers: list[dict],
    metric_name: str,
    min_improvement: float,
) -> None:
    """Display visual comparison of champion vs challengers."""
    console.print("\n[bold cyan]═══ Model Comparison ═══[/bold cyan]\n")

    # Champion info
    console.print(f"[bold green]Champion (Production):[/bold green] v{champion['version']}")
    champion_metric = champion["metrics"].get(metric_name, 0)
    console.print(f"  {metric_name}: {champion_metric:.4f}\n")

    # Challengers table
    table = Table(show_header=True, header_style="bold cyan", title="Challenger Models (Staging)")
    table.add_column("Version", style="white", width=8)
    table.add_column("Metric", justify="right", style="yellow")
    table.add_column("Δ vs Champion", justify="right")
    table.add_column("Δ %", justify="right")
    table.add_column("Status", style="white")

    for challenger in challengers:
        version = challenger["version"]
        challenger_metric = challenger["metrics"].get(metric_name, 0)
        improvement = challenger_metric - champion_metric
        improvement_pct = improvement / champion_metric if champion_metric > 0 else 0

        # Determine status
        if improvement >= min_improvement:
            status = "✓ Eligible"
            status_style = "green"
        elif improvement > 0:
            status = "△ Below threshold"
            status_style = "yellow"
        else:
            status = "✗ Worse"
            status_style = "red"

        delta_style = "green" if improvement >= 0 else "red"

        table.add_row(
            f"v{version}",
            f"{challenger_metric:.4f}",
            f"[{delta_style}]{improvement:+.4f}[/{delta_style}]",
            f"[{delta_style}]{improvement_pct:+.2%}[/{delta_style}]",
            f"[{status_style}]{status}[/{status_style}]",
        )

    console.print(table)
    console.print()


def main():
    """Main script entry point."""
    parser = argparse.ArgumentParser(
        description="Automated model promotion with champion/challenger comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Configuration:")[1],
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Registered model name in MLflow Model Registry",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="test_accuracy",
        help="Metric to optimize (default: test_accuracy)",
    )

    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.005,
        help="Minimum improvement threshold (default: 0.005 = 0.5%%)",
    )

    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Don't archive old champion after promotion",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate only, don't actually promote",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output with detailed metrics",
    )

    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (default: file:./mlruns)",
    )

    args = parser.parse_args()

    # Setup MLflow — default to local file store to avoid network timeouts
    tracking_uri = (
        args.tracking_uri
        or os.environ.get("MLFLOW_TRACKING_URI")
        or f"file:{Path(__file__).resolve().parents[1] / 'mlruns'}"
    )
    mlflow.set_tracking_uri(tracking_uri)

    console.print("\n[bold blue]═══ Automated Model Promotion ═══[/bold blue]\n")
    console.print(f"Model: [cyan]{args.model_name}[/cyan]")
    console.print(f"Metric: [cyan]{args.metric}[/cyan]")
    console.print(f"Min Improvement: [cyan]{args.min_improvement:.2%}[/cyan]")
    console.print(f"Mode: [cyan]{'DRY RUN' if args.dry_run else 'LIVE'}[/cyan]\n")

    # Get current champion
    try:
        production_models = get_production_models(args.model_name)
        if not production_models:
            console.print(
                "[red]✗ No production model found.[/red]\n"
                "Register and promote a baseline model first:\n"
                "  1. Train a model with use_mlflow=True\n"
                "  2. register_model(run_id, model_name)\n"
                "  3. promote_model(model_name, version, stage='Production')\n",
                style="bold",
            )
            return 1

        champion = production_models[0]
    except Exception as e:
        console.print(f"[red]✗ Error getting production model: {e}[/red]")
        return 1

    # Get challengers
    try:
        challengers = get_staging_models(args.model_name)
        if not challengers:
            console.print(
                "[yellow]⚠ No challenger models in Staging.[/yellow]\n"
                "Train and promote models to Staging for evaluation.",
                style="bold",
            )
            return 0
    except Exception as e:
        console.print(f"[red]✗ Error getting staging models: {e}[/red]")
        return 1

    # Display comparison
    if args.verbose:
        display_model_comparison(champion, challengers, args.metric, args.min_improvement)

        # Detailed metrics for each model
        console.print("[bold]Detailed Metrics:[/bold]\n")
        console.print(f"[green]Champion v{champion['version']}:[/green]")
        for metric, value in sorted(champion["metrics"].items()):
            if "accuracy" in metric or "f1" in metric:
                console.print(f"  {metric}: {value:.4f}")
        console.print()

        for challenger in challengers:
            console.print(f"[yellow]Challenger v{challenger['version']}:[/yellow]")
            for metric, value in sorted(challenger["metrics"].items()):
                if "accuracy" in metric or "f1" in metric:
                    champion_val = champion["metrics"].get(metric, 0)
                    diff = value - champion_val
                    console.print(f"  {metric}: {value:.4f} ({diff:+.4f} vs champion)")
            console.print()

    # Run automated promotion
    console.print("[bold]Evaluating promotion...[/bold]\n")

    try:
        result = auto_promote_model(
            model_name=args.model_name,
            metric_name=args.metric,
            min_improvement=args.min_improvement,
            archive_old_champion=not args.no_archive,
            dry_run=args.dry_run,
        )
    except Exception as e:
        console.print(f"[red]✗ Promotion failed: {e}[/red]")
        return 1

    # Display results
    if result["promoted"]:
        console.print("[bold green]✓ Model Promoted![/bold green]\n")
        console.print(f"New Champion: v{result['new_champion_version']}")
        console.print(f"Old Champion: v{result['old_champion_version']}")
        if not args.no_archive:
            console.print(f"  → Archived v{result['old_champion_version']}")

        decision = result["decision"]
        console.print(
            f"\nImprovement: {decision['improvement']:+.4f} ({decision['improvement_pct']:+.2%})"
        )
        console.print(f"Reason: {decision['reason']}\n")

        # Success message
        console.print(
            "[dim]View updated registry:[/dim]\n"
            "  mlflow ui  # Open http://localhost:5000, click Models tab\n"
        )

    else:
        console.print("[yellow]→ No Promotion[/yellow]\n")
        console.print(f"Reason: {result['reason']}\n")

        if args.dry_run and result["new_champion_version"]:
            console.print(
                f"[dim]Dry run complete. Would promote v{result['new_champion_version']}[/dim]\n"
            )
        else:
            console.print(
                "[dim]Current champion remains in production.[/dim]\n"
                "Tips:\n"
                "  - Train models with better hyperparameters\n"
                "  - Lower --min-improvement threshold\n"
                "  - Add more training data\n"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
