#!/usr/bin/env python3
"""
Drift simulation script for MLOps demonstration scenarios.

This script provides convenient wrappers for generating different drift
scenarios to showcase monitoring and retraining capabilities.

Use Cases:
    - Data Drift: Sensor degradation (increased noise, shifted centers)
    - Concept Drift: Process changes (boundary shifts)
    - Gradual Drift: Slow parameter changes over time
    - Sudden Drift: Abrupt distributional shift

Examples:
    # Generate data drift scenario
    python scripts/simulate_drift.py data-drift --n-samples 500

    # Generate concept drift scenario
    python scripts/simulate_drift.py concept-drift --n-samples 500

    # Generate gradual drift progression
    python scripts/simulate_drift.py gradual --n-samples 1000 --n-stages 5
"""

import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Literal

# Ensure repo root is on sys.path so 'src' is importable when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import typer
from rich.console import Console
from rich.progress import track
from rich.table import Table

from src.signal_processing.signal_generator import LabeledSignal, generate_signal

app = typer.Typer(help="Drift simulation for MLOps demonstrations")
console = Console()


def _send_signals_to_api(signals: list[LabeledSignal]) -> tuple[int, int]:
    """
    POST each signal to the /predict endpoint.

    Uses API_URL env var (default: http://api:8000) and X-API-Key header
    (key from API_KEY env var; defaults to dev-key-12345 for local/dev).

    Returns:
        (successes, failures) tuple
    """
    api_url = os.environ.get("API_URL", "http://localhost:80").rstrip("/")
    api_key = os.environ.get("API_KEY", "dev-key-12345")
    _headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key,
    }

    successes = 0
    failures = 0
    for i, signal in enumerate(signals):
        payload = json.dumps(
            {
                "time_values": signal.signal.time,
                "amplitude_values": signal.signal.amplitude,
                "device_id": f"drift-sim-{i:04d}",
            }
        ).encode()

        try:
            req = urllib.request.Request(
                f"{api_url}/predict",
                data=payload,
                headers=_headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                if resp.status in (200, 201):
                    successes += 1
                else:
                    failures += 1
        except Exception:
            failures += 1

    return successes, failures


def save_dataset(
    signals: list[LabeledSignal],
    output_path: Path,
    include_labels: bool = True,
    metadata: dict | None = None,
) -> None:
    """Save signals to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "metadata": metadata or {},
        "signals": [
            {
                "time": signal.signal.time,
                "amplitude": signal.signal.amplitude,
                "shape_type": signal.signal.shape_type,
                "label": signal.label if include_labels else None,
                **signal.metadata,
            }
            for signal in signals
        ],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


@app.command()
def data_drift(
    n_samples: int = typer.Option(500, help="Number of signals to generate"),
    output_dir: Path = typer.Option(Path("data/drift/data_drift"), help="Output directory"),
    seed: int = typer.Option(42, help="Random seed"),
    send_to_api: bool = typer.Option(
        False, "--send-to-api/--no-send-to-api", help="POST signals to /predict"
    ),
) -> None:
    """
    Simulate data drift scenario: sensor degradation.

    Characteristics:
        - Peak centers shift to [35, 42] (outside healthy range)
        - Noise increases to [0.08, 0.12] (8-12% vs baseline 1-2%)
        - Wider peaks (degraded resolution)
        - Lower heights (reduced signal strength)

    Use Case:
        Demonstrate EvidentlyAI detecting feature distribution shifts
        even before model accuracy drops significantly.
    """
    console.print("[bold cyan]Simulating Data Drift Scenario[/bold cyan]")
    console.print("Scenario: Sensor degradation with shifted centers and high noise\n")

    np.random.seed(seed)
    signals = []

    for i in track(range(n_samples), description="Generating signals..."):
        # 50/50 Gaussian/Lorentzian
        shape_type: Literal["gaussian", "lorentzian"] = "gaussian" if i % 2 == 0 else "lorentzian"

        signal = generate_signal(
            shape_type=shape_type,
            drift_scenario="data_drift",
            seed=seed + i,
        )
        signals.append(signal)

    # Save dataset
    save_dataset(
        signals,
        output_dir / "drift_signals.json",
        metadata={
            "drift_type": "data_drift",
            "n_samples": n_samples,
            "description": "Sensor degradation: shifted centers, high noise",
            "seed": seed,
        },
    )

    # Print statistics
    _print_drift_statistics(signals, "Data Drift")
    console.print(f"\n✓ Saved to: {output_dir / 'drift_signals.json'}")

    if send_to_api:
        console.print("\n[cyan]Sending signals to API /predict ...[/cyan]")
        ok, fail = _send_signals_to_api(signals)
        console.print(f"✓ API predictions: {ok} succeeded, {fail} failed")


@app.command()
def concept_drift(
    n_samples: int = typer.Option(500, help="Number of signals to generate"),
    output_dir: Path = typer.Option(Path("data/drift/concept_drift"), help="Output directory"),
    seed: int = typer.Option(42, help="Random seed"),
    send_to_api: bool = typer.Option(
        False, "--send-to-api/--no-send-to-api", help="POST signals to /predict"
    ),
) -> None:
    """
    Simulate concept drift scenario: process boundary changes.

    Characteristics:
        - Parameter ranges shift toward healthy/unhealthy boundary
        - Gaussian: μ∈[46,54] (was [48,52]), σ∈[2.5,4.0] (was [2,3])
        - Lorentzian: μ∈[45,55] (was [42,47]∪[53,58])
        - Increased classification ambiguity

    Use Case:
        Demonstrate sparse label audit revealing accuracy degradation
        when prediction distribution remains stable but ground truth shifts.
    """
    console.print("[bold cyan]Simulating Concept Drift Scenario[/bold cyan]")
    console.print("Scenario: Process change blurring healthy/unhealthy boundary\n")

    np.random.seed(seed)
    signals = []

    for i in track(range(n_samples), description="Generating signals..."):
        shape_type: Literal["gaussian", "lorentzian"] = "gaussian" if i % 2 == 0 else "lorentzian"

        signal = generate_signal(
            shape_type=shape_type,
            drift_scenario="concept_drift",
            seed=seed + i,
        )
        signals.append(signal)

    save_dataset(
        signals,
        output_dir / "drift_signals.json",
        metadata={
            "drift_type": "concept_drift",
            "n_samples": n_samples,
            "description": "Process boundary shift: increased classification ambiguity",
            "seed": seed,
        },
    )

    _print_drift_statistics(signals, "Concept Drift")
    console.print(f"\n✓ Saved to: {output_dir / 'drift_signals.json'}")

    if send_to_api:
        console.print("\n[cyan]Sending signals to API /predict ...[/cyan]")
        ok, fail = _send_signals_to_api(signals)
        console.print(f"✓ API predictions: {ok} succeeded, {fail} failed")


@app.command()
def gradual(
    n_samples: int = typer.Option(1000, help="Total signals to generate"),
    n_stages: int = typer.Option(5, help="Number of drift stages"),
    output_dir: Path = typer.Option(Path("data/drift/gradual"), help="Output directory"),
    seed: int = typer.Option(42, help="Random seed"),
    send_to_api: bool = typer.Option(
        False, "--send-to-api/--no-send-to-api", help="POST all generated signals to /predict"
    ),
) -> None:
    """
    Simulate gradual drift progression over time.

    Generates N stages transitioning from baseline → data_drift:
        - Stage 1: Pure baseline (healthy signals)
        - Stage 2-4: Gradual parameter shifts
        - Stage 5: Full data drift (degraded signals)

    Use Case:
        Demonstrate monitoring detecting gradual degradation over days/weeks.
        Useful for testing drift detection threshold tuning.
    """
    console.print("[bold cyan]Simulating Gradual Drift Progression[/bold cyan]")
    console.print(f"Generating {n_stages} stages transitioning baseline → data_drift\n")

    np.random.seed(seed)
    samples_per_stage = n_samples // n_stages
    all_signals = []

    for stage in range(n_stages):
        console.print(f"[bold]Stage {stage + 1}/{n_stages}[/bold]")

        # Interpolate drift intensity
        drift_ratio = stage / (n_stages - 1) if n_stages > 1 else 0

        stage_signals = []
        for i in range(samples_per_stage):
            shape_type: Literal["gaussian", "lorentzian"] = (
                "gaussian" if i % 2 == 0 else "lorentzian"
            )

            # Mix baseline and drift parameters
            drift_scenario: Literal["baseline", "data_drift", "concept_drift"]
            if drift_ratio == 0:
                drift_scenario = "baseline"
            elif drift_ratio == 1:
                drift_scenario = "data_drift"
            else:
                drift_scenario = "concept_drift"

            signal = generate_signal(
                shape_type=shape_type,
                drift_scenario=drift_scenario,
                seed=seed + stage * 1000 + i,
            )

            # Add stage metadata
            signal.metadata["drift_stage"] = stage + 1
            signal.metadata["drift_ratio"] = drift_ratio

            stage_signals.append(signal)

        all_signals.extend(stage_signals)

        # Save stage dataset
        save_dataset(
            stage_signals,
            output_dir / f"stage_{stage + 1:02d}.json",
            metadata={
                "drift_type": "gradual",
                "stage": stage + 1,
                "total_stages": n_stages,
                "drift_ratio": drift_ratio,
                "n_samples": len(stage_signals),
            },
        )

        console.print(f"  Saved {len(stage_signals)} signals to stage_{stage + 1:02d}.json")

    # Save combined dataset
    save_dataset(
        all_signals,
        output_dir / "all_stages.json",
        metadata={
            "drift_type": "gradual",
            "n_stages": n_stages,
            "total_samples": len(all_signals),
            "description": f"Gradual drift progression over {n_stages} stages",
        },
    )

    console.print(f"\n✓ Saved {len(all_signals)} total signals to {output_dir}")

    if send_to_api:
        console.print("\n[cyan]Sending all stages to API...[/cyan]")
        ok, fail = _send_signals_to_api(all_signals)
        console.print(f"✓ API predictions: {ok} succeeded, {fail} failed")


@app.command()
def sudden(
    n_baseline: int = typer.Option(300, help="Baseline signals before shift"),
    n_drifted: int = typer.Option(200, help="Drifted signals after shift"),
    output_dir: Path = typer.Option(Path("data/drift/sudden"), help="Output directory"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """
    Simulate sudden drift: abrupt distributional shift.

    Generates two distinct datasets:
        - Baseline: Normal healthy/unhealthy distribution
        - Drifted: Abrupt shift to degraded parameters

    Use Case:
        Demonstrate rapid drift detection when system behavior changes overnight
        (e.g., equipment replacement, configuration change).
    """
    console.print("[bold cyan]Simulating Sudden Drift Scenario[/bold cyan]")
    console.print("Scenario: Abrupt shift from baseline to degraded distribution\n")

    np.random.seed(seed)

    # Generate baseline signals
    console.print(f"[bold]Phase 1:[/bold] Generating {n_baseline} baseline signals...")
    baseline_signals = []
    for i in track(range(n_baseline), description="Baseline"):
        shape_type: Literal["gaussian", "lorentzian"] = "gaussian" if i % 2 == 0 else "lorentzian"
        signal = generate_signal(
            shape_type=shape_type,
            drift_scenario="baseline",
            seed=seed + i,
        )
        signal.metadata["phase"] = "baseline"
        baseline_signals.append(signal)

    # Generate drifted signals
    console.print(f"\n[bold]Phase 2:[/bold] Generating {n_drifted} drifted signals...")
    drifted_signals = []
    for i in track(range(n_drifted), description="Drifted"):
        shape_type = "gaussian" if i % 2 == 0 else "lorentzian"
        signal = generate_signal(
            shape_type=shape_type,
            drift_scenario="data_drift",
            seed=seed + 1000 + i,
        )
        signal.metadata["phase"] = "drifted"
        drifted_signals.append(signal)

    # Save datasets
    save_dataset(
        baseline_signals,
        output_dir / "baseline.json",
        metadata={
            "drift_type": "sudden",
            "phase": "baseline",
            "n_samples": len(baseline_signals),
        },
    )

    save_dataset(
        drifted_signals,
        output_dir / "drifted.json",
        metadata={
            "drift_type": "sudden",
            "phase": "drifted",
            "n_samples": len(drifted_signals),
        },
    )

    all_signals = baseline_signals + drifted_signals
    save_dataset(
        all_signals,
        output_dir / "combined.json",
        metadata={
            "drift_type": "sudden",
            "n_baseline": n_baseline,
            "n_drifted": n_drifted,
            "total_samples": len(all_signals),
        },
    )

    _print_drift_statistics(baseline_signals, "Baseline")
    _print_drift_statistics(drifted_signals, "Drifted")
    console.print(f"\n✓ Saved to: {output_dir}")


def _print_drift_statistics(signals: list[LabeledSignal], label: str) -> None:
    """Print statistics for a signal dataset."""
    n_total = len(signals)
    n_healthy = sum(1 for s in signals if s.label == 0)
    n_unhealthy = n_total - n_healthy

    n_gaussian = sum(1 for s in signals if s.signal.shape_type == "gaussian")
    n_lorentzian = n_total - n_gaussian

    table = Table(title=f"{label} Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="magenta")
    table.add_column("Percentage", style="green")

    table.add_row("Total Samples", str(n_total), "100.0%")
    table.add_row("Healthy (label=0)", str(n_healthy), f"{n_healthy / n_total * 100:.1f}%")
    table.add_row("Unhealthy (label=1)", str(n_unhealthy), f"{n_unhealthy / n_total * 100:.1f}%")
    table.add_row("", "", "")
    table.add_row("Gaussian Peaks", str(n_gaussian), f"{n_gaussian / n_total * 100:.1f}%")
    table.add_row("Lorentzian Peaks", str(n_lorentzian), f"{n_lorentzian / n_total * 100:.1f}%")

    console.print(table)


if __name__ == "__main__":
    app()
