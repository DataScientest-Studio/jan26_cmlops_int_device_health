#!/usr/bin/env python3
"""
Model smoke test script for deployment validation.

Critical pre-deployment checks to prevent regressions:
    1. Perfect Gaussian (noise=0) → Must predict Healthy
    2. Heavily degraded Lorentzian (noise>0.08, μ<42) → Must predict Unhealthy
    3. Boundary cases (μ=48.0 vs μ=47.9) → Test decision boundaries
    4. Confidence calibration → No extreme predictions (0.0/1.0)
    5. Feature extraction robustness → Handle edge cases gracefully

Use Cases:
    - CI/CD pipeline: Abort deployment if smoke tests fail
    - Model registry validation: Only promote models passing tests
    - Rollback detection: Identify degraded models in production

Examples:
    # Run all smoke tests on a model
    python scripts/smoke_test_model.py --model models/bootstrap_model.pkl

    # Run specific test suite
    python scripts/smoke_test_model.py --model models/bootstrap_model.pkl --suite edge-cases

    # Strict mode (exit code 1 on any failure)
    python scripts/smoke_test_model.py --model models/bootstrap_model.pkl --strict
"""

import pickle
import sys
from pathlib import Path
from typing import Literal

# Ensure repo root is on sys.path so 'src' is importable when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from src.signal_processing.feature_extractor import extract_features
from src.signal_processing.signal_generator import (
    add_gaussian_noise,
    create_time_array,
    generate_gaussian_peak,
    generate_lorentzian_peak,
)
from src.signal_processing.signal_models import SignalData

app = typer.Typer(help="Model smoke tests for deployment validation")
console = Console()


class SmokeTestResult:
    """Container for smoke test results."""

    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.failures: list[dict] = []

    def add_pass(self, test_name: str):
        self.total += 1
        self.passed += 1
        console.print(f"[green]✓ {test_name}[/green]")

    def add_fail(self, test_name: str, expected: str, actual: str, details: str = ""):
        self.total += 1
        self.failed += 1
        self.failures.append(
            {
                "test": test_name,
                "expected": expected,
                "actual": actual,
                "details": details,
            }
        )
        console.print(f"[red]✗ {test_name}[/red]")
        console.print(f"  Expected: {expected}, Got: {actual}")
        if details:
            console.print(f"  Details: {details}")

    def summary(self):
        """Print test summary."""
        table = Table(title="Smoke Test Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Total Tests", str(self.total))
        table.add_row("Passed", f"[green]{self.passed}[/green]")
        table.add_row("Failed", f"[red]{self.failed}[/red]")
        table.add_row(
            "Success Rate",
            f"{self.passed / self.total * 100:.1f}%" if self.total > 0 else "N/A",
        )

        console.print()
        console.print(table)

        if self.failed > 0:
            console.print("\n[bold red]Failed Tests:[/bold red]")
            for failure in self.failures:
                console.print(f"  • {failure['test']}")

        return self.failed == 0


def load_model(model_path: Path):
    """Load pickled model."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def predict_from_signal_data(model_dict, time: np.ndarray, amplitude: np.ndarray) -> tuple:
    """Extract features and predict from signal."""
    signal = SignalData(time=time.tolist(), amplitude=amplitude.tolist(), shape_type="gaussian")

    features_dict = extract_features(signal)
    if features_dict is None:
        return None, None

    # Convert to feature array (match training order)
    features = np.array(
        [
            [
                features_dict["fwhm"],
                features_dict["peak_height"],
                features_dict["peak_area"],
                features_dict["noise_level"],
                features_dict["snr"],
                features_dict["peak_center"],
            ]
        ]
    )

    # Extract model and scaler from dictionary
    model = model_dict.get("model", model_dict) if isinstance(model_dict, dict) else model_dict
    scaler = model_dict.get("scaler") if isinstance(model_dict, dict) else None

    # Apply scaler if present
    if scaler is not None:
        features = scaler.transform(features)

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    return int(prediction), float(probability[1])


def test_perfect_gaussian(model, results: SmokeTestResult):
    """Test perfect Gaussian signal (should be Healthy)."""
    time = create_time_array(101, "uniform")
    signal = generate_gaussian_peak(time, mu=50.0, sigma=2.5, height=2.8)
    # No noise injection

    pred, prob = predict_from_signal_data(model, time, signal)

    if pred == 0:
        results.add_pass("Perfect Gaussian → Healthy")
    else:
        results.add_fail(
            "Perfect Gaussian → Healthy",
            "Healthy (0)",
            f"Unhealthy ({pred})",
            f"Probability: {prob:.3f}",
        )


def test_degraded_lorentzian(model, results: SmokeTestResult):
    """Test heavily degraded Lorentzian (should be Unhealthy)."""
    time = create_time_array(101, "uniform")
    clean_signal = generate_lorentzian_peak(time, mu=40.0, gamma=5.5, height=1.2)
    signal = add_gaussian_noise(clean_signal, noise_level=0.09, seed=42)

    pred, prob = predict_from_signal_data(model, time, signal)

    if pred == 1:
        results.add_pass("Degraded Lorentzian → Unhealthy")
    else:
        results.add_fail(
            "Degraded Lorentzian → Unhealthy",
            "Unhealthy (1)",
            f"Healthy ({pred})",
            f"Probability: {prob:.3f}",
        )


def test_boundary_cases(model, results: SmokeTestResult):
    """Test decision boundary precision."""
    time = create_time_array(101, "uniform")

    # Boundary case 1: μ=48.0 (just inside healthy range)
    signal1 = generate_gaussian_peak(time, mu=48.0, sigma=2.5, height=2.7)
    pred1, prob1 = predict_from_signal_data(model, time, signal1)

    if pred1 == 0:
        results.add_pass("Boundary: μ=48.0 (healthy threshold) → Healthy")
    else:
        results.add_fail(
            "Boundary: μ=48.0 (healthy threshold) → Healthy",
            "Healthy (0)",
            f"{pred1}",
            f"Prob: {prob1:.3f}",
        )

    # Boundary case 2: μ=52.0 (just inside healthy range, upper bound)
    signal2 = generate_gaussian_peak(time, mu=52.0, sigma=2.5, height=2.7)
    pred2, prob2 = predict_from_signal_data(model, time, signal2)

    if pred2 == 0:
        results.add_pass("Boundary: μ=52.0 (healthy threshold upper) → Healthy")
    else:
        results.add_fail(
            "Boundary: μ=52.0 (healthy threshold upper) → Healthy",
            "Healthy (0)",
            f"{pred2}",
            f"Prob: {prob2:.3f}",
        )


def test_confidence_calibration(model, results: SmokeTestResult):
    """Test confidence score calibration (no extreme 0.0/1.0 unless obvious)."""
    time = create_time_array(101, "uniform")

    # Ambiguous case: moderate noise, borderline parameters
    signal = generate_gaussian_peak(time, mu=51.0, sigma=3.5, height=2.3)
    signal = add_gaussian_noise(signal, noise_level=0.04, seed=42)

    pred, prob = predict_from_signal_data(model, time, signal)

    # Confidence should not be too extreme for ambiguous case
    if 0.1 < prob < 0.9:
        results.add_pass("Confidence calibration: ambiguous case (0.1 < prob < 0.9)")
    else:
        results.add_fail(
            "Confidence calibration: ambiguous case",
            "Probability between 0.1 and 0.9",
            f"{prob:.3f}",
            "Model may be overconfident on borderline cases",
        )


def test_noise_robustness(model, results: SmokeTestResult):
    """Test handling of various noise levels."""
    time = create_time_array(101, "uniform")

    # Low noise Gaussian (should still be healthy)
    signal1 = generate_gaussian_peak(time, mu=50.0, sigma=2.5, height=2.8)
    signal1 = add_gaussian_noise(signal1, noise_level=0.02, seed=42)
    pred1, prob1 = predict_from_signal_data(model, time, signal1)

    if pred1 == 0:
        results.add_pass("Low noise Gaussian (2%) → Healthy")
    else:
        results.add_fail(
            "Low noise Gaussian (2%) → Healthy",
            "Healthy (0)",
            f"{pred1}",
            f"Prob: {prob1:.3f}",
        )

    # High noise degraded Lorentzian (should be unhealthy — Lorentzian shape predicts Unhealthy)
    signal2 = generate_lorentzian_peak(time, mu=44.0, gamma=5.0, height=1.3)
    signal2 = add_gaussian_noise(signal2, noise_level=0.10, seed=43)
    pred2, prob2 = predict_from_signal_data(model, time, signal2)

    if pred2 == 1:
        results.add_pass("High noise Lorentzian (10%) → Unhealthy")
    else:
        results.add_fail(
            "High noise Lorentzian (10%) → Unhealthy",
            "Unhealthy (1)",
            f"{pred2}",
            f"Prob: {prob2:.3f}",
        )


def test_feature_extraction_robustness(model, results: SmokeTestResult):
    """Test that feature extraction handles edge cases."""
    time = create_time_array(101, "uniform")

    # Very wide peak
    signal = generate_gaussian_peak(time, mu=50.0, sigma=8.0, height=1.5)

    try:
        pred, prob = predict_from_signal_data(model, time, signal)
        if pred is not None:
            results.add_pass("Feature extraction: very wide peak")
        else:
            results.add_fail(
                "Feature extraction: very wide peak",
                "Valid prediction",
                "None (feature extraction failed)",
            )
    except Exception as e:
        results.add_fail(
            "Feature extraction: very wide peak",
            "No exception",
            f"Exception: {type(e).__name__}",
            str(e),
        )


def test_shape_type_handling(model, results: SmokeTestResult):
    """Test predictions match expected patterns for each shape type."""
    time = create_time_array(101, "uniform")

    # Clean Gaussian → Healthy expected
    clean_gauss = generate_gaussian_peak(time, mu=50.0, sigma=2.5, height=2.8)
    pred_gauss, prob_gauss = predict_from_signal_data(model, time, clean_gauss)

    # Clean Lorentzian (centered) with moderate params → Could go either way
    clean_lorentz = generate_lorentzian_peak(time, mu=50.0, gamma=3.0, height=2.0)
    pred_lorentz, prob_lorentz = predict_from_signal_data(model, time, clean_lorentz)

    # Off-center Lorentzian → Unhealthy expected
    offcenter_lorentz = generate_lorentzian_peak(time, mu=45.0, gamma=5.0, height=1.3)
    offcenter_lorentz = add_gaussian_noise(offcenter_lorentz, noise_level=0.08, seed=44)
    pred_offcenter, prob_offcenter = predict_from_signal_data(model, time, offcenter_lorentz)

    if pred_gauss == 0:
        results.add_pass("Clean centered Gaussian → Healthy")
    else:
        results.add_fail(
            "Clean centered Gaussian → Healthy",
            "Healthy (0)",
            f"{pred_gauss}",
            f"Prob: {prob_gauss:.3f}",
        )

    if pred_offcenter == 1:
        results.add_pass("Off-center noisy Lorentzian → Unhealthy")
    else:
        results.add_fail(
            "Off-center noisy Lorentzian → Unhealthy",
            "Unhealthy (1)",
            f"{pred_offcenter}",
            f"Prob: {prob_offcenter:.3f}",
        )


@app.command()
def test(
    model: Path = typer.Option(..., help="Path to pickled model file"),
    suite: Literal["all", "critical", "edge-cases", "calibration"] = typer.Option(
        "all", help="Test suite to run"
    ),
    strict: bool = typer.Option(False, help="Exit with code 1 if any test fails (for CI/CD)"),
) -> None:
    """
    Run smoke tests on model.

    Test Suites:
        - critical: Must-pass tests (perfect Gaussian, degraded Lorentzian)
        - edge-cases: Boundary conditions and robustness
        - calibration: Confidence score quality
        - all: Run all tests
    """
    console.print("[bold cyan]Model Smoke Tests[/bold cyan]")
    console.print(f"Model: {model}")
    console.print(f"Suite: {suite}\n")

    if not model.exists():
        console.print(f"[red]Error: Model file not found at {model}[/red]")
        sys.exit(1)

    # Load model
    try:
        model_obj = load_model(model)
        console.print("[green]✓ Model loaded successfully[/green]\n")
    except Exception as e:
        console.print(f"[red]✗ Failed to load model: {e}[/red]")
        sys.exit(1)

    results = SmokeTestResult()

    # Run test suites
    if suite in ["all", "critical"]:
        console.print("[bold]Critical Tests:[/bold]")
        test_perfect_gaussian(model_obj, results)
        test_degraded_lorentzian(model_obj, results)
        console.print()

    if suite in ["all", "edge-cases"]:
        console.print("[bold]Edge Case Tests:[/bold]")
        test_boundary_cases(model_obj, results)
        test_noise_robustness(model_obj, results)
        test_feature_extraction_robustness(model_obj, results)
        test_shape_type_handling(model_obj, results)
        console.print()

    if suite in ["all", "calibration"]:
        console.print("[bold]Calibration Tests:[/bold]")
        test_confidence_calibration(model_obj, results)
        console.print()

    # Summary
    success = results.summary()

    if strict and not success:
        console.print("\n[red]Smoke tests failed: Aborting deployment[/red]")
        sys.exit(1)
    elif success:
        console.print("\n[green]✓ All smoke tests passed![/green]")
    else:
        console.print("\n[yellow]Some tests failed, but not in strict mode (warning only)[/yellow]")


if __name__ == "__main__":
    app()
