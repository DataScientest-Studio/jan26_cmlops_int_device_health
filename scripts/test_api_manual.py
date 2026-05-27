#!/usr/bin/env python3
"""
Manual API Test Script

Starts the FastAPI application and runs manual tests
to verify all endpoints work correctly.

Usage:
    python scripts/test_api_manual.py

Requirements:
    - Trained model at models/bootstrap_model.pkl
    - Database will be created if it doesn't exist
"""

import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path so 'src' is importable when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.signal_processing.signal_generator import generate_signal

# Configuration
API_BASE_URL = "http://localhost:80"
# API key for authentication (X-API-Key header)
# Override with API_KEY environment variable if needed.
import os  # noqa: E402

API_KEY = os.environ.get("API_KEY", "dev-key-12345")
AUTH_HEADERS = {"X-API-Key": API_KEY}
console = Console()


def print_section(title: str):
    """Print a section header."""
    console.print(f"\n[bold cyan]{title}[/bold cyan]")
    console.print("=" * 60)


def print_success(message: str):
    """Print success message."""
    console.print(f"[green]✓[/green] {message}")


def print_error(message: str):
    """Print error message."""
    console.print(f"[red]✗[/red] {message}")


def print_response(response: requests.Response):
    """Print response details."""
    table = Table(show_header=False, box=None)
    table.add_row("[bold]Status:[/bold]", str(response.status_code))
    table.add_row("[bold]Time:[/bold]", f"{response.elapsed.total_seconds():.3f}s")
    console.print(table)

    if response.status_code == 200:
        try:
            data = response.json()
            console.print("\n[bold]Response Data:[/bold]")
            console.print(json.dumps(data, indent=2))
        except Exception:
            console.print(response.text)
    else:
        console.print(f"[red]{response.text}[/red]")


def test_health() -> bool:
    """Test GET /health endpoint."""
    print_section("1. Health Check")

    try:
        response = requests.get(f"{API_BASE_URL}/health", headers=AUTH_HEADERS, timeout=5)
        print_response(response)

        if response.status_code == 200:
            data = response.json()
            if data["status"] == "healthy":
                print_success("API is healthy")
                return True
            elif data["status"] == "degraded":
                # "degraded" means non-critical services (e.g. remote MLflow) are
                # unreachable, but the core API (DB + model) is fully functional.
                # In cloud mode a remote MLflow timeout is expected and does not
                # prevent the API from serving predictions.
                console.print(
                    "[yellow]✓ API functional (status: degraded — "
                    "non-critical services unavailable)[/yellow]"
                )
                return True
            else:
                print_error(f"API status: {data['status']}")
                return False
        else:
            print_error("Health check failed")
            return False
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to API. Is it running?")
        console.print("\n[yellow]Start the API with:[/yellow]")
        console.print("  uvicorn src.api.main:app --reload")
        return False
    except Exception as e:
        print_error(f"Health check error: {e}")
        return False


def test_model_info() -> bool:
    """Test GET /model/info endpoint."""
    print_section("2. Model Information")

    try:
        response = requests.get(f"{API_BASE_URL}/model/info", headers=AUTH_HEADERS, timeout=5)
        print_response(response)

        if response.status_code == 200:
            print_success("Model info retrieved")
            return True
        else:
            print_error("Model info failed")
            return False
    except Exception as e:
        print_error(f"Model info error: {e}")
        return False


def test_predict() -> tuple[bool, int | None]:
    """Test POST /predict endpoint."""
    print_section("3. Prediction")

    try:
        # Generate test signal
        console.print("\n[yellow]Generating test signal (Gaussian, healthy)...[/yellow]")
        signal = generate_signal(
            shape_type="gaussian",
            drift_scenario="baseline",
            n_points=100,
            seed=42,
        )

        # Make prediction request
        request_data = {
            "device_id": "",  # Auto-generate
            "device_name": "Manual Test Device",
            "device_type": "test_sensor",
            "location": "development",
            "time_values": signal.signal.time,
            "amplitude_values": signal.signal.amplitude,
        }

        console.print("[yellow]Sending prediction request...[/yellow]")
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=request_data,
            headers=AUTH_HEADERS,
            timeout=10,
        )
        print_response(response)

        if response.status_code == 200:
            data = response.json()
            prediction_id = data["prediction_id"]

            # Show prediction summary
            table = Table(title="Prediction Summary", show_header=False)
            table.add_row("Prediction ID", str(prediction_id))
            table.add_row("Device ID", data["device_id"])
            table.add_row(
                "Predicted Label",
                f"{data['predicted_label']} ({'Healthy' if data['predicted_label'] == 0 else 'Unhealthy'})",
            )
            table.add_row("Confidence", f"{data['prediction_confidence']:.2%}")
            console.print(table)

            print_success("Prediction successful")
            return True, prediction_id
        else:
            print_error("Prediction failed")
            return False, None
    except Exception as e:
        print_error(f"Prediction error: {e}")
        return False, None


def test_inject_label(prediction_id: int) -> bool:
    """Test POST /labels endpoint."""
    print_section("4. Sparse Label Injection")

    try:
        request_data = {
            "prediction_id": prediction_id,
            "ground_truth_label": 1,  # Healthy (should match Gaussian signal)
            "label_source": "manual_test",
            "injected_by": "test_script",
        }

        console.print(f"[yellow]Injecting label for prediction {prediction_id}...[/yellow]")
        response = requests.post(
            f"{API_BASE_URL}/labels",
            json=request_data,
            headers=AUTH_HEADERS,
            timeout=5,
        )
        print_response(response)

        if response.status_code == 200:
            print_success("Label injected successfully")
            return True
        else:
            print_error("Label injection failed")
            return False
    except Exception as e:
        print_error(f"Label injection error: {e}")
        return False


def test_metrics() -> bool:
    """Test GET /metrics endpoint."""
    print_section("5. Performance Metrics")

    try:
        response = requests.get(
            f"{API_BASE_URL}/metrics?lookback_days=30", headers=AUTH_HEADERS, timeout=5
        )
        print_response(response)

        if response.status_code == 200:
            data = response.json()

            # Show metrics summary
            table = Table(title="System Metrics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Total Predictions", str(data["total_predictions"]))
            table.add_row("Total Labeled", str(data["total_labeled"]))
            table.add_row("Label Coverage", f"{data['label_coverage']:.2%}")

            if data["realized_accuracy"] is not None:
                table.add_row("Realized Accuracy", f"{data['realized_accuracy']:.2%}")

            console.print(table)
            print_success("Metrics retrieved")
            return True
        else:
            print_error("Metrics failed")
            return False
    except Exception as e:
        print_error(f"Metrics error: {e}")
        return False


def test_validation_errors() -> bool:
    """Test input validation."""
    print_section("6. Input Validation Tests")

    tests_passed = 0
    total_tests = 3

    # Test 1: Too short signal
    console.print("\n[yellow]Test: Signal too short (50 points)[/yellow]")
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={
                "device_id": "",
                "time_values": list(range(50)),
                "amplitude_values": [0.1] * 50,
            },
            headers=AUTH_HEADERS,
            timeout=5,
        )
        if response.status_code == 422:
            print_success("Correctly rejected short signal")
            tests_passed += 1
        else:
            print_error(f"Expected 422, got {response.status_code}")
    except Exception as e:
        print_error(f"Error: {e}")

    # Test 2: Length mismatch
    console.print("\n[yellow]Test: Array length mismatch[/yellow]")
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={
                "device_id": "",
                "time_values": list(range(100)),
                "amplitude_values": [0.1] * 50,
            },
            headers=AUTH_HEADERS,
            timeout=5,
        )
        if response.status_code == 422:
            print_success("Correctly rejected mismatched arrays")
            tests_passed += 1
        else:
            print_error(f"Expected 422, got {response.status_code}")
    except Exception as e:
        print_error(f"Error: {e}")

    # Test 3: Too many NaNs
    console.print("\n[yellow]Test: Too many NaN values (10%)[/yellow]")
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={
                "device_id": "",
                "time_values": list(range(100)),
                "amplitude_values": [0.1] * 90 + [None] * 10,
            },
            headers=AUTH_HEADERS,
            timeout=5,
        )
        if response.status_code == 422:
            print_success("Correctly rejected signal with too many NaNs")
            tests_passed += 1
        else:
            print_error(f"Expected 422, got {response.status_code}")
    except Exception as e:
        print_error(f"Error: {e}")

    console.print(f"\n[bold]Validation Tests: {tests_passed}/{total_tests} passed[/bold]")
    return tests_passed == total_tests


def main():
    """Run all manual tests."""
    console.print(
        Panel.fit(
            "[bold cyan]MLOps Device Health API - Manual Test Suite[/bold cyan]\n"
            f"Testing API at: {API_BASE_URL}",
            border_style="cyan",
        )
    )

    results = {}

    # Test 1: Health check
    results["health"] = test_health()
    if not results["health"]:
        console.print("\n[red bold]Cannot proceed without healthy API[/red bold]")
        sys.exit(1)

    # Test 2: Model info
    results["model_info"] = test_model_info()

    # Test 3: Prediction
    results["predict"], prediction_id = test_predict()

    # Test 4: Label injection (only if prediction succeeded)
    if results["predict"] and prediction_id:
        results["labels"] = test_inject_label(prediction_id)
    else:
        results["labels"] = False
        console.print("\n[yellow]Skipping label injection (no prediction_id)[/yellow]")

    # Test 5: Metrics
    results["metrics"] = test_metrics()

    # Test 6: Validation
    results["validation"] = test_validation_errors()

    # Summary
    print_section("Test Summary")
    passed = sum(results.values())
    total = len(results)

    table = Table(title="Results")
    table.add_column("Test", style="cyan")
    table.add_column("Status", style="bold")

    for test_name, passed_test in results.items():
        status = "[green]✓ PASS[/green]" if passed_test else "[red]✗ FAIL[/red]"
        table.add_row(test_name.replace("_", " ").title(), status)

    console.print(table)

    if passed == total:
        console.print(
            Panel(f"[bold green]All {total} tests passed! ✓[/bold green]", border_style="green")
        )
        sys.exit(0)
    else:
        console.print(
            Panel(f"[bold red]{total - passed} test(s) failed[/bold red]", border_style="red")
        )
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Tests interrupted by user[/yellow]")
        sys.exit(130)
