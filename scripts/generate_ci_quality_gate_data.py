"""Generate the golden reference signals for the CI model quality gate.

Writes ``data/ci/quality_gate_signals.csv`` — a fixed, git-committed dataset
used by ``.github/workflows/model-quality-gate.yml``.

Signal generation uses the **exact same defaults** as the Streamlit UI Greenfield
use case (BootstrapConfig in scripts/greenfield_init.py):
  - Healthy (Gaussian):    μ∈[48,52], σ∈[2.0,3.0], H∈[2.5,3.0], noise∈[0.01,0.02]
  - Unhealthy (Lorentzian): μ∈[42,58], γ=σ×1.1775, σ∈[3.8,5.1], H∈[1.0,1.5], noise∈[0.06,0.10]

CSV format (one row per signal):
  label, shape_type, a_0, a_1, ..., a_100

where a_0..a_100 are the 101 amplitude samples at uniform time 0..100.
Time is NOT stored — it is always reconstructed as np.linspace(0, 100, 101).

Usage
-----
    python scripts/generate_ci_quality_gate_data.py

    # Custom output path
    python scripts/generate_ci_quality_gate_data.py --output data/ci/quality_gate_signals.csv

    # Custom sample count (default: 160 = 80 healthy + 80 unhealthy)
    python scripts/generate_ci_quality_gate_data.py --n-samples 200

When to re-run
--------------
Only when the signal generation parameters or feature extraction changes.
After regenerating, commit the updated CSV and verify the quality gate thresholds
in params.yaml still pass (accuracy ≥ min_accuracy, f1 ≥ min_f1).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Signal generation defaults — MUST MATCH BootstrapConfig ────────────────

_SEED = 42
_DEFAULT_N_SAMPLES = 160  # 80 healthy + 80 unhealthy (double of greenfield train set)
_GAUSSIAN_FRACTION = 0.5  # 50/50 split for balanced quality gate test set

_GAUSS_MU_RANGE = (48.0, 52.0)
_GAUSS_SIGMA_RANGE = (2.0, 3.0)
_GAUSS_HEIGHT_RANGE = (2.5, 3.0)
_GAUSS_NOISE_RANGE = (0.01, 0.02)

_LOR_MU_RANGE = (42.0, 58.0)
_LOR_SIGMA_RANGE = (3.8, 5.1)
_LOR_HEIGHT_RANGE = (1.0, 1.5)
_LOR_NOISE_RANGE = (0.06, 0.10)
_GAMMA_SIGMA_FACTOR = 1.1775  # γ = σ × 1.1775

_DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "ci" / "quality_gate_signals.csv"
_N_POINTS = 101  # must match generate_signal default


def _generate_signals(n_samples: int) -> list[tuple[int, str, list[float]]]:
    """Generate synthetic signals with greenfield BootstrapConfig defaults.

    Returns:
        List of (label, shape_type, amplitude_list) tuples.
        label: 0=healthy, 1=unhealthy
        amplitude_list: list of 101 floats
    """
    from src.signal_processing.signal_generator import generate_signal

    rng = np.random.RandomState(_SEED)

    n_gauss = int(n_samples * _GAUSSIAN_FRACTION)
    n_lor = n_samples - n_gauss

    records: list[tuple[int, str, list[float]]] = []

    print(f"Generating {n_gauss} healthy (Gaussian) signals…")
    for i in range(n_gauss):
        mu = rng.uniform(*_GAUSS_MU_RANGE)
        sigma = rng.uniform(*_GAUSS_SIGMA_RANGE)
        height = rng.uniform(*_GAUSS_HEIGHT_RANGE)
        noise = rng.uniform(*_GAUSS_NOISE_RANGE)
        sig = generate_signal(
            shape_type="gaussian",
            mu=mu,
            width_param=sigma,
            height=height,
            noise_level=noise,
            n_points=_N_POINTS,
            seed=_SEED + i,
        )
        records.append((0, "gaussian", list(sig.signal.amplitude)))

    print(f"Generating {n_lor} unhealthy (Lorentzian) signals…")
    for i in range(n_lor):
        mu = rng.uniform(*_LOR_MU_RANGE)
        sigma_l = rng.uniform(*_LOR_SIGMA_RANGE)
        gamma = sigma_l * _GAMMA_SIGMA_FACTOR
        height = rng.uniform(*_LOR_HEIGHT_RANGE)
        noise = rng.uniform(*_LOR_NOISE_RANGE)
        sig = generate_signal(
            shape_type="lorentzian",
            mu=mu,
            width_param=gamma,
            height=height,
            noise_level=noise,
            n_points=_N_POINTS,
            seed=_SEED + n_gauss + i,
        )
        records.append((1, "lorentzian", list(sig.signal.amplitude)))

    # Shuffle with fixed seed for balanced train/test splits downstream
    rng.shuffle(records)  # type: ignore[arg-type]
    return records


def generate(n_samples: int = _DEFAULT_N_SAMPLES, output: Path = _DEFAULT_OUTPUT) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    records = _generate_signals(n_samples)

    n_healthy = sum(1 for r in records if r[0] == 0)
    n_unhealthy = sum(1 for r in records if r[0] == 1)

    # Build CSV
    amp_cols = [f"a_{i}" for i in range(_N_POINTS)]
    header = ["label", "shape_type"] + amp_cols

    with output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for label, shape_type, amplitudes in records:
            row = [label, shape_type] + [f"{v:.6f}" for v in amplitudes]
            writer.writerow(row)

    size_kb = output.stat().st_size / 1024
    print(
        f"\n✅  Written {n_samples} signals to {output}\n"
        f"   ({n_healthy} healthy / {n_unhealthy} unhealthy, {size_kb:.1f} KB)\n"
        f"   Columns: label, shape_type, a_0..a_{_N_POINTS - 1}\n"
        f"   Commit this file to git — do NOT add a .dvc sidecar."
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate CI quality gate reference signals")
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=_DEFAULT_N_SAMPLES,
        help=f"Total number of signals (default: {_DEFAULT_N_SAMPLES})",
    )
    args = parser.parse_args()
    generate(n_samples=args.n_samples, output=args.output)


if __name__ == "__main__":
    main()
