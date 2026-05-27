"""
Feature extraction pipeline stage for DVC.

Reads raw signal JSON datasets produced by generate_data.py, extracts the
6 discriminating features from each signal, and writes two CSV files:
  - features.csv  — feature matrix (one row per signal)
  - labels.csv    — label vector (0 = healthy, 1 = unhealthy)

Usage (standalone):
    python -m src.signal_processing.preprocess \\
        --train-data data/raw/dataset_baseline_train.json \\
        --test-data  data/raw/dataset_baseline_test.json  \\
        --output-dir data/processed

DVC pipeline stage: extract_features
    This script is the intended entry-point for the extract_features DVC stage.
    Feature extraction is deliberately separated from model training so that:
      - DVC can cache the expensive feature-extraction step between training runs
      - Feature matrices can be inspected / visualised independently
      - Downstream stages (train, evaluate) operate on compact tabular data

Params (from params.yaml → preprocess section):
    window_length   Savitzky-Golay filter window length (odd integer, default 11)
    polyorder       Savitzky-Golay polynomial order (default 3)
    peak_prominence Minimum peak prominence for detection (default 0.5)

    Note: window_length and polyorder are accepted as CLI arguments for DVC
    parameter tracking. They are passed to the feature extractor for future
    Savitzky-Golay pre-smoothing support; currently the raw amplitude is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path so 'src' is importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.signal_processing.feature_extractor import extract_features
from src.signal_processing.signal_models import SignalData

# ── Feature column order (must match feature_extractor output) ────────────────
FEATURE_NAMES: list[str] = [
    "peak_height",
    "peak_center",
    "fwhm",
    "snr",
    "peak_area",
    "noise_level",
]


def extract_features_from_json(
    input_path: Path,
    features_output: Path,
    labels_output: Path,
    window_length: int = 11,  # reserved for future SG pre-smoothing
    polyorder: int = 3,  # reserved for future SG pre-smoothing
) -> tuple[int, int]:
    """
    Extract features from a signal JSON file and write CSVs.

    Args:
        input_path:      Path to signal JSON (generate_data.py output).
        features_output: Destination CSV path for the feature matrix.
        labels_output:   Destination CSV path for the label vector.
        window_length:   Savitzky-Golay window (reserved, not yet used).
        polyorder:       Savitzky-Golay order     (reserved, not yet used).

    Returns:
        (n_signals, n_labeled) tuple.

    Raises:
        FileNotFoundError: If input_path does not exist.
        ValueError:        If the JSON structure is invalid.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path) as fh:
        data = json.load(fh)

    if "signals" not in data:
        raise ValueError(f"Invalid JSON structure in {input_path}: missing 'signals' key")

    feature_rows: list[list[float]] = []
    labels: list[int] = []

    for entry in data["signals"]:
        signal = SignalData(
            time=entry["time"],
            amplitude=entry["amplitude"],
            shape_type=entry.get("shape_type", "unknown"),
        )
        raw_features = extract_features(signal)

        # extract_features returns dict[str, float | None] — replace None with 0.0
        row = [float(raw_features.get(name) or 0.0) for name in FEATURE_NAMES]
        feature_rows.append(row)
        labels.append(int(entry.get("label", -1)))

    # Write feature matrix
    features_output.parent.mkdir(parents=True, exist_ok=True)
    with open(features_output, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(FEATURE_NAMES)
        writer.writerows(feature_rows)

    # Write label vector
    labels_output.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_output, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["label"])
        writer.writerows([[lbl] for lbl in labels])

    n_labeled = sum(1 for lbl in labels if lbl >= 0)
    return len(feature_rows), n_labeled


# ── CLI entry point ───────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract features from signal JSON datasets (DVC: extract_features stage).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        required=True,
        help="Path to training signal JSON (e.g. data/raw/dataset_baseline_train.json)",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        help="Path to test signal JSON (optional, e.g. data/raw/dataset_baseline_test.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for output CSV files",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=11,
        help="Savitzky-Golay filter window length (odd integer; reserved for future use)",
    )
    parser.add_argument(
        "--polyorder",
        type=int,
        default=3,
        help="Savitzky-Golay polynomial order (reserved for future use)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    print(f"Feature extraction: {args.train_data} → {args.output_dir}/")

    # --- Train split ---
    n_train, n_labeled_train = extract_features_from_json(
        input_path=args.train_data,
        features_output=args.output_dir / "features_train.csv",
        labels_output=args.output_dir / "labels_train.csv",
        window_length=args.window_length,
        polyorder=args.polyorder,
    )
    print(f"  Train: {n_train} signals, {n_labeled_train} labeled")
    print(f"  → {args.output_dir}/features_train.csv")
    print(f"  → {args.output_dir}/labels_train.csv")

    # --- Test split (optional) ---
    if args.test_data:
        n_test, n_labeled_test = extract_features_from_json(
            input_path=args.test_data,
            features_output=args.output_dir / "features_test.csv",
            labels_output=args.output_dir / "labels_test.csv",
            window_length=args.window_length,
            polyorder=args.polyorder,
        )
        print(f"  Test:  {n_test} signals, {n_labeled_test} labeled")
        print(f"  → {args.output_dir}/features_test.csv")
        print(f"  → {args.output_dir}/labels_test.csv")

    print("Feature extraction complete.")


if __name__ == "__main__":
    main()
