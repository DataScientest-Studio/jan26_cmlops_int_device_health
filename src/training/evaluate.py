"""
Model evaluation pipeline stage for DVC.

Loads a trained model and test data, computes classification metrics, and
writes a metrics JSON file suitable for DVC tracking.

Usage (standalone):
    python -m src.training.evaluate \\
        --model          models/champion_model.pkl \\
        --features-test  data/processed/features_test.csv \\
        --labels-test    data/processed/labels_test.csv \\
        --output         metrics/eval_metrics.json

DVC pipeline stage: evaluate
    This script is the intended entry-point for the evaluate DVC stage.
    Separating evaluation from training allows:
      - Re-evaluating the model on new or different test sets without retraining
      - Champion/Challenger comparison by running evaluate against two model files
      - Clean DVC metric tracking (eval is cached independently of training)

Metrics produced (saved to JSON):
    accuracy    Overall classification accuracy
    precision   Precision for the positive (unhealthy=1) class
    recall      Recall    for the positive (unhealthy=1) class
    f1_score    F1 score  for the positive (unhealthy=1) class
    n_samples   Number of test samples evaluated
    model_path  Path to the model that was evaluated

Params (from params.yaml → evaluate section):
    metrics                    list of metric names to include in output
    generate_confusion_matrix  whether to log the full confusion matrix
    generate_roc_curve         (future) whether to log ROC-AUC data
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

# Ensure repo root is on sys.path so 'src' is importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_from_csv(
    model_path: Path,
    features_path: Path,
    labels_path: Path,
    *,
    generate_confusion_matrix: bool = True,
) -> dict[str, object]:
    """
    Load model and test CSVs, compute metrics, return results dict.

    Args:
        model_path:                Path to trained model pickle (.pkl).
        features_path:             Path to feature matrix CSV from preprocess.py.
        labels_path:               Path to label vector CSV from preprocess.py.
        generate_confusion_matrix: Whether to include the confusion matrix.

    Returns:
        dict with accuracy, precision, recall, f1_score, n_samples, model_path,
        and optionally confusion_matrix.

    Raises:
        FileNotFoundError: If any input file is missing.
        ValueError:        If the model pickle is invalid or features are incompatible.
    """
    # ── Load model ────────────────────────────────────────────────────────────
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, "rb") as fh:
        model_data = pickle.load(fh)

    # Support both raw sklearn estimators and our dict-wrapped models
    if isinstance(model_data, dict):
        model = model_data["model"]
        scaler = model_data.get("scaler")
    else:
        # Plain sklearn model (e.g. from tests)
        model = model_data
        scaler = None

    # ── Load test data ────────────────────────────────────────────────────────
    if not features_path.exists():
        raise FileNotFoundError(f"Features CSV not found: {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_path}")

    X: np.ndarray = pd.read_csv(features_path).values  # noqa: N806
    y: np.ndarray = np.array(pd.read_csv(labels_path)["label"].values).astype(int)

    # Filter out unlabeled samples (-1)
    mask = y >= 0
    X, y = X[mask], y[mask]  # noqa: N806

    if len(y) == 0:
        raise ValueError("No labeled test samples found after filtering unlabeled rows.")

    # ── Scale features ────────────────────────────────────────────────────────
    if scaler is not None:
        X = scaler.transform(X)  # noqa: N806

    # ── Compute predictions ───────────────────────────────────────────────────
    y_pred: np.ndarray = model.predict(X)

    # ── Metrics ───────────────────────────────────────────────────────────────
    results: dict[str, object] = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y, y_pred, zero_division=0)),
        "n_samples": int(len(y)),
        "model_path": str(model_path),
    }

    if generate_confusion_matrix:
        cm = confusion_matrix(y, y_pred).tolist()
        results["confusion_matrix"] = cm

    return results


# ── CLI entry point ───────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model and write DVC-tracked metrics JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model pickle (e.g. models/champion_model.pkl)",
    )
    parser.add_argument(
        "--features-test",
        type=Path,
        required=True,
        help="Path to test feature matrix CSV (from preprocess.py)",
    )
    parser.add_argument(
        "--labels-test",
        type=Path,
        required=True,
        help="Path to test label vector CSV (from preprocess.py)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("metrics/eval_metrics.json"),
        help="Destination for the metrics JSON file",
    )
    parser.add_argument(
        "--no-confusion-matrix",
        action="store_true",
        help="Omit the confusion matrix from the output",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    print(f"Evaluating model: {args.model}")
    print(f"Test features:    {args.features_test}")
    print(f"Test labels:      {args.labels_test}")

    results = evaluate_from_csv(
        model_path=args.model,
        features_path=args.features_test,
        labels_path=args.labels_test,
        generate_confusion_matrix=not args.no_confusion_matrix,
    )

    # Write metrics JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(results, fh, indent=2)

    print("\nEvaluation Results:")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1 Score:  {results['f1_score']:.4f}")
    print(f"  N Samples: {results['n_samples']}")
    print(f"\nMetrics written to: {args.output}")


if __name__ == "__main__":
    main()
