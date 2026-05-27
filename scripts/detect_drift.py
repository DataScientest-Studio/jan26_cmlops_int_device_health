#!/usr/bin/env python3
"""
Drift detection script for MLOps Device Health system.

This script:
1. Loads reference data (training set) from database
2. Loads recent production data from database
3. Runs Evidently drift detection
4. Generates HTML report
5. Records drift events in Prometheus metrics

Usage:
    python scripts/detect_drift.py --output reports/drift_report.html
    python scripts/detect_drift.py --days 7 --min-samples 100
"""

from __future__ import annotations

import json as _json_module
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import typer
from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.database import Database
from src.monitoring.drift_detection import DriftDetector
from src.monitoring.metrics import record_drift_detection

app = typer.Typer(help="Detect data drift using EvidentlyAI")

FEATURE_NAMES = ["fwhm", "peak_height", "peak_area", "noise_level", "snr", "peak_center"]


def load_reference_from_json(file_path: Path) -> pd.DataFrame:
    """Load reference data directly from a labeled signals JSON file.

    This is the preferred method for demos: it does not require labeled
    predictions to already exist in the database.  The baseline JSON file
    (e.g. ``data/raw/dataset_baseline_test.json``) provides the reference
    distribution that the current production predictions are compared against.

    Args:
        file_path: Path to a JSON file in the format written by ``save_dataset``
                   in ``scripts/simulate_drift.py``::

                       {"signals": [{"time": [...], "amplitude": [...],
                                     "shape_type": "gaussian", "label": 0}, ...]}

    Returns:
        DataFrame with feature columns and optional ``ground_truth_label``.

    Raises:
        ValueError: If the file contains no usable signals.
    """
    from src.signal_processing.feature_extractor import extract_features
    from src.signal_processing.signal_models import SignalData

    logger.info(f"Loading reference data from JSON: {file_path}")

    with open(file_path) as fh:
        raw = _json_module.load(fh)

    signals = raw.get("signals", [])
    logger.info(f"Found {len(signals)} signals in reference JSON")

    rows: list[dict] = []
    for s in signals:
        try:
            signal_data = SignalData(
                time=s["time"],
                amplitude=s["amplitude"],
                shape_type=s["shape_type"],
            )
        except Exception as exc:
            logger.debug(f"Skipping invalid signal: {exc}")
            continue

        feats = extract_features(signal_data)
        row: dict = {k: (v if v is not None else float("nan")) for k, v in feats.items()}
        if s.get("label") is not None:
            row["ground_truth_label"] = int(s["label"])
        rows.append(row)

    if not rows:
        raise ValueError(
            f"No usable signals found in reference JSON: {file_path}. "
            "Check that the file contains valid 'signals' entries."
        )

    df = pd.DataFrame(rows)
    logger.info(f"Loaded {len(df)} reference samples from JSON (columns: {list(df.columns)})")
    return df


def load_reference_features(db: Database, limit: int = 10000) -> pd.DataFrame:
    """
    Load reference/training features from database.

    Uses the first N labeled signals as reference data.

    Args:
        db: Database instance
        limit: Maximum number of samples to load

    Returns:
        DataFrame with features and labels
    """
    logger.info(f"Loading up to {limit} reference samples...")

    # Get labeled signal IDs (for reference data)
    signal_ids = db.get_labeled_signal_ids(limit=limit)

    if not signal_ids:
        raise ValueError("No labeled signals found for reference data")

    logger.info(f"Found {len(signal_ids)} labeled signals")

    # Load features and labels
    features_list = []
    for signal_id in signal_ids:
        features = db.get_features_by_signal_id(signal_id)
        label = db.get_label_by_signal_id(signal_id)

        if features and label is not None:
            feature_dict = {
                "signal_id": signal_id,
                **features,
                "ground_truth_label": label,
            }
            features_list.append(feature_dict)

    df = pd.DataFrame(features_list)
    logger.info(f"Loaded {len(df)} reference samples with {len(df.columns)} features")

    return df


def load_current_features(db: Database, days: int = 7, min_samples: int = 100) -> pd.DataFrame:
    """
    Load recent production features from database.

    Args:
        db: Database instance
        days: Number of recent days to include
        min_samples: Minimum number of samples required

    Returns:
        DataFrame with recent features and predictions

    Raises:
        ValueError: If insufficient samples available
    """
    logger.info(f"Loading production data from last {days} days...")

    cutoff_date = datetime.now() - timedelta(days=days)

    # Cursor-based query — JOIN with raw_signals to get signal_id (not on predictions table).
    # Uses p.timestamp (the prediction time) and p.prediction_confidence (correct column name).
    cursor = db.conn.cursor()
    cursor.execute(
        """
        SELECT
            p.prediction_id,
            s.signal_id,
            p.predicted_label,
            p.prediction_confidence,
            p.timestamp
        FROM predictions p
        JOIN raw_signals s ON s.prediction_id = p.prediction_id
        WHERE p.timestamp >= ?
        ORDER BY p.timestamp DESC
        """,
        (cutoff_date.isoformat(),),
    )
    raw_rows = cursor.fetchall()
    predictions = [
        {
            "prediction_id": r["prediction_id"],
            "signal_id": r["signal_id"],
            "predicted_label": r["predicted_label"],
            "confidence": r["prediction_confidence"],
            "created_at": r["timestamp"],
        }
        for r in raw_rows
    ]

    if len(predictions) < min_samples:
        raise ValueError(f"Insufficient samples: found {len(predictions)}, need {min_samples}")

    logger.info(f"Found {len(predictions)} recent predictions")

    # Load features for these predictions
    features_list = []
    for row in predictions:
        features = db.get_features_by_signal_id(row["signal_id"])

        if features:
            feature_dict = {
                "signal_id": row["signal_id"],
                "prediction_id": row["prediction_id"],
                **features,
                "predicted_label": row["predicted_label"],
                "confidence": row["confidence"],
            }

            # Add ground truth label if available
            label = db.get_label_by_signal_id(row["signal_id"])
            if label is not None:
                feature_dict["ground_truth_label"] = label

            features_list.append(feature_dict)

    df = pd.DataFrame(features_list)
    logger.info(f"Loaded {len(df)} current samples with {len(df.columns)} features")

    return df


@app.command()
def detect_drift(
    output_dir: str = typer.Option("reports/drift", help="Output directory for drift reports"),
    days: int = typer.Option(7, help="Number of recent days for current data"),
    min_samples: int = typer.Option(50, help="Minimum samples required"),
    reference_limit: int = typer.Option(10000, help="Maximum reference samples to load"),
    stattest_threshold: float = typer.Option(0.05, help="P-value threshold for drift detection"),
    record_metrics: bool = typer.Option(True, help="Record drift events in Prometheus metrics"),
    reference_json: str | None = typer.Option(
        None,
        "--reference-json",
        help=(
            "Path to a labeled signals JSON file to use as reference data instead of "
            "loading labeled predictions from the database. "
            "Recommended for demos: use data/raw/dataset_baseline_test.json."
        ),
    ),
) -> None:
    """
    Run drift detection and generate report.

    This command:
    1. Loads reference data (labeled training samples)
    2. Loads current production data (recent predictions)
    3. Detects data drift using Evidently
    4. Generates HTML report with visualizations
    5. Records drift events in Prometheus metrics
    """
    logger.info("🔍 Starting drift detection...")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Connect to database — respects DATABASE_URL env var for PostgreSQL
    import os as _os

    _db_url = _os.environ.get("DATABASE_URL", "")
    if _db_url and _db_url.startswith("postgresql"):
        db = Database(db_url=_db_url)
    else:
        db = Database(db_path=Path(__file__).parent.parent / "data" / "database" / "mlops.db")

    try:
        # Load reference data — prefer the JSON file if provided (demo-friendly:
        # works without labeled predictions in the DB)
        if reference_json:
            logger.info(f"📊 Loading reference data from JSON: {reference_json}")
            reference_df = load_reference_from_json(Path(reference_json))
        else:
            logger.info("📊 Loading reference data from DB (labeled predictions)...")
            reference_df = load_reference_features(db, limit=reference_limit)

        logger.info("📊 Loading current production data...")
        current_df = load_current_features(db, days=days, min_samples=min_samples)

        # Determine feature columns (exclude metadata)
        exclude_cols = {
            "signal_id",
            "prediction_id",
            "ground_truth_label",
            "predicted_label",
            "confidence",
        }
        feature_cols = [col for col in reference_df.columns if col not in exclude_cols]

        # Drop columns that are all-NaN in either dataset — Evidently cannot
        # handle empty columns and raises an explicit error for them.
        valid_feature_cols = [
            c
            for c in feature_cols
            if not reference_df[c].isna().all() and not current_df[c].isna().all()
        ]
        if len(valid_feature_cols) < len(feature_cols):
            dropped = set(feature_cols) - set(valid_feature_cols)
            logger.warning(
                f"⚠️ Dropped {len(dropped)} all-NaN column(s) before drift detection: {dropped}"
            )
        feature_cols = valid_feature_cols

        if not feature_cols:
            logger.error(
                "❌ No valid feature columns remain after dropping all-NaN columns.  "
                "Run more predictions to populate feature data before drift detection."
            )
            raise typer.Exit(code=1)

        if len(reference_df) < 2:
            logger.error(
                f"❌ Reference dataset has only {len(reference_df)} sample(s). "
                "Evidently requires at least 2.  Inject more labeled predictions first."
            )
            raise typer.Exit(code=1)

        logger.info(f"🔬 Analyzing {len(feature_cols)} features for drift...")

        # Initialize drift detector
        detector = DriftDetector(
            feature_columns=feature_cols,
            target_column="ground_truth_label"
            if "ground_truth_label" in current_df.columns
            else None,
            prediction_column="predicted_label",
        )

        # Detect data drift
        logger.info("🔍 Running data drift detection...")
        data_drift = detector.detect_data_drift(
            reference_data=reference_df,
            current_data=current_df,
            stattest_threshold=stattest_threshold,
        )

        # Log results
        if data_drift["drift_detected"]:
            logger.warning(
                f"⚠️ Data drift DETECTED! {data_drift['n_drifted_features']}/{data_drift['n_features']} features drifted"
            )
            logger.warning(f"Drifted features: {data_drift['drifted_features']}")

            # Record drift in Prometheus metrics
            if record_metrics:
                record_drift_detection(drift_type="data")
        else:
            logger.info("✅ No data drift detected")

        # Detect target drift (if labels available)
        if detector.target_column and detector.target_column in current_df.columns:
            logger.info("🔍 Running target drift detection...")
            target_drift = detector.detect_target_drift(
                reference_data=reference_df,
                current_data=current_df,
            )

            if target_drift["drift_detected"]:
                logger.warning("⚠️ Target drift DETECTED!")
                if record_metrics:
                    record_drift_detection(drift_type="concept")
            else:
                logger.info("✅ No target drift detected")
        else:
            target_drift = None
            logger.info("⏭️ Skipping target drift (insufficient labels)")

        # Detect prediction drift (only if reference_df has model predictions)
        logger.info("🔍 Running prediction drift detection...")
        pred_col = "predicted_label"
        if pred_col not in reference_df.columns:
            logger.info(
                "⏭️ Skipping prediction drift (reference data has no predicted_label column)"
            )
            prediction_drift = None
        else:
            prediction_drift = detector.detect_prediction_drift(
                reference_data=reference_df,
                current_data=current_df,
            )

            if prediction_drift["drift_detected"]:
                logger.warning("⚠️ Prediction drift DETECTED!")
                if record_metrics:
                    record_drift_detection(drift_type="prediction")
            else:
                logger.info("✅ No prediction drift detected")

        # Generate comprehensive HTML report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_path / f"drift_report_{timestamp}.html"

        logger.info(f"📄 Generating drift report: {report_path}")
        summary = detector.generate_drift_report(
            reference_data=reference_df,
            current_data=current_df,
            output_path=report_path,
        )

        # Save JSON summary.
        # IMPORTANT: write both the short keys (read by API /metrics endpoint) and
        # the long *_details keys (used by other consumers / legacy code).
        summary_path = output_path / f"drift_summary_{timestamp}.json"
        summary.update(
            {
                # Short keys — read by src/api/main.py /metrics endpoint:
                "data_drift": data_drift,
                "target_drift": target_drift,
                # Long keys — backward-compat / other consumers:
                "data_drift_details": data_drift,
                "target_drift_details": target_drift,
                "prediction_drift_details": prediction_drift,
                "reference_samples": len(reference_df),
                "current_samples": len(current_df),
                "days_analyzed": days,
            }
        )

        detector.save_drift_summary(summary, summary_path)

        logger.success("✅ Drift detection complete!")
        logger.success(f"📄 HTML Report: {report_path}")
        logger.success(f"📄 JSON Summary: {summary_path}")

        # Summary output
        typer.echo("\n" + "=" * 60)
        typer.echo("DRIFT DETECTION SUMMARY")
        typer.echo("=" * 60)
        typer.echo(f"Reference samples: {len(reference_df)}")
        typer.echo(f"Current samples: {len(current_df)}")
        typer.echo(f"Features analyzed: {len(feature_cols)}")
        typer.echo(f"Time period: Last {days} days")
        typer.echo("=" * 60)
        typer.echo(f"Data Drift: {'⚠️ DETECTED' if data_drift['drift_detected'] else '✅ OK'}")
        if data_drift["drift_detected"]:
            typer.echo(
                f"  - Drifted features: {data_drift['n_drifted_features']}/{data_drift['n_features']}"
            )
            typer.echo(f"  - Drift share: {data_drift['drift_share']:.2%}")

        if target_drift:
            typer.echo(
                f"Target Drift: {'⚠️ DETECTED' if target_drift['drift_detected'] else '✅ OK'}"
            )

        typer.echo(
            f"Prediction Drift: {'⚠️ DETECTED' if prediction_drift and prediction_drift['drift_detected'] else '✅ OK' if prediction_drift else '⏭️ N/A'}"
        )
        typer.echo("=" * 60)

    except Exception as e:
        logger.error(f"❌ Drift detection failed: {e}")
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
