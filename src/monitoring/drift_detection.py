"""
Drift detection module using EvidentlyAI.

This module provides functionality to detect and report:
- Data drift: Changes in feature distributions
- Prediction drift: Changes in model output distribution
- Concept drift: Changes in feature-target relationships

Reports are generated in HTML format and can be stored locally or uploaded
to cloud storage (S3/DagsHub).

Compatible with Evidently v0.7.x (legacy API via evidently.legacy.*)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# evidently 0.7.x moved the legacy API under evidently.legacy.*;
# evidently 0.4.x uses the top-level namespace.
# Support both so the host venv (0.4.x) and container (0.7.x) stay compatible.
try:
    from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
    from evidently.legacy.pipeline.column_mapping import ColumnMapping
    from evidently.legacy.report import Report
except ImportError:  # evidently < 0.6 (0.4.x API)
    from evidently import ColumnMapping  # type: ignore[no-redef]  # noqa: E402
    from evidently.metric_preset import (  # type: ignore[no-redef]  # noqa: E402
        DataDriftPreset,
        DataQualityPreset,
        TargetDriftPreset,
    )
    from evidently.report import Report  # type: ignore[no-redef]  # noqa: E402


class DriftDetector:
    """
    Detect drift in features, predictions, and targets using EvidentlyAI.

    This class compares reference data (training/baseline) against current
    production data to identify distribution shifts.
    """

    def __init__(
        self,
        feature_columns: list[str],
        target_column: str | None = None,
        prediction_column: str | None = None,
    ):
        """
        Initialize drift detector.

        Args:
            feature_columns: List of feature column names to monitor
            target_column: Name of ground truth label column (optional)
            prediction_column: Name of prediction column (optional)
        """
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.prediction_column = prediction_column

        # Configure column mapping for Evidently
        self.column_mapping = ColumnMapping(
            target=target_column,
            prediction=prediction_column,
            numerical_features=feature_columns,
        )

    def detect_data_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        stattest: str = "ks",
        stattest_threshold: float = 0.05,
    ) -> dict[str, Any]:
        """
        Detect drift in feature distributions.

        Compares feature distributions between reference (baseline) and current
        data using statistical tests.

        Args:
            reference_data: Baseline/training data
            current_data: Recent production data
            stattest: Statistical test ('ks', 'wasserstein', 'chisquare')
            stattest_threshold: P-value threshold for drift detection

        Returns:
            Dict with drift detection results:
            - drift_detected: bool
            - drift_share: float (proportion of drifted features)
            - drifted_features: list[str]
            - n_features: int
            - n_drifted_features: int
        """
        # Create Evidently report with data drift preset
        report = Report(
            metrics=[
                DataDriftPreset(
                    stattest=stattest,
                    stattest_threshold=stattest_threshold,
                )
            ]
        )

        # Run drift detection
        report.run(
            reference_data=reference_data[self.feature_columns],
            current_data=current_data[self.feature_columns],
            column_mapping=self.column_mapping,
        )

        # Extract results
        result = report.as_dict()

        # Parse drift metrics
        drift_metrics = result["metrics"][0]["result"]

        # Evidently 0.4.x: per-column detail is NOT in DataDriftPreset result.
        # Use the top-level counters instead.
        n_drifted = drift_metrics.get("number_of_drifted_columns", 0)
        n_total = drift_metrics.get("number_of_columns", len(self.feature_columns))
        # `drift_share` in Evidently 0.4.x is the configured *threshold*, not
        # the actual computed share.  Use `share_of_drifted_columns` for the
        # real fraction.
        actual_share = drift_metrics.get(
            "share_of_drifted_columns", n_drifted / n_total if n_total else 0.0
        )

        # `dataset_drift` uses Evidently's 50% threshold; override to flag ANY
        # feature drift so mild but real drift is not silently ignored.
        drift_detected = n_drifted > 0

        return {
            "drift_detected": drift_detected,
            "drift_share": actual_share,
            "drifted_features": [],  # per-column names not available in DataDriftPreset
            "n_features": n_total,
            "n_drifted_features": n_drifted,
            "stattest": stattest,
            "threshold": stattest_threshold,
        }

    def detect_target_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Detect drift in target/label distribution.

        Compares ground truth label distribution between reference and current data.
        Requires target_column to be set.

        Args:
            reference_data: Baseline data with labels
            current_data: Current data with labels

        Returns:
            Dict with target drift results:
            - drift_detected: bool
            - drift_score: float
            - target_column: str

        Raises:
            ValueError: If target_column not set
        """
        if self.target_column is None:
            raise ValueError("target_column must be set for target drift detection")

        # Use target drift preset
        report = Report(metrics=[TargetDriftPreset()])

        cols_to_use = [self.target_column] + self.feature_columns
        report.run(
            reference_data=reference_data[cols_to_use],
            current_data=current_data[cols_to_use],
            column_mapping=self.column_mapping,
        )

        result = report.as_dict()
        drift_metrics = result["metrics"][0]["result"]

        return {
            "drift_detected": drift_metrics.get("drift_detected", False),
            "drift_score": drift_metrics.get("drift_score", 0.0),
            "target_column": self.target_column,
        }

    def detect_prediction_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Detect drift in model predictions.

        Compares model output distribution between reference and current data.
        Requires prediction_column to be set.

        Args:
            reference_data: Baseline data with predictions
            current_data: Current data with predictions

        Returns:
            Dict with prediction drift results:
            - drift_detected: bool
            - drift_score: float
            - prediction_column: str

        Raises:
            ValueError: If prediction_column not set
        """
        if self.prediction_column is None:
            raise ValueError("prediction_column must be set for prediction drift detection")

        # Create temporary column mapping with prediction as target
        temp_mapping = ColumnMapping(
            target=self.prediction_column,
            numerical_features=self.feature_columns,
        )

        report = Report(metrics=[TargetDriftPreset()])

        cols_to_use = [self.prediction_column] + self.feature_columns
        report.run(
            reference_data=reference_data[cols_to_use],
            current_data=current_data[cols_to_use],
            column_mapping=temp_mapping,
        )

        result = report.as_dict()
        drift_metrics = result["metrics"][0]["result"]

        return {
            "drift_detected": drift_metrics.get("drift_detected", False),
            "drift_score": drift_metrics.get("drift_score", 0.0),
            "prediction_column": self.prediction_column,
        }

    def generate_drift_report(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        output_path: str | Path,
        include_data_quality: bool = True,
    ) -> dict[str, Any]:
        """
        Generate comprehensive HTML drift report.

        Creates an interactive HTML report with drift visualizations and
        saves it to the specified path.

        Args:
            reference_data: Baseline/training data
            current_data: Recent production data
            output_path: Path where HTML report will be saved
            include_data_quality: Include data quality metrics in report

        Returns:
            Dict with report summary:
            - report_path: str
            - timestamp: str
            - data_drift: dict
            - target_drift: dict (if target_column set)
            - prediction_drift: dict (if prediction_column set)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build metrics list
        metrics = [DataDriftPreset()]

        if include_data_quality:
            metrics.append(DataQualityPreset())

        if self.target_column is not None:
            metrics.append(TargetDriftPreset())

        # Create comprehensive drift report
        report = Report(metrics=metrics)

        # Select relevant columns (only include prediction_column if present in both datasets)
        cols_to_use = self.feature_columns.copy()
        if self.target_column:
            cols_to_use.append(self.target_column)
        if self.prediction_column and self.prediction_column in reference_data.columns:
            cols_to_use.append(self.prediction_column)

        # Run report
        report.run(
            reference_data=reference_data[cols_to_use],
            current_data=current_data[cols_to_use],
            column_mapping=self.column_mapping,
        )

        # Save HTML report
        report.save_html(str(output_path))

        # Extract summary
        result = report.as_dict()

        # Data drift summary (first metric — DataDriftPreset)
        data_drift_metrics = result["metrics"][0]["result"]
        n_drifted = data_drift_metrics.get("number_of_drifted_columns", 0)
        n_total = data_drift_metrics.get("number_of_columns", 6)
        actual_share = data_drift_metrics.get(
            "share_of_drifted_columns",
            n_drifted / n_total if n_total else 0.0,
        )
        # Flag ANY feature drift, not only when >= 50% threshold
        drift_detected = n_drifted > 0

        summary = {
            "report_path": str(output_path),
            "timestamp": datetime.now().isoformat(),
            "data_drift": {
                "drift_detected": drift_detected,
                "drift_share": actual_share,
                "n_drifted_features": n_drifted,
            },
        }

        # Add target drift if available
        if self.target_column and len(result["metrics"]) > 1:
            target_idx = 2 if include_data_quality else 1
            if target_idx < len(result["metrics"]):
                target_metrics = result["metrics"][target_idx]["result"]
                summary["target_drift"] = {
                    "drift_detected": target_metrics.get("drift_detected", False),
                    "drift_score": target_metrics.get("drift_score", 0.0),
                }

        return summary

    def save_drift_summary(
        self,
        summary: dict[str, Any],
        output_path: str | Path,
    ) -> None:
        """
        Save drift detection summary to JSON file.

        Args:
            summary: Drift detection summary dictionary
            output_path: Path where JSON will be saved
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)


def load_reference_data(data_path: str | Path) -> pd.DataFrame:
    """
    Load reference/baseline data from file.

    Supports CSV, Parquet, and Pickle formats.

    Args:
        data_path: Path to reference data file

    Returns:
        DataFrame with reference data

    Raises:
        ValueError: If file format not supported
    """
    data_path = Path(data_path)

    if data_path.suffix == ".csv":
        return pd.read_csv(data_path)
    elif data_path.suffix in [".parquet", ".pq"]:
        return pd.read_parquet(data_path)
    elif data_path.suffix == ".pkl":
        return pd.read_pickle(data_path)
    else:
        raise ValueError(
            f"Unsupported format: {data_path.suffix}. Supported: .csv, .parquet, .pq, .pkl"
        )


def load_current_data(data_path: str | Path) -> pd.DataFrame:
    """
    Load current/production data from file.

    Supports CSV, Parquet, and Pickle formats.

    Args:
        data_path: Path to current data file

    Returns:
        DataFrame with current data

    Raises:
        ValueError: If file format not supported
    """
    return load_reference_data(data_path)
