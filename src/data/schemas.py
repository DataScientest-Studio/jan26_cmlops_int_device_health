"""
Data Contracts — Pandera schemas for MLOps Device Health.

These schemas enforce the feature data contract between the feature extraction
pipeline and the training / inference pipeline.  Any DataFrame that enters the
training pipeline is validated against ``signal_features_schema`` before model
fitting or scoring.

Usage::

    from src.data.schemas import validate_features

    df = extract_features_from_batch(signals)
    validate_features(df)   # raises SchemaError if contract is violated
"""

from __future__ import annotations

import logging

import pandas as pd

_logger = logging.getLogger(__name__)

# Lazily import pandera so that import of this module never breaks when
# pandera is not installed (e.g. lightweight Docker layers without it).
try:
    import pandera.pandas as pa
    from pandera.pandas import Column, DataFrameSchema

    _PANDERA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PANDERA_AVAILABLE = False
    _logger.warning(
        "pandera is not installed — data contract validation will be skipped. "
        "Install with: pip install 'pandera>=0.19'"
    )


# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

#: Canonical feature names produced by ``feature_extractor.extract_features``.
FEATURE_COLUMNS: list[str] = [
    "fwhm",
    "peak_height",
    "peak_area",
    "noise_level",
    "snr",
    "peak_center",
]

#: Optional label column (present in training DataFrames, absent at inference).
LABEL_COLUMN = "label"


def _build_schema(require_label: bool = False) -> DataFrameSchema | None:
    """Build the Pandera DataFrameSchema for signal features."""
    if not _PANDERA_AVAILABLE:
        return None

    columns: dict[str, Column] = {
        "fwhm": Column(float, pa.Check.ge(0.0), nullable=True),
        "peak_height": Column(float, nullable=True),
        "peak_area": Column(float, pa.Check.ge(0.0), nullable=True),
        "noise_level": Column(float, pa.Check.ge(0.0), nullable=True),
        "snr": Column(float, nullable=True),
        "peak_center": Column(float, pa.Check.between(0.0, 100.0), nullable=True),
    }
    if require_label:
        columns[LABEL_COLUMN] = Column(
            int,
            pa.Check.isin([0, 1]),
            nullable=False,
        )

    return DataFrameSchema(
        columns=columns,
        coerce=True,
        strict=False,  # allow extra columns (metadata, model_version, etc.)
    )


#: Schema for unlabeled inference DataFrames (no label column required).
signal_features_schema: DataFrameSchema | None = _build_schema(require_label=False)

#: Schema for labeled training DataFrames (label column must be 0 or 1).
signal_features_training_schema: DataFrameSchema | None = _build_schema(require_label=True)


# ---------------------------------------------------------------------------
# Public validation helpers
# ---------------------------------------------------------------------------


def validate_features(df: pd.DataFrame, *, require_label: bool = False) -> pd.DataFrame:
    """
    Validate a feature DataFrame against the signal features contract.

    Args:
        df: DataFrame produced by ``extract_features`` (one row per signal).
        require_label: If True, also validate the ``label`` column is 0 or 1.

    Returns:
        The validated (and coerced) DataFrame.

    Raises:
        pandera.errors.SchemaError: If the DataFrame violates the contract.
        ValueError: If df is empty or None.
    """
    if df is None or df.empty:
        raise ValueError("validate_features: received empty or None DataFrame")

    if not _PANDERA_AVAILABLE:
        _logger.debug("pandera not available — skipping schema validation")
        return df

    schema = signal_features_training_schema if require_label else signal_features_schema
    if schema is None:
        return df

    try:
        validated = schema.validate(df)
        _logger.debug(
            "Data contract validated: %d rows, %d cols", len(validated), len(validated.columns)
        )
        return validated
    except Exception:
        _logger.exception("Data contract violation detected")
        raise
