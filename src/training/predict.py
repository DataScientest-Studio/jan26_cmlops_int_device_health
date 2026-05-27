"""
Prediction pipeline for MLOps device health monitoring.

Provides:
- predict(): Make prediction from raw signal
- predict_batch(): Batch predictions
- Integration with Database for result storage

Features:
- Automatic feature extraction
- Confidence scores (probability)
- Model loading and caching
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.signal_processing.feature_extractor import extract_features
from src.signal_processing.signal_models import SignalData
from src.training.train import load_model


def predict(
    time_values: list[float],
    amplitude_values: list[float],
    model_path: Path | str | dict[str, Any],  # Can be path or pre-loaded artifact
    return_probabilities: bool = True,
) -> dict[str, Any]:
    """
    Predict device health from raw signal.

    Args:
        time_values: Time array (e.g., [0.0, 1.0, ..., 100.0])
        amplitude_values: Amplitude array (may contain None for NaN)
        model_path: Path to trained model (pickle) or pre-loaded artifact dict
        return_probabilities: Whether to return class probabilities

    Returns:
        Dict with:
        {
            "predicted_label": int (0=healthy, 1=unhealthy),
            "confidence": float (max class probability),
            "probabilities": dict {"healthy": float, "unhealthy": float},  # If return_probabilities=True
            "features": dict {feature_name: value},
            "model_version": str,
            "mlflow_run_id": str | None,
            "git_sha": str | None,
            "dvc_data_hash": str | None,
        }

    Raises:
        ValueError: If signal is invalid (length mismatch, too short, etc.)
        FileNotFoundError: If model not found
    """
    # Validate inputs
    if len(time_values) != len(amplitude_values):
        raise ValueError(
            f"time_values and amplitude_values must have same length: "
            f"{len(time_values)} != {len(amplitude_values)}"
        )

    if len(time_values) < 51:
        raise ValueError(f"Signal too short: {len(time_values)} < 51")

    # Convert None to np.nan for feature extraction
    amplitude_array = [np.nan if val is None else val for val in amplitude_values]

    # Create SignalData (shape_type not used by feature extraction, set to gaussian as placeholder)
    signal_data = SignalData(
        time=time_values,
        amplitude=amplitude_array,
        shape_type="gaussian",  # Placeholder - not used by feature extraction
    )

    # Extract features
    features = extract_features(signal_data)

    # Data contract validation (Task 6) — non-fatal at inference time
    try:
        import pandas as _pd  # noqa: I001
        from src.data.schemas import validate_features as _validate_features

        _feat_df = _pd.DataFrame([features])
        _validate_features(_feat_df, require_label=False)
    except ImportError:
        pass
    except Exception as _cv_exc:
        import logging as _log

        _log.getLogger(__name__).debug("Data contract warning at inference: %s", _cv_exc)

    # Load model (or use pre-loaded artifact)
    model_artifact = model_path if isinstance(model_path, dict) else load_model(model_path)
    model = model_artifact["model"]
    scaler = model_artifact["scaler"]
    feature_names = model_artifact["feature_names"]
    model_version = model_artifact["model_version"]

    # Safety: if model is an sklearn Pipeline that embeds its own scaler,
    # extract the components so the rest of the function works uniformly.
    from sklearn.pipeline import Pipeline as _SkPipeline
    from sklearn.preprocessing import StandardScaler as _SkScaler

    if isinstance(model, _SkPipeline):
        if scaler is None:
            for _sname, _sobj in model.steps:
                if isinstance(_sobj, _SkScaler):
                    scaler = _sobj
                    break
        if feature_names is None and hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)
        # Use the final estimator for predict/predict_proba
        model = model.steps[-1][1]

    if feature_names is None:
        raise ValueError("Model artifact missing 'feature_names'. Cannot build feature vector.")
    if scaler is None:
        raise ValueError("Model artifact missing 'scaler'. Cannot scale features for prediction.")

    # Prepare feature vector (replace None with 0.0 for missing features)
    feature_vector = np.array([[features.get(name) or 0.0 for name in feature_names]])

    # Scale features
    feature_vector_scaled = scaler.transform(feature_vector)

    # Predict
    predicted_label = int(model.predict(feature_vector_scaled)[0])

    # Get probabilities
    probabilities_array = model.predict_proba(feature_vector_scaled)[0]
    confidence = float(max(probabilities_array))

    result = {
        "predicted_label": predicted_label,
        "confidence": confidence,
        "features": features,
        "model_version": model_version,
        "mlflow_run_id": model_artifact.get("mlflow_run_id")
        if isinstance(model_artifact, dict)
        else None,
        "git_sha": model_artifact.get("git_sha") if isinstance(model_artifact, dict) else None,
        "dvc_data_hash": model_artifact.get("dvc_data_hash")
        if isinstance(model_artifact, dict)
        else None,
    }

    if return_probabilities:
        result["probabilities"] = {
            "healthy": float(probabilities_array[0]),
            "unhealthy": float(probabilities_array[1]),
        }

    return result


def predict_batch(
    signals: list[tuple[list[float], list[float]]],
    model_path: Path | str,
    return_probabilities: bool = True,
) -> list[dict[str, Any]]:
    """
    Make predictions for multiple signals.

    Args:
        signals: List of (time_values, amplitude_values) tuples
        model_path: Path to trained model (pickle)
        return_probabilities: Whether to return class probabilities

    Returns:
        List of prediction dicts (same format as predict())

    Raises:
        ValueError: If any signal is invalid
        FileNotFoundError: If model not found
    """
    # Load model once (avoid repeated loading)
    model_artifact = load_model(model_path)
    model = model_artifact["model"]
    scaler = model_artifact["scaler"]
    feature_names = model_artifact["feature_names"]
    model_version = model_artifact["model_version"]

    results = []

    for time_values, amplitude_values in signals:
        # Validate
        if len(time_values) != len(amplitude_values):
            raise ValueError(
                f"time_values and amplitude_values must have same length: "
                f"{len(time_values)} != {len(amplitude_values)}"
            )

        if len(time_values) < 51:
            raise ValueError(f"Signal too short: {len(time_values)} < 51")

        # Convert None to np.nan
        amplitude_array = [np.nan if val is None else val for val in amplitude_values]

        # Create SignalData and extract features
        signal_data = SignalData(
            time=time_values,
            amplitude=amplitude_array,
            shape_type="gaussian",  # Placeholder - shape unknown in prediction
        )
        features = extract_features(signal_data)

        # Prepare feature vector
        feature_vector = np.array([[features.get(name) or 0.0 for name in feature_names]])
        feature_vector_scaled = scaler.transform(feature_vector)

        # Predict
        predicted_label = int(model.predict(feature_vector_scaled)[0])
        probabilities_array = model.predict_proba(feature_vector_scaled)[0]
        confidence = float(max(probabilities_array))

        result = {
            "predicted_label": predicted_label,
            "confidence": confidence,
            "features": features,
            "model_version": model_version,
        }

        if return_probabilities:
            result["probabilities"] = {
                "healthy": float(probabilities_array[0]),
                "unhealthy": float(probabilities_array[1]),
            }

        results.append(result)

    return results


def predict_from_file(
    signal_file: Path | str,
    model_path: Path | str,
    return_probabilities: bool = True,
) -> dict[str, Any]:
    """
    Make prediction from signal stored in JSON file.

    Args:
        signal_file: Path to JSON file with 'time' and 'amplitude' arrays
        model_path: Path to trained model (pickle)
        return_probabilities: Whether to return class probabilities

    Returns:
        Prediction dict (same format as predict())

    Raises:
        ValueError: If signal file format is invalid
        FileNotFoundError: If files not found
    """
    signal_file = Path(signal_file)
    if not signal_file.exists():
        raise FileNotFoundError(f"Signal file not found: {signal_file}")

    # Load signal from JSON
    with open(signal_file) as f:
        signal_data = json.load(f)

    if "time" not in signal_data or "amplitude" not in signal_data:
        raise ValueError(
            f"Invalid signal file format. Expected 'time' and 'amplitude' keys, "
            f"found: {list(signal_data.keys())}"
        )

    time_values = signal_data["time"]
    amplitude_values = signal_data["amplitude"]

    return predict(
        time_values=time_values,
        amplitude_values=amplitude_values,
        model_path=model_path,
        return_probabilities=return_probabilities,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict device health from signal")
    parser.add_argument(
        "--signal-file",
        type=Path,
        required=True,
        help="Path to signal JSON file (with 'time' and 'amplitude' keys)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default="models/trained_model.pkl",
        help="Path to trained model (default: models/trained_model.pkl)",
    )
    parser.add_argument(
        "--no-probabilities",
        action="store_true",
        help="Disable probability output",
    )

    args = parser.parse_args()

    # Make prediction
    result = predict_from_file(
        signal_file=args.signal_file,
        model_path=args.model,
        return_probabilities=not args.no_probabilities,
    )

    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(
        f"Predicted label: {result['predicted_label']} "
        f"({'Healthy' if result['predicted_label'] == 0 else 'Unhealthy'})"
    )
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Model version: {result['model_version']}")

    if "probabilities" in result:
        print("\nProbabilities:")
        print(f"  Healthy (0): {result['probabilities']['healthy']:.2%}")
        print(f"  Unhealthy (1): {result['probabilities']['unhealthy']:.2%}")

    print("\nExtracted Features:")
    for feature_name, feature_value in result["features"].items():
        if feature_value is not None:
            print(f"  {feature_name}: {feature_value:.4f}")
        else:
            print(f"  {feature_name}: None")
