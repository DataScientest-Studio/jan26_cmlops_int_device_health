"""
ML test conftest — fixtures for trained models, datasets, and helper functions.
"""

import json
from pathlib import Path

import pytest

from src.signal_processing.signal_generator import generate_signal
from src.signal_processing.signal_models import LabeledSignal
from src.training import train_model


def save_dataset(
    dataset: list[LabeledSignal], output_path: Path, include_labels: bool = True
) -> None:
    """Serialize LabeledSignal list to JSON (matches scripts/generate_data.py pattern)."""
    dataset_dict = {"n_samples": len(dataset), "signals": []}
    for idx, ls in enumerate(dataset):
        entry = {
            "id": idx,
            "time": ls.signal.time if isinstance(ls.signal.time, list) else ls.signal.time.tolist(),
            "amplitude": ls.signal.amplitude
            if isinstance(ls.signal.amplitude, list)
            else ls.signal.amplitude.tolist(),
            "shape_type": ls.signal.shape_type,
            "metadata": ls.metadata,
        }
        if include_labels:
            entry["label"] = ls.label
        dataset_dict["signals"].append(entry)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset_dict, f, indent=2)


@pytest.fixture
def sample_training_data(tmp_path) -> tuple[Path, list[LabeledSignal]]:
    """40-signal training dataset (20 healthy Gaussian + 20 unhealthy Lorentzian)."""
    signals = []
    for i in range(20):
        signals.append(generate_signal("gaussian", drift_scenario="baseline", seed=i))
    for i in range(20, 40):
        signals.append(generate_signal("lorentzian", drift_scenario="baseline", seed=i))
    train_file = tmp_path / "train_data.json"
    save_dataset(signals, train_file)
    return train_file, signals


@pytest.fixture
def sample_test_data(tmp_path) -> tuple[Path, list[LabeledSignal]]:
    """10-signal test dataset (5 healthy + 5 unhealthy)."""
    signals = []
    for i in range(5):
        signals.append(generate_signal("gaussian", drift_scenario="baseline", seed=100 + i))
    for i in range(5):
        signals.append(generate_signal("lorentzian", drift_scenario="baseline", seed=200 + i))
    test_file = tmp_path / "test_data.json"
    save_dataset(signals, test_file)
    return test_file, signals


@pytest.fixture
def trained_model(sample_training_data, tmp_path) -> Path:
    """Pre-trained model artifact for prediction tests."""
    train_file, _ = sample_training_data
    model_file = tmp_path / "test_model.pkl"
    train_model(
        train_data_path=train_file,
        model_output_path=model_file,
        model_version="test_v1",
        use_mlflow=False,
    )
    return model_file
