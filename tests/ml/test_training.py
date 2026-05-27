"""
Tests for model training: basic training, parameters, evaluation, error handling.
"""

import pytest

from src.signal_processing.signal_generator import generate_signal
from src.training import evaluate_model, load_model, train_model

from .conftest import save_dataset


class TestModelTraining:
    """train_model basic behavior."""

    def test_basic_training(self, sample_training_data, tmp_path):
        train_file, _ = sample_training_data
        model_file = tmp_path / "model.pkl"
        results = train_model(
            train_data_path=train_file,
            model_output_path=model_file,
            model_version="v1.0",
            use_mlflow=False,
        )
        assert results["model_version"] == "v1.0"
        assert results["train_samples"] == 32  # 80% of 40
        assert results["test_samples"] == 8  # 20% of 40
        assert 0.0 <= results["train_accuracy"] <= 1.0
        assert "test_accuracy" in results
        assert model_file.exists()

    def test_includes_gold_standard_split(self, sample_training_data, tmp_path):
        train_file, _ = sample_training_data
        model_file = tmp_path / "model.pkl"
        results = train_model(
            train_data_path=train_file,
            model_output_path=model_file,
            model_version="v1.0",
            use_mlflow=False,
        )
        assert "confusion_matrix" in results
        assert "classification_report" in results

    def test_custom_params(self, sample_training_data, tmp_path):
        train_file, _ = sample_training_data
        model_file = tmp_path / "model.pkl"
        results = train_model(
            train_data_path=train_file,
            model_output_path=model_file,
            model_version="v1.0",
            max_iter=500,
            C=0.5,
            use_mlflow=False,
        )
        assert results["train_samples"] == 32
        assert model_file.exists()

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Training data not found"):
            train_model(
                train_data_path=tmp_path / "nonexistent.json",
                model_output_path=tmp_path / "model.pkl",
                model_version="v1.0",
                use_mlflow=False,
            )

    def test_insufficient_samples(self, tmp_path):
        signals = [generate_signal("gaussian", drift_scenario="baseline", seed=1)]
        train_file = tmp_path / "tiny.json"
        save_dataset(signals, train_file)
        with pytest.raises(ValueError, match="Insufficient training samples"):
            train_model(
                train_data_path=train_file,
                model_output_path=tmp_path / "model.pkl",
                model_version="v1.0",
                use_mlflow=False,
            )


class TestModelLoading:
    """load_model artifact structure."""

    def test_load_model(self, trained_model):
        artifact = load_model(trained_model)
        assert "model" in artifact
        assert "scaler" in artifact
        assert "feature_names" in artifact
        assert artifact["model_version"] == "test_v1"
        expected_features = [
            "fwhm",
            "peak_height",
            "peak_area",
            "noise_level",
            "snr",
            "peak_center",
        ]
        assert artifact["feature_names"] == expected_features

    def test_load_model_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Model not found"):
            load_model(tmp_path / "nonexistent.pkl")


class TestModelEvaluation:
    """evaluate_model on test set."""

    def test_evaluate(self, trained_model, sample_test_data):
        test_file, _ = sample_test_data
        results = evaluate_model(model_path=trained_model, test_data_path=test_file)
        assert results["test_samples"] == 10
        assert 0.0 <= results["test_accuracy"] <= 1.0
        assert isinstance(results["confusion_matrix"], list)
        assert isinstance(results["classification_report"], str)

    def test_evaluate_file_not_found(self, trained_model, tmp_path):
        with pytest.raises(FileNotFoundError, match="Test data not found"):
            evaluate_model(model_path=trained_model, test_data_path=tmp_path / "nope.json")
