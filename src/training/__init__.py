"""
Training package for MLOps device health monitoring.

Provides:
- train_model(): Train classifier from labeled data
- predict(): Make prediction from raw signal
- load_model(): Load trained model from disk
- evaluate_model(): Evaluate model on test set
- predict_batch(): Batch predictions
- predict_from_file(): Predict from JSON signal file
"""

from .predict import predict, predict_batch, predict_from_file
from .train import evaluate_model, load_model, train_model

__all__ = [
    "train_model",
    "load_model",
    "evaluate_model",
    "predict",
    "predict_batch",
    "predict_from_file",
]
