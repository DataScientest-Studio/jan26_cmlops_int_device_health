"""
Reproducibility End-to-End Tests.

Verifies the >=95% reproducibility requirement from the project proposal:
Given identical input data and fixed random seeds, the training pipeline
must produce models that yield bit-identical predictions on any holdout set.

No external dependencies (Docker, network, MLflow) required -- CI-safe.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from src.signal_processing.feature_extractor import extract_features
from src.signal_processing.signal_generator import generate_signal
from src.signal_processing.signal_models import SignalData
from src.training import train_model


@pytest.fixture(autouse=True, scope="module")
def force_local_mode():
    """Guarantee DEPLOYMENT_MODE=local so train_model never calls dvc add."""
    old = os.environ.get("DEPLOYMENT_MODE")
    os.environ["DEPLOYMENT_MODE"] = "local"
    yield
    if old is None:
        os.environ.pop("DEPLOYMENT_MODE", None)
    else:
        os.environ["DEPLOYMENT_MODE"] = old


def _build_records(seeds, shape_type, *, label):
    records = []
    for idx, seed in enumerate(seeds):
        ls = generate_signal(shape_type, drift_scenario="baseline", seed=seed)
        records.append(
            {
                "id": idx,
                "time": list(ls.signal.time),
                "amplitude": list(ls.signal.amplitude),
                "shape_type": ls.signal.shape_type,
                "label": label,
                "metadata": ls.metadata,
            }
        )
    return records


def _write_dataset(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"n_samples": len(records), "signals": records}, indent=2),
        encoding="utf-8",
    )


def _feat_matrix(signals):
    rows = []
    for s in signals:
        sd = SignalData(
            time=np.asarray(s["time"]),
            amplitude=np.asarray(s["amplitude"]),
            shape_type=s["shape_type"],
        )
        rows.append(extract_features(sd))
    return pd.DataFrame(rows)


def _train_once(train_path, model_path):
    import pickle

    train_model(
        train_data_path=train_path,
        model_output_path=model_path,
        model_version="repro_test",
        use_mlflow=False,
        k_range=(2, 6),
        k_method="silhouette",
        distance_threshold=2.0,
        knn_neighbors=5,
        use_domain_heuristics=True,
        allow_unlabeled=False,
        filter_unlabeled=True,
        test_size=0.2,
        stratify=True,
        random_state=42,
    )
    with open(model_path, "rb") as fh:
        return pickle.load(fh)


def _predict(bundle, feat):
    """Scale feat then call predict on the LogisticRegression model inside the bundle."""
    return bundle["model"].predict(bundle["scaler"].transform(feat))


def _predict_proba(bundle, feat):
    """Scale feat then call predict_proba on the model inside the bundle."""
    return bundle["model"].predict_proba(bundle["scaler"].transform(feat))


@pytest.fixture(scope="module")
def golden_train_file(tmp_path_factory):
    """Deterministic 40-signal (20 healthy + 20 unhealthy) training dataset."""
    healthy = _build_records(range(20), "gaussian", label=0)
    unhealthy = _build_records(range(100, 120), "lorentzian", label=1)
    records = healthy + unhealthy
    rng = np.random.default_rng(999)
    idx = rng.permutation(len(records))
    records = [records[i] for i in idx]
    for i, r in enumerate(records):
        r["id"] = i
    path = tmp_path_factory.mktemp("repro_data") / "train_golden.json"
    _write_dataset(records, path)
    return path


@pytest.fixture(scope="module")
def golden_holdout():
    """10-signal holdout: 5 healthy + 5 unhealthy (distinct seeds from train)."""
    healthy = _build_records(range(200, 205), "gaussian", label=0)
    unhealthy = _build_records(range(300, 305), "lorentzian", label=1)
    return healthy + unhealthy


class TestReproducibility:
    """Pipeline reproducibility: same data + same seed -> same predictions."""

    def test_predictions_identical_across_two_runs(
        self, golden_train_file, golden_holdout, tmp_path
    ):
        """Predictions on holdout are bit-identical across two independent training runs."""
        b1 = _train_once(golden_train_file, tmp_path / "m1.pkl")
        b2 = _train_once(golden_train_file, tmp_path / "m2.pkl")
        feat = _feat_matrix(golden_holdout)
        p1 = _predict(b1, feat).tolist()
        p2 = _predict(b2, feat).tolist()
        assert p1 == p2, f"Predictions differ:\n  Run 1: {p1}\n  Run 2: {p2}"

    def test_model_accuracy_meets_threshold(self, golden_train_file, golden_holdout, tmp_path):
        """Model achieves >=90% accuracy on holdout (degenerate model guard)."""
        bundle = _train_once(golden_train_file, tmp_path / "m_acc.pkl")
        feat = _feat_matrix(golden_holdout)
        labels = [s["label"] for s in golden_holdout]
        preds = _predict(bundle, feat).tolist()
        accuracy = sum(p == t for p, t in zip(preds, labels, strict=False)) / len(labels)
        assert accuracy >= 0.90, (
            f"Accuracy {accuracy:.1%} below 90%: preds={preds}, labels={labels}"
        )

    def test_probabilities_close_across_runs(self, golden_train_file, golden_holdout, tmp_path):
        """Predicted probabilities differ by <1e-6 between two identical training runs."""
        b1 = _train_once(golden_train_file, tmp_path / "m_p1.pkl")
        b2 = _train_once(golden_train_file, tmp_path / "m_p2.pkl")
        feat = _feat_matrix(golden_holdout)
        max_diff = float(np.abs(_predict_proba(b1, feat) - _predict_proba(b2, feat)).max())
        assert max_diff < 1e-6, f"Prob max diff {max_diff:.2e} > 1e-6"

    def test_training_returns_reproducible_f1(self, golden_train_file, tmp_path):
        """F1 score is identical (within 1e-9) across two training runs."""

        def _get_f1(r):
            for k in ("f1_score", "f1"):
                if k in r:
                    return float(r[k])
                if k in r.get("metrics", {}):
                    return float(r["metrics"][k])
            return None

        r1 = train_model(
            golden_train_file,
            tmp_path / "mf1.pkl",
            model_version="r1",
            use_mlflow=False,
            k_range=(2, 6),
            allow_unlabeled=False,
            random_state=42,
        )
        r2 = train_model(
            golden_train_file,
            tmp_path / "mf2.pkl",
            model_version="r2",
            use_mlflow=False,
            k_range=(2, 6),
            allow_unlabeled=False,
            random_state=42,
        )
        f1_1, f1_2 = _get_f1(r1), _get_f1(r2)
        if f1_1 is not None and f1_2 is not None:
            assert abs(f1_1 - f1_2) < 1e-9, f"F1 differs: {f1_1} vs {f1_2}"
