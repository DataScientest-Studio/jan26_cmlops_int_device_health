"""
Tests for semi-supervised learning: K optimization, clustering, gold standard split.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.training.semi_supervised import (
    SemiSupervisedTrainer,
    create_gold_standard_split,
    select_sliding_window_data,
)


class TestKOptimization:
    """SemiSupervisedTrainer.optimize_k with different methods."""

    @pytest.fixture
    def synthetic_data(self):
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            n_classes=2,
            n_clusters_per_class=2,
            random_state=42,
        )
        return X, y

    @pytest.mark.parametrize("method", ["silhouette", "elbow", "calinski"])
    def test_optimize_k_methods(self, synthetic_data, method):
        X, y = synthetic_data
        trainer = SemiSupervisedTrainer(k_range=(2, 5), k_method=method)
        optimal_k, scores = trainer.optimize_k(X, np.ones(len(y), dtype=bool))
        assert 2 <= optimal_k <= 5
        assert trainer.optimal_k_ == optimal_k

    def test_insufficient_labeled_returns_min_k(self, synthetic_data):
        X, y = synthetic_data
        trainer = SemiSupervisedTrainer(k_range=(2, 10))
        labeled_mask = np.zeros(len(y), dtype=bool)
        labeled_mask[:5] = True
        optimal_k, scores = trainer.optimize_k(X, labeled_mask)
        assert optimal_k == 2


class TestClusterAndLabel:
    """SemiSupervisedTrainer.cluster_and_label."""

    @pytest.fixture
    def synthetic_data(self):
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            n_classes=2,
            n_clusters_per_class=2,
            random_state=42,
        )
        return X, y

    def test_fully_labeled(self, synthetic_data):
        X, y = synthetic_data
        trainer = SemiSupervisedTrainer(k_range=(2, 4), k_method="silhouette")
        clusters, labels, info = trainer.cluster_and_label(X, y)
        assert len(clusters) == len(y)
        assert len(labels) == len(y)
        assert set(labels).issubset({0, 1})
        for ci in info.values():
            assert "label" in ci
            assert not ci["is_pseudo_label"]

    def test_sparse_labels(self, synthetic_data):
        X, y = synthetic_data
        y_sparse = np.full_like(y, -1)
        rng = np.random.default_rng(42)
        labeled_idx = rng.choice(len(y), size=len(y) // 10, replace=False)
        y_sparse[labeled_idx] = y[labeled_idx]
        trainer = SemiSupervisedTrainer(k_range=(2, 5))
        _, labels, _ = trainer.cluster_and_label(X, y_sparse)
        assert set(labels).issubset({0, 1})

    def test_manual_k(self, synthetic_data):
        X, y = synthetic_data
        trainer = SemiSupervisedTrainer()
        _, _, info = trainer.cluster_and_label(X, y, k=3)
        assert trainer.optimal_k_ == 3
        assert len(info) == 3

    def test_insufficient_labeled_raises(self):
        X = np.random.default_rng(42).standard_normal((10, 5))
        y = np.full(10, -1)
        y[0] = 0
        trainer = SemiSupervisedTrainer()
        with pytest.raises(ValueError, match="at least 2 labeled samples"):
            trainer.cluster_and_label(X, y)

    def test_predict_before_fit_raises(self):
        trainer = SemiSupervisedTrainer()
        X = np.random.default_rng(42).standard_normal((10, 5))
        with pytest.raises(ValueError, match="Model not fitted"):
            trainer.predict_cluster_labels(X)


class TestSlidingWindowDataSelection:
    """select_sliding_window_data."""

    @pytest.fixture
    def ts_data(self):
        dates = pd.date_range("2024-01-01", periods=365, freq="D")
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            {
                "timestamp": dates,
                "ground_truth_label": rng.integers(0, 2, 365),
                "feature1": rng.standard_normal(365),
            }
        )

    def test_by_days(self, ts_data):
        result = select_sliding_window_data(
            ts_data,
            window_days=30,
            timestamp_col="timestamp",
            label_col="ground_truth_label",
        )
        assert 0 < len(result) <= 31

    def test_by_sample_size(self, ts_data):
        result = select_sliding_window_data(
            ts_data,
            window_size=50,
            timestamp_col="timestamp",
            label_col="ground_truth_label",
        )
        assert len(result) == 50

    def test_no_labeled_raises(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10),
                "ground_truth_label": [np.nan] * 10,
                "feature1": np.zeros(10),
            }
        )
        with pytest.raises(ValueError, match="No labeled samples"):
            select_sliding_window_data(df, window_size=5, label_col="ground_truth_label")


class TestGoldStandardSplit:
    """create_gold_standard_split."""

    @pytest.fixture
    def labeled_data(self):
        return make_classification(
            n_samples=200,
            n_features=10,
            n_classes=2,
            weights=[0.7, 0.3],
            random_state=42,
        )

    def test_basic_split(self, labeled_data):
        X, y = labeled_data
        X_tr, X_te, y_tr, y_te = create_gold_standard_split(X, y, test_size=0.2, random_state=42)
        assert len(X_tr) == 160
        assert len(X_te) == 40

    def test_stratified_preserves_distribution(self, labeled_data):
        X, y = labeled_data
        _, _, y_tr, y_te = create_gold_standard_split(
            X, y, test_size=0.2, stratify=True, random_state=42
        )
        train_ratio = np.mean(y_tr == 0)
        test_ratio = np.mean(y_te == 0)
        assert abs(train_ratio - test_ratio) < 0.05

    def test_reproducibility(self, labeled_data):
        X, y = labeled_data
        s1 = create_gold_standard_split(X, y, test_size=0.2, random_state=42)
        s2 = create_gold_standard_split(X, y, test_size=0.2, random_state=42)
        np.testing.assert_array_equal(s1[2], s2[2])
