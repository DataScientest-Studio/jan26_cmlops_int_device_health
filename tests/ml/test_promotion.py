"""
Tests for model promotion: evaluate_promotion, statistical significance, bootstrap CI.
"""

import numpy as np
import pytest

from src.training.promotion import (
    bootstrap_confidence_interval,
    compare_models_statistically,
    evaluate_promotion,
    statistical_significance_test,
)


class TestEvaluatePromotion:
    """Promotion decision logic."""

    def test_better_challenger_promoted(self):
        decision = evaluate_promotion(
            champion_metrics={"test_accuracy": 0.85, "test_f1_score": 0.82},
            challenger_metrics={"test_accuracy": 0.87, "test_f1_score": 0.84},
            min_improvement=0.01,
            metric_name="test_accuracy",
        )
        assert decision["should_promote"] is True
        assert decision["improvement"] == pytest.approx(0.02)

    def test_worse_challenger_rejected(self):
        decision = evaluate_promotion(
            champion_metrics={"test_accuracy": 0.90},
            challenger_metrics={"test_accuracy": 0.85},
            min_improvement=0.01,
            metric_name="test_accuracy",
        )
        assert decision["should_promote"] is False
        assert decision["improvement"] < 0

    def test_below_threshold_rejected(self):
        decision = evaluate_promotion(
            champion_metrics={"test_accuracy": 0.85},
            challenger_metrics={"test_accuracy": 0.855},
            min_improvement=0.01,
            metric_name="test_accuracy",
        )
        assert decision["should_promote"] is False

    def test_require_all_metrics_improve(self):
        decision = evaluate_promotion(
            champion_metrics={"test_accuracy": 0.85, "test_f1_score": 0.82},
            challenger_metrics={"test_accuracy": 0.87, "test_f1_score": 0.80},
            min_improvement=0.01,
            metric_name="test_accuracy",
            require_all_metrics_improve=True,
        )
        assert decision["should_promote"] is False

    def test_metrics_comparison_dict(self):
        decision = evaluate_promotion(
            champion_metrics={"test_accuracy": 0.85, "test_f1_score": 0.82},
            challenger_metrics={"test_accuracy": 0.87, "test_f1_score": 0.84},
            min_improvement=0.01,
        )
        comp = decision["metrics_comparison"]
        assert comp["test_accuracy"]["difference"] == pytest.approx(0.02)
        assert comp["test_accuracy"]["improved"] is True


class TestStatisticalSignificance:
    """statistical_significance_test and bootstrap_confidence_interval."""

    def test_significant_improvement(self):
        champion = np.array([0, 1, 0, 1, 0, 1, 0, 1] * 10)
        challenger = np.array([0, 1, 1, 1, 0, 1, 1, 1] * 10)
        truth = np.array([0, 1, 1, 1, 0, 1, 1, 1] * 10)
        result = statistical_significance_test(champion, challenger, truth, alpha=0.05)
        assert result["is_significant"] is True
        assert result["p_value"] < 0.05

    def test_identical_not_significant(self):
        preds = np.array([0, 1, 0, 1, 0, 1] * 10)
        truth = np.array([0, 1, 1, 1, 0, 1] * 10)
        result = statistical_significance_test(preds, preds, truth, alpha=0.05)
        assert result["is_significant"] is False
        assert result["mean_improvement"] == pytest.approx(0.0)

    def test_bootstrap_ci(self):
        champion = np.array([0, 1, 0, 1, 0, 1] * 10)
        challenger = np.array([0, 1, 1, 1, 0, 1] * 10)
        truth = np.array([0, 1, 1, 1, 0, 1] * 10)
        result = bootstrap_confidence_interval(champion, challenger, truth, n_bootstrap=100)
        assert result["ci_lower"] <= result["mean_difference"] <= result["ci_upper"]

    def test_bootstrap_identical_includes_zero(self):
        preds = np.array([0, 1, 0, 1, 0, 1] * 10)
        truth = np.array([0, 1, 1, 1, 0, 1] * 10)
        result = bootstrap_confidence_interval(preds, preds, truth, n_bootstrap=100)
        assert result["ci_lower"] <= 0 <= result["ci_upper"]
        assert result["is_significant"] is False


class TestCompareModelsStatistically:
    """compare_models_statistically."""

    def test_comparison_structure(self):
        champion = np.array([0, 1, 0, 1, 0, 1] * 10)
        challenger = np.array([0, 1, 1, 1, 0, 1] * 10)
        truth = np.array([0, 1, 1, 1, 0, 1] * 10)
        result = compare_models_statistically(
            model_name="test",
            champion_version=1,
            challenger_version=2,
            test_predictions_champion=champion.tolist(),
            test_predictions_challenger=challenger.tolist(),
            ground_truth=truth.tolist(),
        )
        assert result["champion_version"] == 1
        assert result["challenger_version"] == 2
        assert "statistical_test" in result
        assert "bootstrap_ci" in result
        assert "recommendation" in result
