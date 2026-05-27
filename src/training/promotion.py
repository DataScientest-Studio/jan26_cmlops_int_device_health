"""
Automated Champion/Challenger Model Promotion Logic.

This module provides statistical testing and decision logic for automated
model promotion. It compares challenger models (Staging) against the current
champion (Production) and promotes the best challenger if it meets criteria:

1. **Statistical Significance**: Performance improvement is significant (t-test)
2. **Minimum Improvement**: Meets minimum threshold (e.g., +0.5% accuracy)
3. **Validation Metrics**: All validation metrics pass thresholds

Promotion Decision Flow:
    Champion (Production) vs Challengers (Staging)
        ↓
    Statistical Testing (t-test, bootstrap)
        ↓
    Threshold Check (min improvement)
        ↓
    Promote Best Challenger
        ↓
    Archive Old Champion

Usage:
    >>> from src.training.promotion import evaluate_promotion, auto_promote_model
    >>>
    >>> # Evaluate if promotion should happen
    >>> decision = evaluate_promotion(
    ...     champion_metrics={"test_accuracy": 0.85, "test_f1_score": 0.82},
    ...     challenger_metrics={"test_accuracy": 0.87, "test_f1_score": 0.84},
    ...     min_improvement=0.01,
    ... )
    >>> if decision["should_promote"]:
    ...     print(f"Promote: {decision['reason']}")
    >>>
    >>> # Automated promotion with MLflow registry
    >>> result = auto_promote_model(
    ...     model_name="device_health_classifier",
    ...     metric_name="test_accuracy",
    ...     min_improvement=0.005,
    ...     require_significance=True,
    ... )
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from src.training.registry import (
    get_production_models,
    get_staging_models,
    promote_model,
)


def evaluate_promotion(
    champion_metrics: dict[str, float],
    challenger_metrics: dict[str, float],
    min_improvement: float = 0.02,
    metric_name: str = "test_f1_score",
    require_all_metrics_improve: bool = False,
) -> dict[str, Any]:
    """
    Evaluate whether a challenger should be promoted to production.

    Args:
        champion_metrics: Champion model metrics (e.g., {"test_f1_score": 0.85})
        challenger_metrics: Challenger model metrics
        min_improvement: Minimum required improvement (default: 2% for F1)
        metric_name: Primary metric for comparison (default: "test_f1_score")
        require_all_metrics_improve: If True, all metrics must improve

    Returns:
        Dict with promotion decision:
        {
            "should_promote": bool,
            "reason": str,
            "improvement": float,
            "champion_metric": float,
            "challenger_metric": float,
            "metrics_comparison": dict,
        }

    Example:
        >>> decision = evaluate_promotion(
        ...     champion_metrics={"test_f1_score": 0.82, "test_accuracy": 0.85},
        ...     challenger_metrics={"test_f1_score": 0.84, "test_accuracy": 0.87},
        ...     min_improvement=0.02,
        ... )
        >>> print(decision["should_promote"])  # True
        >>> print(decision["reason"])  # "Challenger outperforms by 2.44%"
    """
    champion_value = champion_metrics.get(metric_name, 0.0)
    challenger_value = challenger_metrics.get(metric_name, 0.0)

    improvement = challenger_value - champion_value
    improvement_pct = improvement / champion_value if champion_value > 0 else 0

    # Compare all metrics
    metrics_comparison = {}
    all_improved = True

    for metric in set(champion_metrics.keys()) | set(challenger_metrics.keys()):
        champ_val = champion_metrics.get(metric, 0.0)
        chall_val = challenger_metrics.get(metric, 0.0)
        diff = chall_val - champ_val

        metrics_comparison[metric] = {
            "champion": champ_val,
            "challenger": chall_val,
            "difference": diff,
            "improved": diff >= 0,
        }

        if diff < 0 and require_all_metrics_improve:
            all_improved = False

    # Decision logic
    should_promote = False
    reason = ""

    if challenger_value <= champion_value:
        reason = f"Challenger does not outperform champion ({challenger_value:.4f} ≤ {champion_value:.4f})"
    elif improvement < min_improvement:
        reason = f"Improvement ({improvement:.4f}) below minimum threshold ({min_improvement:.4f})"
    elif require_all_metrics_improve and not all_improved:
        reason = "Not all metrics improved"
    else:
        should_promote = True
        reason = f"Challenger outperforms by {improvement_pct:.2%} (Δ={improvement:.4f})"

    return {
        "should_promote": should_promote,
        "reason": reason,
        "improvement": improvement,
        "improvement_pct": improvement_pct,
        "champion_metric": champion_value,
        "challenger_metric": challenger_value,
        "metrics_comparison": metrics_comparison,
    }


def statistical_significance_test(
    champion_predictions: list[int] | np.ndarray,
    challenger_predictions: list[int] | np.ndarray,
    ground_truth: list[int] | np.ndarray,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Test if challenger's performance is statistically significantly better than champion.

    Uses paired t-test on per-sample accuracy (correct vs incorrect predictions).

    Args:
        champion_predictions: Champion model predictions
        challenger_predictions: Challenger model predictions
        ground_truth: True labels
        alpha: Significance level (default: 0.05)

    Returns:
        Dict with test results:
        {
            "is_significant": bool,
            "p_value": float,
            "t_statistic": float,
            "champion_correct_rate": float,
            "challenger_correct_rate": float,
            "confidence_level": float,
        }

    Example:
        >>> result = statistical_significance_test(
        ...     champion_predictions=[0, 1, 1, 0, 1],
        ...     challenger_predictions=[0, 1, 1, 1, 1],
        ...     ground_truth=[0, 1, 1, 1, 1],
        ...     alpha=0.05
        ... )
        >>> print(result["is_significant"])
    """
    champion_pred = np.array(champion_predictions)
    challenger_pred = np.array(challenger_predictions)
    y_true = np.array(ground_truth)

    if len(y_true) == 0:
        raise ValueError("Cannot perform statistical test with empty predictions")

    if len(champion_pred) != len(y_true) or len(challenger_pred) != len(y_true):
        raise ValueError("Predictions and ground truth must have same length")

    # Per-sample correctness (1 if correct, 0 if incorrect)
    champion_correct = (champion_pred == y_true).astype(int)
    challenger_correct = (challenger_pred == y_true).astype(int)

    # Paired t-test (challenger - champion)
    differences = challenger_correct - champion_correct

    # Use one-sided t-test (challenger > champion)
    t_statistic, p_value = stats.ttest_1samp(differences, 0, alternative="greater")

    champion_accuracy = champion_correct.mean()
    challenger_accuracy = challenger_correct.mean()

    return {
        "is_significant": bool(p_value < alpha),
        "p_value": float(p_value),
        "t_statistic": float(t_statistic),
        "champion_accuracy": float(champion_accuracy),
        "challenger_accuracy": float(challenger_accuracy),
        "confidence_level": 1 - alpha,
        "mean_improvement": float(differences.mean()),
    }


def bootstrap_confidence_interval(
    champion_predictions: list[int] | np.ndarray,
    challenger_predictions: list[int] | np.ndarray,
    ground_truth: list[int] | np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """
    Compute bootstrap confidence interval for accuracy difference.

    Args:
        champion_predictions: Champion model predictions
        challenger_predictions: Challenger model predictions
        ground_truth: True labels
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default: 0.95)

    Returns:
        Dict with bootstrap results:
        {
            "mean_difference": float,
            "ci_lower": float,
            "ci_upper": float,
            "is_significant": bool,  # True if CI doesn't include 0
        }

    Example:
        >>> result = bootstrap_confidence_interval(
        ...     champion_predictions=[0, 1, 1, 0, 1],
        ...     challenger_predictions=[0, 1, 1, 1, 1],
        ...     ground_truth=[0, 1, 1, 1, 1],
        ... )
        >>> print(f"95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
    """
    champion_pred = np.array(champion_predictions)
    challenger_pred = np.array(challenger_predictions)
    y_true = np.array(ground_truth)

    n_samples = len(y_true)
    differences = []

    rng = np.random.default_rng(42)  # Reproducible

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)

        champ_sample = champion_pred[indices]
        chall_sample = challenger_pred[indices]
        y_sample = y_true[indices]

        # Compute accuracy difference
        champ_acc = (champ_sample == y_sample).mean()
        chall_acc = (chall_sample == y_sample).mean()
        diff = chall_acc - champ_acc

        differences.append(diff)

    differences = np.array(differences)  # type: ignore[assignment]

    # Confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(differences, 100 * alpha / 2)
    ci_upper = np.percentile(differences, 100 * (1 - alpha / 2))

    return {
        "mean_difference": float(differences.mean()),  # type: ignore[attr-defined]
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "is_significant": bool(ci_lower > 0),  # CI doesn't include 0
        "confidence_level": confidence_level,
    }


def auto_promote_model(
    model_name: str,
    metric_name: str = "test_f1_score",
    min_improvement: float = 0.02,
    require_significance: bool = False,
    alpha: float = 0.05,
    archive_old_champion: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Automatically evaluate and promote best challenger model to production.

    Args:
        model_name: Registered model name in MLflow
        metric_name: Primary metric for comparison (default: "test_f1_score")
        min_improvement: Minimum required improvement (default: 2% for F1)
        require_significance: If True, require statistical significance
            (Note: requires access to predictions, not implemented in this version)
        alpha: Significance level for statistical test
        archive_old_champion: Archive old production model after promotion
        dry_run: If True, only evaluate without promoting

    Returns:
        Dict with promotion results:
        {
            "promoted": bool,
            "new_champion_version": int | None,
            "old_champion_version": int | None,
            "decision": dict,
            "reason": str,
        }

    Example:
        >>> result = auto_promote_model(
        ...     model_name="device_health_classifier",
        ...     metric_name="test_f1_score",
        ...     min_improvement=0.02,
        ... )
        >>> if result["promoted"]:
        ...     print(f"Promoted v{result['new_champion_version']}")

    Note:
        Statistical significance testing requires access to model predictions
        and ground truth, which is not available from MLflow metrics alone.
        Set require_significance=False for metric-based promotion only.
    """
    # Get current production model (champion)
    production_models = get_production_models(model_name)

    if not production_models:
        return {
            "promoted": False,
            "new_champion_version": None,
            "old_champion_version": None,
            "decision": {"should_promote": False},
            "reason": "No production model exists. Register and promote a baseline model first.",
        }

    champion = production_models[0]  # Assume latest production
    champion_version = champion["version"]
    champion_metrics = champion["metrics"]

    # Get staging models (challengers)
    staging_models = get_staging_models(model_name)

    if not staging_models:
        return {
            "promoted": False,
            "new_champion_version": None,
            "old_champion_version": champion_version,
            "decision": {"should_promote": False},
            "reason": "No challenger models in Staging.",
        }

    # Evaluate each challenger
    best_challenger = None
    best_decision = None
    best_improvement = -float("inf")

    for challenger in staging_models:
        decision = evaluate_promotion(
            champion_metrics=champion_metrics,
            challenger_metrics=challenger["metrics"],
            min_improvement=min_improvement,
            metric_name=metric_name,
        )

        if decision["should_promote"] and decision["improvement"] > best_improvement:
            best_challenger = challenger
            best_decision = decision
            best_improvement = decision["improvement"]

    # No suitable challenger found
    if best_challenger is None:
        reasons = []
        last_decision: dict[str, Any] = {}
        for challenger in staging_models:
            last_decision = evaluate_promotion(
                champion_metrics=champion_metrics,
                challenger_metrics=challenger["metrics"],
                min_improvement=min_improvement,
                metric_name=metric_name,
            )
            reasons.append(f"v{challenger['version']}: {last_decision['reason']}")

        # Always include champion/challenger metric so callers can display them
        # (avoids downstream champion_f1=0 when no challenger meets the threshold).
        best_staging = staging_models[0] if staging_models else None
        return {
            "promoted": False,
            "new_champion_version": None,
            "old_champion_version": champion_version,
            "decision": {
                "should_promote": False,
                "champion_metric": champion_metrics.get(metric_name, 0.0),
                "challenger_metric": (
                    best_staging["metrics"].get(metric_name, 0.0) if best_staging else 0.0
                ),
            },
            "reason": "No challengers meet promotion criteria. " + "; ".join(reasons),
        }

    # Promote best challenger
    if dry_run:
        return {
            "promoted": False,
            "new_champion_version": best_challenger["version"],
            "old_champion_version": champion_version,
            "decision": best_decision,
            "reason": f"[DRY RUN] Would promote v{best_challenger['version']}: {best_decision['reason']}",  # type: ignore[index]
        }

    # Actual promotion
    promote_model(
        model_name,
        best_challenger["version"],
        stage="Production",
        archive_existing_production=archive_old_champion,
    )

    return {
        "promoted": True,
        "new_champion_version": best_challenger["version"],
        "old_champion_version": champion_version,
        "decision": best_decision,
        "reason": f"Promoted v{best_challenger['version']} to Production. {best_decision['reason']}",  # type: ignore[index]
    }


def compare_models_statistically(
    model_name: str,
    champion_version: int,
    challenger_version: int,
    test_predictions_champion: list[int],
    test_predictions_challenger: list[int],
    ground_truth: list[int],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Compare two model versions with statistical testing.

    Args:
        model_name: Registered model name
        champion_version: Champion model version
        challenger_version: Challenger model version
        test_predictions_champion: Champion predictions on test set
        test_predictions_challenger: Challenger predictions on test set
        ground_truth: True labels for test set
        alpha: Significance level

    Returns:
        Dict with comprehensive comparison:
        {
            "champion_version": int,
            "challenger_version": int,
            "statistical_test": dict,
            "bootstrap_ci": dict,
            "recommendation": str,
        }

    Example:
        >>> comparison = compare_models_statistically(
        ...     model_name="device_health_classifier",
        ...     champion_version=1,
        ...     challenger_version=2,
        ...     test_predictions_champion=[0, 1, 1, 0],
        ...     test_predictions_challenger=[0, 1, 1, 1],
        ...     ground_truth=[0, 1, 1, 1],
        ... )
        >>> print(comparison["recommendation"])
    """
    # Statistical significance test
    stat_test = statistical_significance_test(
        test_predictions_champion, test_predictions_challenger, ground_truth, alpha
    )

    # Bootstrap confidence interval
    bootstrap_ci = bootstrap_confidence_interval(
        test_predictions_champion, test_predictions_challenger, ground_truth
    )

    # Recommendation
    if stat_test["is_significant"] and bootstrap_ci["is_significant"]:
        recommendation = (
            f"Strong evidence to promote v{challenger_version}. "
            f"Improvement is statistically significant (p={stat_test['p_value']:.4f}) "
            f"and 95% CI excludes zero ({bootstrap_ci['ci_lower']:.4f}, {bootstrap_ci['ci_upper']:.4f})."
        )
    elif stat_test["is_significant"]:
        recommendation = (
            f"Moderate evidence to promote v{challenger_version}. "
            f"T-test significant (p={stat_test['p_value']:.4f}) but bootstrap CI includes zero."
        )
    else:
        recommendation = (
            f"Insufficient evidence to promote v{challenger_version}. "
            f"Difference not statistically significant (p={stat_test['p_value']:.4f})."
        )

    return {
        "champion_version": champion_version,
        "challenger_version": challenger_version,
        "statistical_test": stat_test,
        "bootstrap_ci": bootstrap_ci,
        "recommendation": recommendation,
    }
