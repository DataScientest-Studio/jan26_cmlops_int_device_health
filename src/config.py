"""Centralised project-wide constants.

All hard-coded thresholds, label conventions, and domain values live here
so that every module can import from a single authoritative source.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Label convention
# ---------------------------------------------------------------------------

LABEL_HEALTHY: int = 0
"""Ground-truth label for a *healthy* device signal (Gaussian peak)."""

LABEL_UNHEALTHY: int = 1
"""Ground-truth label for an *unhealthy* device signal (Lorentzian peak)."""

# ---------------------------------------------------------------------------
# Drift scenario names
# Canonical keys used across the signal generator, drift provocation use case,
# and the predictions page.  The display names are derived from these.
# ---------------------------------------------------------------------------

DRIFT_SCENARIO_BASELINE: str = "baseline"
DRIFT_SCENARIO_DATA_DRIFT: str = "data_drift"
DRIFT_SCENARIO_CONCEPT_DRIFT: str = "concept_drift"
DRIFT_SCENARIO_FEATURE_DRIFT: str = "feature_drift"
DRIFT_SCENARIO_PRIOR_PROBABILITY_DRIFT: str = "prior_probability_drift"

#: All drift scenarios supported by the Drift Provocation use case, in order.
DRIFT_SCENARIOS: list[str] = [
    DRIFT_SCENARIO_DATA_DRIFT,
    DRIFT_SCENARIO_CONCEPT_DRIFT,
    DRIFT_SCENARIO_FEATURE_DRIFT,
    DRIFT_SCENARIO_PRIOR_PROBABILITY_DRIFT,
]

#: Human-readable display names keyed by scenario key.
DRIFT_SCENARIO_LABELS: dict[str, str] = {
    DRIFT_SCENARIO_BASELINE: "Baseline",
    DRIFT_SCENARIO_DATA_DRIFT: "Data Drift",
    DRIFT_SCENARIO_CONCEPT_DRIFT: "Concept Drift",
    DRIFT_SCENARIO_FEATURE_DRIFT: "Feature Drift",
    DRIFT_SCENARIO_PRIOR_PROBABILITY_DRIFT: "Prior Probability Drift",
}

# ---------------------------------------------------------------------------
# Retraining thresholds
# ---------------------------------------------------------------------------

#: Hard minimum: the validate_data DAG task raises an error if labeled signals
#: in the database fall below this count.  20 is intentionally low so that
#: retraining can be triggered early in sparse-labelling scenarios (≈10 % of
#: predictions receive a ground-truth label).  Statistical quality is governed
#: by the recommended threshold below.
MIN_LABELED_SIGNALS: int = 20

#: Recommended minimum for statistically meaningful retraining.
MIN_LABELED_SIGNALS_RECOMMENDED: int = 100

#: Default fraction of generated signals that receive a sparse ground-truth
#: label (10 % matches realistic production labeling budget).
DEFAULT_LABEL_INJECTION_PCT: int = 10

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

#: Virtual device UUID used by the Drift Simulator (satisfies UUID format).
DRIFT_SIM_DEVICE_ID: str = "00000000-0000-0000-dddd-000000000000"
