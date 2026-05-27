"""Generates a minimal bootstrap model artifact for CI environments.

Called by .github/workflows/test.yml and live-tests.yml when
models/bootstrap_model.pkl is not present (should normally be committed,
but this acts as a safety net for forks / fresh checkouts).

IMPORTANT: Signal generation and model training use **exactly the same defaults**
as the Streamlit UI Greenfield use case (scripts/greenfield_init.py BootstrapConfig).
This ensures the CI bootstrap model is not a synthetic artefact diverging from what
real users get when they run the Greenfield flow.

Defaults mirror BootstrapConfig:
  - Healthy (Gaussian):   μ∈[48,52], σ∈[2.0,3.0], H∈[2.5,3.0], noise∈[0.01,0.02]
  - Unhealthy (Lorentzian): μ∈[42,58], γ=σ*1.1775, σ∈[3.8,5.1], H∈[1.0,1.5], noise∈[0.06,0.10]
  - Classifier: LogisticRegression (C=1.0, penalty=l2, solver=lbfgs, max_iter=1000)
    matching params.yaml train.* settings.

No MLflow overhead — this is a pure CI artefact, not an experiment run.
"""

import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Allow imports from project root when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUT_PATH = Path("models/bootstrap_model.pkl")

FEATURE_NAMES = [
    "fwhm",
    "peak_height",
    "peak_area",
    "noise_level",
    "snr",
    "peak_center",
]

# ── Signal generation defaults — MUST MATCH BootstrapConfig in greenfield_init.py ──
_SEED = 42
_N_SAMPLES = 100  # total signals (matches BootstrapConfig.n_samples default)
_GAUSSIAN_FRACTION = 0.7  # 70 healthy + 30 unhealthy (matches BootstrapConfig default)

# Gaussian (healthy) signal parameter ranges
_GAUSS_MU_RANGE = (48.0, 52.0)
_GAUSS_SIGMA_RANGE = (2.0, 3.0)
_GAUSS_HEIGHT_RANGE = (2.5, 3.0)
_GAUSS_NOISE_RANGE = (0.01, 0.02)

# Lorentzian (unhealthy) signal parameter ranges
_LOR_MU_RANGE = (42.0, 58.0)
_LOR_SIGMA_RANGE = (3.8, 5.1)  # user-facing σ; converted to γ via × 1.1775
_LOR_HEIGHT_RANGE = (1.0, 1.5)
_LOR_NOISE_RANGE = (0.06, 0.10)
_GAMMA_SIGMA_FACTOR = 1.1775  # γ = σ × 1.1775 (FWHM equivalence)

# LogisticRegression — MUST MATCH params.yaml train.* settings
_LR_C = 1.0
_LR_PENALTY = "l2"
_LR_SOLVER = "lbfgs"
_LR_MAX_ITER = 1000


def _build_feature_matrix() -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic signals using identical parameters to BootstrapConfig defaults.

    Returns:
        X: Feature matrix (N, 6)
        y: Labels (N,) — 0 = healthy (Gaussian), 1 = unhealthy (Lorentzian)
    """
    from src.signal_processing.feature_extractor import extract_features
    from src.signal_processing.signal_generator import generate_signal
    from src.signal_processing.signal_models import SignalData

    rng = np.random.RandomState(_SEED)

    n_gauss = int(_N_SAMPLES * _GAUSSIAN_FRACTION)
    n_lor = _N_SAMPLES - n_gauss

    rows: list[list[float]] = []
    labels: list[int] = []

    # Healthy class (Gaussian peaks, label=0) — identical to _make_dataset() in greenfield_init
    for i in range(n_gauss):
        mu = rng.uniform(*_GAUSS_MU_RANGE)
        sigma = rng.uniform(*_GAUSS_SIGMA_RANGE)
        height = rng.uniform(*_GAUSS_HEIGHT_RANGE)
        noise = rng.uniform(*_GAUSS_NOISE_RANGE)
        sig = generate_signal(
            shape_type="gaussian",
            mu=mu,
            width_param=sigma,
            height=height,
            noise_level=noise,
            seed=_SEED + i,
        )
        sd = SignalData(time=sig.signal.time, amplitude=sig.signal.amplitude, shape_type="gaussian")
        feats = extract_features(sd)
        rows.append([feats.get(name) or 0.0 for name in FEATURE_NAMES])
        labels.append(0)

    # Unhealthy class (Lorentzian peaks, label=1) — identical to _make_dataset()
    for i in range(n_lor):
        mu = rng.uniform(*_LOR_MU_RANGE)
        sigma_l = rng.uniform(*_LOR_SIGMA_RANGE)
        gamma = sigma_l * _GAMMA_SIGMA_FACTOR
        height = rng.uniform(*_LOR_HEIGHT_RANGE)
        noise = rng.uniform(*_LOR_NOISE_RANGE)
        sig = generate_signal(
            shape_type="lorentzian",
            mu=mu,
            width_param=gamma,
            height=height,
            noise_level=noise,
            seed=_SEED + n_gauss + i,
        )
        sd = SignalData(
            time=sig.signal.time, amplitude=sig.signal.amplitude, shape_type="lorentzian"
        )
        feats = extract_features(sd)
        rows.append([feats.get(name) or 0.0 for name in FEATURE_NAMES])
        labels.append(1)

    return np.array(rows, dtype=float), np.array(labels, dtype=int)


def generate() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Building feature matrix from synthetic signals (greenfield defaults)…")
    X, y = _build_feature_matrix()  # noqa: N806
    n_healthy = int((y == 0).sum())
    n_unhealthy = int((y == 1).sum())
    print(f"  Training set: {len(y)} samples ({n_healthy} healthy, {n_unhealthy} unhealthy)")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # noqa: N806

    clf = LogisticRegression(
        C=_LR_C,
        solver=_LR_SOLVER,
        max_iter=_LR_MAX_ITER,
        random_state=_SEED,
    )
    clf.fit(X_scaled, y)

    train_acc = clf.score(X_scaled, y)
    print(f"  Training accuracy: {train_acc:.3f}")

    artifact = {
        "model": clf,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "model_version": "bootstrap-ci-0.1",
        "algorithm": "LogisticRegression",
        "trained_at": datetime.now().isoformat(),
        "trainer": "ci-bootstrap",
        "optimal_k": 2,
        "cluster_info": {},
    }

    with OUTPUT_PATH.open("wb") as fh:
        pickle.dump(artifact, fh)

    print(f"✅ CI bootstrap model written to {OUTPUT_PATH}")


if __name__ == "__main__":
    generate()
    sys.exit(0)
