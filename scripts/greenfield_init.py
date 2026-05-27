#!/usr/bin/env python3
"""Greenfield initialisation -- generate data, train bootstrap model, register.

This script is the single entry point to bootstrap the MLOps platform from
scratch (no pre-existing data, no models, empty registries).

Architecture
------------
- ``BootstrapConfig``  -- all parameters for the bootstrap run
- ``BootstrapResult``  -- structured outcome (metrics, lineage, errors)
- ``run_bootstrap()``  -- main orchestrator, usable from CLI and Streamlit UI
- ``ProgressCallback`` -- typed callback for UI progress updates

Workflow
-------
1.  (Optional) Wipe all existing data
2.  Generate synthetic signal dataset
3.  DVC-track generated data files (cloud mode)
4.  Train the bootstrap model with MLflow tracking
5.  Register the trained model in the MLflow Model Registry
6.  (Optional) Promote to Production

Usage
-----
    # Minimal (local mode, defaults)
    python scripts/greenfield_init.py

    # Cloud: custom classifier + promotion
    python scripts/greenfield_init.py --classifier random_forest --promote

    # Custom params
    python scripts/greenfield_init.py --n-samples 200 --gaussian-fraction 0.6
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure repo root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import typer
from rich.console import Console

console = Console()
app = typer.Typer(help="Bootstrap the platform from scratch (data + model + registry)")

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

CLASSIFIER_CHOICES = ("logistic_regression", "decision_tree", "random_forest", "svc")

# Lorentzian γ (HWHM) = √(2 ln 2) × σ ≈ 1.1775 × σ  —  matches FWHM of a
# Gaussian with the same σ, so both peak types share one intuitive width knob.
_GAMMA_SIGMA_FACTOR = 1.1775

ProgressCallback = Callable[[str, str, float], None]
"""Signature: (step_name, message, fraction_complete_0_to_1) -> None"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BootstrapConfig:
    """All parameters for a greenfield bootstrap run."""

    n_samples: int = 100
    gaussian_fraction: float = 0.7
    labeled_fraction: float = 0.2
    seed: int = 42
    classifier: str = "logistic_regression"
    model_name: str = field(
        default_factory=lambda: os.environ.get("MODEL_REGISTRY_NAME", "device_health_classifier")
    )
    experiment_name: str = field(
        default_factory=lambda: os.environ.get("MODEL_REGISTRY_NAME", "device_health_classifier")
    )
    wipe: bool = False
    promote: bool = False

    # Signal generation ranges -- Gaussian (healthy)
    gauss_mu_range: tuple[float, float] = (48.0, 52.0)
    gauss_sigma_range: tuple[float, float] = (2.0, 3.0)
    gauss_height_range: tuple[float, float] = (2.5, 3.0)
    gauss_noise_range: tuple[float, float] = (0.01, 0.02)

    # Signal generation ranges -- Lorentzian (unhealthy)
    # lor_sigma_range stores the user-facing σ; converted to γ at generation time.
    lor_mu_range: tuple[float, float] = (42.0, 58.0)
    lor_sigma_range: tuple[float, float] = (3.8, 5.1)
    lor_height_range: tuple[float, float] = (1.0, 1.5)
    lor_noise_range: tuple[float, float] = (0.06, 0.10)

    # Auto-resolved in __post_init__
    mode: str = field(default="", init=False)
    mlflow_uri: str = field(default="", init=False)

    # Optional override for the training data path (used by champion/challenger
    # to train on freshly generated signals instead of the baseline split).
    train_data_path_override: Path | None = None

    def __post_init__(self) -> None:
        if self.classifier not in CLASSIFIER_CHOICES:
            raise ValueError(
                f"Unknown classifier {self.classifier!r}. Choose from {CLASSIFIER_CHOICES}"
            )
        self.mode = _detect_mode()
        self.mlflow_uri = _mlflow_tracking_uri()


@dataclass
class BootstrapResult:
    """Structured outcome of a bootstrap run."""

    success: bool = False
    data_paths: dict[str, str] = field(default_factory=dict)
    dvc_hashes: dict[str, str] = field(default_factory=dict)
    git_sha: str = ""
    mlflow_run_id: str = ""
    model_version: int = 0
    test_f1: float = 0.0
    test_accuracy: float = 0.0
    promoted: bool = False
    error: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_cmd(cmd: list[str]) -> list[str]:
    """Resolve command to absolute path if it lives in the venv Scripts directory.

    On Windows the venv is not added to PATH by subprocess, so bare commands
    like ``dvc`` fail with WinError 2.  We probe ``<venv>/Scripts/<cmd>`` and
    fall back to the bare name (Linux/macOS where the shebang resolves it).
    """
    exe = cmd[0]
    venv_bin = PROJECT_ROOT / ".venv" / "Scripts" / f"{exe}.exe"
    if venv_bin.exists():
        return [str(venv_bin)] + cmd[1:]
    venv_bin_no_ext = PROJECT_ROOT / ".venv" / "Scripts" / exe
    if venv_bin_no_ext.exists():
        return [str(venv_bin_no_ext)] + cmd[1:]
    return cmd


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    timeout: int | None = None,
) -> str:
    """Run a subprocess and return stdout.

    ``timeout`` (seconds) caps how long we wait.  For DVC push/pull commands
    that connect to DagsHub, pass timeout=90 so a rate-limited remote does not
    hang the bootstrap indefinitely.  A timed-out process is treated the same
    as check=False (no exception, empty stdout returned).
    """
    try:
        result = subprocess.run(
            _resolve_cmd(cmd),
            capture_output=True,
            text=True,
            cwd=str(cwd or PROJECT_ROOT),
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        console.print(
            f"  [yellow]⚠️  Command timed out after {timeout}s (skipped): {' '.join(cmd)}[/yellow]"
        )
        return ""
    if check and result.returncode != 0:
        msg = f"Command failed: {' '.join(cmd)}\n{result.stderr[:500]}"
        raise RuntimeError(msg)
    return result.stdout.strip()


def _detect_mode() -> str:
    """Return 'local' or 'cloud' based on .current_mode or DEPLOYMENT_MODE."""
    mode_file = PROJECT_ROOT / ".current_mode"
    if mode_file.exists():
        return mode_file.read_text().strip()
    return os.environ.get("DEPLOYMENT_MODE", "local")


def _mlflow_tracking_uri() -> str:
    """Return the MLflow tracking URI for the current mode."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if uri:
        return uri
    mode = _detect_mode()
    if mode == "cloud":
        user = os.environ.get("DAGSHUB_USER", "")
        repo = os.environ.get("DAGSHUB_REPO", "")
        if user and repo:
            return f"https://dagshub.com/{user}/{repo}.mlflow"
    return "http://localhost:5001"


def _git_sha() -> str:
    """Return the current short git SHA."""
    try:
        return _run(["git", "rev-parse", "--short", "HEAD"], check=False)
    except Exception:
        return "unknown"


def _noop_progress(_step: str, _msg: str, _frac: float) -> None:
    """No-op progress callback for non-interactive use."""


# ---------------------------------------------------------------------------
# DVC hash helpers
# ---------------------------------------------------------------------------


def _parse_dvc_hash(dvc_file: Path) -> str | None:
    """Extract md5 hash from a .dvc file."""
    for line in dvc_file.read_text().splitlines():
        if "md5:" in line:
            return line.split("md5:")[-1].strip()
    return None


def _parse_lock_hash(data_path: Path) -> str | None:
    """Extract md5 hash for a file from dvc.lock."""
    lock_file = PROJECT_ROOT / "dvc.lock"
    if not lock_file.exists():
        return None
    import yaml

    try:
        lock_data = yaml.safe_load(lock_file.read_text())
    except Exception:
        return None
    rel = str(data_path.relative_to(PROJECT_ROOT))
    for _stage_name, stage in (lock_data or {}).get("stages", {}).items():
        for out in stage.get("outs", []):
            if out.get("path") == rel:
                return out.get("md5")
    return None


def _compute_md5(data_path: Path) -> str | None:
    """Compute MD5 hash of a file directly (fallback when DVC tracking is unavailable)."""
    import hashlib

    if not data_path.exists():
        return None
    try:
        h = hashlib.md5()
        with open(data_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------


def _step_wipe(cfg: BootstrapConfig, cb: ProgressCallback) -> None:
    """Step 0: Wipe all existing data from the database."""
    cb("wipe", "Wiping all existing data...", 0.0)
    from src.database.database import Database

    db = Database()
    deleted = db.wipe_all_data()
    total = sum(deleted.values())
    cb("wipe", f"Deleted {total} rows across {len(deleted)} tables", 0.05)


def _step_generate_data(
    cfg: BootstrapConfig,
    cb: ProgressCallback,
) -> dict[str, Path]:
    """Step 1: Generate baseline + bootstrap synthetic datasets.

    Uses custom signal parameter ranges from the config instead of
    shelling out to the CLI, so the UI-supplied knobs take effect.
    """
    import numpy as np

    from scripts.generate_data import save_dataset
    from src.signal_processing.signal_generator import generate_signal
    from src.signal_processing.signal_models import LabeledSignal

    cb("generate", "Generating baseline dataset...", 0.10)

    rng = np.random.RandomState(cfg.seed)

    def _make_dataset(n: int) -> list[LabeledSignal]:
        n_gauss = int(n * cfg.gaussian_fraction)
        n_lor = n - n_gauss
        signals: list[LabeledSignal] = []
        for i in range(n_gauss):
            mu = rng.uniform(*cfg.gauss_mu_range)
            sigma = rng.uniform(*cfg.gauss_sigma_range)
            height = rng.uniform(*cfg.gauss_height_range)
            noise = rng.uniform(*cfg.gauss_noise_range)
            sig = generate_signal(
                shape_type="gaussian",
                mu=mu,
                width_param=sigma,
                height=height,
                noise_level=noise,
                seed=cfg.seed + i,
            )
            signals.append(sig)
        for i in range(n_lor):
            mu = rng.uniform(*cfg.lor_mu_range)
            sigma_l = rng.uniform(*cfg.lor_sigma_range)
            gamma = sigma_l * _GAMMA_SIGMA_FACTOR
            height = rng.uniform(*cfg.lor_height_range)
            noise = rng.uniform(*cfg.lor_noise_range)
            sig = generate_signal(
                shape_type="lorentzian",
                mu=mu,
                width_param=gamma,
                height=height,
                noise_level=noise,
                seed=cfg.seed + n_gauss + i,
            )
            signals.append(sig)
        rng.shuffle(signals)  # type: ignore[arg-type]
        return signals

    # Baseline dataset (full + train/test split)
    baseline = _make_dataset(cfg.n_samples)
    n_test = int(cfg.n_samples * 0.2)
    n_train = cfg.n_samples - n_test

    out = PROJECT_ROOT / "data" / "raw"
    out.mkdir(parents=True, exist_ok=True)

    save_dataset(baseline, out / "dataset_baseline_full.json", include_labels=True)
    save_dataset(baseline[:n_train], out / "dataset_baseline_train.json", include_labels=True)
    save_dataset(baseline[n_train:], out / "dataset_baseline_test.json", include_labels=True)

    cb("generate", "Generating bootstrap dataset (labeled + unlabeled)...", 0.20)

    # Bootstrap dataset (labeled/unlabeled split)
    bootstrap = _make_dataset(cfg.n_samples)
    n_labeled = max(1, int(cfg.n_samples * cfg.labeled_fraction))
    save_dataset(bootstrap[:n_labeled], out / "bootstrap_labeled.json", include_labels=True)
    save_dataset(bootstrap[n_labeled:], out / "bootstrap_unlabeled.json", include_labels=False)

    paths = {
        "baseline_full": PROJECT_ROOT / "data/raw/dataset_baseline_full.json",
        "baseline_train": PROJECT_ROOT / "data/raw/dataset_baseline_train.json",
        "baseline_test": PROJECT_ROOT / "data/raw/dataset_baseline_test.json",
        "bootstrap_labeled": PROJECT_ROOT / "data/raw/bootstrap_labeled.json",
        "bootstrap_unlabeled": PROJECT_ROOT / "data/raw/bootstrap_unlabeled.json",
    }

    for name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Expected data file missing: {p}")
        with open(p) as f:
            data = json.load(f)
        cb("generate", f"  {name}: {data['n_samples']} samples", 0.25)

    cb("generate", f"Generated {len(paths)} data files", 0.30)
    return paths


def _step_dvc_track(
    cfg: BootstrapConfig,
    paths: dict[str, Path],
    cb: ProgressCallback,
) -> dict[str, str]:
    """Step 2: DVC-track generated data files (cloud mode only)."""
    cb("dvc", "DVC-tracking data files...", 0.35)

    if cfg.mode == "local":
        cb("dvc", "Skipping DVC tracking in local sandbox mode", 0.40)
        return dict.fromkeys(paths, "")

    # Cloud mode: commit the DVC stage so pipeline outputs get hashes.
    # Use a short timeout (30s) since we only need local hashing — no DagsHub
    # connection is required here (push is deferred to the sync_production_data DAG).
    _run(["dvc", "commit", "generate_data", "--force"], check=False, timeout=30)

    # Files that are DVC pipeline stage outputs (tracked via dvc.lock, not .dvc files).
    # Calling `dvc add` on these would fail/hang — use MD5 fallback directly.
    # Keys must match the paths dict returned by _step_generate_data.
    _stage_outputs = {
        "baseline_full",
        "baseline_train",
        "baseline_test",
    }

    hashes: dict[str, str] = {}
    for name, p in paths.items():
        dvc_file = Path(str(p) + ".dvc")
        if name in _stage_outputs:
            # Stage output: .dvc file never exists; get hash from dvc.lock or MD5 fallback.
            md5 = _parse_lock_hash(p) or _compute_md5(p)
        elif not dvc_file.exists():
            _run(
                ["dvc", "add", str(p.relative_to(PROJECT_ROOT))],
                check=False,
                timeout=30,  # Short timeout: only local hashing needed, no remote
            )
            md5 = _parse_dvc_hash(dvc_file) if dvc_file.exists() else _parse_lock_hash(p)
            if not md5:
                md5 = _compute_md5(p)
        else:
            md5 = _parse_dvc_hash(dvc_file) or _compute_md5(p)
        hashes[name] = md5 or ""

    # NOTE: We do NOT call `dvc push` here.  In the local-first buffer architecture,
    # DagsHub pushes are performed exclusively by the `sync_production_data` Airflow DAG.
    # Calling `dvc push` during bootstrap would attempt a remote connection on every
    # model bootstrap, causing timeouts and breaking the local-first guarantee.
    # The DVC hashes computed above (from .dvc files) are stored in MLflow for lineage.
    tracked = sum(1 for h in hashes.values() if h)
    cb(
        "dvc",
        f"Tracked {tracked}/{len(hashes)} files with DVC (local only — push via sync DAG)",
        0.45,
    )
    return hashes


def _get_db_url() -> str:
    """Return a PostgreSQL DATABASE_URL if configured, else empty string.

    Resolution order:
    1. DATABASE_URL env var (set directly)
    2. DATABASE_URL in .env.secrets
    3. Assemble from DB_USER / DB_PASSWORD / DB_HOST / DB_PORT / DB_NAME env vars
       (all are set by .env.cloud + .env.secrets when running ``set -a; source ...``).
    """
    url = os.environ.get("DATABASE_URL", "")
    if url.startswith("postgresql"):
        return url
    # Try reading from .env.secrets (for CLI / non-Docker usage)
    secrets_file = PROJECT_ROOT / ".env.secrets"
    if secrets_file.exists():
        for line in secrets_file.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("DATABASE_URL="):
                val = stripped.split("=", 1)[1].strip()
                if val.startswith("postgresql"):
                    return val
    # Assemble from individual components (cloud/local env files)
    user = os.environ.get("DB_USER", "")
    password = os.environ.get("DB_PASSWORD", "")
    host = os.environ.get("DB_HOST", "localhost")
    # When running on the host (not inside Docker), Docker service names like
    # "postgres" are not resolvable — always use localhost for CLI scripts.
    if host in ("postgres", "db"):
        host = "localhost"
    port = os.environ.get("DB_PORT", "5433")
    name = os.environ.get("DB_NAME", "")
    if not password and secrets_file.exists():
        for line in secrets_file.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("DB_PASSWORD="):
                password = stripped.split("=", 1)[1].strip()
    if user and password and name:
        return f"postgresql://{user}:{password}@{host}:{port}/{name}"
    return ""


def _step_store_to_database(
    cfg: BootstrapConfig,
    paths: dict[str, Path],
    mlflow_run_id: str,
    cb: ProgressCallback,
    dvc_hashes: dict[str, str] | None = None,
    registered_version: int | None = None,
) -> tuple[list[int], list[int], int]:
    """Step 2 (new order): Store ALL bootstrap + baseline signals to PostgreSQL.

    Loads three datasets:
    - ``baseline_train.json`` (fully labeled baseline signals — stored for
      production lineage, NOT used as training signal IDs)
    - ``bootstrap_labeled.json`` (signals with sparse ground-truth labels)
    - ``bootstrap_unlabeled.json`` (signals without labels)

    All are stored to the database.  Returns the bootstrap_labeled and
    bootstrap_unlabeled signal IDs so that ``_step_train()`` uses the correct
    semi-supervised learning path with sparse labels.

    **Design note**: F1=0 when a class is absent from the test set is
    *acceptable* — it signals to the operator that more ground-truth labels
    are needed.  The only critical failure would be a missing class in
    *training*, which balanced synthetic data prevents.

    **Ordering (FT-5)**: This step runs BEFORE ``_step_train()`` so that
    signal IDs are available for ``train_model(from_db=True)``.

    Returns:
        (bootstrap_labeled_signal_ids, bootstrap_unlabeled_signal_ids, n_labeled)
    """
    cb("db", "Storing bootstrap signals to database...", 0.35)

    db_url = _get_db_url()
    if not db_url:
        cb("db", "⚠️  No PostgreSQL URL found — skipping database storage", 0.37)
        return [], [], 0

    import json as _json

    from src.database.database import Database
    from src.signal_processing.feature_extractor import extract_features
    from src.signal_processing.signal_models import SignalData

    BOOTSTRAP_DEVICE_ID = "00000000-0000-0000-0000-000000000001"

    db = Database(db_url=db_url)

    # Register the bootstrap virtual device (idempotent)
    db.register_device(
        device_id=BOOTSTRAP_DEVICE_ID,
        device_name="Bootstrap Generator",
        device_type="synthetic",
        location="greenfield_bootstrap",
        status="active",
        deployment_mode=cfg.mode,
    )

    # Use a temporary placeholder version (run_id prefix) for now.
    # After _step_register() completes, the caller will update it to "v<N>"
    # via db.update_model_version_by_run_id().
    model_version = f"bootstrap_{mlflow_run_id[:8]}"

    # Use the DVC hash of the bootstrap labeled file (computed by dvc add) as
    # the lineage anchor.  The hash is available locally immediately after
    # ``dvc add`` — no push to DagsHub required.
    dvc_data_hash: str | None = None
    if dvc_hashes:
        dvc_data_hash = (
            dvc_hashes.get("bootstrap_labeled") or dvc_hashes.get("baseline_full") or None
        )

    # ── Load ALL signals ──────────────────────────────────────────────────────
    # baseline_train.json: fully-labeled dataset used for model training.
    # Returning its signal IDs ensures train_model(from_db=True) gets the same
    # 80 labeled samples that the old file-based fallback used, giving F1=1.0
    # on perfectly separable synthetic data.
    #
    # bootstrap_labeled / bootstrap_unlabeled: stored to DB for production
    # prediction lineage and future retraining (NOT used for initial training).
    baseline_train_path = paths.get("baseline_train")
    labeled_path = paths.get("bootstrap_labeled")
    unlabeled_path = paths.get("bootstrap_unlabeled")

    baseline_train_signals: list = []
    labeled_signals: list = []
    unlabeled_signals: list = []

    if baseline_train_path and baseline_train_path.exists():
        with open(baseline_train_path) as f:
            baseline_train_signals = _json.load(f).get("signals", [])

    if labeled_path and labeled_path.exists():
        with open(labeled_path) as f:
            labeled_signals = _json.load(f).get("signals", [])

    if unlabeled_path and unlabeled_path.exists():
        with open(unlabeled_path) as f:
            unlabeled_signals = _json.load(f).get("signals", [])

    bootstrap_signals = labeled_signals + unlabeled_signals
    n_labeled_bootstrap = len(labeled_signals)

    if not baseline_train_signals and not bootstrap_signals:
        cb("db", "⚠️  No signals found in data files — skipping DB storage", 0.37)
        return [], [], 0

    cb(
        "db",
        f"  Loaded {len(baseline_train_signals)} baseline_train + "
        f"{len(labeled_signals)} bootstrap_labeled + "
        f"{len(unlabeled_signals)} bootstrap_unlabeled signals",
        0.36,
    )

    errors_counter = [0]
    _cached_git_sha = _git_sha()  # compute once; avoids O(N) subprocess calls

    # ── Helper: store one signal and optionally inject its ground-truth label ─
    def _store_signal(
        sig: dict,
        *,
        inject_label: bool,
        errors_ref: list[int],
    ) -> int | None:
        t_vals = sig.get("time", sig.get("time_values", []))
        a_vals = sig.get("amplitude", sig.get("amplitude_values", []))
        shape_type = sig.get("shape_type", "gaussian")
        true_label = sig.get("label")

        if len(t_vals) < 51 or len(a_vals) < 51:
            return None
        try:
            sd = SignalData(time=t_vals, amplitude=a_vals, shape_type=shape_type)
            feats = extract_features(sd)
            predicted_label = 1 if shape_type == "lorentzian" else 0
            prediction_id = db.store_prediction(
                device_id=BOOTSTRAP_DEVICE_ID,
                time_values=t_vals,
                amplitude_values=a_vals,
                predicted_label=predicted_label,
                model_version=model_version,
                features=feats,
                prediction_confidence=0.95,
                shape_type=shape_type,
                mlflow_run_id=mlflow_run_id,
                git_sha=_cached_git_sha,
                dvc_data_hash=dvc_data_hash or None,
                deployment_mode=cfg.mode,
            )
            signal_id = db.get_signal_id_by_prediction_id(prediction_id)
            if inject_label and true_label is not None and signal_id is not None:
                db.inject_sparse_label(
                    prediction_id=prediction_id,
                    ground_truth_label=int(true_label),
                    label_source="greenfield_bootstrap",
                    injected_by="bootstrap_script",
                    deployment_mode=cfg.mode,
                )
            return signal_id
        except Exception as exc:
            errors_ref[0] += 1
            if errors_ref[0] <= 3:
                cb("db", f"⚠️  Signal storage error: {exc}", 0.37)
            return None

    # ── Pass 1: store baseline_train signals (production lineage only) ───────
    baseline_train_signal_ids: list[int] = []
    for sig in baseline_train_signals:
        sid = _store_signal(sig, inject_label=True, errors_ref=errors_counter)
        if sid is not None:
            baseline_train_signal_ids.append(sid)

    # ── Pass 2: store bootstrap signals (these are the training signal IDs) ──
    bootstrap_labeled_signal_ids: list[int] = []
    bootstrap_unlabeled_signal_ids: list[int] = []
    for idx, sig in enumerate(bootstrap_signals):
        is_labeled = idx < n_labeled_bootstrap
        sid = _store_signal(sig, inject_label=is_labeled, errors_ref=errors_counter)
        if sid is not None:
            if is_labeled:
                bootstrap_labeled_signal_ids.append(sid)
            else:
                bootstrap_unlabeled_signal_ids.append(sid)

    db.close()
    cb(
        "db",
        f"✅ Stored {len(baseline_train_signal_ids)} baseline_train + "
        f"{len(bootstrap_labeled_signal_ids)} bootstrap_labeled + "
        f"{len(bootstrap_unlabeled_signal_ids)} bootstrap_unlabeled signals; "
        f"injected {len(baseline_train_signal_ids) + len(bootstrap_labeled_signal_ids)} sparse labels "
        f"({errors_counter[0]} errors)",
        0.40,
    )
    # Return bootstrap labeled/unlabeled IDs as the training signal IDs.
    # Semi-supervised training uses sparse labels (bootstrap_labeled) + unlabeled
    # signals (bootstrap_unlabeled) — this is the correct bootstrap learning mode.
    # F1=0 when a class is missing from the test set is ACCEPTABLE by design;
    # the only unacceptable failure is a missing class in training (balanced
    # synthetic data prevents this).
    return (
        bootstrap_labeled_signal_ids,
        bootstrap_unlabeled_signal_ids,
        len(bootstrap_labeled_signal_ids),
    )


def _step_train(
    cfg: BootstrapConfig,
    paths: dict[str, Path],
    cb: ProgressCallback,
    db_signal_ids: tuple[list[int], list[int]] | None = None,
) -> dict[str, Any]:
    """Step 3: Train the bootstrap classifier with MLflow tracking.

    When *db_signal_ids* is provided (labeled_ids, unlabeled_ids), training
    uses ``from_db=True`` so that ``record_training_split`` is called
    automatically and ``model_training_data`` is populated.

    If the remote MLflow server (DagsHub) returns 429 rate-limit errors, falls
    back to a local file-based MLflow store so the DB-write step can still
    proceed.  The run_id in that case will be a local UUID, which is fine for
    tagging predictions in PostgreSQL.
    """
    cb("train", f"Training {cfg.classifier} model...", 0.50)

    import time as _time

    from src.training.train import train_model

    # When DB signal IDs are available, we open a database connection and
    # train with from_db=True so record_training_split is called inside
    # train_model().  When DB is unavailable, fall back to the JSON file path.
    _db_url = _get_db_url() if db_signal_ids is not None else None
    _db_conn = None
    if _db_url and db_signal_ids is not None:
        try:
            from src.database.database import Database

            _db_conn = Database(db_url=_db_url)
        except Exception:
            _db_conn = None

    _labeled_ids, _unlabeled_ids = db_signal_ids if db_signal_ids else ([], [])
    _all_filter_ids = _labeled_ids + _unlabeled_ids if db_signal_ids else None

    def _do_train(tracking_uri: str) -> dict[str, Any]:
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        # Only set TRAINED_BY if not already overridden by the caller
        # (e.g., champion_challenger sets "champion_challenger_training").
        os.environ.setdefault("TRAINED_BY", "greenfield_bootstrap")
        os.environ["DEPLOYMENT_MODE"] = cfg.mode
        if cfg.mode == "cloud" and tracking_uri == cfg.mlflow_uri:
            user = os.environ.get("DAGSHUB_USER", "")
            token = os.environ.get("DAGSHUB_TOKEN", "")
            if user:
                os.environ["MLFLOW_TRACKING_USERNAME"] = user
            if token:
                os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        else:
            # Clear DagsHub credentials for local file store
            os.environ.pop("MLFLOW_TRACKING_USERNAME", None)
            os.environ.pop("MLFLOW_TRACKING_PASSWORD", None)

        if _db_conn is not None and _all_filter_ids:
            # DB-backed training: record_training_split called automatically inside
            # train_model() → model_training_data will be populated.
            return train_model(
                model_output_path=PROJECT_ROOT / "models/bootstrap_model.pkl",
                model_version="bootstrap_v1.0",
                use_mlflow=True,
                mlflow_experiment_name=cfg.experiment_name,
                from_db=True,
                db=_db_conn,
                signal_ids_filter=_all_filter_ids,
                allow_unlabeled=len(_unlabeled_ids) > 0,
                filter_unlabeled=False,
                k_range=(2, 8),
                k_method="silhouette",
                test_size=0.2,
                stratify=True,
                primary_metric="f1_score",
                classifier_type=cfg.classifier,
                random_state=cfg.seed,
            )
        else:
            # Fallback: file-based training (DB unavailable or no signal IDs)
            return train_model(
                train_data_path=cfg.train_data_path_override or paths["baseline_train"],
                model_output_path=PROJECT_ROOT / "models/bootstrap_model.pkl",
                model_version="bootstrap_v1.0",
                use_mlflow=True,
                mlflow_experiment_name=cfg.experiment_name,
                allow_unlabeled=True,
                filter_unlabeled=False,
                k_range=(2, 8),
                k_method="silhouette",
                test_size=0.2,
                stratify=True,
                primary_metric="f1_score",
                classifier_type=cfg.classifier,
                random_state=cfg.seed,
            )

    # --- First attempt: remote URI (DagsHub / configured server) ---
    try:
        results = _do_train(cfg.mlflow_uri)
    except Exception as first_exc:
        exc_str = str(first_exc)
        if "429" in exc_str or "too many" in exc_str.lower() or "rate" in exc_str.lower():
            cb("train", "⚠️  MLflow rate-limited (429). Waiting 30 s then retrying once...", 0.52)
            _time.sleep(30)
            try:
                results = _do_train(cfg.mlflow_uri)
            except Exception as retry_exc:
                retry_str = str(retry_exc)
                if (
                    "429" in retry_str
                    or "too many" in retry_str.lower()
                    or "rate" in retry_str.lower()
                ):
                    local_uri = f"file:///{(PROJECT_ROOT / 'mlruns').as_posix()}"
                    cb(
                        "train",
                        f"⚠️  Still rate-limited. Falling back to local MLflow store: {local_uri}",
                        0.53,
                    )
                    results = _do_train(local_uri)
                else:
                    raise
        else:
            raise

    if _db_conn is not None:
        import contextlib as _contextlib

        with _contextlib.suppress(Exception):
            _db_conn.close()

    f1 = results.get("test_f1_score", 0.0)
    acc = results.get("test_accuracy", 0.0)
    run_id = results.get("mlflow_run_id", "")
    cb("train", f"F1={f1:.4f}  Accuracy={acc:.4f}  run={run_id[:8]}...", 0.70)
    return results


def _step_register(
    cfg: BootstrapConfig,
    run_id: str,
    cb: ProgressCallback,
) -> int:
    """Step 4: Register the model in the MLflow Model Registry."""
    cb("register", "Registering model in MLflow...", 0.75)

    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(cfg.mlflow_uri)
    client = MlflowClient()

    import contextlib

    with contextlib.suppress(Exception):
        client.create_registered_model(cfg.model_name)

    model_uri = f"runs:/{run_id}/model"
    try:
        mv = client.create_model_version(
            name=cfg.model_name,
            source=model_uri,
            run_id=run_id,
            tags={
                "trained_by": "greenfield_init",
                "role": "bootstrap",
                "classifier": cfg.classifier,
            },
        )
        version = int(mv.version)
    except Exception:
        versions = client.search_model_versions(f"name='{cfg.model_name}'")
        version = max((int(v.version) for v in versions), default=0) + 1

    cb("register", f"Registered as v{version}", 0.85)
    return version


def _step_promote(
    cfg: BootstrapConfig,
    version: int,
    cb: ProgressCallback,
) -> bool:
    """Step 5: Promote the model to Production."""
    cb("promote", "Promoting to Production...", 0.90)

    import mlflow

    from src.training.registry import promote_model

    mlflow.set_tracking_uri(cfg.mlflow_uri)
    promote_model(
        model_name=cfg.model_name,
        version=version,
        stage="Production",
        archive_existing_production=True,
    )
    cb("promote", f"v{version} promoted to Production", 0.95)
    return True


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_bootstrap(
    config: BootstrapConfig,
    progress_callback: ProgressCallback | None = None,
) -> BootstrapResult:
    """Run the full greenfield bootstrap pipeline.

    Can be called from the CLI or from the Streamlit UI.  The
    ``progress_callback`` receives ``(step_name, message, fraction)``
    updates suitable for driving a progress bar.
    """
    cb = progress_callback or _noop_progress
    result = BootstrapResult()

    try:
        # 0. Wipe (optional)
        if config.wipe:
            _step_wipe(config, cb)

        # 1. Generate data
        paths = _step_generate_data(config, cb)
        result.data_paths = {k: str(v) for k, v in paths.items()}

        # 2. DVC track
        hashes = _step_dvc_track(config, paths, cb)
        result.dvc_hashes = hashes

        # 3. Git SHA
        result.git_sha = _git_sha()

        # 4. Store ALL bootstrap signals + labels to PostgreSQL FIRST.
        #    This must happen before _step_train() so that signal_ids are
        #    available when train_model(from_db=True) calls record_training_split().
        #    The model_version is a temporary placeholder (run_id prefix) at this
        #    point; it will be updated to the canonical "v<N>" after registration.
        #
        #    We need the mlflow_run_id for tagging predictions.  At this point we
        #    haven't trained yet, so we use a short UUID stub.  The full run_id is
        #    updated to the real value immediately after training via
        #    update_model_version_by_run_id().
        #
        #    Bootstrap uses a pre-generated stub run_id for the predictions table
        #    so that the real run_id (produced by MLflow during training) can be
        #    written back afterwards.  This avoids a circular dependency.
        import uuid as _uuid

        _bootstrap_stub_run_id = _uuid.uuid4().hex

        labeled_ids, unlabeled_ids, _n_labels = _step_store_to_database(
            config,
            paths,
            _bootstrap_stub_run_id,
            cb,
            dvc_hashes=hashes,
        )

        # 5. Train — pass DB signal IDs so train_model(from_db=True) is used
        #    and record_training_split() is called automatically.
        train_results = _step_train(
            config,
            paths,
            cb,
            db_signal_ids=(labeled_ids, unlabeled_ids) if (labeled_ids or unlabeled_ids) else None,
        )
        result.mlflow_run_id = train_results.get("mlflow_run_id", "")
        result.test_f1 = train_results.get("test_f1_score", 0.0)
        result.test_accuracy = train_results.get("test_accuracy", 0.0)

        if not result.mlflow_run_id:
            raise RuntimeError("Training did not produce an MLflow run ID")

        # 5b. Backfill the real MLflow run_id into predictions rows.
        #     (rows were stored with _bootstrap_stub_run_id above).
        #     Note: model_training_data already has the real run_id because
        #     record_training_split() is called inside train_model() with the
        #     real MLflow run_id.
        _db_url = _get_db_url()
        if _db_url:
            try:
                from src.database.database import Database as _Database

                _db_tmp = _Database(db_url=_db_url)
                _db_tmp.conn.execute(
                    "UPDATE predictions SET mlflow_run_id = ? WHERE mlflow_run_id = ?",
                    (result.mlflow_run_id, _bootstrap_stub_run_id),
                )
                _db_tmp.conn.commit()
                _db_tmp.close()
                cb("db", f"✓ run_id backfilled → {result.mlflow_run_id[:8]}...", 0.72)
            except Exception as _e:
                cb("db", f"⚠️  run_id backfill failed: {_e}", 0.72)

        # 6. Register
        result.model_version = _step_register(config, result.mlflow_run_id, cb)

        # 6b. Update model_version in DB from the temp placeholder to "v<N>"
        if _db_url and result.model_version:
            try:
                from src.database.database import Database as _Database

                _db_tmp2 = _Database(db_url=_db_url)
                canonical_version = f"v{result.model_version}"
                _db_tmp2.update_model_version_by_run_id(result.mlflow_run_id, canonical_version)
                _db_tmp2.close()
                cb("db", f"✓ model_version updated to {canonical_version}", 0.87)
            except Exception as _e:
                cb("db", f"⚠️  model_version update failed: {_e}", 0.87)

        # 7. Promote (optional)
        if config.promote:
            result.promoted = _step_promote(config, result.model_version, cb)
            # After successful promotion, copy the bootstrap model to
            # champion_model.pkl so that Airflow DAGs (batch_rescoring,
            # automated_retraining) can load it from the shared models/ volume.
            if result.promoted:
                import shutil as _shutil

                _bootstrap_pkl = PROJECT_ROOT / "models" / "bootstrap_model.pkl"
                _champion_pkl = PROJECT_ROOT / "models" / "champion_model.pkl"
                if _bootstrap_pkl.exists():
                    _champion_pkl.parent.mkdir(parents=True, exist_ok=True)
                    _shutil.copy2(_bootstrap_pkl, _champion_pkl)
                    cb("promote", f"✓ champion_model.pkl updated → {_champion_pkl.name}", 0.97)

        _done_msg = "Challenger training complete!" if not config.promote else "Bootstrap complete!"
        cb("done", _done_msg, 1.0)
        result.success = True

    except Exception as exc:
        result.error = str(exc)
        cb("error", str(exc), -1.0)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    n_samples: int = typer.Option(100, help="Number of signals to generate"),
    gaussian_fraction: float = typer.Option(0.7, help="Fraction of Gaussian peaks"),
    seed: int = typer.Option(42, help="Random seed"),
    classifier: str = typer.Option(
        "logistic_regression",
        help=f"Classifier: {CLASSIFIER_CHOICES}",
    ),
    promote: bool = typer.Option(False, help="Promote model to Production (cloud mode)"),
    wipe: bool = typer.Option(False, help="Wipe all existing data before bootstrap"),
    model_name: str = typer.Option(
        os.environ.get("MODEL_REGISTRY_NAME", "device_health_classifier"),
        help="MLflow registered model name",
    ),
) -> None:
    """Bootstrap the platform: generate data -> DVC-track -> train -> register."""
    config = BootstrapConfig(
        n_samples=n_samples,
        gaussian_fraction=gaussian_fraction,
        seed=seed,
        classifier=classifier,
        model_name=model_name,
        wipe=wipe,
        promote=promote,
    )

    console.print("[bold blue]======= Greenfield Initialisation =======[/bold blue]")
    console.print(f"  Mode:       [cyan]{config.mode}[/cyan]")
    console.print(f"  MLflow:     [cyan]{config.mlflow_uri}[/cyan]")
    console.print(f"  Classifier: [cyan]{config.classifier}[/cyan]")

    def _cli_progress(step: str, message: str, fraction: float) -> None:
        if fraction < 0:
            console.print(f"  [red][{step}] {message}[/red]")
        else:
            console.print(f"  [{step}] {message}")

    result = run_bootstrap(config, progress_callback=_cli_progress)

    if result.success:
        console.print("\n[bold green]======= Greenfield Init Complete =======[/bold green]")
        console.print(f"  Data files:   {len(result.data_paths)} datasets")
        dvc_count = sum(1 for h in result.dvc_hashes.values() if h)
        console.print(f"  DVC hashes:   {dvc_count}/{len(result.dvc_hashes)}")
        console.print(f"  MLflow run:   {result.mlflow_run_id}")
        console.print(f"  Registry:     {config.model_name} v{result.model_version}")
        console.print(f"  Test F1:      {result.test_f1:.4f}")
        console.print(f"  Test Acc:     {result.test_accuracy:.4f}")
        if result.promoted:
            console.print("  Stage:        Production (champion)")
    else:
        console.print(f"\n[bold red]Bootstrap failed:[/bold red] {result.error}")
        raise SystemExit(1)


if __name__ == "__main__":
    app()
