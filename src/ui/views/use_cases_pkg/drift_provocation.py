"""Tab 4 — Drift Provocation: generate drifted signals, run KS-tests, visualise."""

from __future__ import annotations

import os

import streamlit as st

from src.ui.components.docker_utils import get_host
from src.ui.logging_ui import get_ui_logger

_logger = get_ui_logger(__name__)

from src.config import (
    DRIFT_SIM_DEVICE_ID,
    LABEL_HEALTHY,
    LABEL_UNHEALTHY,
)

from ._common import GAMMA_SIGMA_FACTOR, SECTION_CSS, get_host_db_url

# ---------------------------------------------------------------------------
# Drift types
# ---------------------------------------------------------------------------

DRIFT_TYPES: dict[str, str] = {
    "data_drift": (
        "**Data Drift (Sensor Degradation)** — Input feature distributions shift "
        "(e.g. peak centres move, noise increases) while the underlying labelling "
        "rule stays the same."
    ),
    "concept_drift": (
        "**Concept Drift (Process Change)** — The relationship between features "
        "and the target label changes: signals that *look* healthy now correspond "
        "to unhealthy devices and vice-versa."
    ),
    "feature_drift": (
        "**Feature Drift** — A single extracted feature shifts its distribution "
        "(e.g. all peak heights decrease) without a broad covariate change."
    ),
    "prior_probability_drift": (
        "**Prior Probability Drift (Label Shift)** — The class balance changes "
        "dramatically (e.g. 90 % unhealthy instead of the usual 30 %)."
    ),
}

# Feature columns used for KS-tests and distribution plots
FEATURE_COLS = ["fwhm", "peak_height", "peak_area", "noise_level", "snr", "peak_center"]


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------


def generate_batch(
    n: int,
    gaussian_fraction: float,
    mu_offset: float = 0.0,
    width_multiplier: float = 1.0,
    noise_multiplier: float = 1.0,
    height_multiplier: float = 1.0,
    swap_labels: bool = False,
    seed: int = 99,
    *,
    include_raw: bool = False,
) -> list[dict]:
    """Generate a batch of signals and return feature dicts.

    When *include_raw* is True each dict also contains:
      - ``time_values``  (list[float])
      - ``amplitude_values``  (list[float])
    so the full chain (raw signal → features → label) can be persisted.

    Parameters are clamped to Pydantic validation bounds after applying
    multipliers so that drift simulation does not trigger validation errors.
    """
    import numpy as np

    from src.signal_processing.feature_extractor import extract_features
    from src.signal_processing.signal_generator import generate_signal

    rng = np.random.RandomState(seed)
    n_gauss = int(n * gaussian_fraction)
    n_lor = n - n_gauss
    rows: list[dict] = []

    # Pydantic validation bounds
    max_noise = 0.15
    max_sigma, min_sigma = 6.0, 2.0
    max_gamma, min_gamma = 7.5, 2.36
    max_height_g, min_height_g = 3.5, 1.0
    max_height_l, min_height_l = 3.5, 0.8
    mu_lo, mu_hi = 35.0, 60.0

    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    for i in range(n_gauss):
        mu = _clamp(rng.uniform(48.0, 52.0) + mu_offset, mu_lo, mu_hi)
        sigma = _clamp(rng.uniform(2.0, 3.0) * width_multiplier, min_sigma, max_sigma)
        height = _clamp(rng.uniform(2.5, 3.0) * height_multiplier, min_height_g, max_height_g)
        noise = _clamp(rng.uniform(0.01, 0.02) * noise_multiplier, 0.0, max_noise)
        sig = generate_signal(
            shape_type="gaussian",
            mu=mu,
            width_param=sigma,
            height=height,
            noise_level=noise,
            seed=seed + i,
        )
        feats = extract_features(sig.signal)
        row: dict[str, object] = {
            **feats,
            "shape_type": "gaussian",
            "label": 1 if swap_labels else sig.label,
        }
        if include_raw:
            row["time_values"] = list(sig.signal.time)
            row["amplitude_values"] = list(sig.signal.amplitude)
        rows.append(row)

    for i in range(n_lor):
        mu = _clamp(rng.uniform(42.0, 58.0) + mu_offset, mu_lo, mu_hi)
        sigma_l = rng.uniform(3.8, 5.1) * width_multiplier
        gamma = _clamp(sigma_l * GAMMA_SIGMA_FACTOR, min_gamma, max_gamma)
        height = _clamp(rng.uniform(1.0, 1.5) * height_multiplier, min_height_l, max_height_l)
        noise = _clamp(rng.uniform(0.06, 0.10) * noise_multiplier, 0.0, max_noise)
        sig = generate_signal(
            shape_type="lorentzian",
            mu=mu,
            width_param=gamma,
            height=height,
            noise_level=noise,
            seed=seed + n_gauss + i,
        )
        feats = extract_features(sig.signal)
        row = dict(**feats, shape_type="lorentzian", label=0 if swap_labels else sig.label)
        if include_raw:
            row["time_values"] = list(sig.signal.time)
            row["amplitude_values"] = list(sig.signal.amplitude)
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# KS-tests
# ---------------------------------------------------------------------------


def ks_tests(ref_df, cur_df, feature_cols: list[str]) -> list[dict]:  # noqa: ANN001
    """Run KS-test per feature and return results."""
    from scipy.stats import ks_2samp

    results = []
    for col in feature_cols:
        r = ref_df[col].dropna()
        c = cur_df[col].dropna()
        if len(r) < 5 or len(c) < 5:
            continue
        stat, pval = ks_2samp(r, c)
        results.append(
            {
                "Feature": col,
                "KS Statistic": round(stat, 4),
                "p-value": round(pval, 6),
                "Drift?": "Yes" if pval < 0.05 else "No",
            }
        )
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _show_approval_outcome(mode: str, ar_run_id: str) -> None:
    """Poll automated_retraining for up to 90s and show whether the model was promoted.

    Called immediately after the user approves/rejects a challenger model from
    the drift provocation use case, so the user gets proper outcome feedback.
    """
    import base64 as _b64
    import json as _j2
    import os as _os2
    import time as _t2
    import urllib.request as _ureq2

    from src.ui.components.docker_utils import get_host, get_service_url

    _af_base = get_service_url("mlops_airflow", 8080)
    _af_user = _os2.environ.get("AIRFLOW_USER", "admin")
    _af_pass = _os2.environ.get("AIRFLOW_PASSWORD", "admin")
    if _af_pass == "admin":
        from src.ui.views.use_cases_pkg._common import PROJECT_ROOT as _PR2

        _sf2 = _PR2 / ".env.secrets"
        if _sf2.exists():
            for _line2 in _sf2.read_text().splitlines():
                _line2 = _line2.strip()
                if _line2.startswith("AIRFLOW_PASSWORD="):
                    _af_pass = _line2.split("=", 1)[1].strip()
                elif _line2.startswith("AIRFLOW_USER="):
                    _af_user = _line2.split("=", 1)[1].strip()
    _tok2 = _b64.b64encode(f"{_af_user}:{_af_pass}".encode()).decode()
    _hdrs2 = {
        "Authorization": f"Basic {_tok2}",
        "Accept": "application/json",
    }

    def _dag_state() -> str:
        req = _ureq2.Request(
            f"{_af_base}/api/v1/dags/automated_retraining/dagRuns/{ar_run_id}",
            headers=_hdrs2,
        )
        try:
            with _ureq2.urlopen(req, timeout=8) as r:
                return _j2.loads(r.read().decode()).get("state", "running")
        except Exception:
            return "running"

    if mode != "cloud":
        return

    with st.status("⏳ Checking promotion outcome…", expanded=False) as _oc_st:
        for _oc_i in range(18):  # up to ~90 s
            _state2 = _dag_state()
            _oc_st.write(f"Poll {_oc_i + 1}: **{_state2}**")
            if _state2 in ("success", "failed"):
                break
            _t2.sleep(5)

    if _state2 == "success":
        # Check MLflow for the current champion using alias lookup.
        # NOTE: search_model_versions() does NOT populate the 'aliases' field in
        # MLflow 3.x — we must use get_model_version_by_alias() instead.
        try:
            import mlflow as _mlflow_oc

            _mlflow_oc.set_tracking_uri(
                _os2.environ.get("MLFLOW_TRACKING_URI", f"http://{get_host()}:5002")
            )
            _client_oc = _mlflow_oc.MlflowClient()
            _reg_oc = _os2.environ.get("MODEL_REGISTRY_NAME", "device_health_classifier_cloud")
            _champ_mv = None
            try:
                # Primary: alias-based lookup (MLflow 3.x)
                _champ_mv = _client_oc.get_model_version_by_alias(_reg_oc, "champion")
            except Exception:
                # Fallback: scan current_stage == "Production" (legacy servers)
                for _mv in _client_oc.search_model_versions(f"name='{_reg_oc}'"):
                    if getattr(_mv, "current_stage", "") == "Production" and (
                        _champ_mv is None or int(_mv.version) > int(_champ_mv.version)
                    ):
                        _champ_mv = _mv
            if _champ_mv:
                st.success(
                    f"🏆 **Model v{_champ_mv.version} promoted to Production!** The new champion is live."
                )
            else:
                st.info(
                    "📊 Retraining completed. Challenger did not outperform champion — champion retained."
                )
        except Exception:
            st.info("📊 Retraining completed successfully. Check MLflow Explorer for model stage.")
    elif _state2 == "failed":
        st.error("❌ automated_retraining DAG failed after approval. Check Airflow for details.")
    else:
        st.info("⏱ Retraining still running. Check Airflow or MLflow Explorer for final outcome.")


# ---------------------------------------------------------------------------
# Tab renderer
# ---------------------------------------------------------------------------


def render_drift_provocation_tab(mode: str) -> None:
    """Tab 4: Provoke drift, visualise it, and see detection results."""
    import pandas as pd

    st.markdown(SECTION_CSS, unsafe_allow_html=True)
    st.markdown(
        "Generate drifted signals, compare feature distributions against a "
        "reference baseline, and see whether drift detection would trigger."
    )

    # Show post-approval feedback that survived the st.rerun() via session state.
    # When the user approves/rejects, _rp_just_approved is set by _render_inline_approval_queue.
    # We also clear the _dp_waiting_approval flag here so the queue disappears.
    if _approved_msg := st.session_state.pop("_rp_just_approved", None):
        st.success(_approved_msg)
        st.session_state.pop("_dp_waiting_approval", None)
        # Only poll for promotion outcome on approval — not on rejection
        _outcome_run_id = st.session_state.pop("_dp_ar_run_id", None)
        if _outcome_run_id and not _approved_msg.startswith("\U0001f6ab"):
            _show_approval_outcome(mode, _outcome_run_id)

    # ── Persistent approval queue (outside the trigger block so it survives reruns) ──
    # When the approval detection sets _dp_waiting_approval, this block renders the
    # queue on EVERY rerun — so the Approve/Reject buttons are always recreated and
    # Streamlit can execute the callback when the user clicks them.
    if st.session_state.get("_dp_waiting_approval"):
        st.info(
            "🔐 The automated retraining DAG is paused at **wait_for_human_approval**. "
            "Approve or reject the challenger model below."
        )
        from src.ui.views.use_cases_pkg.retraining_pipeline import _render_inline_approval_queue

        _render_inline_approval_queue()
        st.markdown("---")

    # ── 1. Drift type selector ─────────────
    drift_type = st.selectbox(
        "Drift Type",
        options=list(DRIFT_TYPES.keys()),
        format_func=lambda k: k.replace("_", " ").title(),
        key="drift_type",
    )
    st.markdown(DRIFT_TYPES[drift_type])
    st.markdown("---")

    # ── 2. Parameters ──────────────────────
    st.markdown("#### ⚙️ Drift Parameters")

    col_ref, col_drift = st.columns(2)
    with col_ref:
        n_ref = st.number_input("Reference samples", 50, 2000, 200, 50, key="drift_n_ref")
    with col_drift:
        n_drifted = st.number_input("Drifted samples", 50, 2000, 150, 50, key="drift_n_drift")

    mu_offset = 0.0
    width_mult = 1.0
    noise_mult = 1.0
    height_mult = 1.0
    swap_labels = False
    gauss_frac_drift = 0.7

    if drift_type == "data_drift":
        st.markdown(
            '<div class="signal-section-general">'
            "<strong>📊 Data Drift — Sensor Degradation</strong></div>",
            unsafe_allow_html=True,
        )
        d1, d2, d3 = st.columns(3)
        with d1:
            mu_offset = st.slider("μ offset", -10.0, 10.0, -5.0, 0.5, key="drift_mu_offset")
        with d2:
            noise_mult = st.slider("Noise multiplier", 1.0, 5.0, 3.0, 0.5, key="drift_noise_mult")
        with d3:
            width_mult = st.slider("Width multiplier", 0.5, 3.0, 1.5, 0.1, key="drift_width_mult")

    elif drift_type == "concept_drift":
        st.markdown(
            '<div class="signal-section-unhealthy">'
            "<strong>🔀 Concept Drift — Label Swap</strong></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "Healthy Gaussian signals will be labelled **unhealthy** and "
            "Lorentzian signals labelled **healthy** — the decision boundary "
            "flips while feature distributions stay the same."
        )
        swap_labels = True

    elif drift_type == "feature_drift":
        st.markdown(
            '<div class="signal-section-general">'
            "<strong>🔧 Feature Drift — Height Reduction</strong></div>",
            unsafe_allow_html=True,
        )
        d1, d2 = st.columns(2)
        with d1:
            height_mult = st.slider(
                "Height multiplier",
                0.2,
                1.0,
                0.5,
                0.05,
                key="drift_height_mult",
            )
        with d2:
            noise_mult = st.slider(
                "Noise multiplier",
                1.0,
                3.0,
                1.5,
                0.25,
                key="drift_feat_noise",
            )

    elif drift_type == "prior_probability_drift":
        st.markdown(
            '<div class="signal-section-unhealthy">'
            "<strong>⚖️ Prior Probability Drift — Class Imbalance</strong></div>",
            unsafe_allow_html=True,
        )
        gauss_frac_drift = st.slider(
            "Healthy fraction (drifted)",
            0.05,
            0.95,
            0.15,
            0.05,
            key="drift_gauss_frac",
        )

    st.markdown("---")

    # ── 2b. Label injection percentage ────
    st.markdown("#### 💾 Label Injection & API Key")
    label_injection_pct = st.slider(
        "Ground truth label injection rate (%)",
        min_value=0,
        max_value=100,
        value=20,
        step=5,
        key="drift_label_pct",
        help=(
            "Percentage of *drifted* signals that will receive a ground truth label "
            "injected into the database.  Labels are derived from the signal type "
            "(Gaussian = Healthy / Lorentzian = Unhealthy) as defined by the drift "
            "scenario generator.  Set to 0 to skip label injection entirely."
        ),
    )
    drift_api_key = st.text_input(
        "API Key (X-API-Key header)",
        value="dev-key-12345",
        type="password",
        key="drift_api_key_input",
        help="Required to store predictions via the API (ensures correct deployment_mode and champion model version).",
    )
    st.markdown("---")

    # ── 3. Generate & Detect ─────────────────────────────────────────────────
    # The DAG trigger button (section 7) is rendered at the BOTTOM of this
    # function, after the generate results.  Its click is handled here — before
    # the generate guard — via session state, so the Airflow HTTP call always
    # executes on the next rerun regardless of whether Generate was also clicked.
    if mode == "cloud" and st.session_state.pop("_dp_trigger_pending", False):
        import time as _time
        import urllib.error as _ue

        # Retrieve the approval-gate flag stored by the on_click callback
        _require_human_approval: bool = bool(st.session_state.pop("_dp_require_approval", False))

        try:
            import base64
            import json as _json
            import urllib.request as _ureq

            from src.ui.components.docker_utils import get_service_url

            _af_base = get_service_url("mlops_airflow", 8080)
            _af_user = os.environ.get("AIRFLOW_USER", "admin")
            _af_pass = os.environ.get("AIRFLOW_PASSWORD", "admin")
            # Fall back to .env.secrets if env var is still at default
            if _af_pass == "admin":
                from src.ui.views.use_cases_pkg._common import PROJECT_ROOT

                _sf = PROJECT_ROOT / ".env.secrets"
                if _sf.exists():
                    for _line in _sf.read_text().splitlines():
                        _line = _line.strip()
                        if _line.startswith("AIRFLOW_PASSWORD="):
                            _af_pass = _line.split("=", 1)[1].strip()
                        elif _line.startswith("AIRFLOW_USER="):
                            _af_user = _line.split("=", 1)[1].strip()
            _token = base64.b64encode(f"{_af_user}:{_af_pass}".encode()).decode()
            _af_headers = {
                "Authorization": f"Basic {_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }

            def _af_get(path: str) -> dict | None:
                req = _ureq.Request(
                    f"{_af_base}/api/v1{path}",
                    headers={k: v for k, v in _af_headers.items() if k != "Content-Type"},
                )
                try:
                    with _ureq.urlopen(req, timeout=10) as r:
                        return _json.loads(r.read().decode())  # type: ignore[return-value]
                except Exception:
                    return None

            # ── Step 0: Run evidently_drift_detection first ───────────────────
            # This writes the drift report (JSON + HTML), updates Prometheus
            # drift_detected_gauge and drift_reports_total, so Grafana can see
            # the drift event.  drift_triggered_retraining is then triggered
            # to retrain the model based on that detected drift.
            _eviurl = f"{_af_base}/api/v1/dags/evidently_drift_detection/dagRuns"
            _evipld = _json.dumps(
                {"conf": {"triggered_by": "streamlit_drift_provocation"}}
            ).encode()
            _evireq = _ureq.Request(_eviurl, data=_evipld, headers=_af_headers)
            _evi_run_id: str | None = None
            try:
                _logger.info("Triggering evidently_drift_detection DAG")
                with _ureq.urlopen(_evireq, timeout=10) as _eviresp:
                    _evidently_result = _json.loads(_eviresp.read().decode())
                _evi_run_id = _evidently_result.get("dag_run_id")
                st.info(f"📊 **evidently_drift_detection** triggered — Run ID: `{_evi_run_id}`")

                _evi_status = st.status("⏳ Running evidently_drift_detection…", expanded=False)
                _evi_prog = st.progress(0.0)
                _evi_max = 30  # ~2.5 min
                _evi_state = "running"
                for _evi_pi in range(_evi_max):
                    _time.sleep(5)
                    _evi_data = _af_get(f"/dags/evidently_drift_detection/dagRuns/{_evi_run_id}")
                    _evi_state = (_evi_data or {}).get("state", "running")
                    _evi_prog.progress(min((_evi_pi + 1) / _evi_max, 0.95))
                    _evi_status.write(f"Poll {_evi_pi + 1}: **{_evi_state}**")
                    if _evi_state in ("success", "failed"):
                        break

                _evi_prog.progress(1.0)
                if _evi_state == "success":
                    _evi_status.update(
                        label="✅ evidently_drift_detection complete — Grafana metrics updated",
                        state="complete",
                    )
                else:
                    _evi_status.update(
                        label=f"evidently_drift_detection ended: {_evi_state}",
                        state="error" if _evi_state == "failed" else "running",
                    )
                    st.warning(
                        "⚠️ evidently_drift_detection did not complete successfully — drift metrics may not be updated in Grafana. Continuing with retraining."
                    )
            except Exception as _evi_exc:
                _logger.warning("evidently_drift_detection trigger failed: {}", _evi_exc)
                st.warning(
                    f"⚠️ Could not trigger evidently_drift_detection ({_evi_exc}). Proceeding directly to retraining (drift metrics will not be updated in Grafana)."
                )

            # ── Step 1: Trigger drift_triggered_retraining ────────────────────
            _url = f"{_af_base}/api/v1/dags/drift_triggered_retraining/dagRuns"
            _payload = _json.dumps(
                {
                    "conf": {
                        "triggered_by": "streamlit_drift_tab",
                        "require_human_approval": _require_human_approval,
                    }
                }
            ).encode()
            _req = _ureq.Request(_url, data=_payload, headers=_af_headers)
            _logger.info("Triggering drift_triggered_retraining DAG")
            with _ureq.urlopen(_req, timeout=10) as _resp:
                _result = _json.loads(_resp.read().decode())
            _dtr_run_id = _result.get("dag_run_id", "N/A")
            _logger.info("drift_triggered_retraining DAG triggered — run_id={}", _dtr_run_id)
            st.info(f"🚀 **drift_triggered_retraining** triggered — Run ID: `{_dtr_run_id}`")

            # ── Step 2: Poll drift_triggered_retraining until complete ────────
            _dtr_status = st.status("⏳ Monitoring drift_triggered_retraining…", expanded=True)
            _dtr_prog = st.progress(0.0)
            _dtr_max = 40  # ~3 min
            _dtr_state = "running"
            for _pi in range(_dtr_max):
                _time.sleep(5)
                _run_data = _af_get(f"/dags/drift_triggered_retraining/dagRuns/{_dtr_run_id}")
                _dtr_state = (_run_data or {}).get("state", "running")
                _dtr_prog.progress(min((_pi + 1) / _dtr_max, 0.95))
                _dtr_status.write(f"Poll {_pi + 1}: **{_dtr_state}**")
                if _dtr_state in ("success", "failed"):
                    break

            _dtr_prog.progress(1.0)
            if _dtr_state == "failed":
                _dtr_status.update(label="drift_triggered_retraining **failed**", state="error")
                st.error(
                    "❌ drift_triggered_retraining DAG failed. "
                    "Check the Airflow Control tab for task logs."
                )
            elif _dtr_state != "success":
                _dtr_status.update(label="Monitoring timed out", state="error")
                st.warning(
                    "⏱ Timed out waiting for drift_triggered_retraining. Check Airflow for status."
                )
            else:
                _dtr_status.update(label="✅ drift_triggered_retraining complete", state="complete")

                # ── Step 3: Find the automated_retraining run it triggered ──
                # Retry up to 6 times (30 s) to let the triggered run appear in Airflow.
                _ar_run_id: str | None = None
                _dtr_run_data = _af_get(f"/dags/drift_triggered_retraining/dagRuns/{_dtr_run_id}")
                _dtr_start_date: str = (_dtr_run_data or {}).get("start_date", "")
                for _find_attempt in range(6):
                    _ar_runs = _af_get(
                        "/dags/automated_retraining/dagRuns?order_by=-start_date&limit=5"
                    )
                    if _ar_runs and _ar_runs.get("dag_runs"):
                        # Prefer the run started on or after drift_triggered_retraining started
                        for _run in _ar_runs["dag_runs"]:
                            _run_start = _run.get("start_date") or ""
                            if not _dtr_start_date or _run_start >= _dtr_start_date:
                                _ar_run_id = _run.get("dag_run_id")
                                break
                        if _ar_run_id is None:
                            # Fall back to the most recent run if none matches
                            _ar_run_id = _ar_runs["dag_runs"][0].get("dag_run_id")
                    if _ar_run_id:
                        break
                    _time.sleep(5)

                if not _ar_run_id:
                    st.info(
                        "ℹ️ drift_triggered_retraining succeeded but no new "
                        "automated_retraining run was found — drift threshold may not "
                        "have been reached (no recent drift batch detected)."
                    )
                else:
                    st.info(f"🔗 **automated_retraining** triggered — Run ID: `{_ar_run_id}`")

                    # ── Step 4: Poll automated_retraining ─────────────────
                    _ar_status = st.status("⏳ Monitoring automated_retraining…", expanded=True)
                    _ar_prog = st.progress(0.0)
                    _ar_max = 60  # ~5 min
                    _ar_state = "running"
                    _ar_waiting_approval = False
                    for _pi2 in range(_ar_max):
                        _time.sleep(5)
                        _ar_data = _af_get(f"/dags/automated_retraining/dagRuns/{_ar_run_id}")
                        _ar_state = (_ar_data or {}).get("state", "running")
                        _ar_prog.progress(min((_pi2 + 1) / _ar_max, 0.95))
                        _ar_status.write(f"Poll {_pi2 + 1}: **{_ar_state}**")

                        # Detect wait_for_human_approval task if approval was requested
                        if _require_human_approval and _ar_state == "running":
                            _tasks = _af_get(
                                f"/dags/automated_retraining/dagRuns/{_ar_run_id}/taskInstances"
                            )
                            if _tasks:
                                _task_states = {
                                    t["task_id"]: t.get("state")
                                    for t in _tasks.get("task_instances", [])
                                }
                                _wait_st = _task_states.get(
                                    "evaluation_group.wait_for_human_approval"
                                ) or _task_states.get("wait_for_human_approval")
                                if _wait_st in (
                                    "running",
                                    "up_for_reschedule",
                                    "up_for_retry",
                                    "deferred",
                                ):
                                    _ar_prog.progress(0.6)
                                    _ar_status.update(
                                        label="⏳ Waiting for human approval…", state="running"
                                    )
                                    _ar_waiting_approval = True
                                    break

                        if _ar_state in ("success", "failed"):
                            break

                    _ar_prog.progress(1.0)
                    if _ar_waiting_approval:
                        # Store state and rerun so the approval queue renders at the TOP
                        # of the page (outside this trigger block) where buttons are
                        # re-created on every rerun and Streamlit can execute callbacks.
                        st.session_state["_dp_waiting_approval"] = True
                        st.session_state["_dp_ar_run_id"] = _ar_run_id
                        st.rerun()
                    elif _ar_state == "success":
                        _ar_status.update(
                            label="✅ automated_retraining complete", state="complete"
                        )
                        # Show new model version from MLflow
                        st.success("🎉 Retraining pipeline completed successfully!")
                        try:
                            import mlflow as _mlflow_dp

                            _tracking_uri = os.environ.get(
                                "MLFLOW_TRACKING_URI",
                                f"http://{get_host()}:5002",
                            )
                            _mlflow_dp.set_tracking_uri(_tracking_uri)
                            _client = _mlflow_dp.MlflowClient()
                            _reg_name = os.environ.get(
                                "MODEL_REGISTRY_NAME",
                                "device_health_classifier_cloud",
                            )
                            _versions = _client.search_model_versions(f"name='{_reg_name}'")
                            if _versions:
                                _latest = max(_versions, key=lambda v: int(v.version))
                                st.markdown("#### 📦 New Model in Registry")
                                mc1, mc2, mc3 = st.columns(3)
                                mc1.metric("Model Name", _reg_name)
                                mc2.metric("Version", f"v{_latest.version}")
                                mc3.metric(
                                    "Stage / Alias",
                                    ", ".join(_latest.aliases) or _latest.current_stage or "—",
                                )
                                st.caption(
                                    f"Run ID: `{_latest.run_id}` · "
                                    f"Created: {_latest.creation_timestamp}"
                                )
                        except Exception as _mlflow_err:
                            _logger.warning(
                                "Could not fetch registry info after retraining: {}",
                                _mlflow_err,
                            )
                            st.info(
                                "Retraining succeeded. Check MLflow Explorer for the new model."
                            )
                    elif _ar_state == "failed":
                        _ar_status.update(label="automated_retraining **failed**", state="error")
                        st.error(
                            "❌ automated_retraining failed. "
                            "Check the Airflow Control tab for details."
                        )
                    else:
                        _ar_status.update(label="Monitoring timed out", state="error")
                        st.warning(
                            "⏱ Timed out waiting for automated_retraining. "
                            "Check Airflow for status."
                        )

        except _ue.HTTPError as _exc:
            _body = _exc.read().decode()[:300]
            _logger.warning("drift_triggered_retraining DAG HTTP {}: {}", _exc.code, _body[:200])
            st.error(f"Airflow returned HTTP {_exc.code}: {_body}")
        except Exception as exc:
            _logger.warning("drift_triggered_retraining DAG trigger failed: {}", exc)
            st.error(f"Failed to trigger DAG: {exc}")

    _gen_clicked = st.button("🔬 Generate & Detect Drift", type="primary", key="drift_run_btn")
    _has_cached = bool(st.session_state.get("_dp_drift_results"))
    if not _gen_clicked and not _has_cached:
        return

    if _gen_clicked:
        with st.spinner("Generating reference signals..."):
            ref_rows = generate_batch(int(n_ref), 0.7, seed=42, include_raw=True)
        with st.spinner("Generating drifted signals..."):
            cur_rows = generate_batch(
                int(n_drifted),
                gauss_frac_drift if drift_type == "prior_probability_drift" else 0.7,
                mu_offset=mu_offset,
                width_multiplier=width_mult,
                noise_multiplier=noise_mult,
                height_multiplier=height_mult,
                swap_labels=swap_labels,
                seed=7777,
                include_raw=True,
            )
        # Cache for re-renders triggered by widget interactions (e.g. checkbox)
        st.session_state["_dp_drift_results"] = {
            "ref_rows": ref_rows,
            "cur_rows": cur_rows,
        }
        # Reset batch-stored flag so DB storage runs again for new generation
        st.session_state.pop("_dp_batch_stored", None)
    else:
        _cached = st.session_state["_dp_drift_results"]
        ref_rows = _cached["ref_rows"]
        cur_rows = _cached.get("cur_rows", [])

    ref_df = pd.DataFrame(ref_rows)
    cur_df = pd.DataFrame(cur_rows)

    # ── 4. KS-test results ─────────────
    st.markdown("#### 📋 Drift Detection Results (KS-test, α = 0.05)")
    ks_results = ks_tests(ref_df, cur_df, FEATURE_COLS)
    n_drifted_feats = 0
    if ks_results:
        ks_df = pd.DataFrame(ks_results)
        n_drifted_feats = int(ks_df["Drift?"].eq("Yes").sum())
        total_feats = len(ks_df)
        st.dataframe(ks_df, hide_index=True)

        if n_drifted_feats > 0:
            st.error(
                f"**Drift detected** in {n_drifted_feats}/{total_feats} features. "
                "The drift-triggered retraining DAG would fire."
            )
        else:
            st.success("No statistically significant drift detected.")
    else:
        st.warning("Insufficient data for KS testing.")

    # ── 4. Persist drift signals to database (first generation only) ─────────
    # Step 4a: store drift-batch metadata in the dedicated drift tables (direct DB)
    # Step 4b: store each drifted signal as a normal prediction via the API
    # Skipped on reruns triggered by widget interactions (e.g. checkbox toggle).

    if _gen_clicked:
        try:
            import random as _random

            from src.database.database import Database

            _drift_params = {
                "mu_offset": mu_offset,
                "width_multiplier": width_mult,
                "noise_multiplier": noise_mult,
                "height_multiplier": height_mult,
                "swap_labels": swap_labels,
                "gauss_frac_drift": gauss_frac_drift
                if drift_type == "prior_probability_drift"
                else 0.7,
            }
            db = Database(db_url=get_host_db_url() or None)
            db.register_device(
                device_id=DRIFT_SIM_DEVICE_ID,
                device_name="Drift Simulator",
                device_type=drift_type,
                location="simulation",
                status="active",
                deployment_mode=mode,
            )
            batch_id = db.store_drift_batch(
                drift_type=drift_type,
                n_reference=int(n_ref),
                n_drifted=int(n_drifted),
                parameters=_drift_params,
                reference_rows=ref_rows,
                drifted_rows=cur_rows,
                ks_results=ks_results if ks_results else None,
                n_drifted_features=n_drifted_feats,
                deployment_mode=mode,
            )
            db.close()
            st.session_state["_dp_batch_stored"] = True
            st.info(
                f"📊 Drift batch #{batch_id} metadata saved ({len(cur_rows)} drifted signals logged)."
            )
        except Exception as exc:
            st.warning(f"⚠️ Could not save drift batch metadata: {exc}")

        # Step 4b: Store each drifted signal through the API so the correct
        # champion model is used for prediction, and deployment_mode is set by
        # the API container (not hardcoded here).
        try:
            import json as _json
            import random as _random
            import urllib.error
            import urllib.request

            from src.ui.components.docker_utils import get_service_url

            _api_url = get_service_url("mlops_nginx", 80)

            def _drift_api_post(path: str, payload: dict) -> dict | None:
                url = f"{_api_url}{path}"
                data = _json.dumps(payload).encode()
                headers = {"Content-Type": "application/json"}
                if drift_api_key:
                    headers["X-API-Key"] = drift_api_key
                req = urllib.request.Request(url, data=data, headers=headers)
                try:
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        return _json.loads(resp.read().decode())
                except Exception:
                    return None

            _valid_indices = [
                i
                for i, row in enumerate(cur_rows)
                if row.get("time_values")
                and row.get("amplitude_values")
                and len(row["time_values"]) >= 51  # type: ignore[arg-type]
            ]
            _n_to_label = max(0, round(len(_valid_indices) * label_injection_pct / 100))
            _rng = _random.Random(42)
            _label_indices: set[int] = (
                set(_rng.sample(_valid_indices, _n_to_label)) if _n_to_label else set()
            )

            _stored_main = 0
            _labels_injected = 0
            _store_errors: list[str] = []

            _prog = st.progress(0, text="Storing drifted signals via API…")
            _total = len(cur_rows)

            for i, row in enumerate(cur_rows):
                t_vals = row.get("time_values")
                a_vals = row.get("amplitude_values")
                if t_vals is None or a_vals is None or len(t_vals) < 51:  # type: ignore[arg-type]
                    _prog.progress((i + 1) / _total)
                    continue
                try:
                    pred_payload = {
                        "device_id": DRIFT_SIM_DEVICE_ID,
                        "device_name": "Drift Simulator",
                        "time_values": t_vals,
                        "amplitude_values": a_vals,
                    }
                    pred_result = _drift_api_post("/predict", pred_payload)
                    if pred_result is None:
                        _store_errors.append(f"Signal {i}: API call failed")
                        _prog.progress((i + 1) / _total)
                        continue

                    _stored_main += 1
                    prediction_id = pred_result.get("prediction_id")

                    if prediction_id is not None and i in _label_indices:
                        gt_label = (
                            LABEL_HEALTHY
                            if row.get("shape_type") == "gaussian"
                            else LABEL_UNHEALTHY
                        )
                        lbl_payload = {
                            "prediction_id": prediction_id,
                            "ground_truth_label": gt_label,
                            "label_source": "drift_simulator",
                            "injected_by": "drift_provocation_ui",
                        }
                        lbl_result = _drift_api_post("/labels", lbl_payload)
                        if lbl_result:
                            _labels_injected += 1
                except Exception as _e:
                    _store_errors.append(str(_e))
                _prog.progress((i + 1) / _total)

            _prog.empty()

            _err_note = (
                f"  ⚠️ {len(_store_errors)} signal(s) failed ({_store_errors[0]})"
                if _store_errors
                else ""
            )
            _label_note = (
                f", {_labels_injected} ground truth label(s) injected ({label_injection_pct}%)"
                if label_injection_pct > 0
                else " (label injection disabled)"
            )
            st.success(
                f"💾 {_stored_main}/{len(cur_rows)} drifted signals stored via API "
                f"(champion model used, correct deployment_mode){_label_note}.{_err_note}"
            )
        except Exception as exc:
            st.error(f"⚠️ Could not store drifted signals via API: {exc}")

    # ── 4c. Label-distribution test (concept drift detection) ──
    if "label" in ref_df.columns and "label" in cur_df.columns:
        from scipy.stats import chi2_contingency

        ref_counts = ref_df["label"].value_counts().sort_index()
        cur_counts = cur_df["label"].value_counts().sort_index()
        # Align indices
        all_labels = sorted(set(ref_counts.index) | set(cur_counts.index))
        ref_aligned = [ref_counts.get(lb, 0) for lb in all_labels]
        cur_aligned = [cur_counts.get(lb, 0) for lb in all_labels]
        contingency = [ref_aligned, cur_aligned]
        try:
            chi2, chi2_p, _, _ = chi2_contingency(contingency)
            label_drift = chi2_p < 0.05

            st.markdown("#### 🏷️ Label Distribution Test (χ² test, α = 0.05)")
            lc1, lc2, lc3 = st.columns(3)
            lc1.metric("χ² statistic", f"{chi2:.4f}")
            lc2.metric("p-value", f"{chi2_p:.6f}")
            lc3.metric("Label drift?", "Yes" if label_drift else "No")

            if drift_type == "concept_drift":
                if label_drift:
                    st.error(
                        "**Label distribution drift detected.** Concept drift swaps the "
                        "mapping between features and labels — the feature distributions "
                        "remain identical (KS tests pass), but now different labels are "
                        "assigned to the same signal shapes."
                    )
                else:
                    st.info(
                        "ℹ️ **By design, pure concept drift is invisible to feature-based "
                        "KS tests** because the input distributions do not change — only the "
                        "label assignment flips. Label-distribution tests (above) can detect "
                        "prior-probability changes if the label swap also alters class balance. "
                        "In this symmetric case (equal class sizes), even the χ² test may not "
                        "detect concept drift. A model-accuracy monitor would catch it at "
                        "prediction time."
                    )
            elif label_drift:
                st.warning("Label distribution shifted between reference and drifted batches.")
        except Exception:
            pass  # Not enough data for chi-square

    # ── 5. Visualisation ───────────────
    st.markdown("#### 📊 Feature Distribution Comparison")
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        usable = [c for c in FEATURE_COLS if c in ref_df.columns and c in cur_df.columns]
        fig = make_subplots(
            rows=2,
            cols=3,
            subplot_titles=[c.replace("_", " ").title() for c in usable[:6]],
        )
        for idx, feat in enumerate(usable[:6]):
            r_idx = idx // 3 + 1
            c_idx = idx % 3 + 1
            fig.add_trace(
                go.Histogram(
                    x=ref_df[feat].dropna(),
                    name="Reference",
                    marker_color="#4caf50",
                    opacity=0.6,
                    showlegend=(idx == 0),
                ),
                row=r_idx,
                col=c_idx,
            )
            fig.add_trace(
                go.Histogram(
                    x=cur_df[feat].dropna(),
                    name="Drifted",
                    marker_color="#e53935",
                    opacity=0.6,
                    showlegend=(idx == 0),
                ),
                row=r_idx,
                col=c_idx,
            )
        fig.update_layout(
            barmode="overlay",
            height=500,
            legend={"orientation": "h", "y": -0.15},
            margin={"t": 40, "b": 60},
        )
        st.plotly_chart(fig)
    except ImportError:
        st.warning("Install plotly for distribution charts: `pip install plotly`")

    # ── 6. Class balance comparison ────
    if "label" in ref_df.columns and "label" in cur_df.columns:
        st.markdown("#### ⚖️ Class Balance")
        bal1, bal2 = st.columns(2)
        with bal1:
            ref_bal = ref_df["label"].value_counts(normalize=True)
            st.markdown("**Reference**")
            st.write(
                f"Healthy (0): {ref_bal.get(0, 0):.0%}  ·  Unhealthy (1): {ref_bal.get(1, 0):.0%}"
            )
        with bal2:
            cur_bal = cur_df["label"].value_counts(normalize=True)
            st.markdown("**Drifted**")
            st.write(
                f"Healthy (0): {cur_bal.get(0, 0):.0%}  ·  Unhealthy (1): {cur_bal.get(1, 0):.0%}"
            )

    # ── 7. Trigger Retraining DAG ─────────────────────────────────────────────
    # Rendered AFTER the generate results so it appears at the bottom of the
    # page.  The click is handled via ``on_click`` callback so the flag is set
    # BEFORE the next rerun starts.  That guarantees the section-3 handler
    # (before the early-return guard) receives the flag even if Generate wasn't
    # just clicked.
    st.markdown("---")
    if mode == "cloud":
        st.markdown("#### 🔄 Trigger Retraining DAG")
        st.markdown(
            "Click the button below to trigger the full drift pipeline:\n\n"
            "1. **evidently_drift_detection** runs first — generates the drift report, "
            "updates Prometheus `drift_detected_gauge` and `drift_reports_total`, "
            "so Grafana dashboards capture the drift event.\n"
            "2. **drift_triggered_retraining** then checks for recent drift batches "
            "and, if found, triggers **automated_retraining**."
        )

        # ── Human Approval Gate option ────────────────────────────────────────
        _dp_require_approval = st.checkbox(
            "🔐 Require human approval before model promotion",
            value=False,
            key="dp_require_approval",
            help=(
                "When enabled, the automated_retraining DAG will pause after "
                "evaluation and wait for a human to approve or reject the challenger "
                "in the MLflow Explorer → Approval Queue tab before promoting."
            ),
        )

        def _queue_drift_dag_trigger() -> None:
            st.session_state["_dp_trigger_pending"] = True
            st.session_state["_dp_require_approval"] = st.session_state.get(
                "dp_require_approval", False
            )

        st.button(
            "🚀 Run drift detection + retraining",
            key="drift_trigger_dag",
            type="primary",
            on_click=_queue_drift_dag_trigger,
        )
        if _dp_require_approval:
            st.info(
                "🔐 Human approval gate is **enabled**. After automated_retraining "
                "evaluates the challenger, it will pause and wait for approval in "
                "**MLflow Explorer → 📋 Approval Queue**."
            )
    else:
        st.info(
            "DAG triggering is only available in **cloud mode** "
            "(Airflow is disabled in local mode)."
        )
