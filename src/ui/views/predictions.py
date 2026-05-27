"""Predictions — file upload, live prediction, history, and system health panel."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from src.config import (
    DEFAULT_LABEL_INJECTION_PCT,
    DRIFT_SCENARIO_LABELS,
    DRIFT_SCENARIOS,
    LABEL_HEALTHY,
    LABEL_UNHEALTHY,
)
from src.ui.components.docker_utils import get_service_url
from src.ui.logging_ui import get_ui_logger
from src.ui.styles import get_global_css, hero_section, metric_card

_logger = get_ui_logger(__name__)


def render() -> None:
    """Render the Predictions page."""
    try:
        _render_content()
    except Exception:
        _logger.exception("Unhandled exception in Predictions.render()")
        raise


def _render_content() -> None:
    """Inner render — separated so exceptions are caught and logged by render()."""
    st.markdown(get_global_css(), unsafe_allow_html=True)
    st.markdown(
        hero_section(
            "Predictions & Health",
            "Upload a signal, predict device health, view history, and check system status.",
        ),
        unsafe_allow_html=True,
    )

    # Use st.radio so the selected sub-tab persists across st.rerun() calls.
    # st.tabs() resets to tab 0 on every rerun (e.g. when batch method changes
    # or a predict button triggers internal state updates).
    pred_tabs = [
        "🎯 Single Signal Prediction",
        "📦 Batch Prediction",
        "📋 Prediction History",
        "💚 System Health",
    ]
    tab_css = """
<style>
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] { padding: 0 !important; }
div[data-baseweb="radio"] > div { gap: 0.25rem; flex-wrap: wrap; }
div[data-baseweb="radio"] > div > label {
    border: 1px solid #e2e8f0;
    border-radius: 8px 8px 0 0;
    padding: 0.45rem 1rem;
    margin-bottom: -1px;
    background: #f8fafc;
    font-size: 0.875rem;
    cursor: pointer;
    transition: background 0.15s, color 0.15s;
}
div[data-baseweb="radio"] > div > label:hover { background: #e0e7ff; }
div[data-baseweb="radio"] > div > label[data-checked="true"],
div[data-baseweb="radio"] > div > label[aria-checked="true"] {
    background: white;
    border-bottom-color: white;
    font-weight: 600;
    color: #4f46e5;
}
</style>
"""
    st.markdown(tab_css, unsafe_allow_html=True)
    active_tab = st.radio(
        "Prediction Tab",
        pred_tabs,
        horizontal=True,
        key="_pred_tab",
        label_visibility="collapsed",
    )
    st.markdown(
        "<hr style='margin:0 0 1rem 0;border-color:#e2e8f0;'>",
        unsafe_allow_html=True,
    )

    if active_tab == pred_tabs[0]:
        _predict_tab()
    elif active_tab == pred_tabs[1]:
        _batch_tab()
    elif active_tab == pred_tabs[2]:
        _history_tab()
    elif active_tab == pred_tabs[3]:
        _health_tab()


# ── Helpers ─────────────────────────────────────────────────────


def _api_base() -> str:
    """Return the API base URL (via Nginx)."""
    return get_service_url("mlops_nginx", 80)


def _api_post(path: str, payload: dict, *, api_key: str = "", silent: bool = False) -> dict | None:
    """POST JSON to the API and return the parsed response (or None on error).

    Args:
        silent: When True, suppress st.error calls (useful for batch mode).

    Retries once after 3 s on HTTP 502/503 (API restarting after model hot-reload).
    """
    import time as _t
    import urllib.error
    import urllib.request

    url = f"{_api_base()}{path}"
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    req = urllib.request.Request(url, data=data, headers=headers)

    for _attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            body = exc.fp.read().decode() if exc.fp else ""
            if exc.code in (502, 503) and _attempt < 2:
                # API is temporarily down (e.g. model reload). Wait and retry.
                _t.sleep(3)
                req = urllib.request.Request(url, data=data, headers=headers)
                continue
            _logger.warning("API POST {} HTTP {}: {}", path, exc.code, body[:200])
            if not silent:
                _show_api_error(exc.code, body)
            return None
        except Exception as exc:
            _logger.warning("API POST {} connection error: {}", path, exc)
            if not silent:
                st.error(f"Connection error: {exc}")
            return None
    return None


def _api_get(path: str) -> dict | None:
    """GET from the API and return parsed JSON."""
    import urllib.error
    import urllib.request

    url = f"{_api_base()}{path}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        _logger.warning("API GET {} failed: {}", path, exc)
        st.error(f"Connection error: {exc}")
    return None


def _show_api_error(code: int, body: str) -> None:
    """Display API errors with structured formatting for Pydantic 422 validation errors."""
    if code == 422:
        st.error("⚠️ Validation error (HTTP 422) — the API rejected the request.")
        try:
            import json as _j

            detail = _j.loads(body).get("detail", [])
            if isinstance(detail, list) and detail:
                with st.expander("Validation details", expanded=True):
                    for item in detail:
                        if isinstance(item, dict):
                            loc_parts = [str(x) for x in item.get("loc", []) if x != "body"]
                            loc = " → ".join(loc_parts) if loc_parts else "request"
                            msg = item.get("msg", "unknown error")
                            st.markdown(f"- **`{loc}`**: {msg}")
                        else:
                            st.markdown(f"- {item}")
            elif isinstance(detail, str):
                st.markdown(f"**Detail:** {detail}")
        except Exception:
            with st.expander("Response body"):
                st.code(body[:500])
    else:
        st.error(f"API error {code}: {body[:300]}")


# ── Predict tab ─────────────────────────────────────────────────


def _predict_tab() -> None:
    st.markdown(
        '<div class="section-header">🎯 Single-Signal Prediction</div>'
        '<div class="section-subheader">Upload a signal file or generate a synthetic signal, '
        "then request a real-time prediction from the API.</div>",
        unsafe_allow_html=True,
    )

    method = st.radio(
        "Signal source",
        ["📁 Upload file (JSON/CSV)", "🧪 Generate synthetic signal"],
        horizontal=True,
        key="single_method",
        label_visibility="collapsed",
    )

    time_vals: list[float] | None = None
    amp_vals: list[float] | None = None
    # Track the inferred true_label from signal generation so label injection
    # can default to "Auto (from signal type)" correctly.
    inferred_true_label: int | None = None

    if method.startswith("🧪"):
        time_vals, amp_vals, inferred_true_label = _generate_signal_ui()
        st.session_state["single_inferred_label"] = inferred_true_label
    else:
        time_vals, amp_vals = _upload_signal()
        st.session_state["single_inferred_label"] = None

    if time_vals is not None and amp_vals is not None:
        _show_signal_plot(time_vals, amp_vals)

        # API key input
        api_key = st.text_input(
            "API Key (X-API-Key header)",
            value="dev-key-12345",
            type="password",
            help=(
                "Development key pre-configured in src/api/security.py.\n"
                "• dev-key-12345         → read + write access\n"
                "• monitoring-key-67890  → read-only access\n"
                "Leave blank only if API key auth is disabled."
            ),
        )

        # ── Label injection controls (single signal) ──────────────────────
        st.markdown("#### 💾 Label Injection")
        inject_label_flag = st.checkbox(
            "Inject ground truth label after prediction",
            value=False,
            key="single_inject_label_cb",
            help=(
                "Saves a sparse ground truth label for this prediction into the database. "
                "Sparse labelling (typically ~10 % of predictions) enables retraining."
            ),
        )
        label_choice: str = "auto"
        if inject_label_flag:
            label_choice = st.radio(
                "Label",
                options=["auto", "healthy", "unhealthy"],
                format_func=lambda x: {
                    "auto": "Auto (from signal type)",
                    "healthy": "Healthy",
                    "unhealthy": "Unhealthy",
                }[x],
                horizontal=True,
                key="single_label_radio",
            )

        if st.button("🚀 Predict", type="primary"):
            payload = {
                "device_id": "",
                "time_values": time_vals,
                "amplitude_values": amp_vals,
            }
            with st.spinner("Calling API…"):
                t0 = time.time()
                result = _api_post("/predict", payload, api_key=api_key)
                elapsed_ms = (time.time() - t0) * 1000

            if result:
                _logger.info(
                    "Prediction OK — label={} confidence={:.2f} elapsed={:.0f}ms",
                    result.get("predicted_label", result.get("label", "?")),
                    result.get("prediction_confidence", result.get("confidence", 0.0)),
                    elapsed_ms,
                )
                _show_prediction_result(result, elapsed_ms)
                st.session_state["single_last_result"] = result
                st.session_state["single_last_api_key"] = api_key

                # ── Inject label if requested ─────────────────────────────
                if inject_label_flag:
                    prediction_id = result.get("prediction_id")
                    if prediction_id is not None:
                        # Determine label value
                        if label_choice == "healthy":
                            gt_label = LABEL_HEALTHY
                        elif label_choice == "unhealthy":
                            gt_label = LABEL_UNHEALTHY
                        else:
                            # auto: use inferred label from signal generation; default healthy
                            gt_label = (
                                inferred_true_label
                                if inferred_true_label is not None
                                else LABEL_HEALTHY
                            )
                        label_payload = {
                            "prediction_id": prediction_id,
                            "ground_truth_label": gt_label,
                            "label_source": "predictions_ui",
                            "injected_by": "single_prediction_tab",
                        }
                        lbl_result = _api_post(
                            "/labels", label_payload, api_key=api_key, silent=True
                        )
                        if lbl_result:
                            label_text = "Healthy" if gt_label == LABEL_HEALTHY else "Unhealthy"
                            st.success(
                                f"💾 Ground truth label **{label_text}** injected "
                                f"(label_id={lbl_result.get('label_id', '?')})."
                            )
                        else:
                            st.warning("⚠️ Could not inject label — check API connectivity.")
                    else:
                        st.warning("⚠️ prediction_id not returned by API; label injection skipped.")


def _upload_signal() -> tuple[list[float] | None, list[float] | None]:
    """Handle file upload (JSON or CSV)."""
    uploaded = st.file_uploader(
        "Upload signal file",
        type=["json", "csv"],
        help='JSON: {"time": [...], "amplitude": [...]}  —  CSV: columns `time`, `amplitude`',
    )
    if uploaded is None:
        return None, None

    try:
        if uploaded.name.endswith(".json"):
            data = json.loads(uploaded.read().decode())
            time_vals = list(map(float, data.get("time", data.get("time_values", []))))
            amp_vals = list(map(float, data.get("amplitude", data.get("amplitude_values", []))))
        else:
            # CSV — two supported layouts:
            #   (A) Wide format: one row per device, time_values and
            #       amplitude_values stored as semicolon-separated lists in a
            #       single cell (e.g. data/samples/healthy/healthy_batch.csv).
            #   (B) Long format: one row per data point with columns
            #       "time" / "amplitude" (or "time_values" / "amplitude_values").
            import csv
            import io

            text = uploaded.read().decode()
            reader = csv.DictReader(io.StringIO(text))
            rows = list(reader)

            if not rows:
                st.error("CSV file is empty.")
                return None, None

            # Detect format by inspecting the first row
            first = rows[0]
            t_raw = first.get("time_values", first.get("time", ""))
            a_raw = first.get("amplitude_values", first.get("amplitude", ""))

            if ";" in str(t_raw):
                # Wide format — use only the first device row
                time_vals = [float(v) for v in str(t_raw).split(";") if v.strip()]
                amp_vals = [float(v) for v in str(a_raw).split(";") if v.strip()]
                device_label = first.get("device_name", first.get("device_id", "row-1"))
                st.info(f"Wide-format CSV: using device **{device_label}**.")
            else:
                # Long format — one data point per row
                time_vals = [float(r.get("time", r.get("time_values", 0))) for r in rows]
                amp_vals = [float(r.get("amplitude", r.get("amplitude_values", 0))) for r in rows]

        if len(time_vals) < 51 or len(amp_vals) < 51:
            st.warning("Signal too short — need at least 51 points.")
            return None, None

        st.success(f"Loaded {len(time_vals)} data points from **{uploaded.name}**")
        return time_vals, amp_vals
    except Exception as exc:
        st.error(f"Failed to parse file: {exc}")
        return None, None


def _generate_signal_ui() -> tuple[list[float], list[float], int | None]:
    """Let the user generate a synthetic signal with optional custom peak parameters.

    Returns:
        (time_vals, amplitude_vals, inferred_true_label)
        inferred_true_label is LABEL_HEALTHY (0) or LABEL_UNHEALTHY (1), or None for 'custom'.
    """
    col1, col2, col3 = st.columns(3)
    with col1:
        shape = st.selectbox("Shape type", ["gaussian", "lorentzian"])
    with col2:
        # Drift scenarios synchronised with the Drift Provocation use case.
        _all_single_scenarios = ["baseline"] + DRIFT_SCENARIOS + ["custom"]
        scenario = st.selectbox(
            "Drift scenario",
            _all_single_scenarios,
            format_func=lambda k: DRIFT_SCENARIO_LABELS.get(k, k.replace("_", " ").title()),
            help=(
                "Scenarios match the Drift Provocation use case.  "
                "'custom' unlocks manual peak parameter controls below."
            ),
        )
    with col3:
        n_points = st.slider("Number of points", 51, 500, 101)

    seed = st.number_input("Random seed", value=42, min_value=0, max_value=99999, key="single_seed")

    # Custom parameter controls (shown only when 'custom' drift scenario is selected)
    custom_mu: float | None = None
    custom_width: float | None = None
    custom_height: float | None = None
    custom_noise: float | None = None

    if scenario == "custom":
        st.markdown("**🎛️ Custom Peak Parameters**")
        width_label = (
            "σ (sigma — Gaussian width)" if shape == "gaussian" else "γ (gamma — Lorentzian HWHM)"
        )
        width_default = 2.5 if shape == "gaussian" else 5.0
        # Use shape-specific keys so Gaussian and Lorentzian parameters are tracked separately
        _key_mu = f"custom_mu_{shape}"
        _key_width = f"custom_width_{shape}"
        _key_height = f"custom_height_{shape}"
        _key_noise = f"custom_noise_{shape}"

        # ── Initialize session state keys with defaults (first run only) ────
        st.session_state.setdefault(_key_mu, 50.0)
        st.session_state.setdefault(_key_width, width_default)
        st.session_state.setdefault(_key_height, 2.0)
        st.session_state.setdefault(_key_noise, 0.02)

        # ── Apply pending reset BEFORE sliders are instantiated ──────────────
        if st.session_state.pop(f"_pred_reset_pending_{shape}", False):
            st.session_state[_key_mu] = 50.0
            st.session_state[_key_width] = width_default
            st.session_state[_key_height] = 2.0
            st.session_state[_key_noise] = 0.02

        p1, p2, p3, p4 = st.columns(4)
        with p1:
            custom_mu = st.slider(
                "μ (peak center)",
                min_value=10.0,
                max_value=90.0,
                step=0.5,
                key=_key_mu,
            )
        with p2:
            custom_width = st.slider(
                width_label,
                min_value=0.5,
                max_value=15.0,
                step=0.5,
                key=_key_width,
            )
        with p3:
            custom_height = st.slider(
                "Height",
                min_value=0.1,
                max_value=5.0,
                step=0.1,
                key=_key_height,
                help="Gaussian requires Height \u2265 1.0; Lorentzian requires Height \u2265 0.8.",
            )
        with p4:
            custom_noise = st.slider(
                "Noise level",
                min_value=0.0,
                max_value=0.20,
                step=0.01,
                key=_key_noise,
            )
        if st.button("\u21ba Reset to defaults", key="_pred_reset_custom"):
            st.session_state[f"_pred_reset_pending_{shape}"] = True
            st.rerun()

    from src.signal_processing.signal_generator import generate_signal

    # ── Map UI scenario to generate_signal parameters ─────────────────────
    # generate_signal() natively supports baseline/data_drift/concept_drift.
    # For feature_drift and prior_probability_drift we translate to explicit params.
    actual_scenario: str | None = None
    extra_height: float | None = custom_height
    extra_shape: str = shape  # may be overridden for prior_probability_drift

    if scenario == "custom":
        actual_scenario = None
    elif scenario == "feature_drift":
        # Feature drift reduces peak height; keep other params at baseline.
        # GaussianParameters.height >= 1.0; LorentzianParameters.height >= 0.8.
        # Use a valid low height that falls below the healthy range (>= 2.5) to
        # represent feature-level distribution shift.
        actual_scenario = None
        extra_height = 1.5 if shape == "gaussian" else 0.9
        st.info(
            "ℹ️ **Feature Drift** — peak height is reduced to simulate "
            "a feature-level distribution shift."
        )
    elif scenario == "prior_probability_drift":
        # Prior probability drift = more unhealthy signals in the population.
        # For a single signal, we generate an unhealthy (Lorentzian) signal to
        # represent the shifted distribution.
        actual_scenario = None
        extra_shape = "lorentzian"
        st.info(
            "ℹ️ **Prior Probability Drift** — the class balance shifts toward "
            "unhealthy signals. A Lorentzian (unhealthy) signal is generated."
        )
    else:
        actual_scenario = scenario  # baseline / data_drift / concept_drift / None(custom)

    try:
        sig = generate_signal(
            shape_type=extra_shape,  # type: ignore[arg-type]
            drift_scenario=actual_scenario,  # type: ignore[arg-type]
            n_points=n_points,
            seed=int(seed),
            mu=custom_mu,
            width_param=custom_width,
            height=extra_height,
            noise_level=custom_noise,
        )
    except Exception as _exc:  # catches pydantic ValidationError and any other errors
        _is_pydantic = type(_exc).__name__ in ("ValidationError", "PydanticValidationError")
        if _is_pydantic:
            st.error(
                "\u26a0\ufe0f **Invalid parameter combination** \u2014 Pydantic model validation "
                f"failed: `{_exc}`  \n\n"
                "Adjust the **Custom** slider values to valid ranges "
                "(Gaussian Height \u2265 1.0, Lorentzian Height \u2265 0.8)."
            )
        else:
            st.error(f"Signal generation failed: {_exc}")
        return [], [], None

    # Infer the true label from the generated signal shape for auto-labelling.
    inferred_true_label: int | None = (
        None
        if scenario == "custom"
        else (LABEL_HEALTHY if extra_shape == "gaussian" else LABEL_UNHEALTHY)
    )

    return list(sig.signal.time), list(sig.signal.amplitude), inferred_true_label


def _show_signal_plot(time_vals: list[float], amp_vals: list[float]) -> None:
    """Plot the signal using Plotly."""
    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=time_vals,
                y=amp_vals,
                mode="lines",
                name="Signal",
                line={"color": "#6366f1", "width": 2},
            )
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#1e293b",
            margin={"l": 40, "r": 20, "t": 30, "b": 40},
            height=300,
            xaxis_title="Time",
            yaxis_title="Amplitude",
        )
        st.plotly_chart(fig, width="stretch", key="predict_signal_plot")
    except ImportError:
        st.line_chart({"amplitude": amp_vals})


def _show_prediction_result(result: dict, elapsed_ms: float) -> None:
    """Display the prediction result."""
    label_raw = result.get("predicted_label", result.get("label", "—"))
    label = str(label_raw)
    # API returns 'prediction_confidence'; 'confidence' kept as legacy fallback
    confidence = result.get("prediction_confidence", result.get("confidence", 0.0))
    device_id = result.get("device_id", "—")

    # Determine label text
    if label in ("0", "healthy"):
        label_text = "✅ Healthy"
        color = "#10b981"
    elif label in ("1", "unhealthy"):
        label_text = "❌ Unhealthy"
        color = "#ef4444"
    else:
        label_text = f"Label: {label}"
        color = "#f59e0b"

    cols = st.columns(4)
    cols[0].markdown(
        metric_card("🏷️", f'<span style="color:{color}">{label_text}</span>', "Prediction"),
        unsafe_allow_html=True,
    )
    cols[1].markdown(
        metric_card("📊", f"{confidence:.1%}", "Confidence"),
        unsafe_allow_html=True,
    )
    cols[2].markdown(
        metric_card("⏱️", f"{elapsed_ms:.0f} ms", "Latency"),
        unsafe_allow_html=True,
    )
    cols[3].markdown(
        metric_card("📱", device_id[:12], "Device ID"),
        unsafe_allow_html=True,
    )

    # Model / lineage info row
    model_version = result.get("model_version", "—")
    run_id = result.get("mlflow_run_id")
    dvc_hash = result.get("dvc_data_hash")
    git_sha = result.get("git_sha")

    info_cols = st.columns(4)
    info_cols[0].markdown(
        metric_card("🤖", str(model_version), "Model Version"),
        unsafe_allow_html=True,
    )
    info_cols[1].markdown(
        metric_card("🧪", (run_id or "—")[:12], "MLflow Run"),
        unsafe_allow_html=True,
    )
    info_cols[2].markdown(
        metric_card("📦", (dvc_hash or "—")[:12], "DVC Hash"),
        unsafe_allow_html=True,
    )
    info_cols[3].markdown(
        metric_card("🔗", (git_sha or "—")[:12], "Git SHA"),
        unsafe_allow_html=True,
    )

    with st.expander("📋 Full API Response", expanded=False):
        st.json(result)


# ── Batch prediction tab ─────────────────────────────────────────


def _batch_tab() -> None:
    """Batch prediction: upload many signals or generate synthetic batch, predict all at once."""
    st.markdown(
        '<div class="section-header">📦 Batch Prediction</div>'
        '<div class="section-subheader">Upload a CSV / JSON with multiple signals or generate a '
        "synthetic batch, run predictions in bulk, and view an overlay chart with per-signal "
        "results.</div>",
        unsafe_allow_html=True,
    )

    method = st.radio(
        "Batch source",
        ["📁 Upload file (CSV / JSON)", "🧪 Generate synthetic batch"],
        horizontal=True,
        key="batch_method",
        label_visibility="collapsed",
    )

    signals: list[dict] = []  # list of {"id": str, "time": [...], "amplitude": [...]}

    signals = _batch_upload() if method.startswith("📁") else _batch_generate()

    if not signals:
        return

    # --- Overlay plot (before prediction — grey traces) ---
    _render_batch_plot(signals, predictions=None)

    # --- API Key + Predict button ---
    api_key = st.text_input(
        "API Key (X-API-Key header)",
        value="dev-key-12345",
        type="password",
        key="batch_api_key",
    )

    # ── Label injection controls (batch) ──────────────────────────────────
    st.markdown("#### 💾 Label Injection")
    lc1, lc2 = st.columns([1, 2])
    with lc1:
        label_inject_pct = st.slider(
            "Ground truth label injection rate (%)",
            min_value=0,
            max_value=100,
            value=DEFAULT_LABEL_INJECTION_PCT,
            step=5,
            key="batch_label_pct",
            help=(
                "Percentage of predictions that will receive a sparse ground truth label. "
                f"Default {DEFAULT_LABEL_INJECTION_PCT} % matches a realistic labelling budget."
            ),
        )
    with lc2:
        batch_label_choice = st.radio(
            "Label",
            options=["auto", "healthy", "unhealthy"],
            format_func=lambda x: {
                "auto": "Auto (from signal type)",
                "healthy": "Healthy",
                "unhealthy": "Unhealthy",
            }[x],
            horizontal=True,
            key="batch_label_radio",
        )

    col_retry, col_predict, col_cancel = st.columns([2, 2, 1])
    with col_retry:
        max_retries = st.number_input(
            "Max retries per signal on error",
            min_value=0,
            max_value=5,
            value=2,
            step=1,
            key="batch_max_retries",
        )
    with col_predict:
        # Push button down to align with the number_input control
        st.markdown('<div style="margin-top:1.6rem"></div>', unsafe_allow_html=True)
        predict_clicked = st.button("🚀 Predict All", type="primary", key="batch_predict_btn")
    with col_cancel:
        st.markdown('<div style="margin-top:1.6rem"></div>', unsafe_allow_html=True)
        if st.button("🛑 Cancel", key="batch_cancel_btn", type="secondary"):
            st.session_state["batch_cancel"] = True

    if predict_clicked:
        import random as _random

        _logger.info("Starting batch prediction: {} signals", len(signals))
        st.session_state["batch_cancel"] = False
        predictions: list[dict] = []
        failed_signals: list[str] = []
        progress = st.progress(0, text="Predicting…")
        status_text = st.empty()
        for i, sig in enumerate(signals):
            if st.session_state.get("batch_cancel"):
                status_text.warning(f"⚠️ Cancelled after {i} signal(s).")
                break
            sig_id = str(sig.get("id", f"signal-{i + 1}"))
            status_text.text(f"⟳ Signal {i + 1}/{len(signals)}: {sig_id[:30]}…")
            payload = {
                "device_id": sig_id,
                "time_values": sig["time"],
                "amplitude_values": sig["amplitude"],
            }
            result = None
            for attempt in range(int(max_retries) + 1):
                result = _api_post("/predict", payload, api_key=api_key, silent=True)
                if result is not None:
                    break
                if attempt < int(max_retries):
                    wait_secs = 2**attempt  # exponential back-off: 1 s, 2 s, 4 s…
                    status_text.text(
                        f"⟳ Signal {i + 1}/{len(signals)}: retry {attempt + 1}/{int(max_retries)} "
                        f"— waiting {wait_secs}s for API recovery…"
                    )
                    time.sleep(wait_secs)
            if result is None:
                failed_signals.append(sig_id)
                result = {"predicted_label": "error", "confidence": 0.0, "probabilities": {}}
            predictions.append(result)
            progress.progress((i + 1) / len(signals))
            if i < len(signals) - 1:
                time.sleep(0.05)
        progress.empty()
        status_text.empty()

        if st.session_state.get("batch_cancel"):
            _logger.info("Batch prediction cancelled after {} signal(s)", len(predictions))
            st.warning(f"Batch prediction cancelled. {len(predictions)} signal(s) processed.")
        elif failed_signals:
            _logger.warning(
                "Batch prediction: {}/{} failed (API errors)",
                len(failed_signals),
                len(signals),
            )
            st.warning(
                f"⚠️ {len(failed_signals)}/{len(signals)} signals failed "
                f"(API unreachable or error). Successful: {len(signals) - len(failed_signals)}."
            )
        else:
            _logger.info("Batch prediction OK — {} signals completed", len(signals))
            st.success(f"✅ All {len(signals)} predictions completed.")

        # ── Sparse label injection (batch) ────────────────────────────────
        if label_inject_pct > 0 and predictions:
            _success_results = [
                (sig, pred)
                for sig, pred in zip(signals, predictions, strict=False)
                if pred.get("prediction_id") is not None
            ]
            n_to_label = max(0, round(len(_success_results) * label_inject_pct / 100))
            _rng2 = _random.Random(42)
            to_label = _rng2.sample(_success_results, min(n_to_label, len(_success_results)))

            _injected = 0
            _inject_errors = 0
            for _sig, _pred in to_label:
                _pid = _pred.get("prediction_id")
                if _pid is None:
                    continue
                # Determine ground truth label
                if batch_label_choice == "healthy":
                    _gt = LABEL_HEALTHY
                elif batch_label_choice == "unhealthy":
                    _gt = LABEL_UNHEALTHY
                else:
                    # auto: use the true_label stored in the signal dict by _batch_generate
                    _tl = _sig.get("true_label")
                    _gt = LABEL_HEALTHY if _tl == "healthy" else LABEL_UNHEALTHY
                lbl_payload = {
                    "prediction_id": _pid,
                    "ground_truth_label": _gt,
                    "label_source": "predictions_ui_batch",
                    "injected_by": "batch_prediction_tab",
                }
                lbl_res = _api_post("/labels", lbl_payload, api_key=api_key, silent=True)
                if lbl_res:
                    _injected += 1
                else:
                    _inject_errors += 1

            if _injected:
                st.info(
                    f"💾 {_injected}/{len(to_label)} ground truth label(s) injected "
                    f"({label_inject_pct} % of successful predictions)."
                    + (f"  ⚠️ {_inject_errors} failed." if _inject_errors else "")
                )
            elif _inject_errors:
                st.warning(f"⚠️ Label injection failed for all {_inject_errors} attempts.")

        # Store in session state so the plot and table re-render with colors
        st.session_state["batch_predictions"] = predictions
        st.session_state["batch_signals"] = signals

    # After prediction — re-render with colors if results are stored
    if "batch_predictions" in st.session_state and "batch_signals" in st.session_state:
        stored_signals = st.session_state["batch_signals"]
        stored_preds = st.session_state["batch_predictions"]
        if stored_signals == signals:  # same batch
            _render_batch_plot(stored_signals, predictions=stored_preds)
            _render_batch_table(stored_signals, stored_preds)


def _batch_upload() -> list[dict]:
    """Handle batch file upload (CSV multi-row or JSON array)."""
    uploaded = st.file_uploader(
        "Upload batch file",
        type=["json", "csv"],
        help=(
            "CSV: rows with time_values and amplitude_values as semicolon-separated lists "
            "(wide format), or long format with `signal_id`, `time`, `amplitude` columns.\n"
            "JSON: array of objects, each with `time` and `amplitude` arrays."
        ),
        key="batch_upload",
    )
    if uploaded is None:
        return []

    try:
        if uploaded.name.endswith(".json"):
            raw = json.loads(uploaded.read().decode())
            # Support both array and {"signals": [...]}
            entries = raw if isinstance(raw, list) else raw.get("signals", [])
            signals = []
            for i, entry in enumerate(entries):
                time_vals = list(map(float, entry.get("time", entry.get("time_values", []))))
                amp_vals = list(
                    map(float, entry.get("amplitude", entry.get("amplitude_values", [])))
                )
                signals.append(
                    {
                        "id": entry.get("id", entry.get("device_id", f"signal-{i + 1}")),
                        "time": time_vals,
                        "amplitude": amp_vals,
                    }
                )
        else:
            import csv
            import io

            text = uploaded.read().decode()
            reader = csv.DictReader(io.StringIO(text))
            rows = list(reader)
            if not rows:
                st.error("CSV file is empty.")
                return []

            first = rows[0]
            if ";" in str(first.get("time_values", first.get("time", ""))):
                # Wide format: one signal per row, time/amplitude as semicolon-separated
                signals = []
                for i, row in enumerate(rows):
                    t_raw = row.get("time_values", row.get("time", ""))
                    a_raw = row.get("amplitude_values", row.get("amplitude", ""))
                    time_vals = [float(v) for v in str(t_raw).split(";") if v.strip()]
                    amp_vals = [float(v) for v in str(a_raw).split(";") if v.strip()]
                    sig_id = row.get("device_name", row.get("device_id", f"signal-{i + 1}"))
                    signals.append({"id": sig_id, "time": time_vals, "amplitude": amp_vals})
            else:
                # Long format grouped by signal_id (or treat whole file as one signal)
                if "signal_id" in first:
                    groups: dict[str, dict] = {}
                    for row in rows:
                        sid = row["signal_id"]
                        t_val = float(row.get("time", row.get("time_values", 0)))
                        a_val = float(row.get("amplitude", row.get("amplitude_values", 0)))
                        if sid not in groups:
                            groups[sid] = {"id": sid, "time": [], "amplitude": []}
                        groups[sid]["time"].append(t_val)
                        groups[sid]["amplitude"].append(a_val)
                    signals = list(groups.values())
                else:
                    # Single signal long format
                    time_vals = [float(r.get("time", r.get("time_values", 0))) for r in rows]
                    amp_vals = [
                        float(r.get("amplitude", r.get("amplitude_values", 0))) for r in rows
                    ]
                    signals = [{"id": "signal-1", "time": time_vals, "amplitude": amp_vals}]

        # Filter out signals that are too short
        valid = [s for s in signals if len(s["time"]) >= 51 and len(s["amplitude"]) >= 51]
        skipped = len(signals) - len(valid)
        if skipped:
            st.warning(f"Skipped {skipped} signal(s) with fewer than 51 points.")
        if not valid:
            st.error("No valid signals found.")
            return []

        st.success(f"Loaded **{len(valid)}** signal(s) from **{uploaded.name}**.")
        return valid

    except Exception as exc:
        st.error(f"Failed to parse batch file: {exc}")
        return []


def _batch_generate() -> list[dict]:
    """Synthetically generate a batch of signals with drift scenario support.

    Signals are generated only when the user clicks "Generate Batch", then
    cached in ``st.session_state["_batch_gen_signals"]`` so they survive reruns
    without triggering expensive re-computation on every widget interaction.
    """
    from src.ui.views.use_cases_pkg.drift_provocation import (
        DRIFT_TYPES,
        generate_batch,
    )

    # ── Drift scenario selector ──────────────────────────────────────────
    _drift_none_key = "none"
    _drift_options = [_drift_none_key] + DRIFT_SCENARIOS
    col_scen, col_n, col_pts = st.columns(3)
    with col_scen:
        drift_scenario = st.selectbox(
            "Drift scenario",
            _drift_options,
            format_func=lambda k: (
                "None (standard mix)"
                if k == _drift_none_key
                else DRIFT_SCENARIO_LABELS.get(k, k.replace("_", " ").title())
            ),
            key="batch_drift_scenario",
            help="Synchronised with the Drift Provocation use case.",
        )
    with col_n:
        n_signals = st.selectbox(
            "Number of signals", [1, 5, 10, 20, 50, 100, 200, 500, 1000], index=1, key="batch_n"
        )
    with col_pts:
        n_points = st.slider("Points per signal", 51, 200, 101, key="batch_n_points")

    # Show drift description if a scenario is selected
    if drift_scenario != _drift_none_key:
        st.markdown(DRIFT_TYPES.get(drift_scenario, ""))

    # Determine healthy fraction based on scenario
    if drift_scenario == "prior_probability_drift":
        healthy_pct = st.slider(
            "% healthy signals (drifted class balance)",
            min_value=5,
            max_value=95,
            value=15,
            step=5,
            key="batch_healthy_pct",
        )
    else:
        healthy_pct = st.slider(
            "% healthy signals",
            min_value=0,
            max_value=100,
            value=50,
            step=10,
            key="batch_healthy_pct",
        )

    n_healthy = round(int(n_signals) * healthy_pct / 100)
    n_unhealthy = int(n_signals) - n_healthy
    st.caption(
        f"Will generate {n_healthy} healthy (Gaussian) + {n_unhealthy} unhealthy (Lorentzian)"
    )

    # ── Drift scenario: collect params (no generation yet) ──────────────
    if drift_scenario != _drift_none_key:
        gauss_frac = healthy_pct / 100.0

        # Map drift scenario to generate_batch kwargs
        mu_offset = 0.0
        width_mult = 1.0
        noise_mult = 1.0
        height_mult = 1.0
        swap_labels = False

        if drift_scenario == "data_drift":
            mu_offset = -5.0
            noise_mult = 3.0
            width_mult = 1.5
        elif drift_scenario == "concept_drift":
            swap_labels = True
        elif drift_scenario == "feature_drift":
            height_mult = 0.5
            noise_mult = 1.5
        # prior_probability_drift: controlled by gauss_frac above

        if st.button("🔄 Generate Batch", key="batch_gen_btn", type="primary"):
            with st.spinner("Generating batch…"):
                rows = generate_batch(
                    int(n_signals),
                    gaussian_fraction=gauss_frac
                    if drift_scenario == "prior_probability_drift"
                    else 0.7,
                    mu_offset=mu_offset,
                    width_multiplier=width_mult,
                    noise_multiplier=noise_mult,
                    height_multiplier=height_mult,
                    swap_labels=swap_labels,
                    seed=7777,
                    include_raw=True,
                )
            _gen_signals: list[dict] = []
            for i, row in enumerate(rows):
                t_vals = row.get("time_values")
                a_vals = row.get("amplitude_values")
                if t_vals is None or a_vals is None or len(t_vals) < 51:  # type: ignore[arg-type]
                    continue
                shape_type = str(row.get("shape_type", "gaussian"))
                _gen_signals.append(
                    {
                        "id": f"{drift_scenario}-{i + 1}",
                        "true_label": "healthy" if shape_type == "gaussian" else "unhealthy",
                        "time": list(t_vals),  # type: ignore[arg-type]
                        "amplitude": list(a_vals),  # type: ignore[arg-type]
                    }
                )
            st.session_state["_batch_gen_signals"] = _gen_signals
        return st.session_state.get("_batch_gen_signals", [])

    # ── Standard batch generation (no drift) ────────────────────────────
    # Per-class parameter controls
    with st.expander("🔬 Gaussian (Healthy) Parameters", expanded=False):
        gc1, gc2, gc3, gc4 = st.columns(4)
        with gc1:
            g_mu = st.slider("μ (center)", 10.0, 90.0, 50.0, step=0.5, key="batch_g_mu")
        with gc2:
            g_sigma = st.slider("σ (width)", 0.5, 10.0, 2.5, step=0.5, key="batch_g_sigma")
        with gc3:
            g_height = st.slider("Height", 0.1, 5.0, 2.75, step=0.1, key="batch_g_height")
        with gc4:
            g_noise = st.slider("Noise", 0.0, 0.15, 0.015, step=0.005, key="batch_g_noise")

    with st.expander("🔴 Lorentzian (Unhealthy) Parameters", expanded=False):
        lc1, lc2, lc3, lc4 = st.columns(4)
        with lc1:
            l_mu = st.slider("μ (center)", 10.0, 90.0, 50.0, step=0.5, key="batch_l_mu")
        with lc2:
            l_gamma = st.slider("γ (HWHM)", 0.5, 15.0, 5.0, step=0.5, key="batch_l_gamma")
        with lc3:
            l_height = st.slider("Height", 0.1, 5.0, 1.25, step=0.1, key="batch_l_height")
        with lc4:
            l_noise = st.slider("Noise", 0.0, 0.20, 0.08, step=0.005, key="batch_l_noise")

    if st.button("🔄 Generate Batch", key="batch_gen_btn", type="primary"):
        from src.signal_processing.signal_generator import generate_signal

        _std_signals: list[dict] = []
        for i in range(n_healthy):
            sig = generate_signal(
                shape_type="gaussian",
                n_points=n_points,
                seed=i,
                mu=g_mu,
                width_param=g_sigma,
                height=g_height,
                noise_level=g_noise,
            )
            _std_signals.append(
                {
                    "id": f"healthy-{i + 1}",
                    "true_label": "healthy",
                    "time": list(sig.signal.time),
                    "amplitude": list(sig.signal.amplitude),
                }
            )
        for i in range(n_unhealthy):
            sig = generate_signal(
                shape_type="lorentzian",
                n_points=n_points,
                seed=i,
                mu=l_mu,
                width_param=l_gamma,
                height=l_height,
                noise_level=l_noise,
            )
            _std_signals.append(
                {
                    "id": f"unhealthy-{i + 1}",
                    "true_label": "unhealthy",
                    "time": list(sig.signal.time),
                    "amplitude": list(sig.signal.amplitude),
                }
            )
        st.session_state["_batch_gen_signals"] = _std_signals

    return st.session_state.get("_batch_gen_signals", [])


def _render_batch_plot(signals: list[dict], predictions: list[dict] | None) -> None:
    """Render overlay Plotly chart for batch signals, color-coded after prediction."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("Plotly not installed — install plotly to view overlay chart.")
        return

    fig = go.Figure()

    for i, sig in enumerate(signals):
        if predictions is None:
            color = "rgba(148, 163, 184, 0.6)"  # grey (pre-prediction)
            name = sig.get("id", f"signal-{i + 1}")
        else:
            pred = predictions[i] if i < len(predictions) else {}
            label = str(pred.get("predicted_label", ""))
            if label in ("0", "healthy"):
                color = "rgba(16, 185, 129, 0.7)"  # green
            elif label in ("1", "unhealthy"):
                color = "rgba(239, 68, 68, 0.7)"  # red
            else:
                color = "rgba(245, 158, 11, 0.7)"  # amber for errors
            conf = pred.get("prediction_confidence", pred.get("confidence", 0))
            name = f"{sig.get('id', f'signal-{i + 1}')} ({label}, {conf:.0%})"

        fig.add_trace(
            go.Scatter(
                x=sig["time"],
                y=sig["amplitude"],
                mode="lines",
                name=name,
                line={"color": color, "width": 1.5},
            )
        )

    title = (
        "Signal Overlay — Pre-Prediction (grey)"
        if predictions is None
        else "Signal Overlay — After Prediction"
    )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#1e293b",
        margin={"l": 40, "r": 20, "t": 45, "b": 40},
        height=380,
        xaxis_title="Time",
        yaxis_title="Amplitude",
        legend={"itemclick": "toggle", "bgcolor": "rgba(0,0,0,0.3)"},
    )
    key = "batch_plot_pre" if predictions is None else "batch_plot_post"
    st.plotly_chart(fig, width="stretch", key=key)


def _render_batch_table(signals: list[dict], predictions: list[dict]) -> None:
    """Render a statistics table of features + prediction results."""
    import pandas as pd

    rows = []
    for i, (sig, pred) in enumerate(zip(signals, predictions, strict=False)):
        label = str(pred.get("predicted_label", "—"))
        label_text = (
            "✅ Healthy"
            if label in ("0", "healthy")
            else ("❌ Unhealthy" if label in ("1", "unhealthy") else label)
        )
        # API returns 'prediction_confidence'; 'confidence' kept as legacy fallback
        conf_raw = pred.get("prediction_confidence", pred.get("confidence", None))
        conf: float | None = float(conf_raw) if conf_raw is not None else None
        feats = pred.get("features", {})
        row = {
            "Signal ID": sig.get("id", f"signal-{i + 1}"),
            "Prediction": label_text,
            "Confidence": f"{conf:.1%}" if isinstance(conf, float) else "—",
        }
        # Add feature columns
        for feat in ("fwhm", "peak_height", "peak_area", "noise_level", "snr", "peak_center"):
            v = feats.get(feat)
            row[feat] = f"{v:.4f}" if isinstance(v, float) else "—"
        rows.append(row)

    df = pd.DataFrame(rows)
    st.markdown("#### 📊 Prediction Results")
    st.dataframe(df, width="stretch")


# ── History tab ─────────────────────────────────────────────────


def _history_tab() -> None:
    """Show recent predictions from the database (via /stats or direct DB query)."""
    st.markdown(
        '<div class="section-header">📋 Prediction History</div>'
        '<div class="section-subheader">Recent predictions stored in the PostgreSQL database</div>',
        unsafe_allow_html=True,
    )

    stats = _api_get("/stats")
    if stats:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Predictions", stats.get("total_predictions", "—"))
        c2.metric("Healthy", stats.get("healthy_count", "—"))
        c3.metric("Unhealthy", stats.get("unhealthy_count", "—"))
        label_cov = stats.get("label_coverage", 0)
        c4.metric(
            "Label Coverage",
            f"{label_cov * 100:.1f}%" if isinstance(label_cov, (int, float)) else "—",
        )

        st.markdown("---")

        # Show a dataframe if the API includes row-level history (forward-compat).
        recent = stats.get("recent_predictions", [])
        if recent:
            import pandas as pd

            st.markdown("**Recent predictions**")
            st.dataframe(pd.DataFrame(recent), width="stretch")
        elif stats.get("total_predictions", 0) == 0:
            st.info(
                "No predictions recorded yet.  Use the **🎯 Predict** or "
                "**📦 Batch Prediction** tabs to run predictions, then refresh here."
            )
        else:
            st.success(
                f"Prediction counts are shown above.  "
                f"{stats.get('total_predictions', 0)} prediction(s) recorded in the last "
                f"{stats.get('lookback_days', 30)} days.  "
                "Run individual predictions in the **🎯 Predict** tab to see per-signal details."
            )
    else:
        st.warning(
            "Could not reach the API at " + _api_base() + "/stats. Is the Docker stack running?"
        )


# ── Health tab ──────────────────────────────────────────────────


def _health_tab() -> None:
    """Show system health from the /health endpoint."""
    st.markdown(
        '<div class="section-header">💚 System Health</div>'
        '<div class="section-subheader">Live status from the /health API endpoint</div>',
        unsafe_allow_html=True,
    )

    if st.button("🔄 Refresh", key="refresh_health"):
        st.cache_data.clear()

    health = _api_get("/health")
    if not health:
        st.error(
            "Could not reach the API at " + _api_base() + "/health. Is the Docker stack running?"
        )
        return

    # Overall status
    overall = health.get("status", "unknown")
    if overall == "healthy":
        st.success(f"**Overall:** {overall.upper()}")
    elif overall == "degraded":
        st.warning(f"**Overall:** {overall.upper()}")
    else:
        st.error(f"**Overall:** {overall.upper()}")

    # Metric cards
    cols = st.columns(5)
    checks = [
        ("🗄️", "Database", health.get("database_connected", False)),
        ("🤖", "Model Loaded", health.get("model_loaded", False)),
        ("✅", "Model Valid", health.get("model_valid", False)),
        ("🔬", "MLflow", health.get("mlflow_accessible", False)),
        ("📦", "DVC Remote", health.get("dvc_remote_accessible")),
    ]
    for col, (icon, label, ok) in zip(cols, checks, strict=False):
        if ok is True:
            text = '<span style="color:#10b981">UP</span>'
        elif ok is False:
            text = '<span style="color:#ef4444">DOWN</span>'
        else:
            text = '<span style="color:#64748b">N/A</span>'
        col.markdown(metric_card(icon, text, label), unsafe_allow_html=True)

    # Service breakdown
    services = health.get("services", {})
    if services:
        st.markdown("---")
        st.markdown("#### Service Details")
        for svc_name, svc_status in services.items():
            status_str = str(svc_status)
            if "healthy" in status_str:
                st.markdown(f"- 🟢 **{svc_name}**: {status_str}")
            elif "degraded" in status_str:
                st.markdown(f"- 🟡 **{svc_name}**: {status_str}")
            else:
                st.markdown(f"- 🔴 **{svc_name}**: {status_str}")

    # Metadata
    st.markdown("---")
    with st.expander("📋 Full /health Response", expanded=False):
        st.json(health)
