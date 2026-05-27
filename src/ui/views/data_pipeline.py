"""Data Pipeline & Signal Visualisation page."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_PROJECT_ROOT = str(_Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from src.ui.logging_ui import get_ui_logger
from src.ui.styles import get_global_css, hero_section

_logger = get_ui_logger(__name__)


def render() -> None:
    """Render the Data Pipeline page."""
    try:
        _render_content()
    except Exception:
        _logger.exception("Unhandled exception in data_pipeline.render()")
        raise


def _render_content() -> None:
    """Inner render — separated so exceptions are caught and logged by render()."""
    st.markdown(get_global_css(), unsafe_allow_html=True)
    st.markdown(
        hero_section(
            "Data Pipeline & Signals",
            "From synthetic IoT signals to extracted features \u2014 the foundation of our ML pipeline.",
        ),
        unsafe_allow_html=True,
    )

    # Use st.radio (keyed) instead of st.tabs() — st.tabs() resets to tab 0
    # on every full Streamlit rerun, causing the annoying "jump to first tab" behaviour.
    _DP_TABS = [
        "\U0001f4e1 Signal Comparison",
        "\U0001f527 Feature Extraction",
        "\U0001f4c8 Drift Scenarios",
    ]
    active_dp = st.radio(
        "Data page tab",
        _DP_TABS,
        horizontal=True,
        key="_dp_tab",
        label_visibility="collapsed",
    )
    st.markdown(
        "<hr style='margin:0 0 1rem 0;border-color:#334155;'>",
        unsafe_allow_html=True,
    )

    if active_dp == _DP_TABS[0]:
        _signal_comparison_tab()
    elif active_dp == _DP_TABS[1]:
        _feature_extraction_tab()
    else:
        _drift_tab()


# ── Signal Comparison tab ────────────────────────────────────────────────────


def _signal_comparison_tab() -> None:
    """Show healthy vs unhealthy signal comparison."""
    st.markdown(
        """<div class="section-header">\U0001f4e1 Healthy vs Unhealthy Signals</div>"""
        """<div class="section-subheader">Synthetic signals with controlled parameters simulate real device telemetry</div>""",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        st.markdown(
            """<div class="info-card">"""
            "<h3>\u2705 Healthy Signal \u2014 Gaussian Peak</h3>"
            "<ul>"
            "<li><b>Shape:</b> Gaussian f(t) = A \u00b7 exp(\u2212(t\u2212\u03bc)\u00b2 / 2\u03c3\u00b2)</li>"
            "<li><b>\u03c3 range:</b> [2.0, 5.0] \u2014 narrow, well-defined peak</li>"
            "<li><b>Noise:</b> Low (\u03c3_noise \u2208 [0.01, 0.05])</li>"
            "<li><b>SNR:</b> Typically > 40 dB</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """<div class="info-card">"""
            "<h3>\u274c Unhealthy Signal \u2014 Lorentzian Peak</h3>"
            "<ul>"
            "<li><b>Shape:</b> Lorentzian f(t) = A \u00b7 \u03b3\u00b2 / ((t\u2212\u03bc)\u00b2 + \u03b3\u00b2)</li>"
            "<li><b>\u03b3:</b> 1.1775 \u00d7 \u03c3 (broader, heavier tails)</li>"
            "<li><b>Noise:</b> High (\u03c3_noise \u2208 [0.06, 0.12])</li>"
            "<li><b>SNR:</b> Typically < 25 dB</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )

    try:
        from src.ui.components.signal_viz import create_signal_comparison_chart

        fig = create_signal_comparison_chart()
        st.plotly_chart(fig, key="signal_comparison")
    except ImportError:
        st.info("Install plotly and scipy to see interactive signal plots.")


# ── Feature Extraction tab ────────────────────────────────────────────────────


def _feature_extraction_tab() -> None:
    """Feature Extraction with two sub-tabs: pipeline overview and live extractor."""
    _FE_TABS = [
        "\U0001f527 Feature Extraction Pipeline",
        "\U0001f9ea Live Feature Extractor",
    ]
    sel = st.radio(
        "Feature extraction view",
        _FE_TABS,
        horizontal=True,
        key="_fe_sel",
        label_visibility="collapsed",
    )
    st.markdown("<hr style='margin:0 0 1rem 0;border-color:#334155;'>", unsafe_allow_html=True)

    if sel == _FE_TABS[0]:
        _feature_pipeline_view()
    else:
        _live_feature_extractor()


def _feature_pipeline_view() -> None:
    """Static feature extraction pipeline overview."""
    st.markdown(
        """<div class="section-header">\U0001f527 Feature Extraction Pipeline</div>"""
        """<div class="section-subheader">6 features extracted from each raw signal for ML classification</div>""",
        unsafe_allow_html=True,
    )

    features = [
        ("\U0001f4cf", "peak_height", "Maximum amplitude of the signal after baseline removal"),
        ("\U0001f4cd", "peak_center", "Time-axis location of the peak maximum (\u03bc)"),
        ("\u2194\ufe0f", "FWHM", "Full Width at Half Maximum \u2014 peak breadth indicator"),
        ("\U0001f4e1", "SNR", "Signal-to-Noise Ratio in dB \u2014 key discriminating feature"),
        ("\U0001f4d0", "peak_area", "Numerical integral under the peak (trapezoidal rule)"),
        ("\U0001f50a", "noise_level", "Standard deviation of residuals outside the peak region"),
    ]

    cols = st.columns(4)
    for i, (icon, name, desc) in enumerate(features):
        with cols[i % 4]:
            st.markdown(
                f"""<div class="info-card" style="min-height:130px">"""
                f"<h3>{icon} {name}</h3>"
                f"<p style='font-size:0.85rem'>{desc}</p></div>",
                unsafe_allow_html=True,
            )

    try:
        from src.ui.components.signal_viz import create_feature_extraction_diagram

        fig = create_feature_extraction_diagram()
        st.plotly_chart(fig, key="feature_extraction")
    except ImportError:
        st.info("Install plotly and scipy to see the feature extraction visualisation.")


def _live_feature_extractor() -> None:
    """Live signal generator + feature extraction + annotated plots."""
    st.markdown(
        """<div class="section-header">\U0001f9ea Live Feature Extractor</div>"""
        """<div class="section-subheader">Generate a synthetic signal and see the extracted features in real time</div>""",
        unsafe_allow_html=True,
    )

    # ── Import dependencies ───────────────────────────────────────────────────
    try:
        import numpy as np
        import plotly.graph_objects as go

        from src.config import DRIFT_SCENARIO_LABELS, DRIFT_SCENARIOS
        from src.signal_processing.feature_extractor import extract_features
        from src.signal_processing.signal_generator import generate_signal
    except ImportError as exc:
        st.error(f"Import error: {exc}. Make sure the project dependencies are installed.")
        return

    # ── Signal generator controls (same as Predictions page) ─────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        shape = st.selectbox("Shape type", ["gaussian", "lorentzian"], key="_lfe_shape")
    with col2:
        _all_scenarios = ["baseline"] + DRIFT_SCENARIOS + ["custom"]
        scenario = st.selectbox(
            "Drift scenario",
            _all_scenarios,
            format_func=lambda k: DRIFT_SCENARIO_LABELS.get(k, k.replace("_", " ").title()),
            key="_lfe_scenario",
        )
    with col3:
        n_points = st.slider("Number of points", 51, 500, 101, key="_lfe_npts")

    seed = st.number_input("Random seed", value=42, min_value=0, max_value=99999, key="_lfe_seed")

    custom_mu = custom_width = custom_height = custom_noise = None
    if scenario == "custom":
        st.markdown("**\U0001f3db\ufe0f Custom Peak Parameters**")
        width_label = "\u03c3 (sigma)" if shape == "gaussian" else "\u03b3 (gamma HWHM)"
        width_default = 2.5 if shape == "gaussian" else 5.0
        # Use shape-specific keys so Gaussian and Lorentzian parameters are tracked separately
        _key_mu = f"_lfe_mu_{shape}"
        _key_width = f"_lfe_width_{shape}"
        _key_height = f"_lfe_height_{shape}"
        _key_noise = f"_lfe_noise_{shape}"

        # ── Initialize session state keys with defaults (first run only) ────
        # Using setdefault ensures the key exists in session_state BEFORE any
        # slider is created, so we can safely omit the value= argument from the
        # slider call.  Passing value= AND having the key in session_state
        # triggers Streamlit's "created with default value but also set via
        # Session State API" warning.  With setdefault + no value=, no warning.
        st.session_state.setdefault(_key_mu, 50.0)
        st.session_state.setdefault(_key_width, width_default)
        st.session_state.setdefault(_key_height, 2.0)
        st.session_state.setdefault(_key_noise, 0.02)

        # ── Apply pending reset BEFORE sliders are instantiated ──────────────
        if st.session_state.pop(f"_lfe_reset_pending_{shape}", False):
            st.session_state[_key_mu] = 50.0
            st.session_state[_key_width] = width_default
            st.session_state[_key_height] = 2.0
            st.session_state[_key_noise] = 0.02

        p1, p2, p3, p4 = st.columns(4)
        with p1:
            # No value= argument: Streamlit reads from st.session_state[_key_mu]
            custom_mu = st.slider("\u03bc (peak center)", 10.0, 90.0, step=0.5, key=_key_mu)
        with p2:
            custom_width = st.slider(width_label, 0.5, 15.0, step=0.5, key=_key_width)
        with p3:
            custom_height = st.slider(
                "Height",
                0.1,
                5.0,
                step=0.1,
                key=_key_height,
                help="Gaussian requires Height \u2265 1.0; Lorentzian requires Height \u2265 0.8.",
            )
        with p4:
            custom_noise = st.slider("Noise level", 0.0, 0.20, step=0.01, key=_key_noise)
        if st.button("\u21ba Reset to defaults", key="_lfe_reset"):
            st.session_state[f"_lfe_reset_pending_{shape}"] = True
            st.rerun()

    # ── Map scenario to generator params ──────────────────────────────────────
    actual_scenario = None
    extra_shape = shape
    extra_height = custom_height
    if scenario == "custom":
        actual_scenario = None
    elif scenario == "feature_drift":
        actual_scenario = None
        # GaussianParameters.height >= 1.0; LorentzianParameters.height >= 0.8.
        # Use a valid low height to simulate feature drift (below healthy range ≥ 2.5).
        extra_height = 1.5 if shape == "gaussian" else 0.9
    elif scenario == "prior_probability_drift":
        actual_scenario = None
        extra_shape = "lorentzian"
    else:
        actual_scenario = scenario

    # ── Generate signal ────────────────────────────────────────────────────────
    try:
        from typing import Literal, cast

        sig = generate_signal(
            shape_type=cast(Literal["gaussian", "lorentzian"], extra_shape),
            drift_scenario=cast(
                "Literal['baseline', 'data_drift', 'concept_drift'] | None", actual_scenario
            ),
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
        return
    time_arr = np.array(sig.signal.time)
    amp_arr = np.array(sig.signal.amplitude)

    # ── Signal plot ────────────────────────────────────────────────────────────
    st.markdown("### Generated Signal")
    sig_fig = go.Figure()
    sig_fig.add_trace(
        go.Scatter(
            x=time_arr,
            y=amp_arr,
            mode="lines",
            name="Signal",
            line={"color": "#6366f1", "width": 2},
        )
    )
    sig_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#1e293b",
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
        height=280,
        xaxis_title="Time",
        yaxis_title="Amplitude",
    )
    st.plotly_chart(sig_fig, key="_lfe_signal_plot")

    # ── Extract features button ─────────────────────────────────────────────────
    if not st.button("\U0001f9ea Extract Features", type="primary", key="_lfe_btn"):
        st.info("Click **Extract Features** to analyse this signal.")
        return

    # Use sig.signal directly — it's already a fully-validated SignalData
    # (with shape_type and all required fields).  Re-constructing from raw
    # arrays would fail because SignalData requires shape_type as well.
    feats = extract_features(sig.signal)

    if any(v is None for v in feats.values()):
        st.warning(
            "Some features could not be extracted (no clear peak detected). Try a different seed or shape."
        )

    # ── Feature table ──────────────────────────────────────────────────────────
    st.markdown("### Extracted Features")
    feat_meta = [
        ("peak_height", "\U0001f4cf", "Peak height", "Max amplitude after baseline removal"),
        ("peak_center", "\U0001f4cd", "Peak center (\u03bc)", "Time coordinate of peak maximum"),
        ("fwhm", "\u2194\ufe0f", "FWHM", "Full Width at Half Maximum"),
        ("peak_area", "\U0001f4d0", "Peak area", "Trapezoidal integral under the peak"),
        ("snr", "\U0001f4e1", "SNR", "Signal-to-Noise Ratio (linear)"),
        ("noise_level", "\U0001f50a", "Noise level", "Std dev of residuals outside peak"),
    ]
    tbl_cols = st.columns(3)
    for i, (key, icon, label, desc) in enumerate(feat_meta):
        val = feats.get(key)
        disp = f"{val:.4f}" if val is not None else "N/A"
        with tbl_cols[i % 3]:
            st.markdown(
                f"""<div class="info-card" style="min-height:90px">"""
                f"<h3>{icon} {label}</h3>"
                f"<p style='font-size:1.1rem;font-weight:700;color:#a5b4fc'>{disp}</p>"
                f"<p style='font-size:0.78rem;color:#94a3b8'>{desc}</p></div>",
                unsafe_allow_html=True,
            )

    # ── Annotated feature plots ────────────────────────────────────────────────
    st.markdown("### Feature Visualisation")
    plot_col_left, plot_col_right = st.columns(2)

    # Compute helpers
    peak_h = feats.get("peak_height")
    peak_c = feats.get("peak_center")
    fwhm = feats.get("fwhm")
    peak_a = feats.get("peak_area")
    noise_l = feats.get("noise_level")
    snr_val = feats.get("snr")

    # ── Left plot: peak_height, peak_center, FWHM, peak_area ─────────────────
    with plot_col_left:
        st.markdown("**Peak Geometry Features**")
        fig_left = go.Figure()
        # Signal trace
        fig_left.add_trace(
            go.Scatter(
                x=time_arr,
                y=amp_arr,
                mode="lines",
                name="Signal",
                line={"color": "#6366f1", "width": 2},
            )
        )
        # Peak height vertical line
        if peak_h is not None and peak_c is not None:
            fig_left.add_trace(
                go.Scatter(
                    x=[peak_c, peak_c],
                    y=[0, peak_h],
                    mode="lines",
                    name=f"peak_height = {peak_h:.3f}",
                    line={"color": "#f59e0b", "width": 2, "dash": "dash"},
                )
            )
            fig_left.add_annotation(
                x=peak_c,
                y=peak_h * 1.05,
                text=f"\u25b2 height={peak_h:.3f}",
                showarrow=False,
                font={"color": "#f59e0b", "size": 11},
            )
        # FWHM horizontal line at half-max
        if peak_h is not None and fwhm is not None and peak_c is not None:
            half_max = peak_h / 2.0
            fig_left.add_trace(
                go.Scatter(
                    x=[peak_c - fwhm / 2, peak_c + fwhm / 2],
                    y=[half_max, half_max],
                    mode="lines",
                    name=f"FWHM = {fwhm:.3f}",
                    line={"color": "#10b981", "width": 2.5},
                )
            )
        # Peak area fill
        if peak_a is not None:
            fig_left.add_trace(
                go.Scatter(
                    x=list(time_arr) + list(time_arr[::-1]),
                    y=list(amp_arr) + [0] * len(amp_arr),
                    fill="toself",
                    mode="none",
                    name=f"peak_area = {peak_a:.3f}",
                    fillcolor="rgba(139,92,246,0.15)",
                )
            )
        # Peak center dot
        if peak_c is not None and peak_h is not None:
            fig_left.add_trace(
                go.Scatter(
                    x=[peak_c],
                    y=[peak_h],
                    mode="markers",
                    name=f"peak_center = {peak_c:.2f}",
                    marker={"color": "#ef4444", "size": 10, "symbol": "circle"},
                )
            )
        fig_left.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#1e293b",
            margin={"l": 40, "r": 20, "t": 30, "b": 40},
            height=340,
            xaxis_title="Time",
            yaxis_title="Amplitude",
            legend={"orientation": "v", "font": {"size": 10}},
        )
        st.plotly_chart(fig_left, key="_lfe_left_plot")

    # ── Right plot: noise_level baseline + SNR visualisation ──────────────────
    with plot_col_right:
        st.markdown("**SNR & Noise Level**")
        fig_right = go.Figure()
        # Signal trace
        fig_right.add_trace(
            go.Scatter(
                x=time_arr,
                y=amp_arr,
                mode="lines",
                name="Signal",
                line={"color": "#6366f1", "width": 2},
            )
        )
        if noise_l is not None:
            # Noise band around zero baseline
            fig_right.add_trace(
                go.Scatter(
                    x=list(time_arr) + list(time_arr[::-1]),
                    y=[noise_l] * len(time_arr) + [-noise_l] * len(time_arr),
                    fill="toself",
                    mode="none",
                    name=f"noise band (\u00b1{noise_l:.4f})",
                    fillcolor="rgba(239,68,68,0.18)",
                )
            )
            fig_right.add_hline(
                y=noise_l,
                line_dash="dot",
                line_color="#ef4444",
                annotation_text=f"noise = {noise_l:.4f}",
            )
        if peak_h is not None and noise_l is not None:
            snr_disp = (
                snr_val
                if snr_val is not None
                else (peak_h / noise_l if noise_l > 0 else float("inf"))
            )
            fig_right.add_hline(
                y=peak_h,
                line_dash="dot",
                line_color="#f59e0b",
                annotation_text=f"peak = {peak_h:.3f}  |  SNR = {snr_disp:.1f}x",
            )
            # Arrow from noise band to peak to illustrate SNR
            fig_right.add_shape(
                type="line",
                x0=time_arr[len(time_arr) // 4],
                y0=noise_l,
                x1=time_arr[len(time_arr) // 4],
                y1=peak_h,
                line={"color": "#a78bfa", "width": 1.5, "dash": "dot"},
            )
        fig_right.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#1e293b",
            margin={"l": 40, "r": 20, "t": 30, "b": 40},
            height=340,
            xaxis_title="Time",
            yaxis_title="Amplitude",
            legend={"orientation": "v", "font": {"size": 10}},
        )
        st.plotly_chart(fig_right, key="_lfe_right_plot")

    if snr_val is not None:
        snr_note = (
            "\u2705 High SNR \u2014 healthy-like signal"
            if snr_val > 10
            else "\u26a0\ufe0f Low SNR \u2014 unhealthy-like signal"
        )
        st.caption(f"SNR interpretation: {snr_val:.2f}x ({snr_val:.1f} linear)  {snr_note}")


# ── Drift Scenarios tab ───────────────────────────────────────────────────────


def _drift_tab() -> None:
    """Show 4 drift scenario explanations with individual plots."""
    st.markdown(
        """<div class="section-header">\U0001f4c8 Drift Scenarios</div>"""
        """<div class="section-subheader">Four types of drift that can affect model performance in production</div>""",
        unsafe_allow_html=True,
    )

    _DRIFT_TABS = [
        "\U0001f4ca Data Drift",
        "\U0001f9e0 Concept Drift",
        "\U0001f527 Feature Drift",
        "\u2696\ufe0f Prior Probability Drift",
    ]
    sel = st.radio(
        "Drift scenario",
        _DRIFT_TABS,
        horizontal=True,
        key="_drift_sel",
        label_visibility="collapsed",
    )
    st.markdown("<hr style='margin:0 0 1rem 0;border-color:#334155;'>", unsafe_allow_html=True)

    if sel == _DRIFT_TABS[0]:
        _drift_data_drift()
    elif sel == _DRIFT_TABS[1]:
        _drift_concept_drift()
    elif sel == _DRIFT_TABS[2]:
        _drift_feature_drift()
    else:
        _drift_prior_drift()

    _drift_monitoring_info()


def _drift_monitoring_info() -> None:
    """Show EvidentlyAI and Airflow drift monitoring info."""
    st.markdown("---")
    st.markdown("#### \U0001f6e1\ufe0f Drift Monitoring in Production")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """<div class="info-card">"""
            "<h3>\U0001f50d evidently_drift_detection DAG</h3>"
            "<p>Runs daily at 06:00 UTC. Uses EvidentlyAI to compare current feature "
            "distributions against the reference baseline. Publishes drift scores "
            "to Prometheus. Triggers Alertmanager if thresholds are breached.</p>"
            "<p><b>Metrics:</b> Wasserstein distance (SNR, FWHM, peak_height), "
            "Jensen-Shannon divergence (categorical distributions)</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """<div class="info-card">"""
            "<h3>\U0001f4c9 drift_triggered_retraining DAG</h3>"
            "<p>Triggered when drift scores exceed configured thresholds. "
            "Validates drift severity (avoids retraining on transient noise), "
            "pulls fresh production data, retrains the model, and conditionally "
            "promotes to Production if the new model outperforms the champion.</p>"
            "<p><b>Threshold:</b> Configurable in Airflow Variables: "
            "<code>drift_threshold_wasserstein</code></p>"
            "</div>",
            unsafe_allow_html=True,
        )


def _drift_plot(
    reference_label: str,
    drifted_label: str,
    ref_mu: float,
    ref_sigma: float,
    drift_mu: float,
    drift_sigma: float,
    ref_color: str = "#6366f1",
    drift_color: str = "#ef4444",
) -> None:
    """Generic drift visualisation plot using two Gaussian distributions."""
    try:
        import numpy as np
        import plotly.graph_objects as go

        x = np.linspace(0, 100, 300)
        ref_dist = np.exp(-0.5 * ((x - ref_mu) / ref_sigma) ** 2)
        drift_dist = np.exp(-0.5 * ((x - drift_mu) / drift_sigma) ** 2)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=ref_dist,
                mode="lines",
                name=reference_label,
                fill="tozeroy",
                fillcolor=f"rgba{ref_color[1:]}26".replace("26", ",0.15)").replace(
                    "rgba",
                    "rgba(int(ref_color[1:3],16),int(ref_color[3:5],16),int(ref_color[5:],16),0.15)",
                ),
                line={"color": ref_color, "width": 2},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=drift_dist,
                mode="lines",
                name=drifted_label,
                fill="tozeroy",
                line={"color": drift_color, "width": 2, "dash": "dash"},
            )
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#1e293b",
            margin={"l": 40, "r": 20, "t": 30, "b": 40},
            height=260,
            xaxis_title="Feature value",
            yaxis_title="Density",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.0},
        )
        st.plotly_chart(fig, key=f"_drift_plot_{ref_mu}_{drift_mu}")
    except ImportError:
        st.info("Install plotly to see drift visualisation plots.")


def _drift_data_drift() -> None:
    st.markdown("## \U0001f4ca Data Drift")
    st.markdown(
        """
**What is it?**
Data drift (also called *covariate shift*) occurs when the **distribution of input features
changes** while the underlying relationship between features and labels stays the same.

**In our system:**
- Peak center (\u03bc) shifts from 50 to ~38 (device mounting changes)
- Background noise increases (sensor degradation)
- Feature distributions of SNR and FWHM shift significantly
- The Logistic Regression decision boundary is still valid \u2014 but predictions become less confident

**Effect:**
The model was trained on one distribution and now sees a different one.
Accuracy may gradually degrade even though the *rules* for healthy/unhealthy haven't changed.

**Detection:**
EvidentlyAI computes the Wasserstein distance between reference and current SNR, FWHM,
and peak_height distributions. Alerts fire if distance exceeds the threshold.
        """
    )
    st.markdown("**Visualisation: SNR feature distribution shift**")
    _drift_plot_simple(
        "Reference (SNR \u223c 45 dB)",
        "Drifted (SNR \u223c 28 dB)",
        mu1=50,
        sigma1=8,
        mu2=32,
        sigma2=10,
        color1="#6366f1",
        color2="#ef4444",
        key="_dd_plot",
    )


def _drift_concept_drift() -> None:
    st.markdown("## \U0001f9e0 Concept Drift")
    st.markdown(
        """
**What is it?**
Concept drift (also called *posterior shift*) occurs when the **relationship between features
and labels changes**. The feature distributions may stay the same, but what constitutes
"healthy" vs "unhealthy" has shifted.

**In our system:**
- Previously: Gaussian peak (narrow, high SNR) = Healthy
- After drift: Some devices now show Gaussian-like signals but are failing (different failure mode)
- Intermediate \u03c3/\u03b3 values blur the separation between classes
- The model's decision boundary is now wrong even for signals it was trained on

**Effect:**
Accuracy drops. The model is confidently wrong. This is the hardest drift to detect
without ground truth labels (requires sparse labelling).

**Detection:**
Accuracy drop tracked via sparse label comparison. Prediction distribution shift
(fraction of healthy vs unhealthy predictions over time) monitored in Grafana.
        """
    )
    st.markdown("**Visualisation: Overlapping class distributions after concept drift**")
    _drift_plot_simple(
        "Healthy (before drift)",
        "Unhealthy (before drift)",
        mu1=45,
        sigma1=5,
        mu2=25,
        sigma2=7,
        color1="#10b981",
        color2="#ef4444",
        key="_cd_plot1",
    )
    _drift_plot_simple(
        "Healthy (after drift)",
        "Unhealthy (after drift)",
        mu1=40,
        sigma1=9,
        mu2=35,
        sigma2=9,
        color1="#10b981",
        color2="#f59e0b",
        key="_cd_plot2",
        note="Note: distributions overlap \u2014 boundary is no longer clear",
    )


def _drift_feature_drift() -> None:
    st.markdown("## \U0001f527 Feature Drift")
    st.markdown(
        """
**What is it?**
Feature drift is a subtype of data drift where **specific engineered features change
their distribution** independently of the raw signal. This can happen due to changes
in the feature extraction pipeline itself or in the physical properties the features measure.

**In our system:**
- `peak_height` decreases gradually (sensor calibration drift)
- `FWHM` widens (mechanical wear increases broadening)
- `peak_area` shifts (sensitivity loss)
- `peak_center` stays stable (mounting unchanged)

**Effect:**
Some features become less informative for classification. Feature importance changes.
The model may start relying on features that are now noise.

**Detection:**
Per-feature distribution monitoring in EvidentlyAI. Each feature gets its own
drift score. A feature with Wasserstein distance > threshold triggers an alert.
        """
    )
    st.markdown("**Visualisation: peak_height feature drift**")
    _drift_plot_simple(
        "Reference (peak_height \u223c 2.5)",
        "Drifted (peak_height \u223c 1.2)",
        mu1=55,
        sigma1=7,
        mu2=30,
        sigma2=8,
        color1="#6366f1",
        color2="#f59e0b",
        key="_fd_plot",
    )


def _drift_prior_drift() -> None:
    st.markdown("## \u2696\ufe0f Prior Probability Drift")
    st.markdown(
        """
**What is it?**
Prior probability drift (also called *label shift* or *class imbalance drift*) occurs
when the **proportion of classes in incoming data changes**, even if the per-class
feature distributions stay the same.

**In our system:**
- Baseline: 70% Gaussian (healthy), 30% Lorentzian (unhealthy)
- After drift: 30% Gaussian (healthy), 70% Lorentzian (unhealthy) — production batch failed
- The model was trained on 70/30; it now sees a 30/70 population

**Effect:**
Even a perfectly calibrated model will produce more false negatives (missing unhealthy devices)
if it was tuned for a 70/30 prior. The *recall* for unhealthy devices drops.
The overall accuracy number may look fine while the real-world miss rate rises.

**Detection:**
Prediction distribution monitoring: the fraction of healthy vs unhealthy predictions
over a rolling window (e.g., last 24h vs. baseline). Grafana dashboard: "Class Balance".
        """
    )
    st.markdown("**Visualisation: class balance shift**")
    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Reference (training)",
                x=["Healthy", "Unhealthy"],
                y=[70, 30],
                marker_color=["#6366f1", "#ef4444"],
            )
        )
        fig.add_trace(
            go.Bar(
                name="After drift (production)",
                x=["Healthy", "Unhealthy"],
                y=[30, 70],
                marker_color=["#818cf8", "#fca5a5"],
            )
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#1e293b",
            margin={"l": 40, "r": 20, "t": 30, "b": 40},
            height=260,
            barmode="group",
            yaxis_title="Percentage (%)",
            legend={"orientation": "h"},
        )
        st.plotly_chart(fig, key="_ppd_plot")
    except ImportError:
        st.info("Install plotly to see the class balance visualisation.")


def _hex_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Convert a #RRGGBB hex string to an rgba() string Plotly accepts."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _drift_plot_simple(
    ref_label: str,
    drift_label: str,
    mu1: float,
    sigma1: float,
    mu2: float,
    sigma2: float,
    color1: str,
    color2: str,
    key: str,
    note: str = "",
) -> None:
    """Plot two Gaussian distribution curves to visualise drift."""
    try:
        import numpy as np
        import plotly.graph_objects as go

        x = np.linspace(0, 100, 300)
        d1 = np.exp(-0.5 * ((x - mu1) / sigma1) ** 2)
        d2 = np.exp(-0.5 * ((x - mu2) / sigma2) ** 2)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=d1,
                mode="lines",
                name=ref_label,
                fill="tozeroy",
                fillcolor=_hex_rgba(color1),
                line={"color": color1, "width": 2},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=d2,
                mode="lines",
                name=drift_label,
                fill="tozeroy",
                fillcolor=_hex_rgba(color2),
                line={"color": color2, "width": 2, "dash": "dash"},
            )
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#1e293b",
            margin={"l": 40, "r": 20, "t": 30, "b": 40},
            height=240,
            xaxis_title="Feature value",
            yaxis_title="Density",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.0},
        )
        st.plotly_chart(fig, key=key)
        if note:
            st.caption(note)
    except ImportError:
        st.info("Install plotly to see drift visualisation.")
