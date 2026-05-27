"""Tab 6 — Prediction & Monitoring: send test signals, see predictions, track metrics.

Interactive prediction testing against the live FastAPI endpoint:
  1. Generate or customize test signals
  2. Send to the API and display results
  3. Batch-test multiple signals, show confusion-matrix style breakdown
  4. Track prediction distribution and model confidence over a batch
"""

from __future__ import annotations

import json

import streamlit as st

from ._common import GAMMA_SIGMA_FACTOR, SECTION_CSS


def _api_base() -> str:
    """Get FastAPI base URL."""
    from src.ui.components.docker_utils import get_service_url

    return get_service_url("mlops_nginx", 80)


def _api_post(path: str, payload: dict, *, api_key: str = "") -> dict | None:
    """POST JSON to the API."""
    import urllib.error
    import urllib.request

    url = f"{_api_base()}{path}"
    data = json.dumps(payload).encode()
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode() if exc.fp else ""
        st.error(f"API error {exc.code}: {body[:300]}")
    except Exception as exc:
        st.error(f"Connection error: {exc}")
    return None


def _api_get(path: str) -> dict | None:
    """GET from the API."""
    import urllib.error
    import urllib.request

    url = f"{_api_base()}{path}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        st.error(f"Connection error: {exc}")
    return None


def _render_api_health() -> bool:
    """Show API health status. Return True if healthy."""
    data = _api_get("/health")
    if data and data.get("status") == "healthy":
        st.success(
            f"API is **healthy** — model `{data.get('model_type', '?')}` "
            f"v{data.get('model_version', '?')}  ·  "
            f"uptime {data.get('uptime_seconds', 0):.0f}s"
        )
        return True
    st.error(
        "API is unreachable or unhealthy. Ensure the Docker stack is running "
        "(`make local` or `docker compose up -d`) and a model has been bootstrapped."
    )
    return False


def _generate_test_signals(n: int, seed: int, include_unhealthy: bool) -> list[dict]:
    """Generate test signals and return list of {time, amplitude, expected_label, shape_type}."""
    import numpy as np

    from src.signal_processing.signal_generator import generate_signal

    rng = np.random.RandomState(seed)
    signals = []

    for i in range(n):
        if include_unhealthy and rng.random() < 0.3:
            shape = "lorentzian"
            sigma_l = rng.uniform(3.8, 5.1)
            gamma = sigma_l * GAMMA_SIGMA_FACTOR
            sig = generate_signal(
                shape_type="lorentzian",
                mu=rng.uniform(42, 58),
                width_param=float(gamma),
                height=rng.uniform(1.0, 1.5),
                noise_level=rng.uniform(0.06, 0.10),
                seed=seed + i,
            )
            expected = 0  # unhealthy
        else:
            shape = "gaussian"
            sig = generate_signal(
                shape_type="gaussian",
                mu=rng.uniform(48, 52),
                width_param=rng.uniform(2.0, 3.0),
                height=rng.uniform(2.5, 3.0),
                noise_level=rng.uniform(0.01, 0.02),
                seed=seed + i,
            )
            expected = 1  # healthy

        signals.append(
            {
                "time": list(sig.signal.time),
                "amplitude": list(sig.signal.amplitude),
                "expected_label": expected,
                "shape_type": shape,
            }
        )

    return signals


def _render_single_prediction() -> None:
    """Send a single signal to the API and display prediction details."""
    st.markdown(
        '<div class="signal-section-general"><strong>🎯 Single Signal Prediction</strong></div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        shape = st.selectbox("Signal type", ["gaussian", "lorentzian"], key="pm_shape")
    with col2:
        seed = st.number_input("Seed", 0, 99999, 42, key="pm_seed")

    from src.signal_processing.signal_generator import generate_signal

    if shape == "lorentzian":
        sig = generate_signal(
            shape_type="lorentzian",
            mu=50.0,
            width_param=5.88,
            height=1.2,
            noise_level=0.08,
            seed=int(seed),
        )
        expected = "unhealthy (0)"
    else:
        sig = generate_signal(
            shape_type="gaussian",
            mu=50.0,
            width_param=2.5,
            height=2.8,
            noise_level=0.015,
            seed=int(seed),
        )
        expected = "healthy (1)"

    # Plot
    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(sig.signal.time),
                y=list(sig.signal.amplitude),
                mode="lines",
                name="Signal",
                line={"color": "#4caf50" if shape == "gaussian" else "#e53935", "width": 2},
            )
        )
        fig.update_layout(
            height=250,
            margin={"t": 20, "b": 30, "l": 40, "r": 20},
            xaxis_title="Time",
            yaxis_title="Amplitude",
        )
        st.plotly_chart(fig)
    except ImportError:
        st.line_chart({"amplitude": list(sig.signal.amplitude)})

    st.caption(f"Expected: **{expected}** ({shape})")

    if st.button("🚀 Predict", type="primary", key="pm_predict_single"):
        payload = {
            "device_id": "",
            "time_values": list(sig.signal.time),
            "amplitude_values": list(sig.signal.amplitude),
        }
        with st.spinner("Calling API..."):
            import time as _time

            t0 = _time.time()
            result = _api_post("/predict", payload, api_key="dev-key-12345")
            elapsed_ms = (_time.time() - t0) * 1000

        if result:
            pred_label = result.get("prediction", "—")
            confidence = result.get("confidence", 0)
            is_correct = (pred_label == 1 and shape == "gaussian") or (
                pred_label == 0 and shape == "lorentzian"
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Prediction", f"{'healthy' if pred_label == 1 else 'unhealthy'}")
            c2.metric(
                "Confidence",
                f"{confidence:.2%}" if isinstance(confidence, float) else str(confidence),
            )
            c3.metric("Correct?", "✅ Yes" if is_correct else "❌ No")
            c4.metric("Latency", f"{elapsed_ms:.0f} ms")

            with st.expander("Full API response"):
                st.json(result)


def _render_batch_test() -> None:
    """Batch-send signals and show aggregated metrics."""
    st.markdown(
        '<div class="signal-section-general"><strong>📦 Batch Prediction Test</strong></div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        n_batch = st.number_input("Batch size", 10, 200, 50, 10, key="pm_batch_n")
    with col2:
        batch_seed = st.number_input("Seed", 0, 99999, 123, key="pm_batch_seed")
    with col3:
        st.markdown("<div style='margin-bottom: 2.1rem'></div>", unsafe_allow_html=True)
        include_unhealthy = st.checkbox("Include unhealthy signals", value=True, key="pm_incl_unh")

    if not st.button("📦 Run Batch Test", type="primary", key="pm_batch_run"):
        return

    signals = _generate_test_signals(int(n_batch), int(batch_seed), include_unhealthy)

    progress = st.progress(0.0)
    status = st.status(f"Sending {len(signals)} predictions...", expanded=True)

    results = []
    for i, sig_data in enumerate(signals):
        payload = {
            "device_id": "",
            "time_values": sig_data["time"],
            "amplitude_values": sig_data["amplitude"],
        }
        resp = _api_post("/predict", payload, api_key="dev-key-12345")
        pred = resp.get("prediction", -1) if resp else -1
        confidence = resp.get("confidence", 0) if resp else 0
        results.append(
            {
                "shape_type": sig_data["shape_type"],
                "expected": sig_data["expected_label"],
                "predicted": pred,
                "confidence": confidence,
                "correct": pred == sig_data["expected_label"],
            }
        )
        progress.progress((i + 1) / len(signals))
        if (i + 1) % 10 == 0:
            status.write(f"Processed {i + 1}/{len(signals)}...")

    status.update(label="Batch Test Complete!", state="complete")
    progress.progress(1.0)

    import pandas as pd

    df = pd.DataFrame(results)

    # ── Summary metrics ────
    st.markdown("#### 📊 Batch Results")
    n_correct = int(df["correct"].sum())
    n_total = len(df)
    accuracy = n_correct / n_total if n_total > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", n_total)
    c2.metric("Correct", n_correct)
    c3.metric("Accuracy", f"{accuracy:.2%}")
    c4.metric("Avg Confidence", f"{df['confidence'].mean():.2%}")

    # ── Confusion breakdown ─
    st.markdown("#### 🎯 Confusion Breakdown")
    tp = int(((df["expected"] == 1) & (df["predicted"] == 1)).sum())
    tn = int(((df["expected"] == 0) & (df["predicted"] == 0)).sum())
    fp = int(((df["expected"] == 0) & (df["predicted"] == 1)).sum())
    fn = int(((df["expected"] == 1) & (df["predicted"] == 0)).sum())

    cm1, cm2 = st.columns(2)
    with cm1:
        st.markdown("**Healthy signals (Gaussian)**")
        st.write(f"- True Positive (predicted healthy): {tp}")
        st.write(f"- False Negative (predicted unhealthy): {fn}")
    with cm2:
        st.markdown("**Unhealthy signals (Lorentzian)**")
        st.write(f"- True Negative (predicted unhealthy): {tn}")
        st.write(f"- False Positive (predicted healthy): {fp}")

    # ── Confidence distribution ─
    try:
        import plotly.graph_objects as go

        st.markdown("#### 📈 Prediction Confidence Distribution")
        correct_conf = df[df["correct"]]["confidence"]
        wrong_conf = df[~df["correct"]]["confidence"]

        fig = go.Figure()
        if len(correct_conf) > 0:
            fig.add_trace(
                go.Histogram(
                    x=correct_conf,
                    name="Correct",
                    marker_color="#4caf50",
                    opacity=0.7,
                )
            )
        if len(wrong_conf) > 0:
            fig.add_trace(
                go.Histogram(
                    x=wrong_conf,
                    name="Incorrect",
                    marker_color="#e53935",
                    opacity=0.7,
                )
            )
        fig.update_layout(
            barmode="overlay",
            height=300,
            xaxis_title="Confidence",
            yaxis_title="Count",
            margin={"t": 20, "b": 40},
        )
        st.plotly_chart(fig)
    except ImportError:
        pass

    # ── Detail table ─
    with st.expander("Detailed results", expanded=False):
        st.dataframe(df, hide_index=True)


def _render_model_info() -> None:
    """Show model info from the API."""
    st.markdown("#### ℹ️ Current Model Info")
    data = _api_get("/health")
    if data:
        st.json(
            {
                "model_type": data.get("model_type"),
                "model_version": data.get("model_version"),
                "total_predictions": data.get("total_predictions"),
                "uptime_seconds": data.get("uptime_seconds"),
            }
        )


def render_prediction_monitor_tab(mode: str) -> None:
    """Tab 6: Prediction & Monitoring — test the live API, batch-test, track metrics."""
    st.markdown(SECTION_CSS, unsafe_allow_html=True)
    st.markdown(
        "Send test signals to the live FastAPI prediction endpoint. "
        "Single-signal testing shows detailed results; batch testing "
        "provides confusion breakdown and confidence distributions."
    )

    if not _render_api_health():
        st.info(
            "Start the stack and run a **Greenfield Bootstrap** to create "
            "a model before testing predictions."
        )
        return

    _render_model_info()
    st.markdown("---")
    _render_single_prediction()
    st.markdown("---")
    _render_batch_test()
