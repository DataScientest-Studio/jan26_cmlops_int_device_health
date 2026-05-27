"""Tab 6 — A/B Testing: direct evaluation of champion vs. challenger.

Architecture
~~~~~~~~~~~~
Nginx is **never modified**.  The champion is reached through Nginx
(port 80) as usual, and the challenger container publishes its own host
port (default 8001).  Streamlit sends test signals to **both** APIs
and compares the results side-by-side.
"""

from __future__ import annotations

import subprocess
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st

from ._common import (
    GAMMA_SIGMA_FACTOR,
    SECTION_CSS,
    fetch_champion_info,
    get_mlflow_client,
    get_model_name,
)

# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------


def _start_challenger_container(mode: str) -> tuple[str, int]:
    """Start the api_challenger container via docker compose."""
    overlay = "docker-compose.local.yml" if mode == "local" else "docker-compose.cloud.yml"
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            "docker-compose.yml",
            "-f",
            overlay,
            "--profile",
            "ab_testing",
            "up",
            "-d",
            "--build",
            "api_challenger",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    return result.stdout + result.stderr, result.returncode


def _stop_challenger_container(mode: str) -> None:
    """Stop and remove the api_challenger container."""
    overlay = "docker-compose.local.yml" if mode == "local" else "docker-compose.cloud.yml"
    subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            "docker-compose.yml",
            "-f",
            overlay,
            "--profile",
            "ab_testing",
            "stop",
            "api_challenger",
        ],
        capture_output=True,
        timeout=30,
    )


def _container_health(container: str) -> str:
    """Return 'healthy', 'starting', 'running' (no healthcheck), or 'stopped'."""
    result = subprocess.run(
        [
            "docker",
            "inspect",
            "--format",
            "{{.State.Running}}|{{if .State.Health}}{{.State.Health.Status}}{{else}}no-hc{{end}}",
            container,
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        return "stopped"
    parts = result.stdout.strip().split("|")
    running = parts[0] == "true"
    if not running:
        return "stopped"
    hc = parts[1] if len(parts) > 1 else "no-hc"
    if hc == "healthy":
        return "healthy"
    if hc in ("starting", "unhealthy"):
        return "starting"
    return "running"


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _get_model_info(base_url: str) -> dict | None:
    """Fetch /model/info from an API endpoint."""
    try:
        resp = requests.get(f"{base_url}/model/info", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _send_prediction(
    api_url: str, time_values: list[float], amplitude_values: list[float]
) -> tuple[dict | None, float]:
    """Send a single prediction; return (response_json, latency_seconds)."""
    t0 = time.monotonic()
    try:
        resp = requests.post(
            f"{api_url}/predict",
            json={
                "device_id": "",
                "time_values": time_values,
                "amplitude_values": amplitude_values,
            },
            headers={"X-API-Key": "dev-key-12345"},
            timeout=15,
        )
        elapsed = time.monotonic() - t0
        resp.raise_for_status()
        return resp.json(), elapsed
    except Exception:
        return None, time.monotonic() - t0


def _evaluate_batch(
    api_url: str, signals: list[dict], store_predictions: bool = False
) -> tuple[dict | None, float]:
    """Call /evaluate with a full batch; return (response_json, total_elapsed_seconds)."""
    payload = {
        "signals": [
            {
                "device_id": "",
                "time_values": sig["time_values"],
                "amplitude_values": sig["amplitude"],
                "expected_label": sig["expected_label"],
            }
            for sig in signals
        ],
        "store_predictions": store_predictions,
    }
    t0 = time.monotonic()
    try:
        resp = requests.post(
            f"{api_url}/evaluate",
            json=payload,
            headers={"X-API-Key": "dev-key-12345"},
            timeout=120,
        )
        elapsed = time.monotonic() - t0
        resp.raise_for_status()
        return resp.json(), elapsed
    except Exception:
        return None, time.monotonic() - t0


# ---------------------------------------------------------------------------
# Signal generation (reuses project signal library)
# ---------------------------------------------------------------------------


def _generate_test_signals(n: int, seed: int) -> list[dict]:
    """Generate *n* mixed healthy / unhealthy signals for evaluation."""
    from src.signal_processing.signal_generator import generate_signal

    rng = np.random.RandomState(seed)
    signals: list[dict] = []
    for i in range(n):
        if rng.random() < 0.3:
            shape = "lorentzian"
            gamma = float(rng.uniform(3.8, 5.1) * GAMMA_SIGMA_FACTOR)
            sig = generate_signal(
                shape_type="lorentzian",
                mu=rng.uniform(42, 58),
                width_param=gamma,
                height=rng.uniform(1.0, 1.5),
                noise_level=rng.uniform(0.06, 0.10),
                seed=seed + i,
            )
            expected = 1  # lorentzian = unhealthy = label 1
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
            expected = 0  # gaussian = healthy = label 0
        signals.append(
            {
                "time_values": list(sig.signal.time),
                "amplitude": list(sig.signal.amplitude),
                "expected_label": expected,
                "shape_type": shape,
            }
        )
    return signals


# ---------------------------------------------------------------------------
# Evaluation engine
# ---------------------------------------------------------------------------


def _run_evaluation(
    champion_url: str,
    challenger_url: str,
    n_signals: int,
    seed: int,
    progress_container,
) -> dict:
    """Evaluate champion and challenger via a single /evaluate call each."""
    signals = _generate_test_signals(n_signals, seed)

    bar = progress_container.progress(0, text="Calling champion API…")
    # Store champion predictions to DB — champion signals become the canonical DB record
    champ_resp, _ = _evaluate_batch(champion_url, signals, store_predictions=True)
    bar.progress(0.5, text="Calling challenger API\u2026")
    # Challenger runs in evaluation mode only — not stored to avoid conflicting raw_signal entries
    chall_resp, _ = _evaluate_batch(challenger_url, signals, store_predictions=False)
    bar.empty()

    # ── Parse champion results ──
    champ_preds: list[int] = []
    champ_lats: list[float] = []
    champ_errors = 0
    champ_ok: list[int] = []

    if champ_resp is not None:
        for p in champ_resp.get("predictions", []):
            label = int(p.get("predicted_label", -1))
            champ_preds.append(label)
            champ_lats.append(float(p.get("latency_ms", 0.0)))
            if label != -1:
                champ_ok.append(label)
        champ_errors = int(champ_resp.get("n_errors", 0))
    else:
        champ_preds = [-1] * n_signals
        champ_lats = [0.0] * n_signals
        champ_errors = n_signals

    # ── Parse challenger results ──
    chall_preds: list[int] = []
    chall_lats: list[float] = []
    chall_errors = 0
    chall_ok: list[int] = []

    if chall_resp is not None:
        for p in chall_resp.get("predictions", []):
            label = int(p.get("predicted_label", -1))
            chall_preds.append(label)
            chall_lats.append(float(p.get("latency_ms", 0.0)))
            if label != -1:
                chall_ok.append(label)
        chall_errors = int(chall_resp.get("n_errors", 0))
    else:
        chall_preds = [-1] * n_signals
        chall_lats = [0.0] * n_signals
        chall_errors = n_signals

    expected = [sig["expected_label"] for sig in signals]
    shapes = [sig["shape_type"] for sig in signals]

    agreement = sum(1 for a, b in zip(champ_preds, chall_preds, strict=True) if a == b and a != -1)
    both_valid = sum(
        1 for a, b in zip(champ_preds, chall_preds, strict=True) if a != -1 and b != -1
    )
    champ_correct = sum(1 for p, e in zip(champ_preds, expected, strict=True) if p == e)
    chall_correct = sum(1 for p, e in zip(chall_preds, expected, strict=True) if p == e)

    # F1 scores (binary, label=1 is positive)
    def _f1(preds: list[int], truth: list[int]) -> float:
        tp = sum(1 for p, t in zip(preds, truth, strict=True) if p == 1 and t == 1)
        fp = sum(1 for p, t in zip(preds, truth, strict=True) if p == 1 and t == 0)
        fn = sum(1 for p, t in zip(preds, truth, strict=True) if p == 0 and t == 1)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    champ_f1 = _f1(champ_preds, expected)
    chall_f1 = _f1(chall_preds, expected)

    champ_ms = np.array(champ_lats) if champ_lats else np.array([0.0])
    chall_ms = np.array(chall_lats) if chall_lats else np.array([0.0])

    return {
        "n_signals": n_signals,
        "champion_accuracy": champ_correct / n_signals if n_signals else 0,
        "challenger_accuracy": chall_correct / n_signals if n_signals else 0,
        "champion_f1": champ_f1,
        "challenger_f1": chall_f1,
        "agreement_rate": agreement / both_valid if both_valid else 0,
        "champion_mean_latency_ms": float(np.mean(champ_ms)),
        "challenger_mean_latency_ms": float(np.mean(chall_ms)),
        "champion_p95_latency_ms": float(np.percentile(champ_ms, 95)),
        "challenger_p95_latency_ms": float(np.percentile(chall_ms, 95)),
        "champion_errors": champ_errors,
        "challenger_errors": chall_errors,
        "champion_healthy_pct": sum(1 for p in champ_ok if p == 0) / len(champ_ok) * 100
        if champ_ok
        else 0,
        "challenger_healthy_pct": sum(1 for p in chall_ok if p == 0) / len(chall_ok) * 100
        if chall_ok
        else 0,
        "details": pd.DataFrame(
            {
                "shape": shapes,
                "expected": expected,
                "champion_pred": champ_preds,
                "challenger_pred": chall_preds,
                "champion_ms": [round(v, 1) for v in champ_lats],
                "challenger_ms": [round(v, 1) for v in chall_lats],
            }
        ),
    }


# ---------------------------------------------------------------------------
# Results renderer
# ---------------------------------------------------------------------------


def _render_results(results: dict) -> None:
    """Display side-by-side metrics and detail table."""
    st.markdown("#### 📊 Evaluation Results")

    col_c, col_x = st.columns(2)
    with col_c:
        st.markdown("**Champion**")
        st.metric("Accuracy", f"{results['champion_accuracy']:.1%}")
        st.metric("F1 Score", f"{results['champion_f1']:.4f}")
        st.metric("Mean latency", f"{results['champion_mean_latency_ms']:.0f} ms")
        st.metric("p95 latency", f"{results['champion_p95_latency_ms']:.0f} ms")
        st.metric("Errors", results["champion_errors"])
        st.metric("Healthy %", f"{results['champion_healthy_pct']:.1f}%")
    with col_x:
        st.markdown("**Challenger**")
        st.metric("Accuracy", f"{results['challenger_accuracy']:.1%}")
        st.metric("F1 Score", f"{results['challenger_f1']:.4f}")
        st.metric("Mean latency", f"{results['challenger_mean_latency_ms']:.0f} ms")
        st.metric("p95 latency", f"{results['challenger_p95_latency_ms']:.0f} ms")
        st.metric("Errors", results["challenger_errors"])
        st.metric("Healthy %", f"{results['challenger_healthy_pct']:.1f}%")

    st.metric("Model agreement", f"{results['agreement_rate']:.1%}")

    with st.expander("Per-signal details", expanded=False):
        st.dataframe(results["details"])


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def render_ab_testing_tab(mode: str) -> None:
    """Render the A/B Testing use case tab."""
    import os as _os

    st.markdown(SECTION_CSS, unsafe_allow_html=True)

    # In K8s mode, Docker Compose containers cannot be started — port-forwards
    # already occupy port 5434 (postgres) and 8001 (challenger) would require
    # a K8s deployment instead of a Docker container.
    if _os.environ.get("DEPLOYMENT_MODE", "") == "k8s":
        st.info(
            "⚙️ **A/B Testing (Docker mode) is not available in K8s mode.**\n\n"
            "This tab launches a Docker Compose challenger container which conflicts "
            "with active K8s port-forwards. "
            "In K8s mode, use the **Nginx Traffic Split** tab for traffic-based A/B testing, "
            "or run `make local` / `make cloud` to switch to a Docker stack.",
            icon="ℹ️",
        )
        return

    st.markdown(
        "Deploy **champion** and **challenger** models side-by-side.  "
        "Streamlit sends the same batch of synthetic signals to both API "
        "containers and compares accuracy, latency, and prediction agreement.  "
        "**Nginx is never modified** — the challenger container publishes "
        "its own port (default 8001)."
    )

    try:
        client, uri = get_mlflow_client()
    except Exception as exc:
        st.error(f"Cannot connect to MLflow: {exc}")
        return

    # ── Model info ──────────────────────────────
    champ_mv, champ_run = fetch_champion_info(client)

    chall_mv, chall_run = None, None
    try:
        chall_mv = client.get_model_version_by_alias(get_model_name(), "challenger")
        chall_run = client.get_run(chall_mv.run_id) if chall_mv else None
    except Exception:
        pass

    col_c, col_ch = st.columns(2)
    with col_c:
        st.markdown("#### 🏆 Champion")
        if champ_mv:
            champ_metrics = champ_run.data.metrics if champ_run else {}
            champ_params = champ_run.data.params if champ_run else {}
            m1, m2, m3 = st.columns(3)
            m1.metric("Version", f"v{champ_mv.version}")
            m2.metric("Classifier", champ_params.get("classifier_type", "—"))
            m3.metric("Test F1", f"{champ_metrics.get('test_f1_score', 0):.4f}")
        else:
            st.warning("No champion found. Run **Greenfield Bootstrap** first.")

    with col_ch:
        st.markdown("#### 🥊 Challenger")
        if chall_mv:
            chall_metrics = chall_run.data.metrics if chall_run else {}
            chall_params = chall_run.data.params if chall_run else {}
            m1, m2, m3 = st.columns(3)
            m1.metric("Version", f"v{chall_mv.version}")
            m2.metric("Classifier", chall_params.get("classifier_type", "—"))
            m3.metric("Test F1", f"{chall_metrics.get('test_f1_score', 0):.4f}")
        else:
            st.warning("No challenger found. Train one in the **Champion / Challenger** tab first.")

    if not champ_mv or not chall_mv:
        st.info(
            "Both a **champion** and a **challenger** model must exist in the "
            "MLflow registry before you can start an A/B test."
        )
        return

    st.markdown("---")

    # ── Endpoint URLs ───────────────────────────
    from src.ui.components.docker_utils import get_host, get_service_url

    champion_url = get_service_url("mlops_nginx", 80)
    challenger_url = get_service_url("mlops_api_challenger", 8000)
    if not challenger_url or challenger_url == f"http://{get_host()}:8000":
        # Fallback when port cache misses: use the published default.
        challenger_url = f"http://{get_host()}:8001"

    # ── A/B Configuration ──────────────────────
    st.markdown("#### ⚖️ Evaluation Configuration")

    n_signals = st.slider(
        "Number of test signals",
        min_value=10,
        max_value=500,
        value=50,
        step=10,
        key="_ab_n_signals",
    )
    seed = st.number_input("Random seed", value=42, step=1, key="_ab_seed")

    st.markdown("---")

    # ── Deploy / Teardown ──────────────────────
    ab_active = st.session_state.get("_ab_active", False)

    # Auto-clear stale state: if _ab_active is True but the challenger container
    # is no longer running (e.g., after a page reload or Docker restart), reset
    # to the initial "Deploy" view so the user isn't stuck in an inconsistent state.
    if ab_active and _container_health("mlops_api_challenger") == "stopped":
        st.session_state.pop("_ab_active", None)
        st.session_state.pop("_ab_results", None)
        ab_active = False

    if not ab_active:
        st.markdown("#### 🚀 Deploy A/B Test")
        st.markdown(
            "This will:\n"
            "1. Build and start a **second API container** (`api_challenger`) "
            "serving the Staging model on port 8001\n"
            "2. Nginx stays untouched — champion traffic is unaffected"
        )
        if st.button("🚀 Start A/B Test", key="ab_start_btn", type="primary"):
            _deploy_ab_test(mode)
    else:
        # ── Live status ──
        champ_health = _container_health("mlops_api")
        chall_health = _container_health("mlops_api_challenger")

        sc1, sc2 = st.columns(2)
        with sc1:
            _health_badge("Champion API", champ_health, champion_url)
        with sc2:
            _health_badge("Challenger API", chall_health, challenger_url)

        st.markdown("---")

        col_run, col_stop, col_clear, _ = st.columns([1, 1, 1, 1])
        with col_run:
            if st.button("▶️ Run Evaluation", key="ab_run_btn", type="primary"):
                progress = st.container()
                results = _run_evaluation(
                    champion_url,
                    challenger_url,
                    n_signals,
                    int(seed),
                    progress,
                )
                st.session_state["_ab_results"] = results
        with col_stop:
            if st.button("⛔ Stop A/B Test", key="ab_stop_btn", type="secondary"):
                _teardown_ab_test(mode)
        with col_clear:
            if st.button("🔄 Clear Results", key="ab_clear_btn"):
                st.session_state.pop("_ab_results", None)
                st.rerun()

        if "_ab_results" in st.session_state:
            _render_results(st.session_state["_ab_results"])


# ---------------------------------------------------------------------------
# Health badge
# ---------------------------------------------------------------------------


def _health_badge(label: str, status: str, url: str) -> None:
    info = _get_model_info(url)
    if status == "healthy" and info:
        # Strip the "_semi_supervised" suffix for a cleaner display name.
        algo_raw = info.get("algorithm", "?")
        algo_display = algo_raw.replace("_semi_supervised", "").replace("_", " ").title()
        st.success(f"{label} — healthy  \nv{info.get('registry_version', '?')} ({algo_display})")
    elif status == "healthy":
        st.success(f"{label} — healthy")
    elif status in ("starting", "running"):
        st.warning(f"{label} — {status}…")
    else:
        st.error(f"{label} — not running")


# ---------------------------------------------------------------------------
# Deploy / teardown
# ---------------------------------------------------------------------------


def _deploy_ab_test(mode: str) -> None:
    """Start challenger container, mark A/B active."""
    status = st.status("Deploying A/B test…", expanded=True)

    status.write("Building and starting challenger API container…")
    out, rc = _start_challenger_container(mode)
    if rc != 0:
        status.update(label="Deployment Failed", state="error")
        st.error(f"Failed to start challenger container:\n```\n{out}\n```")
        return
    status.write("Challenger container started.")

    status.write("Waiting for challenger health-check…")
    for attempt in range(40):
        h = _container_health("mlops_api_challenger")
        if h == "healthy":
            break
        time.sleep(3)
        if attempt % 5 == 4:
            status.write(f"  Still waiting… ({h})")
    else:
        h = _container_health("mlops_api_challenger")
        if h == "stopped":
            status.update(label="Deployment Failed", state="error")
            st.error("Challenger container did not start. Check Docker logs.")
            return
        status.write(f"Challenger not yet healthy ({h}) — proceeding anyway.")

    status.update(label="A/B Test Deployed!", state="complete")
    st.session_state["_ab_active"] = True
    st.balloons()
    st.rerun()


def _teardown_ab_test(mode: str) -> None:
    """Stop challenger container, mark A/B inactive."""
    status = st.status("Stopping A/B test…", expanded=True)

    status.write("Stopping challenger API container…")
    _stop_challenger_container(mode)
    status.write("Challenger container stopped.")

    status.update(label="A/B Test Stopped", state="complete")
    st.session_state.pop("_ab_active", None)
    st.session_state.pop("_ab_results", None)
    st.rerun()
