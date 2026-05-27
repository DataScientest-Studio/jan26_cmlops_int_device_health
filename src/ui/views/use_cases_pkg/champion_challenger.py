"""Tab 3 — Champion / Challenger: train a challenger, compare, promote."""

from __future__ import annotations

import os

import streamlit as st

from src.ui.logging_ui import get_ui_logger

_logger = get_ui_logger(__name__)

from ._common import (
    SECTION_CSS,
    fetch_champion_info,
    get_mlflow_client,
    get_model_name,
    restart_api,
)


def _render_challenger_signal_params() -> dict:
    """Signal-param controls matching Greenfield layout."""
    st.markdown("##### 📡 Signal Generation Parameters")

    # ── Default values ──
    _cc_defaults = {
        "cc_n_samples": 100,
        "cc_gf": 0.7,
        "cc_lf": 0.2,
        "cc_seed": 42,
        "cc_g_mu": (48.0, 52.0),
        "cc_g_sigma": (2.0, 3.0),
        "cc_g_height": (2.5, 3.0),
        "cc_g_noise": (0.01, 0.02),
        "cc_l_mu": (42.0, 58.0),
        "cc_l_sigma": (3.8, 5.1),
        "cc_l_height": (1.0, 1.5),
        "cc_l_noise": (0.06, 0.10),
    }
    # Initialise session-state keys once (avoids Streamlit dual-source warning).
    for _k, _v in _cc_defaults.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    # ── Restore defaults button ──
    if st.button("🔄 Restore Defaults", key="cc_restore_defaults"):
        for k, v in _cc_defaults.items():
            st.session_state[k] = v
        st.rerun()

    # ── General parameters card ──
    with st.container(border=True):
        st.markdown(
            '<div class="signal-section-general"><strong>📊 General Parameters</strong></div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            n_samples = st.number_input(
                "Number of samples",
                min_value=20,
                max_value=5000,
                step=50,
                help="Total signals to generate per dataset",
                key="cc_n_samples",
            )
        with c2:
            gaussian_fraction = st.slider(
                "Healthy / Unhealthy ratio",
                min_value=0.1,
                max_value=0.9,
                step=0.05,
                help="Fraction of healthy (Gaussian) signals",
                key="cc_gf",
            )
        with c3:
            labeled_fraction = st.slider(
                "Labeled ratio",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                help="0 = no labels, 1.0 = all labeled",
                key="cc_lf",
            )
        with c4:
            seed = st.number_input(
                "Random seed",
                min_value=0,
                max_value=99999,
                help="Seed for reproducible generation",
                key="cc_seed",
            )

    # ── Gaussian parameters card ──
    with st.container(border=True):
        st.markdown(
            '<div class="signal-section-healthy">'
            "<strong>🟢 Healthy Signals — Gaussian Parameters</strong></div>",
            unsafe_allow_html=True,
        )
        g1, g2, g3, g4 = st.columns(4)
        with g1:
            g_mu = st.slider(
                "μ range (center)",
                min_value=35.0,
                max_value=60.0,
                step=0.5,
                key="cc_g_mu",
            )
        with g2:
            g_sigma = st.slider(
                "σ range (width)",
                min_value=0.5,
                max_value=6.0,
                step=0.1,
                help="Standard deviation (narrower = healthier)",
                key="cc_g_sigma",
            )
        with g3:
            g_height = st.slider(
                "Height range",
                min_value=1.0,
                max_value=5.0,
                step=0.1,
                key="cc_g_height",
            )
        with g4:
            g_noise = st.slider(
                "Noise range",
                min_value=0.0,
                max_value=0.15,
                step=0.005,
                format="%.3f",
                key="cc_g_noise",
            )

    # ── Lorentzian parameters card ──
    with st.container(border=True):
        st.markdown(
            '<div class="signal-section-unhealthy">'
            "<strong>🔴 Unhealthy Signals — Lorentzian Parameters</strong></div>",
            unsafe_allow_html=True,
        )
        l1, l2, l3, l4 = st.columns(4)
        with l1:
            l_mu = st.slider(
                "μ range (center)",
                min_value=35.0,
                max_value=60.0,
                step=0.5,
                key="cc_l_mu",
            )
        with l2:
            l_sigma = st.slider(
                "σ range (width)",
                min_value=2.0,
                max_value=6.0,
                step=0.1,
                help="Equivalent Gaussian σ — converted to Lorentzian γ via γ\u2009=\u20091.18\u2009σ",
                key="cc_l_sigma",
            )
        with l3:
            l_height = st.slider(
                "Height range",
                min_value=0.8,
                max_value=3.5,
                step=0.1,
                key="cc_l_height",
            )
        with l4:
            l_noise = st.slider(
                "Noise range",
                min_value=0.0,
                max_value=0.15,
                step=0.005,
                format="%.3f",
                key="cc_l_noise",
            )

    return {
        "n_samples": int(n_samples),
        "gaussian_fraction": gaussian_fraction,
        "labeled_fraction": labeled_fraction,
        "seed": int(seed),
        # Sliders are range-sliders returning a 2-tuple; mypy infers float because
        # value= is omitted (session_state pre-seeds the default).  # type: ignore comments
        # are safe here — session_state is initialised with (float, float) above.
        "gauss_mu_range": tuple(g_mu),  # type: ignore[arg-type]
        "gauss_sigma_range": tuple(g_sigma),  # type: ignore[arg-type]
        "gauss_height_range": tuple(g_height),  # type: ignore[arg-type]
        "gauss_noise_range": tuple(g_noise),  # type: ignore[arg-type]
        "lor_mu_range": tuple(l_mu),  # type: ignore[arg-type]
        "lor_sigma_range": tuple(l_sigma),  # type: ignore[arg-type]
        "lor_height_range": tuple(l_height),  # type: ignore[arg-type]
        "lor_noise_range": tuple(l_noise),  # type: ignore[arg-type]
    }


def _run_challenger_training(signal_params: dict, classifier: str, mode: str) -> None:
    """Train a challenger model and store results in session_state."""
    # Project root needed for the train_data_path_override below.
    from pathlib import Path as _Path

    from scripts.greenfield_init import BootstrapConfig, run_bootstrap

    from ._common import get_experiment_name, get_model_name

    _project_root = _Path(__file__).resolve().parents[4]

    config = BootstrapConfig(
        n_samples=signal_params["n_samples"],
        gaussian_fraction=signal_params["gaussian_fraction"],
        labeled_fraction=signal_params["labeled_fraction"],
        seed=signal_params["seed"],
        classifier=classifier,
        model_name=get_model_name(),
        experiment_name=get_experiment_name(),
        promote=False,
        wipe=False,
        gauss_mu_range=signal_params["gauss_mu_range"],
        gauss_sigma_range=signal_params["gauss_sigma_range"],
        gauss_height_range=signal_params["gauss_height_range"],
        gauss_noise_range=signal_params["gauss_noise_range"],
        lor_mu_range=signal_params["lor_mu_range"],
        lor_sigma_range=signal_params["lor_sigma_range"],
        lor_height_range=signal_params["lor_height_range"],
        lor_noise_range=signal_params["lor_noise_range"],
    )
    # Train on the freshly generated bootstrap signals (bootstrap_labeled.json)
    # rather than the baseline split, so model lineage correctly reflects that
    # champion/challenger models are trained on newly generated representative data.
    config.train_data_path_override = _project_root / "data/raw/bootstrap_labeled.json"

    status = st.status("Training Challenger model...", expanded=True)
    progress_bar = st.progress(0.0)
    log_lines: list[str] = []
    log_area = st.empty()

    def _cb(step: str, msg: str, frac: float) -> None:
        log_lines.append(f"[{step}] {msg}")
        log_area.code("\n".join(log_lines[-15:]), language="text")
        if 0.0 <= frac <= 1.0:
            progress_bar.progress(frac)

    _logger.info(
        "Starting challenger training: classifier={} n_samples={}", classifier, config.n_samples
    )
    # Override TRAINED_BY so MLflow lineage shows the correct source rather
    # than "greenfield_bootstrap" (which is the default inside run_bootstrap).
    _prev_trained_by = os.environ.get("TRAINED_BY", "")
    os.environ["TRAINED_BY"] = "champion_challenger_training"
    try:
        result = run_bootstrap(config, progress_callback=_cb)
    finally:
        if _prev_trained_by:
            os.environ["TRAINED_BY"] = _prev_trained_by
        else:
            os.environ.pop("TRAINED_BY", None)
    st.session_state["_cc_running"] = False

    if result.success:
        _logger.info(
            "Challenger training OK — version={} f1={:.4f} accuracy={:.4f} run_id={}",
            result.model_version,
            result.test_f1,
            result.test_accuracy,
            result.mlflow_run_id,
        )
        status.update(label="Challenger Trained!", state="complete")
        progress_bar.progress(1.0)
        st.session_state["_cc_result"] = {
            "version": result.model_version,
            "f1": result.test_f1,
            "accuracy": result.test_accuracy,
            "run_id": result.mlflow_run_id,
            "classifier": classifier,
        }
        # Set the "challenger" alias so A/B Testing can find this model
        try:
            client, _uri = get_mlflow_client()
            client.set_registered_model_alias(
                get_model_name(),
                "challenger",
                str(result.model_version),
            )
        except Exception as _alias_err:
            st.warning(f"Could not set challenger alias: {_alias_err}")
    else:
        _logger.warning("Challenger training FAILED: {}", result.error)
        status.update(label="Training Failed", state="error")
        st.error(f"Challenger training failed: {result.error}")


def render_champion_challenger_tab(mode: str) -> None:
    """Tab 3: Champion / Challenger comparison and promotion."""
    st.markdown(SECTION_CSS, unsafe_allow_html=True)
    st.markdown(
        "Train a new model (challenger) and compare it against the current "
        "production champion. If the challenger outperforms, promote it."
    )

    try:
        client, uri = get_mlflow_client()
    except Exception as exc:
        st.error(f"Cannot connect to MLflow: {exc}")
        return

    # ── Champion card ──────────────────────
    champ_mv, champ_run = fetch_champion_info(client)

    st.markdown("#### 🏆 Current Champion")
    if champ_mv is None:
        st.warning("No production champion found. Run a **Greenfield Bootstrap** first.")
        champ_f1, champ_acc = 0.0, 0.0
    else:
        champ_metrics = champ_run.data.metrics if champ_run else {}
        champ_params = champ_run.data.params if champ_run else {}
        champ_f1 = champ_metrics.get("test_f1_score", 0.0)
        champ_acc = champ_metrics.get("test_accuracy", 0.0)
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("Version", f"v{champ_mv.version}")
        cc2.metric("Classifier", champ_params.get("classifier_type", "—"))
        cc3.metric("Test F1", f"{champ_f1:.4f}")
        cc4.metric("Test Accuracy", f"{champ_acc:.4f}")

    st.markdown("---")

    # ── Challenger configuration ───────────
    st.markdown("#### 🥊 Train a Challenger")
    signal_params = _render_challenger_signal_params()

    classifier = st.selectbox(
        "Classifier",
        options=["logistic_regression", "decision_tree", "random_forest", "svc"],
        format_func=lambda x: {
            "logistic_regression": "Logistic Regression",
            "decision_tree": "Decision Tree",
            "random_forest": "Random Forest",
            "svc": "Support Vector Classifier (SVC)",
        }[x],
        index=2,
        key="cc_classifier",
    )

    if mode == "local":
        st.info(
            "**Local sandbox mode** — DVC and DagsHub are skipped. "
            "The challenger will be registered in the local MLflow container."
        )

    running_key = "_cc_running"
    if st.session_state.get(running_key, False):
        _run_challenger_training(signal_params, classifier, mode)

    if st.button("🥊 Train Challenger", key="cc_train_btn", type="primary"):
        st.session_state.pop("_cc_result", None)
        st.session_state[running_key] = True
        st.rerun()

    # ── Comparison & Promotion ─────────────
    cc_result = st.session_state.get("_cc_result")
    if cc_result is None:
        st.caption("Train a challenger above to see the comparison.")
        return

    st.markdown("---")
    # Allow user to clear the stale result from a previous session
    _ccr1, _ccr2 = st.columns([8, 2])
    with _ccr1:
        st.markdown("#### 📊 Comparison")
    with _ccr2:
        if st.button("🔄 Clear Results", key="cc_clear_btn"):
            st.session_state.pop("_cc_result", None)
            st.session_state.pop("_cc_confirm_promote", None)
            st.rerun()

    col_champ, col_chall = st.columns(2)
    with col_champ:
        st.markdown("**🏆 Champion**")
        if champ_mv:
            st.metric("Version", f"v{champ_mv.version}")
            st.metric("Test F1", f"{champ_f1:.4f}")
            st.metric("Test Accuracy", f"{champ_acc:.4f}")
        else:
            st.write("No champion")
    with col_chall:
        st.markdown("**🥊 Challenger**")
        st.metric("Version", f"v{cc_result['version']}")
        st.metric(
            "Test F1",
            f"{cc_result['f1']:.4f}",
            delta=f"{cc_result['f1'] - champ_f1:+.4f}" if champ_mv else None,
        )
        st.metric(
            "Test Accuracy",
            f"{cc_result['accuracy']:.4f}",
            delta=f"{cc_result['accuracy'] - champ_acc:+.4f}" if champ_mv else None,
        )

    improvement = cc_result["f1"] - champ_f1
    if improvement > 0.005:
        st.success(
            f"Challenger outperforms champion by **{improvement:+.4f}** F1. Promotion recommended."
        )
    elif improvement > -0.005:
        st.warning("Performance is roughly equivalent. Promotion is optional.")
    else:
        st.error(
            f"Challenger underperforms champion by **{improvement:+.4f}** F1. "
            "Promotion not recommended."
        )

    promote_key = "_cc_confirm_promote"
    if st.button("⬆️ Promote Challenger to Production", key="cc_promote_btn"):
        st.session_state[promote_key] = True

    if st.session_state.get(promote_key, False):
        st.warning(
            f"This will promote **v{cc_result['version']}** to Production "
            "and archive the current champion. Continue?"
        )
        p1, p2, _ = st.columns([1, 1, 3])
        with p1:
            if st.button("✅ Yes, promote", key="cc_yes"):
                st.session_state[promote_key] = False
                with st.spinner(f"Promoting v{cc_result['version']} to Production..."):
                    try:
                        import mlflow

                        from src.training.registry import promote_model

                        mlflow.set_tracking_uri(uri)
                        promote_model(
                            get_model_name(),
                            cc_result["version"],
                            stage="Production",
                            archive_existing_production=True,
                        )
                        st.success(
                            f"v{cc_result['version']} promoted to Production! "
                            "Restart the API to serve the new model."
                        )
                    except Exception as exc:
                        st.error(f"Promotion failed: {exc}")
                        st.stop()
                with st.spinner("Restarting FastAPI..."):
                    restart_api()
                st.success("FastAPI restarted with new Production model.")
                # Clear challenger result and rerun to refresh champion card
                st.session_state.pop("_cc_result", None)
                st.rerun()
        with p2:
            if st.button("❌ Cancel", key="cc_cancel"):
                st.session_state[promote_key] = False
                st.rerun()
