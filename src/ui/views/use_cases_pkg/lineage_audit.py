"""Tab 2 — Model Lineage Audit: prediction → model → run → data → git."""

from __future__ import annotations

from datetime import UTC

import streamlit as st

from src.ui.views.mlflow_explorer import _rest_get_model_aliases

from ._common import get_host_db_url, get_mlflow_client


def _resolve_stage(version_str: str, alias_map: dict[str, str]) -> str:
    """Return display stage from alias map; any unaliased version is Archived."""
    alias = alias_map.get(version_str, "")
    if alias == "champion":
        return "Production"
    if alias == "challenger":
        return "Staging"
    return "Archived"


def render_lineage_audit_tab() -> None:
    """Trace any model version back through its full lineage."""
    st.markdown(
        "Trace any model version back through its full lineage: training run, "
        "metrics, parameters, data hashes, and git commit. Select a registered "
        "model and version to inspect."
    )

    try:
        client, uri = get_mlflow_client()
    except Exception as exc:
        st.error(f"Cannot connect to MLflow: {exc}")
        return

    # ── 1. Select registered model ─────────
    try:
        models = client.search_registered_models()
    except Exception as exc:
        st.error(f"Failed to fetch models: {exc}")
        return

    if not models:
        st.info("No registered models found. Run a Greenfield Bootstrap first.")
        return

    model_names = [m.name for m in models]
    selected_model = st.selectbox("Registered Model", options=model_names, key="lineage_model_name")

    # ── 2. Select version ──────────────────
    try:
        versions = client.search_model_versions(f"name='{selected_model}'")
    except Exception as exc:
        st.error(f"Failed to fetch versions: {exc}")
        return

    if not versions:
        st.info("No versions found for this model.")
        return

    sorted_versions = sorted(versions, key=lambda v: int(v.version), reverse=True)
    # Use REST alias map — reliable across all MLflow SDK / server versions.
    alias_map = _rest_get_model_aliases(selected_model)
    version_labels = []
    for v in sorted_versions:
        stage = _resolve_stage(str(v.version), alias_map)
        version_labels.append(f"v{v.version} — {stage}")

    selected_idx = st.selectbox(
        "Model Version",
        options=range(len(sorted_versions)),
        format_func=lambda i: version_labels[i],
        key="lineage_version",
    )
    mv = sorted_versions[selected_idx]

    # ── 3. Split into tabs ─────────────────
    tab_lineage, tab_reproduce = st.tabs(["📊 Lineage", "🔬 Reproduce Training"])

    with tab_lineage:
        _render_lineage_tab(mv, selected_model, client, uri, alias_map)

    with tab_reproduce:
        _render_reproducibility_tab(mv, client)


def _render_lineage_tab(mv, selected_model, client, uri, alias_map) -> None:  # type: ignore[no-untyped-def]
    st.markdown("---")
    st.markdown(f"### 🔍 Lineage for `{selected_model}` v{mv.version}")
    stage = _resolve_stage(str(mv.version), alias_map)
    stage_icons = {"Production": "✅", "Staging": "🧪", "Archived": "🗄️"}

    col1, col2, col3 = st.columns(3)
    col1.metric("Version", f"v{mv.version}")
    col2.metric("Stage", f"{stage_icons.get(stage, '⚪')} {stage}")
    if hasattr(mv, "creation_timestamp") and mv.creation_timestamp:
        from datetime import datetime

        ts = datetime.fromtimestamp(mv.creation_timestamp / 1000, tz=UTC)
        col3.metric("Created", ts.strftime("%Y-%m-%d %H:%M UTC"))
    else:
        col3.metric("Created", "—")

    st.markdown(f"**Source:** `{mv.source}`")

    # ── 4. Training run details ────────────
    if not mv.run_id:
        st.warning("No training run linked to this model version.")
        return

    try:
        run = client.get_run(mv.run_id)
    except Exception as exc:
        st.error(f"Failed to fetch training run: {exc}")
        return

    st.markdown("#### 🏋️ Training Run")
    run_link = f"{uri}/#/experiments/{run.info.experiment_id}/runs/{mv.run_id}"
    st.markdown(
        f"**Run ID:** [`{mv.run_id[:12]}…`]({run_link})  ·  "
        f"**Experiment:** {run.info.experiment_id}  ·  "
        f"**Status:** {run.info.status}"
    )

    metrics = run.data.metrics
    if metrics:
        # Group metrics logically
        train_metrics = {k: v for k, v in metrics.items() if k.startswith("train_")}
        test_metrics = {
            k: v
            for k, v in metrics.items()
            if k.startswith("test_") or k in ("gold_standard_test_size",)
        }
        cm_metrics = {
            k: v
            for k, v in metrics.items()
            if k in ("true_positives", "true_negatives", "false_positives", "false_negatives")
        }
        other_metrics = {
            k: v
            for k, v in metrics.items()
            if k not in train_metrics
            and k not in test_metrics
            and k not in cm_metrics
            and k != "primary_metric"
        }

        st.markdown("**Training Metrics**")
        if train_metrics:
            cols = st.columns(min(len(train_metrics), 4))
            for i, (k, val) in enumerate(sorted(train_metrics.items())):
                display = f"{val:.4f}" if isinstance(val, float) else str(val)
                cols[i % len(cols)].metric(k, display)
        else:
            st.caption("No training metrics recorded.")

        st.markdown("**Test Metrics**")
        if test_metrics:
            cols = st.columns(min(len(test_metrics), 4))
            for i, (k, val) in enumerate(sorted(test_metrics.items())):
                display = f"{val:.4f}" if isinstance(val, float) else str(val)
                cols[i % len(cols)].metric(k, display)
        else:
            st.caption("No test metrics recorded.")

        if cm_metrics:
            st.markdown("**Confusion Matrix**")
            tp = int(cm_metrics.get("true_positives", 0))
            tn = int(cm_metrics.get("true_negatives", 0))
            fp = int(cm_metrics.get("false_positives", 0))
            fn = int(cm_metrics.get("false_negatives", 0))
            cm1, cm2 = st.columns(2)
            with cm1:
                st.markdown(
                    f"| | Predicted Healthy | Predicted Unhealthy |\n"
                    f"|---|---|---|\n"
                    f"| **Actual Healthy** | {tp} (TP) | {fn} (FN) |\n"
                    f"| **Actual Unhealthy** | {fp} (FP) | {tn} (TN) |"
                )
            with cm2:
                total = tp + tn + fp + fn
                if total > 0:
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    st.metric("Precision", f"{precision:.4f}")
                    st.metric("Recall", f"{recall:.4f}")

        if other_metrics:
            with st.expander("Other Metrics", expanded=False):
                cols = st.columns(min(len(other_metrics), 4))
                for i, (k, val) in enumerate(sorted(other_metrics.items())):
                    display = f"{val:.4f}" if isinstance(val, float) else str(val)
                    cols[i % len(cols)].metric(k, display)

    params = run.data.params
    if params:
        with st.expander("📋 Training Parameters", expanded=False):
            st.json(params)

    tags = run.data.tags
    st.markdown("#### 🔗 Lineage Chain")

    git_sha = (
        tags.get("git_sha") or tags.get("git_commit") or tags.get("mlflow.source.git.commit", "—")
    )
    # train.py logs dvc_data_hash as a param for DB-backed training; as a tag
    # for file-backed training via mlflow_utils. Check both so lineage shows the
    # hash regardless of which path trained the model.
    dvc_hash = (
        tags.get("dvc_data_hash")
        or tags.get("dvc_data_version")
        or params.get("dvc_data_hash")
        or "—"
    )
    classifier_type = params.get("classifier_type") or params.get("model_type", "—")
    data_path = params.get("train_data_path", "—")
    trained_by = tags.get("trained_by", "—")
    deployment_mode = tags.get("deployment_mode", "—")

    lineage = {
        "Model": f"{selected_model} v{mv.version}",
        "Stage": stage,
        "Training Run": mv.run_id,
        "Classifier": classifier_type,
        "Train Data": (
            "PostgreSQL DB (automated_retraining / sliding window)"
            if data_path == "__from_database__"
            else data_path
        ),
        "Git SHA": git_sha,
        "DVC Data Hash": dvc_hash,
        "Trained By": trained_by,
        "Deployment Mode": deployment_mode,
    }
    st.json(lineage)

    # ── 5. Artifacts list ──────────────────
    try:
        artifacts = client.list_artifacts(mv.run_id)
        if artifacts:
            with st.expander("📦 Artifacts", expanded=False):
                for a in artifacts:
                    icon = "📁" if a.is_dir else "📄"
                    size = f" ({a.file_size} B)" if hasattr(a, "file_size") and a.file_size else ""
                    st.write(f"{icon} `{a.path}`{size}")
    except Exception:
        pass

    # ── 6. DB cross-reference ──────────────
    st.markdown("#### 🗄️ Database Cross-Reference")
    st.markdown("Check if predictions in the PostgreSQL database reference this model version.")
    try:
        from src.database.database import Database

        with Database(db_url=get_host_db_url() or None) as db:
            cursor = db.conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) AS n FROM predictions WHERE model_version = %s",
                (f"v{mv.version}",),
            )
            row = cursor.fetchone()
            count = int(row["n"]) if row else 0
        if count > 0:
            st.success(f"**{count}** predictions in DB reference model v{mv.version}.")
        else:
            st.info(f"No predictions in DB reference model v{mv.version} yet.")
    except Exception as exc:
        st.caption(f"DB lookup unavailable: {exc}")


def _render_reproducibility_tab(mv, client) -> None:  # type: ignore[no-untyped-def]
    """Run an exact reproduction of a model training run and compare metrics."""
    from pathlib import Path

    st.markdown(
        "Re-train the model using the **exact same data and parameters** as the original "
        "run to verify reproducibility. No DagsHub or remote storage required — "
        "uses local training data only."
    )

    if not mv.run_id:
        st.warning("No training run linked to this model version. Cannot reproduce.")
        return

    try:
        run = client.get_run(mv.run_id)
    except Exception as exc:
        st.error(f"Failed to fetch run: {exc}")
        return

    params = run.data.params
    orig_metrics = run.data.metrics

    # ── Show what we'll reproduce ──────────────────────────────────────────
    data_path_str = params.get("train_data_path", "")
    classifier_type = params.get("classifier_type", "logistic_regression")
    k_range_min = int(params.get("k_range_min", 2))
    k_range_max = int(params.get("k_range_max", 10))
    k_method = params.get("k_method", "silhouette")
    _random_state = int(params.get("random_state", 42))
    git_sha = run.data.tags.get("git_sha") or run.data.tags.get("mlflow.source.git.commit", "—")
    dvc_hash = run.data.tags.get("dvc_data_hash") or run.data.params.get("dvc_data_hash") or "—"

    st.markdown("#### 📋 Reproduction Plan")
    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Classifier", classifier_type)
    rc2.metric("K-Range", f"{k_range_min}–{k_range_max} ({k_method})")
    rc3.metric("Git SHA", git_sha[:10] if git_sha != "—" else "—")

    _data_path_display = (
        "PostgreSQL DB (automated_retraining / sliding window)"
        if data_path_str == "__from_database__"
        else (data_path_str or "(not recorded)")
    )
    st.markdown(f"**Training data:** `{_data_path_display}`")
    st.markdown(f"**DVC data hash:** `{dvc_hash}`")

    if not data_path_str:
        st.warning(
            "Training data path was not recorded in this run. "
            "Cannot reproduce without knowing the original dataset."
        )
        return

    data_path = Path(data_path_str)
    if not data_path.exists():
        from ._common import PROJECT_ROOT

        # Try path relative to project root (in case it was stored as relative)
        alt = PROJECT_ROOT / data_path_str.lstrip("/\\")
        if alt.exists():
            data_path = alt
        elif data_path_str == "__from_database__":
            # ── DB-backed training (from_db=True) ─────────────────────────
            # The automated_retraining DAG trained directly from PostgreSQL
            # (no intermediate file). The exact signals are recorded in the
            # model_training_data table and/or exported as split JSON files
            # under data/processed/training_splits/<run_id>/.
            _window_days_str = params.get("window_days", "")
            _window_days_int: int | None = (
                int(_window_days_str) if _window_days_str and _window_days_str.isdigit() else None
            )
            _run_ts_ms = getattr(run.info, "start_time", None)
            _run_id = mv.run_id or ""

            # Try to find pre-exported split JSON (most accurate reproduction)
            from ._common import PROJECT_ROOT as _PR  # noqa: PLC0415

            _split_dir = _PR / "data" / "processed" / "training_splits" / _run_id
            _train_json_path = _split_dir / "train.json"
            _test_json_path = _split_dir / "test.json"
            _has_split_files = _train_json_path.exists() and _test_json_path.exists()

            _run_ts_label = "—"
            if _run_ts_ms:
                from datetime import datetime as _dt_cls

                _run_ts_label = _dt_cls.fromtimestamp(_run_ts_ms / 1000, tz=UTC).strftime(
                    "%Y-%m-%d %H:%M UTC"
                )

            if _has_split_files:
                import json as _json_size

                _n_train = len(_json_size.loads(_train_json_path.read_text()).get("signals", []))
                _n_test = len(_json_size.loads(_test_json_path.read_text()).get("signals", []))
                st.info(
                    f"✅ **Exact training data found** — Split JSON files are stored at  \n"
                    f"`{_split_dir.relative_to(_PR)}`  \n"
                    f"Train: **{_n_train}** signals · Test: **{_n_test}** signals.  \n"
                    "Reproduction will use these exact signals."
                )
            else:
                st.info(
                    "ℹ️ **DB-backed training** — This model was trained directly from PostgreSQL "
                    f"(`from_db=True`). Split JSON files are not available for run `{_run_id[:12]}…`.  \n"
                    "Reproduction will use the current DB state (result may differ slightly if new "
                    "signals were added since training)."
                )

            st.markdown("#### 📋 Original Training Context")
            _oc1, _oc2, _oc3 = st.columns(3)
            _oc1.metric("Trained at", _run_ts_label)
            _oc2.metric("Window", f"{_window_days_str or '—'} days")
            _oc3.metric("Classifier", classifier_type)

            orig_f1 = orig_metrics.get("test_f1_score")
            orig_acc = orig_metrics.get("test_accuracy")
            if orig_f1 is not None:
                _om1, _om2 = st.columns(2)
                _om1.metric("Original Test F1", f"{orig_f1:.4f}")
                if orig_acc is not None:
                    _om2.metric("Original Test Accuracy", f"{orig_acc:.4f}")

            st.markdown("---")
            _btn_label = (
                "▶️ Reproduce from Stored Split Files"
                if _has_split_files
                else "🔄 Reconstruct from Database & Reproduce"
            )
            if st.button(_btn_label, key="db_from_db_reproduce_btn", type="primary"):
                import contextlib
                import json as _json_mod
                import os as _os
                import tempfile
                from pathlib import Path as _Path

                with st.spinner("Reproducing training run… this may take 60–120 seconds."):
                    try:
                        from src.training.train import train_model

                        with tempfile.NamedTemporaryFile(
                            suffix=".pkl", delete=False
                        ) as _tmp_model_f:
                            _tmp_model_path = _Path(_tmp_model_f.name)

                        if _has_split_files:
                            # Combine train + test signals into one file for re-training
                            _all_sigs: list[dict] = []
                            _all_sigs.extend(
                                _json_mod.loads(_train_json_path.read_text()).get("signals", [])
                            )
                            _all_sigs.extend(
                                _json_mod.loads(_test_json_path.read_text()).get("signals", [])
                            )
                            with tempfile.NamedTemporaryFile(
                                mode="w", suffix=".json", delete=False
                            ) as _tmp_data_f:
                                _json_mod.dump({"signals": _all_sigs}, _tmp_data_f)
                                _tmp_data_path = _Path(_tmp_data_f.name)

                            try:
                                _result = train_model(
                                    train_data_path=_tmp_data_path,
                                    model_output_path=_tmp_model_path,
                                    model_version=f"reproduce_v{mv.version}",
                                    use_mlflow=False,
                                    classifier_type=classifier_type,
                                    k_range=(k_range_min, k_range_max),
                                    k_method=k_method,
                                    random_state=_random_state,
                                    allow_unlabeled=True,
                                    filter_unlabeled=False,
                                )
                            finally:
                                with contextlib.suppress(Exception):
                                    _tmp_data_path.unlink(missing_ok=True)
                                with contextlib.suppress(Exception):
                                    _tmp_model_path.unlink(missing_ok=True)
                        else:
                            # Fall back: query DB for current signals
                            from src.database.database import Database

                            _pg_url = _os.environ.get("DATABASE_URL", "")
                            if not _pg_url:
                                st.error("DATABASE_URL not set. Start the stack with `make cloud`.")
                                return  # type: ignore[return-value]

                            _db = Database(db_url=_pg_url)
                            try:
                                _labeled_ids = _db.get_labeled_signal_ids()
                                _unlabeled_ids = _db.get_unlabeled_signal_ids()
                                _signals: list[dict] = []
                                for _sid in _labeled_ids:
                                    _raw = _db.get_signal_data_by_id(_sid)
                                    _lbl = _db.get_label_by_signal_id(_sid)
                                    if _raw is None or _lbl is None:
                                        continue
                                    _signals.append(
                                        {
                                            "id": _sid,
                                            "time": _raw["time_values"],
                                            "amplitude": _raw["amplitude_values"],
                                            "label": _lbl if _lbl in (0, 1) else None,
                                            "shape_type": "gaussian",
                                        }
                                    )
                                for _sid in _unlabeled_ids:
                                    _raw = _db.get_signal_data_by_id(_sid)
                                    if _raw is None:
                                        continue
                                    _signals.append(
                                        {
                                            "id": _sid,
                                            "time": _raw["time_values"],
                                            "amplitude": _raw["amplitude_values"],
                                            "label": None,
                                            "shape_type": "gaussian",
                                        }
                                    )
                            finally:
                                _db.close()

                            with tempfile.NamedTemporaryFile(
                                mode="w", suffix=".json", delete=False
                            ) as _tmp_data_f:
                                _json_mod.dump({"signals": _signals}, _tmp_data_f)
                                _tmp_data_path = _Path(_tmp_data_f.name)

                            try:
                                _result = train_model(
                                    train_data_path=_tmp_data_path,
                                    model_output_path=_tmp_model_path,
                                    model_version=f"reproduce_v{mv.version}",
                                    use_mlflow=False,
                                    classifier_type=classifier_type,
                                    k_range=(k_range_min, k_range_max),
                                    k_method=k_method,
                                    window_days=_window_days_int,
                                    random_state=_random_state,
                                    allow_unlabeled=True,
                                    filter_unlabeled=False,
                                )
                            finally:
                                with contextlib.suppress(Exception):
                                    _tmp_data_path.unlink(missing_ok=True)
                                with contextlib.suppress(Exception):
                                    _tmp_model_path.unlink(missing_ok=True)

                        _repro_f1 = _result.get("test_f1_score")
                        _repro_acc = _result.get("test_accuracy")

                        st.markdown("#### 📊 Reproduction Results")
                        _rc = st.columns(3)
                        if orig_f1 is not None and _repro_f1 is not None:
                            _delta_f1 = _repro_f1 - orig_f1
                            _rc[0].metric("Original F1", f"{orig_f1:.4f}")
                            _rc[1].metric("Reproduced F1", f"{_repro_f1:.4f}")
                            _match = abs(_delta_f1) < 0.02
                            _rc[2].metric(
                                "Match",
                                "✅ Yes" if _match else f"⚠️ Delta {_delta_f1:+.4f}",
                            )
                            if _match:
                                st.success(
                                    f"DB-backed training **verified** — F1 within 2% of original "
                                    f"({orig_f1:.4f} vs {_repro_f1:.4f})."
                                )
                            else:
                                st.warning(
                                    f"Reproduction **diverged** — F1 delta = {_delta_f1:+.4f}. "
                                    "This may occur if the DB has changed since training."
                                )
                        elif _repro_f1 is not None:
                            _rc[0].metric("Reproduced F1", f"{_repro_f1:.4f}")
                            if _repro_acc is not None:
                                _rc[1].metric("Reproduced Accuracy", f"{_repro_acc:.4f}")
                    except Exception as _exc:
                        st.error(f"Reproduction failed: {_exc}")
            return  # Don't fall through to file-based path section

        elif data_path_str.startswith("/tmp/") or data_path_str.startswith("\\tmp\\"):
            # ── Container-generated ephemeral temp file ───────────────────
            # The automated_retraining DAG creates a temp JSON inside the Docker
            # container, trains, then deletes it. The file never exists on the host.
            # HOWEVER: the signal data IS stored persistently in PostgreSQL, so we
            # can reconstruct the exact same training set by re-querying the DB.
            _window_days_str = params.get("window_days", "")
            _window_days_int = (  # type: ignore[assignment]
                int(_window_days_str) if _window_days_str and _window_days_str.isdigit() else None
            )
            _run_ts_ms = getattr(run.info, "start_time", None)

            st.info(
                "ℹ️ **DB-backed reproducibility** — This model was trained on signals "
                f"fetched directly from PostgreSQL inside the Docker container. "
                f"The temp file `{data_path_str}` was deleted after training.  \n\n"
                f"The **original signal data is still in PostgreSQL** and can be "
                f"reconstructed using the same query the DAG ran. "
                f"Below you can reproduce training using the current DB state — "
                f"the result may differ slightly only if new signals were added or "
                f"removed since the original run."
            )

            _run_ts_label = "—"
            if _run_ts_ms:
                from datetime import datetime as _dt_cls

                _run_ts_label = _dt_cls.fromtimestamp(_run_ts_ms / 1000, tz=UTC).strftime(
                    "%Y-%m-%d %H:%M UTC"
                )

            st.markdown("#### 📋 Original Training Context")
            _oc1, _oc2, _oc3 = st.columns(3)
            _oc1.metric("Trained at", _run_ts_label)
            _oc2.metric("Window", f"{_window_days_str or '—'} days")
            _oc3.metric("Classifier", classifier_type)

            orig_f1 = orig_metrics.get("test_f1_score")
            orig_acc = orig_metrics.get("test_accuracy")
            if orig_f1 is not None:
                _om1, _om2 = st.columns(2)
                _om1.metric("Original Test F1", f"{orig_f1:.4f}")
                if orig_acc is not None:
                    _om2.metric("Original Test Accuracy", f"{orig_acc:.4f}")

            st.markdown("---")
            if st.button(
                "🔄 Reconstruct from Database & Reproduce",
                key="db_reproduce_btn",
                type="primary",
            ):
                import contextlib
                import json as _json_mod
                import os as _os
                import tempfile
                from pathlib import Path as _Path

                with st.spinner(
                    "Fetching signals from database and reproducing training…  "
                    "This may take 60–120 seconds."
                ):
                    try:
                        from src.database.database import Database
                        from src.training.train import train_model

                        _pg_url = _os.environ.get("DATABASE_URL", "")
                        if not _pg_url:
                            st.error(
                                "DATABASE_URL not set. Start the stack with `make cloud` "
                                "so the database is reachable."
                            )
                            return  # type: ignore[return-value]

                        _db = Database(db_url=_pg_url)
                        try:
                            _labeled_ids = _db.get_labeled_signal_ids()
                            _unlabeled_ids = _db.get_unlabeled_signal_ids()

                            _signals = []  # type: ignore[assignment]
                            for _sid in _labeled_ids:
                                _raw = _db.get_signal_data_by_id(_sid)
                                _lbl = _db.get_label_by_signal_id(_sid)
                                if _raw is None or _lbl is None:
                                    continue
                                _eff_lbl = _lbl if _lbl in (0, 1) else None
                                _signals.append(
                                    {
                                        "id": _sid,
                                        "time": _raw["time_values"],
                                        "amplitude": _raw["amplitude_values"],
                                        "label": _eff_lbl,
                                        "shape_type": "gaussian",
                                    }
                                )
                            for _sid in _unlabeled_ids:
                                _raw = _db.get_signal_data_by_id(_sid)
                                if _raw is None:
                                    continue
                                _signals.append(
                                    {
                                        "id": _sid,
                                        "time": _raw["time_values"],
                                        "amplitude": _raw["amplitude_values"],
                                        "label": None,
                                        "shape_type": "gaussian",
                                    }
                                )
                        finally:
                            _db.close()

                        if not _signals:
                            st.error(
                                "No signals found in the database. "
                                "Run Greenfield → Bootstrap first to populate the DB."
                            )
                            return  # type: ignore[return-value]

                        _n_labeled_db = sum(1 for _s in _signals if _s.get("label") is not None)
                        st.info(
                            f"Fetched **{_n_labeled_db} labeled** + "
                            f"**{len(_signals) - _n_labeled_db} unlabeled** signals "
                            f"from the database."
                        )

                        # Write temp data file (same format as the DAG)
                        with tempfile.NamedTemporaryFile(
                            mode="w", suffix=".json", delete=False
                        ) as _tmp_data_f:
                            _json_mod.dump({"signals": _signals}, _tmp_data_f)
                            _tmp_data_path = _Path(_tmp_data_f.name)

                        # Temp model output path
                        with tempfile.NamedTemporaryFile(
                            suffix=".pkl", delete=False
                        ) as _tmp_model_f:
                            _tmp_model_path = _Path(_tmp_model_f.name)

                        try:
                            _result = train_model(
                                train_data_path=_tmp_data_path,
                                model_output_path=_tmp_model_path,
                                model_version=f"reproduce_v{mv.version}",
                                use_mlflow=False,
                                classifier_type=classifier_type,
                                k_range=(k_range_min, k_range_max),
                                k_method=k_method,
                                window_days=_window_days_int,
                                random_state=_random_state,
                                allow_unlabeled=True,
                                filter_unlabeled=False,
                            )
                        finally:
                            with contextlib.suppress(Exception):
                                _tmp_data_path.unlink(missing_ok=True)
                            with contextlib.suppress(Exception):
                                _tmp_model_path.unlink(missing_ok=True)

                        _repro_f1 = _result.get("test_f1_score")
                        _repro_acc = _result.get("test_accuracy")

                        st.markdown("#### 📊 Reproduction Results")
                        _rc = st.columns(3)

                        if orig_f1 is not None and _repro_f1 is not None:
                            _delta_f1 = _repro_f1 - orig_f1
                            _rc[0].metric("Original F1", f"{orig_f1:.4f}")
                            _rc[1].metric("Reproduced F1", f"{_repro_f1:.4f}")
                            _match = abs(_delta_f1) < 0.02  # 2% tolerance for DB reconstruction
                            _rc[2].metric(
                                "Match",
                                "✅ Yes" if _match else f"⚠️ Delta {_delta_f1:+.4f}",
                            )
                            if _match:
                                st.success(
                                    f"DB-reconstruction **verified** — F1 within 2% of original "
                                    f"({orig_f1:.4f} vs {_repro_f1:.4f}). "
                                    f"The model is reproducible from the stored signals."
                                )
                            else:
                                st.warning(
                                    f"Reproduction **diverged** — F1 delta = {_delta_f1:+.4f}. "
                                    "This typically means new signals were added to the DB since "
                                    "the original training run, changing the window coverage. "
                                    "The algorithm and parameters were identical."
                                )
                        elif _repro_f1 is not None:
                            _rc[0].metric("Reproduced F1", f"{_repro_f1:.4f}")
                            if _repro_acc is not None:
                                _rc[1].metric("Reproduced Accuracy", f"{_repro_acc:.4f}")

                    except Exception as _exc:
                        st.error(f"Reproduction failed: {_exc}")
            return  # Don't fall through to file-based path section
        else:
            st.error(
                f"Training data not found at `{data_path_str}`.  \n"
                "Ensure the file is available locally. If the data was stored on "
                "DagsHub/S3, run `dvc pull` first."
            )
            return

    # ── Informational note for bootstrap_labeled.json (champion/challenger) ─
    if "bootstrap_labeled" in data_path.name:
        st.info(
            "ℹ️ **Champion/Challenger design** — These models are trained on the "
            "bootstrap labeled dataset to compare different classifier architectures "
            "(LogisticRegression, RandomForest, SVC, etc.) on a common reference "
            "dataset. This is intentional: the same data, different algorithms."
        )

    st.success(f"Training data found: `{data_path}` ({data_path.stat().st_size // 1024} KB)")

    # ── Key original metrics ──────────────────────────────────────────────
    orig_f1 = orig_metrics.get("test_f1_score", None)
    orig_acc = orig_metrics.get("test_accuracy", None)

    if orig_f1 is not None:
        om1, om2 = st.columns(2)
        om1.metric("Original Test F1", f"{orig_f1:.4f}")
        if orig_acc is not None:
            om2.metric("Original Test Accuracy", f"{orig_acc:.4f}")

    # ── Run reproduction ──────────────────────────────────────────────────
    st.markdown("---")
    if st.button("▶️ Run Reproduction", key="reproduce_btn", type="primary"):
        import contextlib
        import tempfile
        from pathlib import Path as _Path

        with st.spinner("Reproducing training run… this may take 30–90 seconds."):
            try:
                from src.training.train import train_model

                with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
                    tmp_path = _Path(tmp.name)

                result = train_model(
                    train_data_path=data_path,
                    model_output_path=tmp_path,
                    model_version=f"reproduce_v{mv.version}",
                    use_mlflow=False,
                    classifier_type=classifier_type,
                    k_range=(k_range_min, k_range_max),
                    k_method=k_method,
                )

                # Clean up temp model file
                with contextlib.suppress(Exception):
                    tmp_path.unlink(missing_ok=True)

                repro_f1 = result.get("test_f1_score")
                repro_acc = result.get("test_accuracy")

                st.markdown("#### 📊 Reproduction Results")
                res_cols = st.columns(3)

                if orig_f1 is not None and repro_f1 is not None:
                    delta_f1 = repro_f1 - orig_f1
                    res_cols[0].metric("Original F1", f"{orig_f1:.4f}")
                    res_cols[1].metric("Reproduced F1", f"{repro_f1:.4f}")
                    match = abs(delta_f1) < 0.01
                    res_cols[2].metric(
                        "Match",
                        "✅ Yes" if match else f"⚠️ Delta {delta_f1:+.4f}",
                    )
                    if match:
                        st.success(
                            f"Reproduction **verified** — F1 within 1% of original "
                            f"({orig_f1:.4f} vs {repro_f1:.4f})."
                        )
                    else:
                        st.warning(
                            f"Reproduction **diverged** — F1 delta = {delta_f1:+.4f}. "
                            "This may be due to non-deterministic clustering, different "
                            "random seeds, or data preprocessing differences."
                        )
                elif repro_f1 is not None:
                    res_cols[0].metric("Reproduced F1", f"{repro_f1:.4f}")
                    if repro_acc is not None:
                        res_cols[1].metric("Reproduced Accuracy", f"{repro_acc:.4f}")
                    st.info("Original metrics not available for comparison.")

                with st.expander("Full reproduction result", expanded=False):
                    st.json(result)

            except Exception as exc:
                st.error(f"Reproduction failed: {exc}")
