"""Tab 5 — Automated Retraining Pipeline.

End-to-end retraining workflow visualisation:
  1. Show the Airflow DAG structure (automated_retraining)
  2. Trigger the retraining DAG manually
  3. Monitor the DAG run status
  4. Display the new model metrics after retraining completes
"""

from __future__ import annotations

import os
import time

import streamlit as st

from src.ui.components.docker_utils import get_host
from src.ui.logging_ui import get_ui_logger

_logger = get_ui_logger(__name__)

from datetime import UTC

from src.config import (
    DEFAULT_LABEL_INJECTION_PCT,
    DRIFT_SCENARIO_LABELS,
    DRIFT_SCENARIOS,
    LABEL_HEALTHY,
    LABEL_UNHEALTHY,
    MIN_LABELED_SIGNALS,
    MIN_LABELED_SIGNALS_RECOMMENDED,
)

from ._common import (
    fetch_champion_info,
    get_mlflow_client,
    reload_api_model,
)

_AIRFLOW_API = os.environ.get("AIRFLOW_API_URL", f"http://{get_host()}:8081") + "/api/v1"

_AIRFLOW_AUTH = (
    os.environ.get("AIRFLOW_USER", "admin"),
    os.environ.get("AIRFLOW_PASSWORD", "admin"),
)

_DAG_ID = "automated_retraining"

# Pipeline stages shown in the progress tracker
_PIPELINE_STAGES = [
    (
        "check_data",
        "Check Data Availability",
        "Query PostgreSQL for new prediction records since the last training run. "
        "Minimum threshold: 50 new samples.",
    ),
    (
        "extract_features",
        "Extract Features",
        "Run the feature extraction pipeline (FWHM, peak height, area, noise, SNR, "
        "peak centre) on the new signal data and merge with the existing gold-standard set.",
    ),
    (
        "train_model",
        "Train Challenger Model",
        "Train a new model using the semi-supervised pipeline (K-means label "
        "propagation → classifier). Classifier type and hyper-parameters are "
        "inherited from the current champion's configuration.",
    ),
    (
        "evaluate",
        "Evaluate Champion vs Challenger",
        "Compare the challenger against the production champion on the held-out "
        "gold-standard test set. Primary metric: F1 score "
        "(improvement threshold: +0.005).",
    ),
    (
        "promote",
        "Promote (if improved)",
        "If the challenger exceeds the champion's F1 score by the threshold, "
        "transition it to the 'Production' stage in the MLflow Model Registry "
        "and archive the old champion.",
    ),
    (
        "notify",
        "Notify & Log",
        "Log the comparison results as an MLflow artefact and send a summary "
        "notification (Slack webhook if configured).",
    ),
]


def _airflow_get(path: str) -> dict | None:
    """GET from Airflow REST API."""
    import requests

    try:
        resp = requests.get(
            f"{_AIRFLOW_API}{path}",
            auth=_AIRFLOW_AUTH,
            timeout=10,
        )
        if resp.ok:
            return resp.json()
    except Exception as exc:
        _logger.warning("Airflow GET {} failed: {}", path, exc)
    return None


def _airflow_post(path: str, payload: dict | None = None) -> dict | None:
    """POST to Airflow REST API."""
    import requests

    try:
        resp = requests.post(
            f"{_AIRFLOW_API}{path}",
            json=payload or {},
            auth=_AIRFLOW_AUTH,
            timeout=10,
        )
        if resp.ok:
            return resp.json()
        _logger.warning("Airflow POST {} HTTP {}: {}", path, resp.status_code, resp.text[:200])
        st.error(f"Airflow returned {resp.status_code}: {resp.text[:300]}")
    except Exception as exc:
        _logger.warning("Airflow POST {} failed: {}", path, exc)
        st.error(f"Failed to reach Airflow: {exc}")
    return None


def _render_dag_overview() -> None:
    """Show the retraining DAG structure."""
    st.markdown("#### 📋 Retraining DAG Overview")
    st.markdown(
        "The **automated_retraining** DAG runs weekly (or on-demand) and "
        "executes the following pipeline:"
    )

    for i, (stage_id, label, detail) in enumerate(_PIPELINE_STAGES, 1):
        st.markdown(f"**{i}.** `{stage_id}` — {label}")
        st.caption(f"   {detail}")

    st.markdown(
        "\n> The DAG trains a **challenger** model on the latest data, "
        "compares it against the current **production champion**, and "
        "promotes the challenger only if it exceeds the improvement threshold."
    )


def _render_current_champion() -> None:
    """Show current champion model info (non-blocking — uses 5-minute cache)."""
    import threading

    st.markdown("#### 🏆 Current Production Champion")

    # Use session-state cache to avoid blocking ALL tab renders on every call.
    # The buffer connection is local (localhost:5002) but threading avoids a
    # blocking call that would delay rendering of Data Generation / DAG Trigger tabs.
    _cache_key = "_retraining_champion_cache"
    _cache_ttl = 300  # 5 minutes

    import time as _time_mod

    cached = st.session_state.get(_cache_key)
    if cached and (_time_mod.time() - cached.get("ts", 0)) < _cache_ttl:
        champ_mv, champ_run = cached.get("mv"), cached.get("run")
    else:
        champ_mv, champ_run = None, None
        _result: list = []

        def _fetch():
            try:
                client, _ = get_mlflow_client()
                mv, run = fetch_champion_info(client)
                _result.append((mv, run))
            except Exception as exc:
                _result.append((None, str(exc)))

        _t = threading.Thread(target=_fetch, daemon=True)
        _t.start()
        _t.join(timeout=10)  # 10s max — don't block tab rendering

        if _result:
            if isinstance(_result[0], tuple):
                champ_mv, champ_run = _result[0]
            else:
                st.error(f"Cannot fetch champion info: {_result[0]}")
                return
        else:
            st.warning("Champion info is loading (buffer connecting). Refresh the page to retry.")
            return

        st.session_state[_cache_key] = {"mv": champ_mv, "run": champ_run, "ts": _time_mod.time()}

    if champ_mv is None:
        st.warning("No production champion found.")
        return

    champ_metrics = champ_run.data.metrics if champ_run else {}
    champ_params = champ_run.data.params if champ_run else {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Version", f"v{champ_mv.version}")
    c2.metric("Classifier", champ_params.get("classifier_type", "—"))
    c3.metric("Test F1", f"{champ_metrics.get('test_f1_score', 0):.4f}")
    c4.metric("Test Accuracy", f"{champ_metrics.get('test_accuracy', 0):.4f}")


def _render_latest_challenger() -> None:
    """Show the most recently registered challenger model (if any)."""
    import threading
    import time as _time_mod

    _cache_key = "_retraining_challenger_cache"
    _cache_ttl = 60  # 1-minute cache

    cached = st.session_state.get(_cache_key)
    if cached and (_time_mod.time() - cached.get("ts", 0)) < _cache_ttl:
        challenger_mv, challenger_run = cached.get("mv"), cached.get("run")
    else:
        challenger_mv, challenger_run = None, None
        _result: list = []

        def _fetch() -> None:
            try:
                from src.ui.views.use_cases_pkg._common import get_model_name

                client, _ = get_mlflow_client()
                model_name = get_model_name()
                # Check for challenger alias first; fall back to latest non-champion version
                try:
                    mv = client.get_model_version_by_alias(model_name, "challenger")
                    run = client.get_run(mv.run_id) if mv.run_id else None
                    _result.append((mv, run))
                    return
                except Exception:
                    pass
                # Alias not found — pick the latest version that isn't the champion
                try:
                    champ_mv = client.get_model_version_by_alias(model_name, "champion")
                    champ_ver = champ_mv.version
                except Exception:
                    champ_ver = None
                versions = client.search_model_versions(f"name='{model_name}'")
                non_champ = [v for v in versions if v.version != champ_ver]
                if non_champ:
                    # sort descending by version integer
                    non_champ.sort(key=lambda v: int(v.version), reverse=True)
                    mv = non_champ[0]
                    run = client.get_run(mv.run_id) if mv.run_id else None
                    _result.append((mv, run))
                else:
                    _result.append((None, None))
            except Exception as exc:
                _result.append((None, str(exc)))

        t = threading.Thread(target=_fetch, daemon=True)
        t.start()
        t.join(timeout=10)

        if _result:
            first = _result[0]
            if isinstance(first, tuple):
                challenger_mv, challenger_run = first
            else:
                return  # error — silently skip

        st.session_state[_cache_key] = {
            "mv": challenger_mv,
            "run": challenger_run,
            "ts": _time_mod.time(),
        }

    if challenger_mv is None:
        return  # no challenger yet

    st.markdown("#### 🧪 Latest Challenger")
    chall_metrics = challenger_run.data.metrics if challenger_run else {}
    chall_params = challenger_run.data.params if challenger_run else {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Version", f"v{challenger_mv.version}")
    c2.metric("Classifier", chall_params.get("classifier_type", "—"))
    c3.metric("Test F1", f"{chall_metrics.get('test_f1_score', 0):.4f}")
    c4.metric("Test Accuracy", f"{chall_metrics.get('test_accuracy', 0):.4f}")


def _render_recent_runs() -> None:
    """Show recent DAG run history."""
    st.markdown("#### 📜 Recent Retraining Runs")

    data = _airflow_get(f"/dags/{_DAG_ID}/dagRuns?limit=5&order_by=-start_date")
    if data is None:
        st.warning("Cannot fetch DAG runs from Airflow.")
        return

    runs = data.get("dag_runs", [])
    if not runs:
        st.info("No retraining runs found yet.")
        return

    import pandas as pd

    rows = []
    for run in runs:
        rows.append(
            {
                "Run ID": run.get("dag_run_id", "—")[:30],
                "State": run.get("state", "—"),
                "Start": run.get("start_date", "—")[:19] if run.get("start_date") else "—",
                "End": run.get("end_date", "—")[:19] if run.get("end_date") else "—",
                "Triggered By": run.get("conf", {}).get("triggered_by", "schedule"),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def _show_reload_result() -> bool:
    """Display a persisted reload result from session_state. Returns True if displayed."""
    if st.session_state.get("_retrain_reload_ok") is None:
        return False
    ok_r = st.session_state["_retrain_reload_ok"]
    msg_r = st.session_state.get("_retrain_reload_msg", "")
    prev_v = st.session_state.get("_retrain_reload_prev_v", "?")
    new_v = st.session_state.get("_retrain_reload_new_v", "?")
    if ok_r:
        st.success(f"✅ Reload succeeded: {msg_r}")
        if prev_v != new_v:
            st.info(f"🔄 Version swapped: `{prev_v}` → `{new_v}`")
        else:
            st.info(
                f"ℹ️ Model unchanged: still `{new_v}`. "
                "The API background poller may already have reloaded it."
            )
    else:
        st.error(f"❌ Reload failed: {msg_r}")
        st.info(
            "Tip: Try using the Docker Control page → Restart Individual Service → api "
            "for a hard container restart."
        )
    return True


def _post_completion_ui() -> None:
    """Render the post-retraining UI (champion/challenger + reload button)."""
    # Invalidate champion cache so fresh data is fetched
    st.session_state.pop("_retraining_champion_cache", None)

    st.success("✅ Retraining pipeline completed successfully!")
    st.markdown("#### 📊 Post-Retraining Status")
    _render_current_champion()
    _render_latest_challenger()

    col_reload, col_reset = st.columns(2)
    with col_reload:
        if st.button("🔄 Reload Model in API", type="primary", key="retrain_reload_api"):
            with st.spinner("🔄 Hot-reloading model (no restart required)…"):
                ok, msg, data = reload_api_model()
            prev_v = data.get("previous_version") or data.get("previous_source", "?")
            new_v = data.get("model_version", "?")
            # Persist result so it survives the next rerun
            st.session_state["_retrain_reload_ok"] = ok
            st.session_state["_retrain_reload_msg"] = msg
            st.session_state["_retrain_reload_prev_v"] = prev_v
            st.session_state["_retrain_reload_new_v"] = new_v
            st.rerun()
    with col_reset:
        if st.button("🔁 Trigger Another Run", key="retrain_new_run"):
            for _k in [
                "_retrain_dag_complete",
                "_retrain_reload_ok",
                "_retrain_reload_msg",
                "_retrain_reload_prev_v",
                "_retrain_reload_new_v",
            ]:
                st.session_state.pop(_k, None)
            st.rerun()


def _trigger_and_monitor() -> None:
    """Trigger the retraining DAG and poll for status."""
    st.markdown("#### 🚀 Trigger Retraining")

    # ── Show persisted reload result (survives reruns) ──────────────────────
    if _show_reload_result():
        if st.button("↩️ Back to retraining", key="retrain_back_after_reload"):
            for _k in [
                "_retrain_reload_ok",
                "_retrain_reload_msg",
                "_retrain_reload_prev_v",
                "_retrain_reload_new_v",
            ]:
                st.session_state.pop(_k, None)
            st.rerun()
        return

    # ── DAG already completed — show post-completion UI ──────────────────────
    if st.session_state.get("_retrain_dag_complete"):
        _post_completion_ui()
        return

    # ── Human Approval Gate option ─────────────────────────────────────────
    require_approval = st.checkbox(
        "🔐 Require human approval before model promotion",
        value=False,
        key="retrain_require_approval",
        help=(
            "When enabled, the DAG will pause after evaluation and wait for a "
            "human to approve or reject the challenger in the MLflow Explorer → "
            "Approval Queue tab before promoting to production."
        ),
    )

    # ── Normal trigger flow ───────────────────────────────────────────────────
    col1, _col2 = st.columns([1, 3])
    with col1:
        if not st.button("🔄 Trigger Retraining DAG", type="primary", key="retrain_trigger"):
            return

    _logger.info("Triggering Airflow DAG: {}", _DAG_ID)
    result = _airflow_post(
        f"/dags/{_DAG_ID}/dagRuns",
        {
            "conf": {
                "triggered_by": "streamlit_retraining_tab",
                "require_human_approval": require_approval,
            }
        },
    )
    if result is None:
        return

    if require_approval:
        st.info(
            "🔐 Human approval gate is **enabled**. After evaluation, the DAG will pause "
            "and wait for approval in **MLflow Explorer → 📋 Approval Queue**."
        )

    dag_run_id = result.get("dag_run_id", "")
    _logger.info("Retraining DAG {} triggered — run_id={}", _DAG_ID, dag_run_id)
    st.info(f"DAG triggered: `{dag_run_id}`")

    # Poll for completion
    status_container = st.status("Monitoring retraining pipeline...", expanded=True)
    progress = st.progress(0.0)

    max_polls = 60  # ~5 minutes with 5s interval
    for poll in range(max_polls):
        time.sleep(5)

        run_data = _airflow_get(f"/dags/{_DAG_ID}/dagRuns/{dag_run_id}")
        if run_data is None:
            status_container.write(f"Poll {poll + 1}: waiting for response...")
            continue

        state = run_data.get("state", "running")
        progress.progress(min((poll + 1) / max_polls, 0.95))
        status_container.write(f"Poll {poll + 1}: state = **{state}**")

        # When human approval is required, detect the wait-for-approval task
        # and break out early so the inline approval queue becomes visible.
        if require_approval and state == "running":
            tasks_data = _airflow_get(f"/dags/{_DAG_ID}/dagRuns/{dag_run_id}/taskInstances")
            if tasks_data:
                task_states = {
                    t["task_id"]: t.get("state") for t in tasks_data.get("task_instances", [])
                }
                wait_state = task_states.get(
                    "evaluation_group.wait_for_human_approval"
                ) or task_states.get("wait_for_human_approval")
                if wait_state in ("running", "up_for_reschedule", "up_for_retry", "deferred"):
                    progress.progress(0.6)
                    status_container.update(label="⏳ Waiting for human approval…", state="running")
                    st.info(
                        "🔐 The DAG is paused at **wait_for_human_approval**. "
                        "Use the approval queue below to approve or reject."
                    )
                    # Persist dag_run_id so post-refresh code can re-show this section
                    st.session_state["_retrain_dag_run_id"] = dag_run_id
                    st.session_state["_retrain_waiting_approval"] = True
                    break

        if state == "success":
            progress.progress(1.0)
            status_container.update(label="Retraining Complete!", state="complete")
            # Persist completion so the post-completion UI survives reruns
            st.session_state["_retrain_dag_complete"] = True
            st.rerun()

        if state == "failed":
            progress.progress(1.0)
            status_container.update(label="Retraining Failed", state="error")
            st.error("Retraining pipeline failed. Check Airflow logs for details.")

            # Show task instance states
            tasks_data = _airflow_get(f"/dags/{_DAG_ID}/dagRuns/{dag_run_id}/taskInstances")
            if tasks_data:
                failed_tasks = [
                    t for t in tasks_data.get("task_instances", []) if t.get("state") == "failed"
                ]
                if failed_tasks:
                    st.markdown("**Failed tasks:**")
                    for t in failed_tasks:
                        st.write(f"- `{t.get('task_id')}`: {t.get('state')}")
            return

    else:
        status_container.update(label="Monitoring timed out", state="error")
        st.warning("Monitoring timed out after 5 minutes. Check Airflow for status.")


def _push_dvc_snapshots() -> None:
    """Section: push DVC pointer files to GitHub after a sync DAG run."""
    import subprocess

    st.markdown("---")
    st.markdown("#### 📦 Push DVC Snapshots to GitHub")
    st.markdown(
        "After the `sync_production_data` DAG has pushed raw data to DagsHub, "
        "run this step to commit the updated `.dvc` pointer files to GitHub.  \n"
        "The commit is tagged `[skip ci]` so the full lint/test/build pipeline "
        "is **not** triggered."
    )

    # Show what would be staged
    dry_run = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "data/sync.dvc",
            "data/raw_signals.dvc",
            "dvc.lock",
            "data/.gitignore",
        ],
        capture_output=True,
        text=True,
    )
    changed_files = [ln.strip() for ln in dry_run.stdout.splitlines() if ln.strip()]

    if changed_files:
        st.info(
            "Files with pending DVC pointer changes:\n"
            + "\n".join(f"- `{f}`" for f in changed_files)
        )
    else:
        st.success("DVC pointer files are already up to date — nothing to push.")
        return

    if not st.button("🚀 Push DVC Snapshots to GitHub", key="dvc_snapshot_push"):
        return

    with st.spinner("Staging and pushing DVC pointer files..."):
        # Stage DVC pointer files only (not raw data)
        add_result = subprocess.run(
            [
                "git",
                "add",
                "--ignore-missing",
                "data/sync.dvc",
                "data/raw_signals.dvc",
                "dvc.lock",
                "data/.gitignore",
            ],
            capture_output=True,
            text=True,
        )
        if add_result.returncode != 0:
            st.error(f"git add failed: {add_result.stderr.strip()}")
            return

        # Commit
        commit_result = subprocess.run(
            ["git", "commit", "-m", "[skip ci] chore(data): update DVC snapshots"],
            capture_output=True,
            text=True,
        )
        if commit_result.returncode not in (0, 1):  # 1 = nothing to commit
            st.error(f"git commit failed: {commit_result.stderr.strip()}")
            return
        if "nothing to commit" in commit_result.stdout:
            st.info("Nothing to commit.")
            return

        # Push
        push_result = subprocess.run(
            ["git", "push"],
            capture_output=True,
            text=True,
        )
        if push_result.returncode != 0:
            st.error(f"git push failed: {push_result.stderr.strip()}")
            return

    st.success("DVC pointer files pushed to GitHub with `[skip ci]`.")


def _render_inline_approval_queue(mode: str | None = None) -> None:
    """Inline approval queue shown on the DAG Trigger sub-tab when human approval is required.

    Shows only pending approvals (no history). The full history remains in MLflow Explorer.
    """
    import contextlib
    import os
    import sqlite3
    from datetime import datetime
    from pathlib import Path

    from src.ui.components.docker_utils import get_host

    pg_url = os.environ.get("DATABASE_URL", "")
    if not pg_url:
        _pg_host = os.environ.get("POSTGRES_HOST") or os.environ.get("DB_HOST") or get_host()
        _pg_port = os.environ.get("POSTGRES_PORT") or os.environ.get("DB_PORT") or "5433"
        _pg_user = os.environ.get("POSTGRES_USER") or os.environ.get("DB_USER") or "mlops_user"
        _pg_pass = (
            os.environ.get("POSTGRES_PASSWORD") or os.environ.get("DB_PASSWORD") or "changeme"
        )
        _pg_db = os.environ.get("POSTGRES_DB") or os.environ.get("DB_NAME") or "mlops_prod"
        pg_url = (
            f"postgresql://{_pg_user}:{_pg_pass}@{_pg_host}:{_pg_port}/{_pg_db}?connect_timeout=5"
        )

    def _load_pending() -> list[dict]:
        rows: list[dict] = []
        if pg_url:
            try:
                import psycopg2

                conn = psycopg2.connect(pg_url)
                with conn, conn.cursor() as cur:
                    cur.execute(
                        "SELECT id, model_version, mlflow_run_id, challenger_f1, "
                        "champion_f1, champion_f1_on_challenger_test, status, created_at "
                        "FROM model_approvals WHERE status='pending' ORDER BY id DESC"
                    )
                    cols = [d[0] for d in cur.description]
                    rows = [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]
                conn.close()
                return rows
            except Exception as _pg_exc:
                st.warning(
                    f"⚠️ Could not load pending approvals from PostgreSQL: {_pg_exc}  \n"
                    "Check that `DATABASE_URL` is set correctly in the Streamlit environment."
                )
        _project_root = Path(__file__).resolve().parents[4]
        db_path = _project_root / "data" / "mlops.db"
        if not db_path.exists():
            return []
        with contextlib.closing(sqlite3.connect(str(db_path))) as con:
            con.row_factory = sqlite3.Row
            rows = [
                dict(r)
                for r in con.execute(
                    "SELECT id, model_version, mlflow_run_id, challenger_f1, "
                    "champion_f1, champion_f1_on_challenger_test, status, created_at "
                    "FROM model_approvals WHERE status='pending' ORDER BY id DESC"
                ).fetchall()
            ]
        return rows

    def _update_status(approval_id: int, new_status: str, decided_by: str) -> bool:
        """Update approval status in PostgreSQL. Returns True on success."""
        now_iso = datetime.now(UTC).isoformat()
        if not pg_url:
            st.error("⚠️ DATABASE_URL not set — cannot update approval status.")
            return False
        try:
            import psycopg2

            conn = psycopg2.connect(pg_url)
            try:
                cur = conn.cursor()
                cur.execute(
                    "UPDATE model_approvals SET status=%s, decided_at=%s, decided_by=%s "
                    "WHERE id=%s",
                    (new_status, now_iso, decided_by, approval_id),
                )
                cur.close()
                conn.commit()
            finally:
                conn.close()
            return True
        except Exception as _pg_err:
            st.error(
                f"⚠️ PostgreSQL update failed: {_pg_err}. "
                "Check that the database is reachable and try again."
            )
            return False

    st.markdown("---")
    st.markdown("#### 📋 Pending Model Approvals")

    # Show post-approval confirmation (survives the st.rerun() via session state)
    if _approved_msg := st.session_state.pop("_rp_just_approved", None):
        st.success(_approved_msg)
        _rp_ar_run_id = st.session_state.pop("_rp_ar_run_id", None)
        if _rp_ar_run_id and mode:
            from src.ui.views.use_cases_pkg.drift_provocation import (  # noqa: PLC0415
                _show_approval_outcome,
            )

            _show_approval_outcome(mode, _rp_ar_run_id)

    pending = _load_pending()
    if not pending:
        st.info(
            "No pending approvals. The DAG will populate this queue once "
            "challenger evaluation is complete."
        )
        return
    st.caption("Full approval history is available in **MLflow Explorer → 📋 Approval Queue**.")
    for appr in pending:
        _chall_f1 = appr.get("challenger_f1") or 0.0
        # Use champion_f1_on_challenger_test when available (fair comparison)
        _champ_f1_display = (
            appr.get("champion_f1_on_challenger_test")
            if appr.get("champion_f1_on_challenger_test") is not None
            else appr.get("champion_f1") or 0.0
        )
        _fair_comparison = appr.get("champion_f1_on_challenger_test") is not None
        with st.expander(
            f"🔔 #{appr['id']} — {appr['model_version']}  (challenger F1 {_chall_f1:.4f})",
            expanded=True,
        ):
            c1, c2, c3 = st.columns(3)
            c1.metric("Challenger F1", f"{appr.get('challenger_f1') or 0:.4f}")
            c2.metric(
                "Champion F1" + (" \u2020" if _fair_comparison else ""),
                f"{_champ_f1_display:.4f}",
            )
            delta = _chall_f1 - (_champ_f1_display or 0.0)
            c3.metric("Delta", f"{delta:+.4f}", delta_color="normal")
            if _fair_comparison:
                st.caption(
                    "\u2020 Champion re-evaluated on the same test signals as challenger "
                    "(fair comparison)."
                )
            st.caption(
                f"MLflow run: `{appr.get('mlflow_run_id') or 'n/a'}` · "
                f"Requested: {appr.get('created_at', '')}"
            )
            reviewer = st.text_input(
                "Reviewer name / team", value="", key=f"rp_reviewer_{appr['id']}"
            )
            btn_col1, btn_col2 = st.columns(2)
            if btn_col1.button("✅ Approve", key=f"rp_approve_{appr['id']}", type="primary"):
                ok = _update_status(appr["id"], "approved", reviewer or "streamlit_user")
                if ok:
                    _chall_f1 = appr.get("challenger_f1") or 0.0
                    _champ_f1_cmp = (
                        appr.get("champion_f1_on_challenger_test")
                        if appr.get("champion_f1_on_challenger_test") is not None
                        else appr.get("champion_f1") or 0.0
                    )
                    _delta = _chall_f1 - (_champ_f1_cmp or 0.0)
                    if _delta > 0:
                        _outcome_hint = (
                            f"Challenger F1 {_chall_f1:.4f} > Champion F1 {_champ_f1_cmp:.4f} — "
                            "**model should be promoted to Production** within 60 seconds."
                        )
                    else:
                        _outcome_hint = (
                            f"Challenger F1 {_chall_f1:.4f} ≤ Champion F1 {_champ_f1_cmp:.4f} — "
                            "approval submitted but champion may be retained by the DAG."
                        )
                    st.session_state["_rp_just_approved"] = (
                        f"✅ Model **{appr['model_version']}** approved. {_outcome_hint}"
                    )
                    st.session_state["_rp_ar_run_id"] = st.session_state.get(
                        "_retrain_dag_run_id", ""
                    )
                    st.rerun()
                # else: error already shown by _update_status, no rerun so message stays visible
            if btn_col2.button("🚫 Reject", key=f"rp_reject_{appr['id']}"):
                ok = _update_status(appr["id"], "rejected", reviewer or "streamlit_user")
                if ok:
                    st.session_state["_rp_just_approved"] = (
                        f"🚫 Model **{appr['model_version']}** rejected — "
                        "the challenger will remain as Staged/Challenger. Champion is retained."
                    )
                    # No outcome polling on rejection — the message already explains the result
                    st.rerun()


def render_retraining_pipeline_tab(mode: str) -> None:
    """Tab 5: Automated Retraining Pipeline — trigger, monitor, review."""
    if mode != "cloud":
        st.info(
            "**Local sandbox mode** — The Automated Retraining Pipeline "
            "(data generation and DAG trigger) is only available in cloud mode.  \n"
            "Use the **Champion / Challenger** tab for manual model training in local mode."
        )
        return

    st.markdown(
        "Trigger and monitor the automated retraining pipeline. "
        "The pipeline trains a challenger model on latest data, "
        "compares against the champion, and promotes if improved."
    )

    _rp_tabs = ["\U0001f4d6 Workflow", "\U0001f52c Data Generation", "\U0001f680 DAG Trigger"]
    _rp_sel = st.radio(
        "Retraining tab",
        _rp_tabs,
        horizontal=True,
        key="_rp_inner_tab",
        label_visibility="collapsed",
    )
    st.markdown("<hr style='margin:0 0 1rem 0;border-color:#e2e8f0;'>", unsafe_allow_html=True)

    if _rp_sel == _rp_tabs[0]:
        _render_dag_overview()
        st.markdown("---")
        _render_current_champion()
        st.markdown("---")
        _render_recent_runs()
    elif _rp_sel == _rp_tabs[1]:
        _render_data_generation(mode)
    else:
        _trigger_and_monitor()
        _push_dvc_snapshots()
        # Show inline approval queue when human approval gate is enabled
        if st.session_state.get("retrain_require_approval"):
            _render_inline_approval_queue(mode=mode)


def _render_data_generation(mode: str) -> None:
    """Sub-tab: generate synthetic signals and inject sparse labels to prepare for retraining."""
    st.markdown("#### 🔬 Generate Training Data")
    st.markdown(
        "Generate synthetic signals, run predictions via the API, and inject a "
        "sparse subset of ground truth labels into the database.  "
        f"The retraining DAG requires at least **{MIN_LABELED_SIGNALS} labeled signals** "
        f"(recommended: **{MIN_LABELED_SIGNALS_RECOMMENDED}**) before it will proceed."
    )

    # ── Controls ─────────────────────────────────────────────────────────
    from src.ui.views.use_cases_pkg.drift_provocation import (
        DRIFT_TYPES,
        generate_batch,
    )

    col_scen, col_n, col_pts = st.columns(3)
    with col_scen:
        drift_scenario = st.selectbox(
            "Drift scenario",
            ["none"] + DRIFT_SCENARIOS,
            format_func=lambda k: (
                "None (standard mix)"
                if k == "none"
                else DRIFT_SCENARIO_LABELS.get(k, k.replace("_", " ").title())
            ),
            key="retrain_drift_scenario",
        )
    with col_n:
        n_signals = st.number_input(
            "Number of signals to generate",
            min_value=10,
            max_value=2000,
            value=150,
            step=10,
            key="retrain_n_signals",
        )
    with col_pts:
        healthy_pct = st.slider(
            "% healthy signals",
            min_value=5,
            max_value=95,
            value=70,
            step=5,
            key="retrain_healthy_pct",
            disabled=(drift_scenario == "prior_probability_drift"),
        )
        if drift_scenario == "prior_probability_drift":
            healthy_pct = 15  # override for PPD

    if drift_scenario != "none":
        st.markdown(DRIFT_TYPES.get(drift_scenario, ""))

    lci1, lci2 = st.columns(2)
    with lci1:
        label_inject_pct = st.slider(
            "Ground truth label injection rate (%)",
            min_value=0,
            max_value=100,
            value=20,
            step=5,
            key="retrain_label_pct",
            help=(
                f"Default {DEFAULT_LABEL_INJECTION_PCT} % — inject at least enough to reach "
                f"{MIN_LABELED_SIGNALS} labeled signals."
            ),
        )
    with lci2:
        label_choice = st.radio(
            "Label",
            options=["auto", "healthy", "unhealthy"],
            format_func=lambda x: {
                "auto": "Auto (from signal type)",
                "healthy": "Healthy",
                "unhealthy": "Unhealthy",
            }[x],
            horizontal=True,
            key="retrain_label_radio",
        )

    api_key = st.text_input(
        "API Key (X-API-Key header)",
        value="dev-key-12345",
        type="password",
        key="retrain_api_key",
    )

    n_to_label = max(0, round(int(n_signals) * label_inject_pct / 100))
    _label_ok = n_to_label >= MIN_LABELED_SIGNALS
    st.info(
        f"Will generate **{n_signals}** signals → predict all → inject **{n_to_label}** label(s).  \n"
        f"The retraining DAG requires ≥ **{MIN_LABELED_SIGNALS}** labeled signals "
        f"(recommended ≥ **{MIN_LABELED_SIGNALS_RECOMMENDED}**).  "
        + (
            "✅ Sufficient."
            if _label_ok
            else f"⚠️ Not enough — increase signals or injection rate to reach {MIN_LABELED_SIGNALS}."
        )
    )

    if not st.button("⚡ Generate, Predict & Inject Labels", type="primary", key="retrain_gen_btn"):
        return

    # ── Generate signals ─────────────────────────────────────────────────
    gauss_frac = healthy_pct / 100.0

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

    with st.spinner("Generating signals…"):
        rows = generate_batch(
            int(n_signals),
            gaussian_fraction=gauss_frac if drift_scenario == "prior_probability_drift" else 0.7,
            mu_offset=mu_offset,
            width_multiplier=width_mult,
            noise_multiplier=noise_mult,
            height_multiplier=height_mult,
            swap_labels=swap_labels,
            seed=2024,
            include_raw=True,
        )

    # ── Predict each signal via the API ──────────────────────────────────
    from src.ui.views.predictions import _api_post

    progress = st.progress(0, text="Running predictions…")
    results: list[dict] = []
    errors = 0
    _total = len(rows)
    for i, row in enumerate(rows):
        t_vals = row.get("time_values")
        a_vals = row.get("amplitude_values")
        if t_vals is None or a_vals is None or len(t_vals) < 51:  # type: ignore[arg-type]
            errors += 1
            if (i + 1) % 5 == 0 or i == _total - 1:
                progress.progress((i + 1) / _total)
            continue
        payload = {
            "device_id": f"retrain-gen-{i + 1}",
            "time_values": t_vals,
            "amplitude_values": a_vals,
        }
        r = _api_post("/predict", payload, api_key=api_key, silent=True)
        if r:
            r["_shape_type"] = row.get("shape_type", "gaussian")
            results.append(r)
        else:
            errors += 1
        if (i + 1) % 5 == 0 or i == _total - 1:
            progress.progress((i + 1) / _total)
    progress.progress(1.0)
    progress.empty()

    st.success(f"✅ {len(results)}/{len(rows)} predictions stored.")
    if errors:
        st.warning(f"⚠️ {errors} signals skipped (too short or API error).")

    # ── Inject labels ─────────────────────────────────────────────────────
    if label_inject_pct > 0 and results:
        import random as _random

        _rng = _random.Random(42)
        valid_for_label = [r for r in results if r.get("prediction_id") is not None]
        n_actual = min(n_to_label, len(valid_for_label))
        to_label = _rng.sample(valid_for_label, n_actual)

        injected = 0
        inject_errors = 0
        for pred in to_label:
            pid = pred.get("prediction_id")
            if pid is None:
                continue
            if label_choice == "healthy":
                gt = LABEL_HEALTHY
            elif label_choice == "unhealthy":
                gt = LABEL_UNHEALTHY
            else:
                gt = LABEL_HEALTHY if pred.get("_shape_type") == "gaussian" else LABEL_UNHEALTHY
            lbl_payload = {
                "prediction_id": pid,
                "ground_truth_label": gt,
                "label_source": "retraining_ui",
                "injected_by": "retraining_pipeline_tab",
            }
            lbl_res = _api_post("/labels", lbl_payload, api_key=api_key, silent=True)
            if lbl_res:
                injected += 1
            else:
                inject_errors += 1

        st.info(
            f"💾 {injected}/{len(to_label)} ground truth label(s) injected "
            f"({label_inject_pct} % rate)."
            + (f"  ⚠️ {inject_errors} failed." if inject_errors else "")
        )

        # Check whether minimum labeled threshold is now reachable
        if injected >= MIN_LABELED_SIGNALS:
            st.success(
                f"✅ At least {injected} labels injected — the retraining DAG "
                "should now pass the validation step. Switch to the **🚀 DAG Trigger** tab."
            )
        elif injected > 0:
            st.warning(
                f"⚠️ Only {injected} label(s) injected so far. "
                f"The DAG requires {MIN_LABELED_SIGNALS} labeled signals. "
                "Run data generation again or increase the injection rate."
            )
