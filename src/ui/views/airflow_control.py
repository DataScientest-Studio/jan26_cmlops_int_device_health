"""Airflow Control Plane — live DAG status, trigger runs, task metrics.

Connects to the Airflow REST API (v1) to show real-time orchestration
state.  Falls back gracefully when the Airflow webserver is unreachable
(e.g. Docker stack is down).

Key features:
- DAG overview with pause/unpause toggle and per-DAG trigger button
- Recent runs with expandable details and duration
- Task instance statistics (without unsupported ``order_by``)
- DAG pipeline visualisation via Mermaid diagrams
- Health check for metadatabase + scheduler
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from base64 import b64encode
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
import streamlit.components.v1 as components

from src.ui.components.docker_utils import get_service_url
from src.ui.logging_ui import get_ui_logger
from src.ui.styles import get_global_css, hero_section

_logger = get_ui_logger(__name__)

# ── Airflow connection helpers ──────────────────────────────────


def _airflow_base_url() -> str:
    return get_service_url("mlops_airflow", 8080)


def _airflow_auth_header() -> str:
    user = os.environ.get("AIRFLOW_USER", "admin")
    pwd = os.environ.get("AIRFLOW_PASSWORD", "admin")
    token = b64encode(f"{user}:{pwd}".encode()).decode()
    return f"Basic {token}"


def _airflow_get(path: str, *, timeout: int = 10) -> dict | None:
    """GET from Airflow REST API.  Returns parsed JSON or *None*."""
    url = f"{_airflow_base_url()}/api/v1{path}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": _airflow_auth_header(),
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        _logger.warning("Airflow GET {} failed: {}", path, exc)
        return None


def _airflow_post(path: str, payload: dict, *, timeout: int = 15) -> dict | None:
    """POST JSON to Airflow REST API.  Returns parsed JSON or *None*."""
    url = f"{_airflow_base_url()}/api/v1{path}"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": _airflow_auth_header(),
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        _logger.warning("Airflow POST {} failed: {}", path, exc)
        return None


def _airflow_patch(path: str, payload: dict, *, timeout: int = 10) -> dict | None:
    """PATCH JSON to Airflow REST API (used for pause/unpause)."""
    url = f"{_airflow_base_url()}/api/v1{path}"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        method="PATCH",
        headers={
            "Authorization": _airflow_auth_header(),
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        _logger.warning("Airflow PATCH {} failed: {}", path, exc)
        return None


# ── State helpers ───────────────────────────────────────────────

_STATE_ICONS: dict[str, str] = {
    "success": "\U0001f7e2",
    "running": "\U0001f535",
    "failed": "\U0001f534",
    "queued": "\U0001f7e1",
    "up_for_retry": "\U0001f7e0",
    "skipped": "\u26aa",
    "no_status": "\u26ab",
}


def _state_icon(state: str | None) -> str:
    return _STATE_ICONS.get(state or "no_status", "\u26ab")


def _fmt_ts(ts: str | None) -> str:
    """Format an ISO timestamp to a human-readable string."""
    if not ts:
        return "\u2014"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except (ValueError, AttributeError):
        return ts


# ── Mermaid rendering ───────────────────────────────────────────


def _render_mermaid(diagram: str, height: int = 300) -> None:
    """Render a Mermaid diagram via embedded Mermaid.js with auto-sizing."""
    html_content = f"""
    <html><head>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    </head><body style="background:transparent;margin:0;overflow:hidden">
    <div id="diagram" class="mermaid" style="background:#0f172a;border-radius:12px;padding:1rem;height:{height}px;overflow:auto;margin-bottom:2.5rem;">
{diagram.strip()}
    </div>
    <script>
      (function tryRender(n) {{
        if (typeof mermaid !== 'undefined') {{
          mermaid.initialize({{startOnLoad:false,theme:'dark',securityLevel:'loose'}});
          mermaid.run({{nodes:[document.getElementById('diagram')]}});
          var _a = 0;
          var _p = setInterval(function() {{
            _a++;
            var svg = document.querySelector('#diagram svg');
            if (svg || _a > 100) {{
              clearInterval(_p);
              var el = document.getElementById('diagram');
              if (el && window.frameElement) {{
                window.frameElement.style.height = (el.scrollHeight + 32) + 'px';
              }}
            }}
          }}, 100);
        }} else if (n < 100) {{
          setTimeout(function() {{ tryRender(n + 1); }}, 100);
        }}
      }})(0);
    </script>
    </body></html>
    """
    components.html(html_content, height=height, scrolling=False)


# ── Mermaid DAG graphs ──────────────────────────────────────────

_DAG_MERMAID: dict[str, str] = {
    "automated_retraining": (
        "graph LR\n"
        "  A[validate_data] --> B[extract_features]\n"
        "  B --> C[train_challenger]\n"
        "  C --> D[compare_models]\n"
        "  D --> E[request_human_approval]\n"
        "  E --> F[wait_for_human_approval]\n"
        "  F -->|approved| G[promote_unconditional]\n"
        "  F -->|rejected| G2[skip_promotion]\n"
        "  G --> H[send_notification]\n"
        "  G2 --> H"
    ),
    "drift_triggered_retraining": (
        "graph LR\n"
        "  A[validate_drift] --> B[pull_data]\n"
        "  B --> C[retrain]\n"
        "  C --> D[compare_champion]\n"
        "  D --> E[promote_or_reject]\n"
        "  E --> F[update_monitoring]"
    ),
    "evidently_drift_detection": (
        "graph LR\n"
        "  A[fetch_predictions] --> B[load_reference]\n"
        "  B --> C[data_drift_report]\n"
        "  B --> D[model_perf_report]\n"
        "  C --> E[publish_metrics]\n"
        "  D --> E\n"
        "  E --> F[evaluate_thresholds]\n"
        "  F --> G[trigger_alerts]"
    ),
    "model_promotion": (
        "graph LR\n"
        "  A[load_champion] --> C[prepare_test]\n"
        "  B[load_challenger] --> C\n"
        "  C --> D[evaluate_both]\n"
        "  D --> E[compare_metrics]\n"
        "  E --> F[promote_or_reject]\n"
        "  F --> G[update_registry]"
    ),
    "sync_production_data": (
        "graph LR\n"
        "  A[export_predictions] --> D[dvc_add]\n"
        "  B[export_features] --> D\n"
        "  C[export_signals] --> D\n"
        "  D --> E[dvc_push_dagshub]"
    ),
    "database_backup": ("graph LR\n  A[pg_dump] --> B[rotate_backups]\n  B --> C[log_result]"),
    "mlflow_pull_dagshub": (
        "graph LR\n  A[check_prerequisites] --> B[pull_from_dagshub]\n  B --> C[log_result]"
    ),
    "batch_rescoring": (
        "graph LR\n"
        "  A[load_champion] --> B[fetch_predictions]\n"
        "  B --> C[rescore_predictions]\n"
        "  C --> D{dry_run?}\n"
        "  D -->|no| E[write_labels]\n"
        "  D -->|yes| F[preview_only]\n"
        "  E --> G[write_audit_record]"
    ),
}


# ── Pause/unpause callbacks ─────────────────────────────────────


def _do_toggle(dag_id: str, set_paused: bool) -> None:
    """on_click callback: PATCH the DAG and store result in session_state.

    Because this runs as a callback, the PATCH fires *before* the rerun
    re-fetches dags_data — so ``set_paused`` still reflects the user's
    original intent.
    """
    action = "pause" if set_paused else "unpause"
    _logger.info("Airflow DAG {} {}", dag_id, action)
    result = _airflow_patch(f"/dags/{dag_id}", {"is_paused": set_paused})
    new_word = "paused" if set_paused else "active"
    if result is not None:
        _logger.info("Airflow DAG {} {} OK", dag_id, action)
        st.session_state[f"_toggle_msg_{dag_id}"] = (
            "success",
            f"DAG `{dag_id}` is now **{new_word}**.",
        )
    else:
        _logger.warning("Airflow DAG {} {} FAILED", dag_id, action)
        st.session_state[f"_toggle_msg_{dag_id}"] = (
            "error",
            "Failed to update DAG state.",
        )
    # Keep the expander open after the rerun
    st.session_state[f"_exp_open_{dag_id}"] = True


def _do_trigger(dag_id: str) -> None:
    """on_click callback: trigger a DAG run and store result in session_state."""
    note = st.session_state.get(f"note_{dag_id}", "")
    payload: dict = {"conf": {}}
    if note:
        payload["note"] = note
    _logger.info("Triggering Airflow DAG: {}", dag_id)
    result = _airflow_post(f"/dags/{dag_id}/dagRuns", payload)
    if result and result.get("dag_run_id"):
        _logger.info("Airflow DAG {} triggered — run_id={}", dag_id, result["dag_run_id"])
        st.session_state[f"_trigger_msg_{dag_id}"] = (
            "success",
            f"Triggered `{dag_id}` — run `{result['dag_run_id']}`",
            result["dag_run_id"],
        )
    else:
        _logger.warning("Airflow DAG {} trigger FAILED", dag_id)
        st.session_state[f"_trigger_msg_{dag_id}"] = (
            "error",
            "Failed to trigger DAG run.",
            None,
        )
    st.session_state[f"_exp_open_{dag_id}"] = True


# ── Page sections ───────────────────────────────────────────────


def _render_dag_list(dags_data: dict) -> None:
    """Render DAG overview with pause/unpause toggle and trigger button."""
    st.markdown(
        '<div class="section-header">\U0001f4cb Registered DAGs</div>',
        unsafe_allow_html=True,
    )
    dags = dags_data.get("dags", [])
    if not dags:
        st.info("No DAGs found in the Airflow instance.")
        return

    for dag in sorted(dags, key=lambda d: d.get("dag_id", "")):
        dag_id = dag.get("dag_id", "unknown")
        is_paused = dag.get("is_paused", True)
        schedule = dag.get("schedule_interval") or dag.get("timetable_description", "\u2014")
        owners_list = dag.get("owners", [])
        owner = ", ".join(owners_list) if owners_list else "\u2014"

        pause_icon = "\u23f8\ufe0f" if is_paused else "\U0001f7e2"
        exp_open = st.session_state.pop(f"_exp_open_{dag_id}", False)
        with st.expander(
            f"{pause_icon} **{dag_id}**  \u00b7  `{schedule}`  \u00b7  {owner}",
            expanded=exp_open,
        ):
            # Show any pending toggle message from the callback
            msg_key = f"_toggle_msg_{dag_id}"
            if msg_key in st.session_state:
                msg_type, msg_text = st.session_state.pop(msg_key)
                if msg_type == "success":
                    st.success(msg_text)
                else:
                    st.error(msg_text)

            # Show any pending trigger message
            trigger_key = f"_trigger_msg_{dag_id}"
            if trigger_key in st.session_state:
                t_type, t_text, t_run_id = st.session_state.pop(trigger_key)
                if t_type == "success":
                    st.success(t_text)
                    if t_run_id:
                        _poll_run_status(dag_id, t_run_id)
                else:
                    st.error(t_text)

            # -- Pause / Unpause toggle --
            col_toggle, col_trigger, col_note = st.columns([1, 1, 2])
            with col_toggle:
                label = "\u25b6\ufe0f Unpause" if is_paused else "\u23f8\ufe0f Pause"
                st.button(
                    label,
                    key=f"toggle_{dag_id}",
                    on_click=_do_toggle,
                    args=(dag_id, not is_paused),
                )

            # -- Trigger button --
            with col_note:
                st.text_input("Run note", key=f"note_{dag_id}", placeholder="optional")
            with col_trigger:
                st.button(
                    "\U0001f680 Trigger",
                    key=f"trigger_{dag_id}",
                    on_click=_do_trigger,
                    args=(dag_id,),
                )

            # -- Mermaid diagram --
            mermaid = _DAG_MERMAID.get(dag_id)
            if mermaid:
                st.markdown("**Pipeline graph:**")
                _render_mermaid(mermaid, height=200)


def _poll_run_status(dag_id: str, dag_run_id: str, max_polls: int = 6) -> None:
    """Poll a DAG run status a few times to show progress."""
    status_placeholder = st.empty()
    for i in range(max_polls):
        time.sleep(2)
        data = _airflow_get(f"/dags/{dag_id}/dagRuns/{dag_run_id}")
        if data is None:
            status_placeholder.warning("Lost connection to Airflow.")
            return
        state = data.get("state", "unknown")
        icon = _state_icon(state)
        elapsed = ""
        if data.get("start_date"):
            try:
                started = datetime.fromisoformat(data["start_date"].replace("Z", "+00:00"))
                elapsed = f" ({(datetime.now(started.tzinfo) - started).total_seconds():.0f}s)"
            except (ValueError, TypeError):
                pass
        status_placeholder.info(f"{icon} **{state}**{elapsed} \u2014 polling {i + 1}/{max_polls}")
        if state in ("success", "failed"):
            if state == "success":
                status_placeholder.success(f"\U0001f7e2 DAG run completed successfully!{elapsed}")
            else:
                status_placeholder.error(f"\U0001f534 DAG run failed.{elapsed}")
            return
    status_placeholder.info(
        f"Run still in progress. Check the [Airflow UI]({_airflow_base_url()}) for live updates."
    )


def _render_recent_runs(dag_id: str | None = None) -> None:
    """Fetch and render recent DAG runs."""
    st.markdown(
        '<div class="section-header">\U0001f552 Recent DAG Runs</div>',
        unsafe_allow_html=True,
    )

    path = "/dags/~/dagRuns?limit=25&order_by=-start_date"
    if dag_id and dag_id != "All DAGs":
        path = f"/dags/{dag_id}/dagRuns?limit=15&order_by=-start_date"

    data = _airflow_get(path)
    if data is None:
        st.warning("Could not fetch DAG runs \u2014 Airflow API unreachable.")
        return

    runs = data.get("dag_runs", [])
    if not runs:
        st.info("No DAG runs recorded yet.")
        return

    # Sort descending: latest runs first
    def _run_sort_key(run: dict) -> str:
        return run.get("start_date") or run.get("execution_date") or ""

    runs = sorted(runs, key=_run_sort_key, reverse=True)

    for run in runs:
        state = run.get("state", "no_status")
        icon = _state_icon(state)
        rid = run.get("dag_run_id", "\u2014")
        dag = run.get("dag_id", "\u2014")
        started = _fmt_ts(run.get("start_date"))
        ended = _fmt_ts(run.get("end_date"))
        duration = "\u2014"
        if run.get("start_date") and run.get("end_date"):
            try:
                s = datetime.fromisoformat(run["start_date"].replace("Z", "+00:00"))
                e = datetime.fromisoformat(run["end_date"].replace("Z", "+00:00"))
                secs = (e - s).total_seconds()
                duration = f"{secs:.1f}s"
            except (ValueError, TypeError):
                pass

        with st.expander(f"{icon} {dag}  \u00b7  {state}  \u00b7  {started}", expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.metric("State", f"{icon} {state}")
            c2.metric("Duration", duration)
            c3.metric("Run ID", rid[:30])
            st.markdown(f"**Started:** {started}  \u00b7  **Ended:** {ended}")
            if run.get("note"):
                st.markdown(f"**Note:** {run['note']}")


def _render_task_stats(all_dag_ids: list[str] | None = None) -> None:
    """Show aggregate task-instance statistics."""
    st.markdown(
        '<div class="section-header">\U0001f4ca Task Instance Statistics</div>',
        unsafe_allow_html=True,
    )
    data = _airflow_get("/dags/~/dagRuns/~/taskInstances?limit=2000")
    if data is None:
        st.warning("Could not fetch task instances \u2014 Airflow API unreachable.")
        return

    tasks = data.get("task_instances", [])
    if not tasks:
        st.info("No task instances found.")
        return

    # Count by state
    counts: dict[str, int] = {}
    for t in tasks:
        s = t.get("state") or "no_status"
        counts[s] = counts.get(s, 0) + 1

    # Show ALL state metrics in a single row
    num_states = len(counts)
    cols = st.columns(max(num_states, 1))
    for i, (state, count) in enumerate(sorted(counts.items(), key=lambda x: -x[1])):
        cols[i].metric(f"{_state_icon(state)} {state}", count)

    # Per-DAG breakdown
    dag_counts: dict[str, dict[str, int]] = {}
    for t in tasks:
        d = t.get("dag_id", "unknown")
        s = t.get("state") or "no_status"
        dag_counts.setdefault(d, {})
        dag_counts[d][s] = dag_counts[d].get(s, 0) + 1

    st.markdown("**Per-DAG breakdown:**")
    for dag_name, states in sorted(dag_counts.items()):
        parts = [f"{_state_icon(s)} {s}: {c}" for s, c in sorted(states.items())]
        st.markdown(f"- **{dag_name}** \u2014 {', '.join(parts)}")

    # Show DAGs with NO task instances
    if all_dag_ids:
        missing = sorted(set(all_dag_ids) - set(dag_counts.keys()))
        if missing:
            st.markdown("**DAGs with no task instances (never triggered or scheduled):**")
            for dag_name in missing:
                st.markdown(f"- **{dag_name}** \u2014 \u26aa no runs yet")
            st.caption(
                "💡 DAGs appear here if Airflow knows about them but no task instances "
                "have been recorded yet. This can happen when a DAG was just triggered "
                "and tasks are still queued, when the DAG has only been created but never "
                "scheduled, or when the Airflow API task-instance limit (2000) has been "
                "reached. Check the Recent Runs tab or open Airflow UI for details."
            )


def _render_health_check() -> None:
    """Quick health indicator for the Airflow webserver."""
    data = _airflow_get("/health")
    if data is None:
        _logger.warning("Airflow health-check: webserver unreachable")
        st.error("❌ Airflow webserver is **unreachable**.")
        return

    meta_db = data.get("metadatabase", {}).get("status", "unknown")
    scheduler = data.get("scheduler", {}).get("status", "unknown")
    _logger.debug("Airflow health: metadatabase={} scheduler={}", meta_db, scheduler)
    sched_latest = data.get("scheduler", {}).get("latest_scheduler_heartbeat", "")
    c1, c2, c3 = st.columns(3)
    c1.metric("Metadatabase", f"{'\U0001f7e2' if meta_db == 'healthy' else '\U0001f534'} {meta_db}")
    c2.metric(
        "Scheduler",
        f"{'\U0001f7e2' if scheduler == 'healthy' else '\U0001f534'} {scheduler}",
    )
    if sched_latest:
        c3.metric("Last Heartbeat", _fmt_ts(sched_latest))


# ── Main render ─────────────────────────────────────────────────


def _detect_mode() -> str:
    """Detect deployment mode (local or cloud).

    Priority: .current_mode file → DEPLOYMENT_MODE env var → default local.
    File takes priority because DEPLOYMENT_MODE is baked in at Streamlit startup
    and becomes stale when the user switches modes from within the UI.
    """
    # 1. .current_mode file — always updated on mode switch (authoritative)
    mode_file = Path(_PROJECT_ROOT) / ".current_mode"
    if mode_file.exists():
        val = mode_file.read_text().strip()
        if val in ("local", "cloud", "k8s"):
            return val
    # 2. DEPLOYMENT_MODE env var — set by make ui at startup, may be stale
    env_mode = os.environ.get("DEPLOYMENT_MODE", "").strip()
    if env_mode in ("local", "cloud", "k8s"):
        return env_mode
    return "local"


def render() -> None:
    """Render the Airflow Control Plane page."""
    try:
        _render_content()
    except Exception:
        _logger.exception("Unhandled exception in airflow_control.render()")
        raise


def _render_content() -> None:
    """Inner render — separated so exceptions are caught and logged by render()."""
    st.markdown(get_global_css(), unsafe_allow_html=True)
    st.markdown(
        hero_section(
            "\u2708\ufe0f Airflow Control Plane",
            "Live orchestration status \u2014 trigger DAG runs, monitor task "
            "execution, and view pipeline health.",
        ),
        unsafe_allow_html=True,
    )

    # In local mode, Airflow is intentionally disabled (scale: 0)
    mode = _detect_mode()
    if mode == "local":
        st.info(
            "**Airflow is disabled in local sandbox mode.**\n\n"
            "In local mode, the Airflow container is scaled to zero "
            "(`scale: 0` in `docker-compose.local.yml`). This is by design — "
            "local mode focuses on data exploration and model experimentation "
            "without orchestration overhead.\n\n"
            "Airflow orchestration is available in **cloud mode** where it manages "
            "scheduled retraining, drift detection, data sync, and backup DAGs.\n\n"
            "Switch to cloud mode with `make cloud` to enable Airflow."
        )
        st.markdown("---")
        st.markdown(
            "**Available in cloud mode:**\n"
            "- 🔄 Automated retraining (weekly + drift-triggered)\n"
            "- 🔍 EvidentlyAI drift detection (daily)\n"
            "- ☁️ Production data sync to DagsHub (daily)\n"
            "- 🗄️ Database backups (daily)\n"
            "- 🏆 Champion/Challenger model promotion"
        )
        return

    # -- Health check row --
    _render_health_check()

    # -- Load DAG list (shared across sections) --
    dags_data = _airflow_get("/dags?limit=50")
    if dags_data is None:
        _is_k8s = mode == "k8s"
        if _is_k8s:
            st.warning(
                "⚠️ Cannot connect to the Airflow REST API.  \n"
                "Make sure the Kubernetes cluster is running and port-forwards are active:  \n"
                "```\nmake k8s-up      # start the cluster\n"
                "make k8s-ports   # forward Airflow → localhost:8080\n```"
            )
        else:
            st.warning(
                "⚠️ Cannot connect to the Airflow REST API.  "
                "Make sure the Docker stack is running (`make up` or `docker compose up -d`)."
            )
        st.markdown(
            f"**Airflow URL:** `{_airflow_base_url()}`  \n"
            f"[Open Airflow UI ↗]({_airflow_base_url()})"
        )
        return

    # Use st.radio (keyed) instead of st.tabs() to prevent tab-jump on rerun.
    _AF_TABS = ["\U0001f4cb DAGs", "\U0001f552 Recent Runs", "\U0001f4ca Task Stats"]
    active_af = st.radio(
        "Airflow tab",
        _AF_TABS,
        horizontal=True,
        key="_af_tab",
        label_visibility="collapsed",
    )
    st.markdown(
        "<hr style='margin:0 0 1rem 0;border-color:#334155;'>",
        unsafe_allow_html=True,
    )

    if active_af == _AF_TABS[0]:
        _render_dag_list(dags_data)
    elif active_af == _AF_TABS[1]:
        dag_ids = ["All DAGs"] + [d["dag_id"] for d in dags_data.get("dags", [])]
        selected_dag = st.selectbox("Filter by DAG", dag_ids, key="airflow_runs_filter")
        _render_recent_runs(selected_dag if selected_dag != "All DAGs" else None)
    else:
        all_dag_ids = [d["dag_id"] for d in dags_data.get("dags", [])]
        _render_task_stats(all_dag_ids=all_dag_ids)

    # -- Footer link --
    st.markdown("---")
    st.markdown(f"[Open full Airflow UI \u2197]({_airflow_base_url()})")
