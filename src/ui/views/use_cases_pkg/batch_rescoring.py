"""Batch Re-Scoring use case tab — Task 3.

Shows rescoring_runs history from the database and allows the user to trigger
the batch_rescoring Airflow DAG manually (cloud mode) or display a dry-run
summary in local mode.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
from pathlib import Path

import streamlit as st

from src.ui.components.docker_utils import get_host
from src.ui.logging_ui import get_ui_logger

_logger = get_ui_logger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _load_rescoring_history() -> list[dict]:
    """Load rescoring_runs table from PostgreSQL or SQLite."""
    pg_url = os.environ.get("DATABASE_URL", "")
    if pg_url:
        try:
            import psycopg2

            conn = psycopg2.connect(pg_url)
            with conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT id, model_version, rescored_at, n_predictions, "
                    "n_changed, change_rate, triggered_by, status "
                    "FROM rescoring_runs ORDER BY id DESC LIMIT 100"
                )
                cols = [d[0] for d in cur.description]
                rows = [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]
            conn.close()
            return rows
        except Exception as exc:
            st.warning(f"PostgreSQL unavailable ({exc}) — using SQLite fallback")

    db_path = _PROJECT_ROOT / "data" / "mlops.db"
    if not db_path.exists():
        return []
    with contextlib.closing(sqlite3.connect(str(db_path))) as con:
        con.row_factory = sqlite3.Row
        rows = [
            dict(r)
            for r in con.execute(
                "SELECT id, model_version, rescored_at, n_predictions, "
                "n_changed, change_rate, triggered_by, status "
                "FROM rescoring_runs ORDER BY id DESC LIMIT 100"
            ).fetchall()
        ]
    return rows


def _trigger_rescoring_dag(lookback_days: int, dry_run: bool) -> str | None:
    """Trigger the batch_rescoring Airflow DAG via its REST API.

    Returns the dag_run_id on success, None on failure.
    """
    import json
    import urllib.error
    import urllib.request

    airflow_url = os.environ.get("AIRFLOW_API_URL", f"http://{get_host()}:8081")
    url = f"{airflow_url}/api/v1/dags/batch_rescoring/dagRuns"
    payload = json.dumps(
        {
            "conf": {"lookback_days": lookback_days, "dry_run": dry_run},
        }
    ).encode()

    airflow_user = os.environ.get("AIRFLOW_USERNAME", "admin")
    airflow_pass = os.environ.get("AIRFLOW_PASSWORD", "admin")
    import base64

    token = base64.b64encode(f"{airflow_user}:{airflow_pass}".encode()).decode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Basic {token}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode())
            run_id: str = body.get("dag_run_id", "?")
            st.success(f"✅ Batch re-scoring DAG triggered (run_id: `{run_id}`)")
            return run_id
    except urllib.error.HTTPError as exc:
        st.error(f"Airflow API error {exc.code}: {exc.read().decode()}")
    except Exception as exc:
        st.error(f"Could not reach Airflow: {exc}")
    return None


def render_batch_rescoring_tab(mode: str) -> None:
    """Render the Batch Re-Scoring use case tab."""
    st.markdown(
        "<div class='section-header'>⏮️ Batch Re-Scoring</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        After promoting a new champion model, **batch re-scoring** re-runs predictions on
        historical signals so the database always reflects the current model's view.
        The `batch_rescoring` Airflow DAG handles this automatically.

        | What it does | Why it matters |
        |:-------------|:---------------|
        | Re-scores last N days of predictions | Old predictions used a stale model |
        | Writes audit record to `rescoring_runs` | Full traceability of re-scoring events |
        | Computes change rate | Quantifies impact of new model |
        """
    )

    st.divider()

    # ── Trigger section ──────────────────────────────────────────────────────
    st.subheader("Trigger Re-Scoring")

    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        lookback_days = st.number_input(
            "Lookback (days)", min_value=1, max_value=365, value=30, step=1
        )
    with col2:
        dry_run = st.checkbox("Dry run (no DB writes)", value=False)

    if mode == "cloud":
        if st.button("▶ Trigger Airflow DAG", type="primary"):
            run_id = _trigger_rescoring_dag(int(lookback_days), dry_run)
            if run_id:
                # Poll DAG status and auto-refresh the page when it completes.
                airflow_url = os.environ.get("AIRFLOW_API_URL", f"http://{get_host()}:8081")
                import base64 as _b64
                import json as _json
                import time
                import urllib.request as _req

                _user = os.environ.get("AIRFLOW_USERNAME", "admin")
                _pass = os.environ.get("AIRFLOW_PASSWORD", "admin")
                _tok = _b64.b64encode(f"{_user}:{_pass}".encode()).decode()
                _status_url = f"{airflow_url}/api/v1/dags/batch_rescoring/dagRuns/{run_id}"
                _headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Basic {_tok}",
                }

                def _get_dag_state() -> str:
                    try:
                        _r = _req.Request(_status_url, headers=_headers, method="GET")
                        with _req.urlopen(_r, timeout=10) as _resp:
                            return _json.loads(_resp.read().decode()).get("state", "unknown")
                    except Exception:
                        return "unknown"

                _terminal_states = {"success", "failed", "upstream_failed"}
                _br_max = 60  # max ~5 min
                with st.status("⏳ Waiting for batch_rescoring to finish…", expanded=True) as _s:
                    _br_prog = st.progress(0.0)
                    for _poll in range(_br_max):  # max ~5 min
                        time.sleep(5)
                        _state = _get_dag_state()
                        _br_prog.progress(min((_poll + 1) / _br_max, 0.95))
                        _s.write(f"Poll {_poll + 1}: **{_state}**")
                        if _state in _terminal_states:
                            break
                    _br_prog.progress(1.0)
                    if _state == "success":
                        _s.update(label="✅ Batch re-scoring complete", state="complete")
                    else:
                        _s.update(label=f"Batch re-scoring ended: {_state}", state="error")
                st.rerun()  # refresh history table to show latest run
    else:
        st.info(
            "**Local mode:** Airflow DAG triggering requires cloud mode (`make cloud-up`).  "
            f"Use the Airflow UI at http://{get_host()}:8081 to trigger `batch_rescoring` manually."
        )

    st.divider()

    # ── History ──────────────────────────────────────────────────────────────
    st.subheader("Re-Scoring History")
    history = _load_rescoring_history()

    if not history:
        st.info("No re-scoring runs recorded yet. Trigger the DAG or run it from Airflow.")
        return

    import pandas as pd

    df = pd.DataFrame(history)
    if "change_rate" in df.columns:
        df["change_rate"] = df["change_rate"].apply(lambda x: f"{x:.1%}" if x is not None else "–")
    st.dataframe(df, width="stretch")

    # Summary metrics
    total_runs = len(history)
    avg_change = sum(
        h.get("change_rate") or 0.0 for h in history if isinstance(h.get("change_rate"), float)
    ) / max(1, sum(1 for h in history if isinstance(h.get("change_rate"), float)))
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Total Re-Scoring Runs", total_runs)
    mc2.metric("Avg Change Rate", f"{avg_change:.1%}")
    _last_run = history[0].get("rescored_at", "–") if history else "–"
    # PostgreSQL returns datetime objects; st.metric requires str/number
    if hasattr(_last_run, "strftime"):
        _last_run = _last_run.strftime("%Y-%m-%d %H:%M")
    elif _last_run and _last_run != "–":
        _last_run = str(_last_run)[:16]
    mc3.metric(
        "Last Run",
        _last_run,
    )
