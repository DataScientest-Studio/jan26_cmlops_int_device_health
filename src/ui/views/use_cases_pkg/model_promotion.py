"""Tab — Model Promotion & Lifecycle Management.

Provides a Streamlit UI for manually managing model versions in the MLflow
registry via the ``model_promotion`` Airflow DAG:

  * **Promote** — sets a chosen version as the ``champion`` alias.
  * **Rollback** — re-assigns ``champion`` to a previous version.
  * **Archive** — tags old versions (no active alias) older than N days.

Cloud-mode only: the DAG requires Airflow and a remote MLflow registry.
"""

from __future__ import annotations

import os
import time

import streamlit as st

from src.ui.components.docker_utils import get_host
from src.ui.logging_ui import get_ui_logger

_logger = get_ui_logger(__name__)

from ._common import (
    PROJECT_ROOT,
    fetch_champion_info,
    get_mlflow_client,
    get_model_name,
)

_AIRFLOW_API = os.environ.get("AIRFLOW_API_URL", f"http://{get_host()}:8081") + "/api/v1"
_DAG_ID = "model_promotion"

# ---------------------------------------------------------------------------
# Airflow helpers (same pattern as retraining_pipeline.py)
# ---------------------------------------------------------------------------


def _af_auth() -> tuple[str, str]:
    user = os.environ.get("AIRFLOW_USER", "admin")
    passwd = os.environ.get("AIRFLOW_PASSWORD", "admin")
    if passwd == "admin":
        sf = PROJECT_ROOT / ".env.secrets"
        if sf.exists():
            for line in sf.read_text().splitlines():
                line = line.strip()
                if line.startswith("AIRFLOW_PASSWORD="):
                    passwd = line.split("=", 1)[1].strip()
                elif line.startswith("AIRFLOW_USER="):
                    user = line.split("=", 1)[1].strip()
    return user, passwd


def _af_post(path: str, payload: dict | None = None) -> dict | None:
    import requests

    try:
        resp = requests.post(
            f"{_AIRFLOW_API}{path}",
            json=payload or {},
            auth=_af_auth(),
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


def _af_get(path: str) -> dict | None:
    import requests

    try:
        resp = requests.get(
            f"{_AIRFLOW_API}{path}",
            auth=_af_auth(),
            timeout=10,
        )
        if resp.ok:
            return resp.json()
    except Exception as exc:
        _logger.warning("Airflow GET {} failed: {}", path, exc)
    return None


# ---------------------------------------------------------------------------
# MLflow helpers
# ---------------------------------------------------------------------------


def _fetch_all_versions(mode: str) -> list:
    """Return all registered model versions sorted descending."""
    try:
        client, _ = get_mlflow_client()
        model_name = get_model_name(mode)
        versions = client.search_model_versions(f"name='{model_name}'")
        return sorted(versions, key=lambda v: int(v.version), reverse=True)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# DAG trigger + polling
# ---------------------------------------------------------------------------


def _trigger_and_monitor(conf: dict) -> None:
    """Trigger the model_promotion DAG with ``conf`` and poll until done."""
    action = conf.get("action", "promote")
    _logger.info("Triggering model_promotion DAG — action={} conf={}", action, conf)
    result = _af_post(f"/dags/{_DAG_ID}/dagRuns", {"conf": conf})
    if result is None:
        return

    run_id = result.get("dag_run_id") or result.get("run_id", "")
    _logger.info("model_promotion DAG triggered — run_id={}", run_id)
    st.success(f"✅ DAG triggered — run ID: `{run_id}`")

    if not run_id:
        return

    st.markdown("**Monitoring run…** (refreshes every 5 s for up to 3 min)")
    status_slot = st.empty()
    progress_slot = st.empty()

    _terminal = {"success", "failed", "upstream_failed"}
    deadline = time.time() + 180  # 3-minute timeout

    progress_val = 0.0
    while time.time() < deadline:
        data = _af_get(f"/dags/{_DAG_ID}/dagRuns/{run_id}")
        state = (data or {}).get("state", "unknown")

        progress_val = min(progress_val + 0.08, 0.95)
        progress_slot.progress(progress_val, text=f"State: **{state}**")
        status_slot.info(f"Run state: **{state}**")

        if state in _terminal:
            break
        time.sleep(5)

    # Final status
    progress_slot.empty()
    data = _af_get(f"/dags/{_DAG_ID}/dagRuns/{run_id}") or {}
    final_state = data.get("state", "unknown")

    if final_state == "success":
        status_slot.success(f"🎉 DAG completed successfully  (state: `{final_state}`)")
        # Store result so it survives the rerun, then refresh to show updated registry
        st.session_state["_mp_action_banner"] = (
            "success",
            "🎉 Action completed — registry updated.",
        )
        st.rerun()
    elif final_state in ("failed", "upstream_failed"):
        status_slot.error(f"❌ DAG run {final_state}. Check Airflow logs for details.")
        st.session_state["_mp_action_banner"] = (
            "error",
            f"❌ DAG run {final_state}. Check Airflow logs.",
        )
        st.rerun()
    else:
        status_slot.warning(
            f"⏳ Run still in state `{final_state}` after timeout. "
            "Check the Airflow UI for the final result."
        )


# ---------------------------------------------------------------------------
# Sub-action renderers
# ---------------------------------------------------------------------------


def _render_promote(versions: list, champion_version: str | None, mode: str) -> None:
    st.markdown("#### 🚀 Promote a Version to Champion")
    st.markdown(
        "Sets the **`champion`** alias on the selected version.  "
        "The API container will automatically serve the new champion on its next request."
    )

    if not versions:
        st.warning("No model versions found in the registry.")
        return

    version_nums = [int(v.version) for v in versions]
    # Default to the latest non-champion version, or latest overall
    _default_idx = 0
    for i, v in enumerate(versions):
        v_aliases = list(getattr(v, "aliases", None) or [])
        if "champion" not in v_aliases:
            _default_idx = i
            break

    selected_v = st.selectbox(
        "Select version to promote",
        version_nums,
        index=_default_idx,
        format_func=lambda v: (
            f"v{v}" + (" 🏆 current champion" if str(v) == str(champion_version) else "")
        ),
        key="mp_promote_version",
    )

    # Show metrics for selected version
    _show_version_metrics(versions, selected_v)

    force = st.checkbox(
        "Force promotion (bypass F1 improvement check)",
        value=False,
        key="mp_promote_force",
        help="Skip the challenger-vs-champion F1 comparison. Use with care.",
    )

    if st.button("🚀 Promote to Champion", type="primary", key="mp_promote_btn"):
        _trigger_and_monitor(
            {"action": "promote", "model_version": int(selected_v), "force": force}
        )


def _render_rollback(versions: list, champion_version: str | None, mode: str) -> None:
    st.markdown("#### ⏪ Rollback Champion to a Previous Version")
    st.markdown(
        "Re-assigns the **`champion`** alias to a selected previous version.  "
        "Useful when the current champion regresses in production."
    )

    if not versions:
        st.warning("No model versions found in the registry.")
        return

    version_nums = [int(v.version) for v in versions]
    _default_idx = 1 if len(version_nums) > 1 else 0  # second-latest by default

    selected_v = st.selectbox(
        "Rollback to version",
        version_nums,
        index=_default_idx,
        format_func=lambda v: (
            f"v{v}" + (" 🏆 current champion" if str(v) == str(champion_version) else "")
        ),
        key="mp_rollback_version",
    )

    _show_version_metrics(versions, selected_v)

    if champion_version and str(selected_v) == str(champion_version):
        st.info("This is already the current champion — no action needed.")

    if st.button("⏪ Rollback to this Version", type="primary", key="mp_rollback_btn"):
        _trigger_and_monitor(
            {"action": "rollback", "model_version": int(selected_v), "force": False}
        )


def _render_archive(versions: list, champion_version: str | None, mode: str) -> None:
    st.markdown("#### 🗄️ Archive Model Versions")
    st.markdown(
        "Tags model versions **without** an active alias (not champion/challenger) "
        "as archived.  Archived models remain in the registry but are excluded "
        "from serving and the active model pool."
    )

    archive_mode = st.radio(
        "Archive mode",
        ["By retention period", "By explicit version"],
        horizontal=True,
        key="mp_archive_mode",
    )

    if archive_mode == "By explicit version":
        if not versions:
            st.warning("No model versions found in the registry.")
            return

        version_nums = [int(v.version) for v in versions]
        # Default to first non-champion version
        _default_idx = 0
        for i, v in enumerate(versions):
            if str(v.version) != str(champion_version):
                _default_idx = i
                break

        selected_v = st.selectbox(
            "Select version to archive",
            version_nums,
            index=_default_idx,
            format_func=lambda v: (
                f"v{v}" + (" 🏆 current champion" if str(v) == str(champion_version) else "")
            ),
            key="mp_archive_version",
        )

        _show_version_metrics(versions, selected_v)

        if str(selected_v) == str(champion_version):
            st.error("⚠️ Cannot archive the current champion. Promote another version first.")
        else:
            st.warning(
                f"⚠️ This will tag **v{selected_v}** as archived and remove the "
                "challenger alias if it holds one.  The version stays in the registry."
            )
            if st.button("🗄️ Archive this Version", type="primary", key="mp_archive_explicit_btn"):
                _trigger_and_monitor(
                    {
                        "action": "archive",
                        "model_version": None,
                        "force": False,
                        "retention_days": 90,
                        "archive_model_version": int(selected_v),
                    }
                )
    else:
        retention_days = st.number_input(
            "Retention days",
            min_value=1,
            max_value=3650,
            value=90,
            step=7,
            key="mp_archive_retention",
            help="Versions without an alias older than this many days will be archived.",
        )

        st.warning(
            f"⚠️ This will archive all unaliased versions older than **{retention_days} days**.  "
            "Reversible via the MLflow UI but not in Airflow."
        )

        if st.button("🗄️ Archive by Retention Period", type="primary", key="mp_archive_btn"):
            _trigger_and_monitor(
                {
                    "action": "archive",
                    "model_version": None,
                    "force": False,
                    "retention_days": int(retention_days),
                    "archive_model_version": None,
                }
            )


# ---------------------------------------------------------------------------
# Version metrics helper
# ---------------------------------------------------------------------------


def _show_version_metrics(versions: list, selected_version: int) -> None:
    """Show a compact metrics row for the selected model version."""
    mv = next((v for v in versions if int(v.version) == selected_version), None)
    if mv is None:
        return

    try:
        client, _ = get_mlflow_client()
        model_name = get_model_name()
        run = client.get_run(mv.run_id)
        m = run.data.metrics
        p = run.data.params
        alias_map = _build_alias_map(client, model_name)
        aliases = alias_map.get(mv.version, [])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Version", f"v{mv.version}")
        c2.metric("Classifier", p.get("classifier_type", "—"))
        c3.metric("Test F1", f"{m.get('test_f1_score', 0):.4f}")
        c4.metric(
            "Aliases",
            ", ".join(aliases) if aliases else "none",
        )
    except Exception:
        pass  # metrics not critical, don't block the UI


# ---------------------------------------------------------------------------
# Registry overview helper
# ---------------------------------------------------------------------------


def _build_alias_map(client, model_name: str) -> dict:
    """Return {version_str: [aliases]} by querying known aliases via the API.

    ``search_model_versions`` from DagsHub does NOT populate the ``aliases``
    field on returned objects.  Instead we call ``get_model_version_by_alias``
    for each well-known alias and build a lookup map.
    """
    alias_map: dict = {}  # version_str -> list[str]
    for alias in ("champion", "challenger"):
        try:
            mv = client.get_model_version_by_alias(model_name, alias)
            alias_map.setdefault(mv.version, []).append(alias)
        except Exception:
            pass
    return alias_map


def _stage_from_aliases(aliases: list) -> str:
    """Derive a human-readable stage label from alias list."""
    if "champion" in aliases:
        return "Production"
    if "challenger" in aliases:
        return "Staging"
    return "Archived"


def _render_registry_overview(versions: list, champion_version: str | None) -> None:
    """Compact table of all registered versions with their key metrics."""
    if not versions:
        st.info("No versions found in the registry.")
        return

    import pandas as pd

    # Build alias map once (search_model_versions doesn't return aliases on DagsHub)
    try:
        client, _ = get_mlflow_client()
        model_name = get_model_name()
        alias_map = _build_alias_map(client, model_name)
    except Exception:
        alias_map = {}
        client = None

    rows = []
    for v in versions:
        aliases = alias_map.get(v.version, [])
        try:
            if client is not None:
                run = client.get_run(v.run_id)
                f1 = run.data.metrics.get("test_f1_score", None)
                clf = run.data.params.get("classifier_type", "—")
            else:
                f1, clf = None, "—"
        except Exception:
            f1, clf = None, "—"

        rows.append(
            {
                "Version": f"v{v.version}",
                "Aliases": ", ".join(aliases) if aliases else "—",
                "Stage": _stage_from_aliases(aliases),
                "Classifier": clf,
                "Test F1": f"{f1:.4f}" if f1 is not None else "—",
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def render_model_promotion_tab(mode: str) -> None:
    """Render the Model Promotion & Lifecycle Management use-case tab."""
    from src.ui.views.use_cases_pkg._common import SECTION_CSS

    st.markdown(SECTION_CSS, unsafe_allow_html=True)

    # ── Persistent action banner (survives rerun) ─────────────────────────────
    banner = st.session_state.pop("_mp_action_banner", None)
    if banner:
        level, msg = banner
        if level == "success":
            st.success(msg)
        elif level == "error":
            st.error(msg)
        else:
            st.info(msg)

    st.markdown(
        '<div class="signal-section-general">'
        "<strong>🏆 Model Promotion & Lifecycle Management</strong><br/>"
        "Manually promote, rollback, or archive model versions in the MLflow registry "
        "via the <code>model_promotion</code> Airflow DAG."
        "</div>",
        unsafe_allow_html=True,
    )

    if mode != "cloud":
        st.info(
            "ℹ️ **Cloud mode required.**  "
            "Model promotion uses the DagsHub MLflow registry and the Airflow scheduler, "
            "which are only available in cloud mode.  "
            "Switch to cloud mode via the sidebar to use this feature."
        )
        return

    # ── Current champion overview ────────────────────────────────────────
    st.markdown("#### 🏆 Current Registry State")

    versions = _fetch_all_versions(mode)
    champion_version: str | None = None

    try:
        client, _ = get_mlflow_client()
        champ_mv, champ_run = fetch_champion_info(client, mode)

        if champ_mv is not None:
            champion_version = str(champ_mv.version)
            champ_metrics = champ_run.data.metrics if champ_run else {}
            champ_params = champ_run.data.params if champ_run else {}

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Champion Version", f"v{champ_mv.version}")
            c2.metric("Classifier", champ_params.get("classifier_type", "—"))
            c3.metric("Test F1", f"{champ_metrics.get('test_f1_score', 0):.4f}")
            c4.metric("Test Accuracy", f"{champ_metrics.get('test_accuracy', 0):.4f}")
        else:
            st.warning("No champion version found in the registry.")
    except Exception as exc:
        st.error(f"Cannot fetch champion info: {exc}")

    if versions:
        with st.expander(f"📋 All registered versions ({len(versions)} total)", expanded=False):
            _render_registry_overview(versions, champion_version)

    st.markdown("---")

    # ── Action selector ──────────────────────────────────────────────────
    st.markdown("#### ⚙️ Lifecycle Action")
    action = st.radio(
        "Action",
        ["🚀 Promote", "⏪ Rollback", "🗄️ Archive"],
        horizontal=True,
        key="mp_action",
        label_visibility="collapsed",
    )

    st.markdown("")

    if action == "🚀 Promote":
        _render_promote(versions, champion_version, mode)
    elif action == "⏪ Rollback":
        _render_rollback(versions, champion_version, mode)
    else:
        _render_archive(versions, champion_version, mode)
