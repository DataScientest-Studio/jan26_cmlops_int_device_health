"""App Console — live tail of the Streamlit UI application log.

This page displays the contents of the rotating log file written by
``src.ui.logging_ui``.  It shows the last N lines of the file and
auto-refreshes when the user clicks the Refresh button.

Key limitations (Streamlit architecture)
-----------------------------------------
Streamlit only executes the currently-visible page while the user is on it.
The log *file* is written continuously by loguru's file sink (even when the
user is on a different page), so no entries are lost.  When the user navigates
here, all log lines accumulated since startup are displayed.

Auto-refresh every few seconds is possible with the ``streamlit-autorefresh``
package.  It is wired in here when available but the page functions without it.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from src.ui.logging_ui import get_log_file_path, get_ui_logger
from src.ui.styles import get_global_css, hero_section

_logger = get_ui_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────


def render() -> None:
    """Render the App Console page."""
    st.markdown(get_global_css(), unsafe_allow_html=True)
    st.markdown(
        hero_section(
            "App Console",
            "Live tail of the Streamlit UI application log.  "
            "Updated on every page visit.  Cleared when Streamlit restarts.",
        ),
        unsafe_allow_html=True,
    )

    log_path = get_log_file_path()

    # ── Auto-refresh via streamlit-autorefresh ───────────────────────────────
    st_autorefresh(interval=5_000, key="app_console_autorefresh")
    st.caption("⚡ Auto-refreshing every 5 seconds while this page is open.")

    # ── Controls row ──────────────────────────────────────────────────────────
    col_refresh, col_lines, col_clear = st.columns([1, 3, 1])
    with col_refresh:
        if st.button("🔄 Refresh", help="Re-read the log file now"):
            st.rerun()
    with col_lines:
        tail_lines = st.slider(
            "Lines to show",
            min_value=100,
            max_value=5_000,
            value=500,
            step=100,
            help="Number of lines to display (most recent)",
        )
    with col_clear:
        if (
            st.button("🗑️ Clear log", help="Truncate the log file")
            and log_path
            and log_path.exists()
        ):
            log_path.write_text("", encoding="utf-8")
            st.success("Log cleared.")
            st.rerun()

    # ── Log display ──────────────────────────────────────────────────────────
    if log_path is None:
        st.info(
            "Logging has not been configured yet.  "
            "The log file will appear here after the first page render.",
            icon="ℹ️",
        )
        return

    if not log_path.exists():
        st.info(
            f"Log file `{log_path}` does not exist yet.  "
            "Interact with any page to generate log entries.",
            icon="ℹ️",
        )
        return

    try:
        raw = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        st.error(f"Cannot read log file: {exc}")
        return

    lines = raw.splitlines()
    shown = lines[-tail_lines:] if len(lines) > tail_lines else lines
    shown_text = "\n".join(shown)

    if not shown:
        st.caption("Log is empty.")
        return

    # Summary metrics
    error_count = sum(1 for ln in shown if " ERROR    " in ln or " CRITICAL " in ln)
    warning_count = sum(1 for ln in shown if " WARNING  " in ln)
    info_count = sum(1 for ln in shown if " INFO     " in ln)

    m_cols = st.columns(4)
    m_cols[0].metric("Total lines", len(lines))
    m_cols[1].metric("Shown", len(shown))
    m_cols[2].metric("⚠️ Warnings", warning_count)
    m_cols[3].metric("🔴 Errors", error_count)

    # Filter controls
    filter_col1, filter_col2 = st.columns([1, 2])
    with filter_col1:
        level_filter = st.selectbox(
            "Filter by level",
            options=["ALL", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            index=0,
            key="console_level_filter",
        )
    with filter_col2:
        text_filter = st.text_input(
            "Search text (case-insensitive)",
            value="",
            placeholder="e.g. render, exception, docker_control …",
            key="console_text_filter",
            help="Filter lines containing this string (or leave empty to show all).",
        )

    filtered = list(shown)
    if level_filter != "ALL":
        filter_map = {
            "DEBUG": "DEBUG   ",
            "INFO": "INFO    ",
            "WARNING": "WARNING ",
            "ERROR": "ERROR   ",
            "CRITICAL": "CRITICAL",
        }
        token = filter_map.get(level_filter, level_filter)
        filtered = [ln for ln in filtered if token in ln]

    if text_filter.strip():
        needle = text_filter.strip().lower()
        filtered = [ln for ln in filtered if needle in ln.lower()]

    shown_text = "\n".join(filtered)

    active_filters = []
    if level_filter != "ALL":
        active_filters.append(f"level={level_filter}")
    if text_filter.strip():
        active_filters.append(f'text="{text_filter.strip()}"')

    if active_filters:
        st.caption(
            f"Showing last {len(shown)} lines → "
            f"filtered by {', '.join(active_filters)}: "
            f"**{len(filtered)} matching lines**."
        )
    else:
        st.caption(
            f"Showing last {len(shown)} of {len(lines)} total lines "
            f"({error_count} errors, {warning_count} warnings, {info_count} infos)."
        )

    import html as _html

    escaped = _html.escape(shown_text)
    st.markdown(
        f'<div style="height:600px;overflow-y:auto;background:#0d1117;color:#e6edf3;'
        f"font-family:'SFMono-Regular',Consolas,'Liberation Mono',Menlo,monospace;"
        f"font-size:12px;padding:12px 16px;border-radius:8px;"
        f'border:1px solid #30363d;white-space:pre-wrap;word-break:break-all;">'
        f"{escaped}</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"Log file: `{log_path.resolve()}`")
