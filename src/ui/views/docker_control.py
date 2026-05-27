"""Docker Control Center — start/stop stack, container status, log viewer."""

from __future__ import annotations

import datetime
import html as _html
import sys
import time as _time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
import streamlit.components.v1 as components

from src.ui.components.docker_utils import (
    _get_active_services,
    _read_secret,
    compose_restart,
    detect_current_mode,
    get_container_logs,
    get_container_statuses,
    get_host,
    get_host_port,
    get_stack_health_summary,
    make_streaming,
)
from src.ui.logging_ui import get_ui_logger
from src.ui.styles import get_global_css, hero_section, metric_card, status_badge

_logger = get_ui_logger(__name__)


def _is_inside_container() -> bool:
    """Return True when this Streamlit process is running inside a Docker container.

    When Streamlit runs as the mlops_streamlit container, ``make`` and the
    Docker socket are not available so stack-control operations cannot work.
    We detect this via the ``/.dockerenv`` sentinel file that Docker always
    creates inside containers.
    """
    return Path("/.dockerenv").exists()


# ── Streaming helper ──────────────────────────────────────────────────────────


def _run_make_with_live_output(
    target: str,
    op_label: str,
    console_ph: Any,  # DeltaGenerator returned by _docker_console()
    extra_vars: dict[str, str] | None = None,
    timeout: int = 900,
    session_rc_key: str = "stack_op_rc",
    session_op_key: str = "stack_op_last",
    op_type: str = "start",
) -> None:
    """Run a Makefile target, streaming live output directly into the console panel.

    ``console_ph`` is the ``st.empty()`` returned by ``_docker_console()``.
    During execution, streaming lines are rendered inside that placeholder so
    the live log appears inside the permanent console scrollable box rather
    than in a separate floating element.  After the process exits the console
    is re-rendered with the completed entry appended to the history.

    No ``st.rerun()`` is called here — callers must call ``st.rerun()`` after
    this returns to refresh the container status table.
    """
    buf: list[str] = []
    last_ui_update = 0.0

    _logger.info("Starting operation: {} (target={})", op_label, target)
    gen = make_streaming(target, extra_vars=extra_vars, timeout=timeout)
    rc = -1
    try:
        while True:
            line = next(gen)
            buf.append(line)
            # Throttle UI updates to at most 4 per second to avoid WebSocket spam
            now = _time.monotonic()
            if now - last_ui_update >= 0.25:
                _render_console_in_ph(console_ph, live_lines=buf[-200:])
                last_ui_update = now
    except StopIteration as exc:
        rc = exc.value

    _console_append(op_label, "".join(buf), rc)
    st.session_state[session_rc_key] = rc
    st.session_state[session_op_key] = op_type

    # Re-render the console with the completed entry (no live_lines)
    _render_console_in_ph(console_ph)

    # After a successful stop, remove .current_mode so detect_current_mode()
    # returns "unknown" on the next rerun rather than the stale "cloud"/"local".
    if rc == 0 and op_type == "stop":
        import contextlib

        _mode_file = Path(__file__).resolve().parents[3] / ".current_mode"
        with contextlib.suppress(OSError):
            _mode_file.unlink(missing_ok=True)

    if rc == 0:
        _logger.info("Operation completed successfully: {} (rc={})", op_label, rc)
    else:
        _logger.error(
            "Operation failed: {} rc={}\nLast output:\n{}",
            op_label,
            rc,
            "".join(buf[-50:]),
        )


# ── Console helpers ────────────────────────────────────────────────────────────
# The console stores timestamped docker operation entries in session_state.
# Both tabs share the same console so the user always has a complete picture
# regardless of which tab triggered the last operation.


def _console_append(operation: str, output: str, rc: int | None = None) -> None:
    """Append one docker operation result to the persistent in-session console."""
    if "docker_console" not in st.session_state:
        st.session_state.docker_console = []

    ts = datetime.datetime.now().strftime("%H:%M:%S")
    if rc is None:
        rc_badge = ""
    elif rc == 0:
        rc_badge = "  ✓  rc=0"
    else:
        rc_badge = f"  ✗  rc={rc}"

    sep = "─" * 64
    header = f"[{ts}]  {operation}{rc_badge}"
    entry_lines = [sep, header]
    stripped = output.strip()
    if stripped:
        entry_lines.append(stripped)
    entry_lines.append("")  # trailing blank line for readability

    st.session_state.docker_console.append("\n".join(entry_lines))
    # Keep the last 15 operation blocks to bound memory usage
    if len(st.session_state.docker_console) > 15:
        st.session_state.docker_console = st.session_state.docker_console[-15:]


_CONSOLE_DIV_STYLE = (
    "height:280px;overflow-y:auto;background:#0d1117;color:#e6edf3;"
    "font-family:'SFMono-Regular',Consolas,'Liberation Mono',Menlo,monospace;"
    "font-size:12px;padding:12px 16px;border-radius:8px;"
    "border:1px solid #30363d;white-space:pre-wrap;word-break:break-all;"
)


def _render_console_in_ph(console_ph: Any, live_lines: list[str] | None = None) -> None:
    """Render the console scrollable box into the ``console_ph`` st.empty().

    If ``live_lines`` is provided an "in-progress" entry is prepended at the
    top of the scrollable box so streaming output appears inside the console
    panel itself rather than in a separate floating element.
    """
    parts: list[str] = []

    if live_lines:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        sep = "─" * 64
        parts.append(f"{sep}\n[{ts}]  ... running ...\n{''.join(live_lines)}")

    history = list(reversed(st.session_state.get("docker_console", [])))
    parts.extend(history)

    if not parts:
        console_ph.caption(
            "Console is empty.  Start / Stop / Restart operations will appear here automatically."
        )
        return

    full_output = "\n".join(parts)
    escaped = _html.escape(full_output)
    console_ph.markdown(
        f'<div style="{_CONSOLE_DIV_STYLE}">{escaped}</div>',
        unsafe_allow_html=True,
    )


def _docker_console(key_suffix: str = "") -> Any:
    """Render the permanent Docker Operations Console.

    Returns an ``st.empty()`` placeholder (``console_ph``) that holds the
    scrollable content box.  Pass this to ``_run_make_with_live_output()``
    so that live streaming output is rendered inside the console panel — not
    in a separate floating element above the buttons.

    Always visible at the bottom of each tab.  Displays every start / stop /
    restart / pull operation from this session, newest-first.
    """
    st.markdown("---")
    col_title, col_clear = st.columns([5, 1])
    with col_title:
        st.markdown(
            '<div class="section-header">🖥️ Docker Operations Console</div>',
            unsafe_allow_html=True,
        )
    with col_clear:
        if st.button("🗑️ Clear", key=f"clear_console{key_suffix}", help="Clear console output"):
            st.session_state.docker_console = []

    console_ph = st.empty()
    _render_console_in_ph(console_ph)
    return console_ph


def render() -> None:
    """Render the Docker Control Center page."""
    try:
        _render_content()
    except Exception:
        _logger.exception("Unhandled exception in docker_control.render()")
        raise


def _render_content() -> None:
    """Inner render — separated so exceptions are caught and logged."""
    st.markdown(get_global_css(), unsafe_allow_html=True)
    st.markdown(
        hero_section(
            "Docker Control Center",
            "Manage the complete container stack — switch modes, view status, "
            "and inspect logs in real time.",
        ),
        unsafe_allow_html=True,
    )

    # When running inside the mlops_streamlit Docker container, make and the
    # Docker socket are not available.  Stack-control buttons cannot work in
    # this context.  Show a clear notice so the user knows to use the host
    # Streamlit (make ui) or the Makefile directly for stack management.
    if _is_inside_container():
        st.info(
            "ℹ️ **Docker Control is read-only in container mode.**\n\n"
            "This Streamlit instance is running inside the `mlops_streamlit` Docker container. "
            "Stack start/stop operations require the Docker socket and `make`, which are not "
            "available inside the container.\n\n"
            "To control the stack, use the host Streamlit (`make ui`) or run "
            "`make cloud`, `make cloud-rebuild`, `make ghcr`, `make down`, etc. directly "
            "from a terminal on the host machine."
        )
        # Still render the read-only status table so the user can see container health.
        st.markdown("---")
        _container_status()
        return

    _TAB_LABELS = ["🐳 Stack Control", "📦 GHCR Registry"]
    # ── Session-state radio navigation ──────────────────────────────────────
    # st.tabs renders ALL tab code on every pass, including hidden tabs.
    # components.html() inside a hidden tab creates a zero-dimension iframe
    # that Mermaid cannot render into. _ghcr_tab() is therefore ONLY called
    # when the GHCR tab is the active selection, ensuring the Mermaid diagram
    # is always rendered into a visible container.
    _restore_label = st.session_state.pop("_restore_tab_label", None)
    if _restore_label in _TAB_LABELS:
        st.session_state["docker_active_tab"] = _TAB_LABELS.index(_restore_label)
    if "docker_active_tab" not in st.session_state:
        st.session_state["docker_active_tab"] = 0

    _active_label = st.radio(
        "##docker_nav",
        _TAB_LABELS,
        index=st.session_state["docker_active_tab"],
        horizontal=True,
        key="docker_tab_radio",
        label_visibility="collapsed",
    )
    st.session_state["docker_active_tab"] = _TAB_LABELS.index(_active_label)
    st.markdown("---")
    _active_tab = st.session_state["docker_active_tab"]

    if _active_tab == 0:
        # ── Stack Control ────────────────────────────────────────────────────
        ctrl_mode_area = st.container()
        ctrl_mid_area = st.container()
        ctrl_console_area = st.container()

        with ctrl_console_area:
            ctrl_console_ph = _docker_console(key_suffix="_ctrl")

        # Mode-control buttons — stores pending op in session_state, does NOT run it yet
        with ctrl_mode_area:
            _mode_control(ctrl_console_ph)

        # Status table and log viewer
        with ctrl_mid_area:
            st.markdown("---")
            _container_status()
            st.markdown("---")
            _log_viewer()

        # Run any pending Stack Control operation here so ctrl_console_ph is in scope.
        _pending = st.session_state.pop("_ctrl_pending_op", None)
        if _pending:
            _run_make_with_live_output(
                _pending["target"],
                op_label=_pending["label"],
                console_ph=ctrl_console_ph,
                timeout=_pending["timeout"],
                session_rc_key="stack_op_rc",
                session_op_key="stack_op_last",
                op_type=_pending["op_type"],
            )
            st.rerun()

    elif _active_tab == 1:
        # ── GHCR Registry ────────────────────────────────────────────────────
        # _ghcr_tab() — and its Mermaid diagram — is only called when this tab
        # is the ACTIVE selection, i.e., always visible on screen.
        ghcr_main_area = st.container()
        ghcr_console_area = st.container()

        with ghcr_console_area:
            ghcr_console_ph = _docker_console(key_suffix="_ghcr")

        with ghcr_main_area:
            _ghcr_tab(ghcr_console_ph)


# ── Stack Control tab ──────────────────────────────────────────────────────────


def _mode_control(console_ph: Any) -> None:
    """Render mode selector and compose control buttons.

    ``console_ph`` is the ``st.empty()`` returned by ``_docker_console()``.
    It is passed through to ``_run_make_with_live_output()`` so that live
    streaming output is rendered inside the console panel.

    When a button is clicked the pending operation is stored in ``session_state``
    as ``_ctrl_pending_op``.  The caller runs it OUTSIDE any container context
    so the live output goes to the correct position in the layout.
    """
    current_mode = detect_current_mode()

    col_mode, col_info, col_up, col_down = st.columns([2, 2, 1, 1])

    with col_mode:
        default_index = 0 if current_mode in ("local", "unknown") else 1
        mode = st.radio(
            "Deployment Mode",
            options=["local", "cloud"],
            index=default_index,
            horizontal=True,
            help="Local: all services on localhost. Cloud: DagsHub MLflow + DVC remote.",
        )

    with col_info:
        if current_mode == "unknown":
            st.markdown(
                status_badge("warning", "Stack stopped / not detected"),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                status_badge("running", f"Mode: {current_mode}"),
                unsafe_allow_html=True,
            )

    with col_up:
        rebuild = st.checkbox(
            "Rebuild images",
            value=True,
            help=(
                "Full no-cache rebuild before starting (default: on). "
                "Equivalent to 'make local-rebuild' / 'make cloud-rebuild': stops all "
                "containers, rebuilds images without Docker cache, then restarts. "
                "Ensures the latest code and docker-compose changes are always active. "
                "Uncheck for a quick restart (uses Docker layer cache)."
            ),
        )
        start_clicked = st.button("🚀 Start Stack", type="primary")

    with col_down:
        stop_clicked = st.button("⏹️ Stop Stack")

    if start_clicked:
        target_map = {
            ("local", False): "local",
            ("local", True): "local-rebuild",
            ("cloud", False): "cloud",
            ("cloud", True): "cloud-rebuild",
        }
        target = target_map[(mode, rebuild)]
        st.session_state["_ctrl_pending_op"] = {
            "target": target,
            "label": f"make {target}",
            "timeout": 900 if rebuild else 300,
            "op_type": "start",
        }

    if stop_clicked:
        st.session_state["_ctrl_pending_op"] = {
            "target": "down",
            "label": "make down",
            "timeout": 180,
            "op_type": "stop",
        }

    # Brief result badge shown below the button row
    op_rc = st.session_state.get("stack_op_rc")
    op_last = st.session_state.get("stack_op_last", "start")
    if op_rc == 0:
        summary = get_stack_health_summary()
        if op_last == "stop":
            st.success(
                f"Stack stopped — {summary['stopped']} containers down.  "
                "Full output is in the **Docker Operations Console** at the bottom of this tab."
            )
        else:
            st.success(
                f"Stack started — {summary['running']} running, {summary['healthy']} healthy.  "
                "Nginx / Grafana become healthy ~1–2 min after the API passes its health-check.  "
                "Full output is in the **Docker Operations Console** at the bottom of this tab."
            )
    elif op_rc is not None:
        st.error(
            f"Last operation failed (rc={op_rc}).  "
            "Full output is in the **Docker Operations Console** at the bottom of this tab."
        )


def _container_status(key_suffix: str = "") -> None:
    """Render the container status table (static — refreshed only by st.rerun() calls)."""
    st.markdown(
        '<div class="section-header">📦 Container Status</div>',
        unsafe_allow_html=True,
    )
    _render_container_status_content(key_suffix)


def _render_container_status_content(key_suffix: str = "") -> None:  # noqa: F811
    """Internal: renders the full container status table (called by _container_status)."""
    statuses = get_container_statuses()
    running = sum(1 for s in statuses if s.is_up)
    healthy = sum(1 for s in statuses if s.health == "healthy")
    stopped = len(statuses) - running

    if st.button("🔄 Refresh", key=f"refresh_status{key_suffix}"):
        st.rerun()

    cols = st.columns(4)
    cols[0].markdown(metric_card("📦", str(len(statuses)), "Total"), unsafe_allow_html=True)
    cols[1].markdown(metric_card("🟢", str(running), "Running"), unsafe_allow_html=True)
    cols[2].markdown(metric_card("💚", str(healthy), "Healthy"), unsafe_allow_html=True)
    cols[3].markdown(metric_card("🔴", str(stopped), "Stopped"), unsafe_allow_html=True)

    st.markdown("")

    header = "| Status | Service | Port | State | Health | Image | Started |"
    sep = "|:------:|:--------|:----:|:------|:-------|:------|:--------|"
    rows = [header, sep]
    for s in statuses:
        if s.name == "mlops_api":
            nginx_port = get_host_port("mlops_nginx", 80)
            port_str = (
                f'<a href="http://{get_host()}:{nginx_port}/health" target="_blank" '
                f'style="color:#818cf8;text-decoration:none">via :{nginx_port}</a>'
                if s.is_up
                else f":{s.port}"
            )
        elif s.name in ("mlops_postgres", "mlops_postgres_mlflow"):
            port_str = f":{s.port} <small>(TCP)</small>"
        else:
            port_str = (
                f'<a href="http://{get_host()}:{s.port}" target="_blank" '
                f'style="color:#818cf8;text-decoration:none">:{s.port}</a>'
                if s.is_up
                else f":{s.port}"
            )
        rows.append(
            f"| {s.status_emoji} | {s.icon} **{s.display_name}** | {port_str} | "
            f"{s.state} | {s.health} | `{s.image}` | {s.uptime} |"
        )
    st.markdown("\n".join(rows), unsafe_allow_html=True)

    st.markdown("#### Restart Individual Service")
    active_svcs = _get_active_services()
    svc_names = [s[2] for s in active_svcs]
    svc_map = {s[2]: s[0] for s in active_svcs}
    selected = st.selectbox("Select service", svc_names, key=f"restart_svc{key_suffix}")
    if st.button("🔄 Restart Selected", key=f"restart_btn{key_suffix}"):
        with st.spinner(f"Restarting {selected}…"):
            out, rc = compose_restart(svc_map[selected])
        _console_append(f"docker compose restart {svc_map.get(selected, selected)}", out, rc)
        if rc == 0:
            st.success(f"{selected} restarted.")
        else:
            st.error(f"Failed (rc={rc}) — see console below.")


def _log_viewer(key_suffix: str = "") -> None:
    """Render container log viewer."""
    st.markdown(
        '<div class="section-header">📋 Container Logs</div>',
        unsafe_allow_html=True,
    )

    col_svc, col_lines = st.columns([3, 1])
    active_svcs = _get_active_services()
    container_names = {s[2]: s[1] for s in active_svcs}
    with col_svc:
        display_name = st.selectbox(
            "Container",
            list(container_names.keys()),
            key=f"log_container{key_suffix}",
        )
    with col_lines:
        tail = st.number_input(
            "Lines", min_value=10, max_value=1000, value=100, step=50, key=f"log_lines{key_suffix}"
        )

    if st.button("📜 Fetch Logs", key=f"fetch_logs{key_suffix}"):
        container = container_names[display_name]
        with st.spinner(f"Fetching logs from {container}…"):
            logs = get_container_logs(container, tail=tail)
        st.markdown(
            f'<div class="log-viewer"><pre>{logs}</pre></div>',
            unsafe_allow_html=True,
        )


# ── GHCR Registry tab ──────────────────────────────────────────────────────────

_GHCR_LIFECYCLE_MERMAID = r"""
flowchart LR
    push_code["Push to main / tag"]
    push_code --> build_api["Build API image"]
    push_code --> build_air["Build Airflow image"]
    push_code --> build_ui["Build Streamlit image"]
    build_api --> scan["Trivy security scan"]
    scan --> push_api["Push to GHCR"]
    build_air --> push_air["Push to GHCR"]
    build_ui --> push_ui["Push to GHCR"]
    push_api --> img_api["mlops-device-health-api"]
    push_air --> img_air["mlops-device-health-airflow"]
    push_ui --> img_ui["mlops-device-health-streamlit"]
    img_api --> pull["docker compose pull"]
    img_air --> pull
    img_ui --> pull
    pull --> up["docker compose up"]
    up --> api_c["API container"]
    up --> air_c["Airflow container"]
    up --> ui_c["Streamlit container"]
    up --> mon_c["Monitoring stack"]
    api_c --> ready["Stack ready"]
    air_c --> ready
    ui_c --> ready
    mon_c --> ready
    classDef ci fill:#1e3a5f,stroke:#3b82f6,color:#e2e8f0
    classDef reg fill:#1e293b,stroke:#6366f1,color:#e2e8f0
    classDef deploy fill:#1e3a2f,stroke:#10b981,color:#e2e8f0
    classDef ok fill:#14532d,stroke:#22c55e,color:#e2e8f0
    class push_code,build_api,build_air,build_ui,scan,push_api,push_air,push_ui ci
    class img_api,img_air,img_ui reg
    class pull,up,api_c,air_c,ui_c,mon_c deploy
    class ready ok
"""


def _render_mermaid(diagram: str, height: int = 350) -> None:
    """Render a Mermaid diagram via embedded Mermaid.js.

    Uses a retry loop so the diagram renders correctly even on the first visit
    (before the CDN script is cached) and even when the iframe is inside a
    hidden tab that becomes visible after page load.
    """
    html_content = f"""
    <html><head>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
      body {{ margin:0; padding:0; background:transparent; overflow:hidden; }}
      #diagram-wrap {{
        background:#0f172a; border-radius:12px; padding:1rem;
        overflow-y:auto; overflow-x:auto;
        height:{height - 8}px; box-sizing:border-box;
      }}
    </style>
    </head><body>
    <div id="diagram-wrap">
      <div id="diagram" class="mermaid">
{diagram.strip()}
      </div>
    </div>
    <script>
      (function() {{
        var attempts = 0;
        function tryRender() {{
          attempts++;
          var el = document.getElementById('diagram');
          if (!el) {{ if (attempts < 40) setTimeout(tryRender, 100); return; }}
          if (typeof mermaid === 'undefined') {{ if (attempts < 40) setTimeout(tryRender, 100); return; }}
          if (el.getAttribute('data-processed')) return;
          try {{
            mermaid.initialize({{startOnLoad:false,theme:'dark',securityLevel:'loose'}});
            mermaid.run({{nodes:[el]}});
          }} catch(e) {{
            if (attempts < 40) setTimeout(tryRender, 200);
          }}
        }}
        document.addEventListener('DOMContentLoaded', function() {{ setTimeout(tryRender, 50); }});
        window.addEventListener('load', function() {{ setTimeout(tryRender, 50); }});
        setTimeout(tryRender, 100);
      }})();
    </script>
    </body></html>
    """
    components.html(html_content, height=height, scrolling=False)


def _ghcr_tab(console_ph: Any) -> None:
    """Render the GHCR Registry tab.

    ``console_ph`` is the ``st.empty()`` returned by ``_docker_console()``.
    Streaming output from pull / stop operations is rendered inside it.
    """
    st.markdown("### 📦 GHCR Registry Mode")
    st.markdown(
        "Pull production-ready images directly from **GitHub Container Registry** instead of "
        "building locally. Images are published automatically by the CI/CD pipeline on every "
        "push to a branch or version tag."
    )

    owner = _read_secret("GITHUB_OWNER") or "your-github-username"
    ghcr_tag = _read_secret("GHCR_TAG") or "main"
    base = f"ghcr.io/{owner}"
    st.info(
        f"**Registry owner:** `{owner}`  \n"
        f"**Image tag:** `{ghcr_tag}`  \n"
        f"Override `GITHUB_OWNER` / `GHCR_TAG` in `.env.secrets` to change, "
        f"or on the command line: `make GHCR_TAG=feature-my-branch ghcr`",
        icon="ℹ️",
    )

    with st.expander("📐 CI/CD → GHCR → Deploy lifecycle", expanded=True):
        _render_mermaid(_GHCR_LIFECYCLE_MERMAID, height=380)

    with st.expander("📋 Pull commands reference", expanded=False):
        st.markdown("**Pull specific tags:**")
        st.code(
            f"docker pull {base}/mlops-device-health-api:{ghcr_tag}\n"
            f"docker pull {base}/mlops-device-health-airflow:{ghcr_tag}\n"
            f"docker pull {base}/mlops-device-health-streamlit:{ghcr_tag}",
            language="bash",
        )
        st.markdown("**Pull and start via Makefile (recommended):**")
        st.code(
            "make ghcr                             # pull tag :main (Makefile default) and start\n"
            "make ghcr-rebuild                     # down → pull :main → restart\n"
            "make GHCR_TAG=<branch-name> ghcr      # pull a specific branch tag\n"
            "# or set GHCR_TAG=<tag> in .env.secrets, then run: make ghcr",
            language="bash",
        )
        st.info(
            f"📌 **Current session tag:** `{ghcr_tag}` "
            f"(from `GHCR_TAG` in `.env.secrets`).  "
            "The ‘Pull & Start from GHCR’ button above uses this tag.  "
            "Set `GHCR_TAG=main` in `.env.secrets` to pull the `:main` default.",
            icon="ℹ️",
        )
        st.markdown("**Or manually via compose:**")
        st.code(
            "docker compose \\\n"
            "  --env-file .env.cloud --env-file .env.secrets \\\n"
            "  -f docker-compose.yml -f docker-compose.cloud.yml \\\n"
            "  -f docker-compose.ghcr.yml \\\n"
            "  pull && docker compose ... up -d --force-recreate",
            language="bash",
        )

    st.markdown("---")

    # Pull & Start / Stop buttons
    st.markdown("#### 🚀 Pull & Start from GHCR")
    col_btn, col_stop = st.columns([2, 1])
    with col_btn:
        start_ghcr = st.button("⬇️ Pull & Start from GHCR", type="primary", key="ghcr_start")

    with col_stop:
        stop_ghcr = st.button("⏹️ Stop Stack", key="ghcr_stop")

    if start_ghcr:
        _run_make_with_live_output(
            "ghcr",
            op_label=f"make ghcr  (GHCR_TAG={ghcr_tag})",
            console_ph=console_ph,
            extra_vars={"GHCR_TAG": ghcr_tag},
            timeout=600,
            session_rc_key="ghcr_op_rc",
            session_op_key="ghcr_op_last",
            op_type="start-ghcr",
        )
        st.session_state["_restore_tab_label"] = "\U0001f4e6 GHCR Registry"
        st.rerun()

    if stop_ghcr:
        _run_make_with_live_output(
            "down",
            op_label="make down (GHCR tab)",
            console_ph=console_ph,
            timeout=180,
            session_rc_key="ghcr_op_rc",
            session_op_key="ghcr_op_last",
            op_type="stop",
        )
        st.session_state["_restore_tab_label"] = "\U0001f4e6 GHCR Registry"
        st.rerun()

    # Brief result badge
    ghcr_rc = st.session_state.get("ghcr_op_rc")
    ghcr_last = st.session_state.get("ghcr_op_last", "start")
    if ghcr_rc == 0:
        summary = get_stack_health_summary()
        if ghcr_last == "stop":
            st.success(
                f"Stack stopped — {summary['stopped']} containers down.  "
                "Full output is in the **Docker Operations Console** at the bottom of this tab."
            )
        else:
            st.success(
                f"GHCR pull completed — {summary['running']} running, {summary['healthy']} healthy.  "
                "Nginx / Grafana become healthy ~1–2 min after the API passes its health-check.  "
                "Full output is in the **Docker Operations Console** at the bottom of this tab."
            )
    elif ghcr_rc is not None:
        st.error(
            f"Last GHCR operation failed (rc={ghcr_rc}).  "
            "Full output is in the **Docker Operations Console** at the bottom of this tab."
        )

    st.markdown("---")

    # GHCR Registry Explorer
    try:
        from src.ui.views.github_dashboard import _render_ghcr_images

        _render_ghcr_images()
    except ImportError:
        st.warning("GHCR explorer unavailable — github_dashboard module not found.")

    st.markdown("---")

    # Container status & log viewer (reused)
    _container_status(key_suffix="_ghcr")
    st.markdown("---")
    _log_viewer(key_suffix="_ghcr")
