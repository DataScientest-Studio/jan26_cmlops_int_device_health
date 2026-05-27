"""Monitoring Dashboard — Prometheus metrics + Grafana embedding.

Queries the Prometheus HTTP API directly for key operational metrics
and optionally embeds Grafana panels via iframe.  Falls back
gracefully when services are unreachable.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from datetime import UTC

import streamlit as st

from src.ui.components.docker_utils import (
    get_container_statuses,
    get_k8s_pod_statuses,
    get_service_url,
)
from src.ui.logging_ui import get_ui_logger
from src.ui.styles import get_global_css, hero_section

# ── Mode detection ──────────────────────────────────────────────

# Containers/pods expected to be absent per mode (not degraded).
_EXPECTED_ABSENT: dict[str, set[str]] = {
    "local": {"mlops_airflow", "mlops_airflow_scheduler"},
    "cloud": {"mlops_mlflow"},
    # In K8s mode all Docker containers are absent; pod statuses are shown instead.
    "k8s": set(),
}


_logger = get_ui_logger(__name__)


def _detect_mode() -> str:
    mode_file = Path(_PROJECT_ROOT) / ".current_mode"
    if mode_file.exists():
        return mode_file.read_text().strip()
    import os

    return os.environ.get("DEPLOYMENT_MODE", "local")


# ── Connection helpers ──────────────────────────────────────────


def _prometheus_url() -> str:
    return get_service_url("mlops_prometheus", 9090)


def _grafana_url() -> str:
    return get_service_url("mlops_grafana", 3000)


def _alertmanager_url() -> str:
    return get_service_url("mlops_alertmanager", 9093)


def _prom_query(promql: str, *, timeout: int = 8) -> list[dict]:
    """Execute PromQL instant query, return result list."""
    import urllib.parse

    url = f"{_prometheus_url()}/api/v1/query?query={urllib.parse.quote(promql)}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            if data.get("status") == "success":
                return data.get("data", {}).get("result", [])
    except (urllib.error.URLError, TimeoutError, OSError):
        pass
    return []


def _prom_up() -> bool:
    """Quick check: is Prometheus reachable?"""
    url = f"{_prometheus_url()}/-/healthy"
    try:
        with urllib.request.urlopen(url, timeout=5):
            return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


# ── Rendering helpers ───────────────────────────────────────────


def _render_infrastructure() -> None:
    """Combined: container health + system metrics + scrape targets."""
    st.markdown(
        '<div class="section-header">🏗️ Infrastructure Overview</div>',
        unsafe_allow_html=True,
    )

    # ── System gauges at top ──
    cpu = _prom_query('100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)')
    mem = _prom_query("100 * (1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)")
    disk = _prom_query(
        '100 * (1 - node_filesystem_avail_bytes{fstype!~"tmpfs|overlay"} '
        '/ node_filesystem_size_bytes{fstype!~"tmpfs|overlay"})'
    )

    def _extract_pct(results: list[dict]) -> str:
        if results:
            val = results[0].get("value", [None, "0"])
            pct = float(val[1]) if isinstance(val, list) and len(val) > 1 else 0.0
            return f"{pct:.1f}%"
        return "—"

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("🖥️ CPU Usage", _extract_pct(cpu))
    sc2.metric("🧠 Memory Usage", _extract_pct(mem))
    sc3.metric("💾 Disk Usage", _extract_pct(disk))

    st.markdown("---")

    # ── Container/pod status cards ──
    mode = _detect_mode()
    statuses = get_k8s_pod_statuses() if mode == "k8s" else get_container_statuses()
    # cAdvisor container resource data
    mem_results = _prom_query('container_memory_usage_bytes{name=~"mlops_.*"}')
    cpu_results = _prom_query('rate(container_cpu_usage_seconds_total{name=~"mlops_.*"}[5m])')

    cadvisor_mem: dict[str, float] = {}
    for r in mem_results:
        name = r.get("metric", {}).get("name", "")
        val = r.get("value", [None, "0"])
        cadvisor_mem[name] = (
            float(val[1]) / (1024 * 1024) if isinstance(val, list) and len(val) > 1 else 0.0
        )

    cadvisor_cpu: dict[str, float] = {}
    for r in cpu_results:
        name = r.get("metric", {}).get("name", "")
        val = r.get("value", [None, "0"])
        cadvisor_cpu[name] = float(val[1]) * 100 if isinstance(val, list) and len(val) > 1 else 0.0

    if statuses:
        label = "Pod Health:" if mode == "k8s" else "Container Health:"
        st.markdown(f"**{label}**")
        expected_absent = _EXPECTED_ABSENT.get(mode, set())

        # Filter out containers that are expected to be absent in this mode
        filtered_statuses = [cs for cs in statuses if cs.name not in expected_absent]

        if not filtered_statuses:
            st.info("No monitored containers in this mode.")
        else:
            # Render in rows of 4
            for row_start in range(0, len(filtered_statuses), 4):
                row_items = filtered_statuses[row_start : row_start + 4]
                cols = st.columns(len(row_items))
                for i, cs in enumerate(row_items):
                    is_up = cs.state == "running"
                    color = "#22c55e" if is_up else "#ef4444"
                    icon = "🟢" if is_up else "🔴"
                    container_mem = cadvisor_mem.get(cs.name, 0)
                    container_cpu = cadvisor_cpu.get(cs.name, 0)
                    resource_line = ""
                    if container_mem > 0 or container_cpu > 0:
                        resource_line = (
                            f"<br><span style='font-size:0.75rem;color:#94a3b8'>"
                            f"CPU {container_cpu:.1f}% · RAM {container_mem:.0f}MB</span>"
                        )
                    cols[i].markdown(
                        f"<div style='background:#1e293b;border-left:4px solid {color};"
                        f"border-radius:8px;padding:0.6rem 0.8rem;margin-bottom:0.3rem'>"
                        f"<strong>{icon} {cs.display_name}</strong><br>"
                        f"<span style='font-size:0.85rem;color:#cbd5e1'>"
                        f"{cs.state}</span>"
                        f"{resource_line}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # Note which containers are excluded
        if expected_absent:
            absent_names = ", ".join(sorted(expected_absent))
            st.caption(f"ℹ️ Excluded from monitoring ({mode} mode): {absent_names}")
    else:
        no_status_msg = (
            "No K8s pods detected. Ensure port-forwards are active."
            if mode == "k8s"
            else "No Docker containers detected."
        )
        st.info(no_status_msg)

    # ── Prometheus scrape targets ──
    results = _prom_query("up")
    if results:
        st.markdown("**Prometheus Scrape Targets:**")
        target_cols = st.columns(min(len(results), 4))
        for i, r in enumerate(results):
            labels = r.get("metric", {})
            job = labels.get("job", labels.get("instance", "unknown"))
            instance = labels.get("instance", "")
            val = r.get("value", [None, "0"])
            is_up = val[1] == "1" if isinstance(val, list) and len(val) > 1 else False
            # Friendly label: append instance when it differs from job name
            display_label = job
            if instance and instance != job:
                display_label = f"{job} ({instance})"
            # Explain mlops_host: it's the relabeled node_exporter instance
            if instance == "mlops_host":
                display_label = "node_exporter (host)"
            target_cols[i % len(target_cols)].metric(
                display_label,
                "\U0001f7e2 UP" if is_up else "\U0001f534 DOWN",
            )
        # Explain expected DOWN targets
        st.caption(
            "\u2139\ufe0f **node_exporter** shows DOWN on Windows \u2014 "
            "node_exporter is a Linux-only host metrics exporter and is not running on this system. "
            "Container metrics are still available via cAdvisor.  "
            "\u00b7  **mlops_host** is the relabeled `instance` label for the node_exporter job "
            "(set via `relabel_configs` in prometheus.yml to give the host a friendly name)."
        )


def _render_api_metrics() -> None:
    """Show HTTP request metrics with visual cards, activity highlights, and count deltas."""
    st.markdown(
        '<div class="section-header">⚡ API & Model Metrics</div>',
        unsafe_allow_html=True,
    )

    api_total = _prom_query("api_requests_total")
    pred_total = _prom_query("model_predictions_total")
    # 1-minute rate to detect recent activity
    api_rate = _prom_query("rate(api_requests_total[1m])")
    pred_rate = _prom_query("rate(model_predictions_total[1m])")
    process_cpu = _prom_query("process_cpu_seconds_total")
    process_mem = _prom_query("process_resident_memory_bytes")

    has_data = any([api_total, pred_total, process_cpu, process_mem])

    # Build rate lookup: (method, endpoint, status) -> rate_per_sec
    rate_map: dict[tuple[str, str, str], float] = {}
    for r in api_rate:
        lbl = r.get("metric", {})
        val = r.get("value", [None, "0"])
        rps = float(val[1]) if isinstance(val, list) and len(val) > 1 else 0.0
        rate_map[(lbl.get("method", ""), lbl.get("endpoint", ""), lbl.get("status_code", ""))] = rps

    pred_rate_map: dict[str, float] = {}
    for r in pred_rate:
        lbl = r.get("metric", {})
        val = r.get("value", [None, "0"])
        rps = float(val[1]) if isinstance(val, list) and len(val) > 1 else 0.0
        key = lbl.get("model_version", lbl.get("predicted_label", "total"))
        pred_rate_map[key] = rps

    # ── API Request cards ──
    if api_total:
        st.markdown(
            "**API Requests** "
            "<span style='font-size:0.8rem;color:#94a3b8'>"
            "(counts HTTP requests, not individual signals — a batch of 100 signals = 1 request)"
            "</span>",
            unsafe_allow_html=True,
        )
        for row_start in range(0, len(api_total), 3):
            row_items = api_total[row_start : row_start + 3]
            cols = st.columns(len(row_items))
            for i, r in enumerate(row_items):
                labels = r.get("metric", {})
                val = r.get("value", [None, "0"])
                count = int(float(val[1])) if isinstance(val, list) and len(val) > 1 else 0
                method = labels.get("method", "?")
                endpoint = labels.get("endpoint", "?")
                status = labels.get("status_code", "?")
                rps = rate_map.get((method, endpoint, status), 0.0)
                is_active = rps > 0.001
                border_color = "#22c55e" if is_active else "#334155"
                glow = "box-shadow:0 0 8px #22c55e55;" if is_active else ""
                activity_badge = (
                    f"<span style='background:#22c55e;color:#000;font-size:0.7rem;"
                    f"padding:1px 6px;border-radius:8px;margin-left:6px'>ACTIVE "
                    f"{rps:.2f} req/s</span>"
                    if is_active
                    else "<span style='color:#64748b;font-size:0.7rem;margin-left:6px'>idle</span>"
                )
                cols[i].markdown(
                    f"<div style='background:#1e293b;border-left:4px solid {border_color};"
                    f"border-radius:8px;padding:0.7rem 1rem;margin-bottom:0.4rem;{glow}'>"
                    f"<strong style='font-size:1.3rem'>{count:,}</strong>{activity_badge}<br>"
                    f"<span style='color:#818cf8;font-weight:600'>{method}</span> "
                    f"<code style='color:#e2e8f0'>{endpoint}</code> "
                    f"<span style='color:#94a3b8'>→ {status}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ── Model Predictions cards ──
    if pred_total:
        st.markdown("**Model Predictions** (per model version / predicted label)")
        pred_cols = st.columns(min(len(pred_total), 4))
        for i, r in enumerate(pred_total):
            labels = r.get("metric", {})
            val = r.get("value", [None, "0"])
            count = int(float(val[1])) if isinstance(val, list) and len(val) > 1 else 0
            # Build a human-readable label from whichever Prometheus labels are present
            parts: list[str] = []
            if labels.get("model_version"):
                parts.append(f"model: {labels['model_version']}")
            if labels.get("predicted_label"):
                parts.append(f"label: {labels['predicted_label']}")
            label = " · ".join(parts) if parts else "total"
            rps = pred_rate_map.get(
                labels.get("model_version", labels.get("predicted_label", "total")), 0.0
            )
            is_active = rps > 0.001
            border_color = "#6366f1" if is_active else "#334155"
            glow = "box-shadow:0 0 8px #6366f155;" if is_active else ""
            badge = (
                f"<span style='background:#6366f1;color:#fff;font-size:0.7rem;"
                f"padding:1px 6px;border-radius:8px;margin-left:6px'>"
                f"{rps:.2f}/s</span>"
                if is_active
                else ""
            )
            pred_cols[i % len(pred_cols)].markdown(
                f"<div style='background:#1e293b;border-left:4px solid {border_color};"
                f"border-radius:8px;padding:0.7rem 1rem;margin-bottom:0.4rem;{glow}'>"
                f"<strong style='font-size:1.3rem'>{count:,}</strong>{badge}<br>"
                f"<span style='color:#cbd5e1'>{label}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Process metrics — one card per service ──
    if process_cpu or process_mem:
        st.markdown("**Process Metrics** (one card per service: CPU seconds + RSS memory)")

        # Build a per-job dict so CPU and memory are shown together
        service_stats: dict[str, dict[str, float]] = {}
        for r in process_cpu:
            job = r.get("metric", {}).get("job", "unknown")
            val = r.get("value", [None, "0"])
            service_stats.setdefault(job, {})["cpu_s"] = (
                float(val[1]) if isinstance(val, list) and len(val) > 1 else 0.0
            )
        for r in process_mem:
            job = r.get("metric", {}).get("job", "unknown")
            val = r.get("value", [None, "0"])
            mem_bytes = float(val[1]) if isinstance(val, list) and len(val) > 1 else 0.0
            service_stats.setdefault(job, {})["mem_mb"] = mem_bytes / (1024 * 1024)

        # One card per service, up to 4 columns
        jobs = sorted(service_stats.keys())
        card_cols = st.columns(min(len(jobs), 4))
        for i, job in enumerate(jobs):
            stats = service_stats[job]
            cpu_s = stats.get("cpu_s", 0.0)
            mem_mb = stats.get("mem_mb", 0.0)
            short_name = job.replace("mlops_", "").replace("_", " ").title()
            card_cols[i % len(card_cols)].markdown(
                f"<div style='background:#1e293b;border:1px solid #334155;"
                f"border-radius:8px;padding:0.8rem 1rem;margin-bottom:0.4rem'>"
                f"<div style='color:#94a3b8;font-size:0.75rem;font-weight:600;letter-spacing:0.05em'>{short_name.upper()}</div>"
                f"<div style='margin-top:0.4rem;display:flex;gap:1rem'>"
                f"<div><div style='color:#e2e8f0;font-size:1.2rem;font-weight:700'>{cpu_s:.1f}s</div>"
                f"<div style='color:#64748b;font-size:0.7rem'>CPU seconds</div></div>"
                f"<div><div style='color:#e2e8f0;font-size:1.2rem;font-weight:700'>{mem_mb:.0f} MB</div>"
                f"<div style='color:#64748b;font-size:0.7rem'>RSS memory</div></div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

    if not has_data:
        st.info(
            "No API or model metrics recorded yet.  "
            "Send some predictions to the `/predict` endpoint to populate these counters."
        )

    # ── Governance KPI cards ──────────────────────────────────────────────────
    deploy_time_raw = _prom_query("model_deploy_time_seconds")
    auto_rate_raw = _prom_query("automation_rate_gauge")
    mttd_raw = _prom_query("mttd_seconds")

    if any([deploy_time_raw, auto_rate_raw, mttd_raw]):
        st.markdown("**Governance KPIs** (set by Airflow DAGs on each retraining cycle)")
        kc1, kc2, kc3 = st.columns(3)

        def _first_value(results: list, fallback: str = "–") -> str:
            if results:
                val = results[0].get("value", [None, None])
                if isinstance(val, list) and len(val) > 1 and val[1] is not None:
                    return val[1]
            return fallback

        # Model deploy time
        dt_raw = _first_value(deploy_time_raw)
        dt_display = f"{float(dt_raw):.0f}s" if dt_raw != "–" else "–"
        dt_color = "#22c55e" if dt_raw != "–" and float(dt_raw) < 3600 else "#f59e0b"
        kc1.markdown(
            f"<div style='background:#1e293b;border-left:4px solid {dt_color};"
            f"border-radius:8px;padding:0.8rem 1rem'>"
            f"<div style='color:#94a3b8;font-size:0.75rem;font-weight:600'>MODEL DEPLOY TIME</div>"
            f"<div style='color:#e2e8f0;font-size:1.6rem;font-weight:700;margin-top:0.3rem'>{dt_display}</div>"
            f"<div style='color:#64748b;font-size:0.7rem'>trigger → production</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Automation rate
        ar_raw = _first_value(auto_rate_raw)
        ar_display = f"{float(ar_raw):.0%}" if ar_raw != "–" else "–"
        ar_color = "#22c55e" if ar_raw != "–" and float(ar_raw) >= 0.8 else "#f59e0b"
        kc2.markdown(
            f"<div style='background:#1e293b;border-left:4px solid {ar_color};"
            f"border-radius:8px;padding:0.8rem 1rem'>"
            f"<div style='color:#94a3b8;font-size:0.75rem;font-weight:600'>AUTOMATION RATE</div>"
            f"<div style='color:#e2e8f0;font-size:1.6rem;font-weight:700;margin-top:0.3rem'>{ar_display}</div>"
            f"<div style='color:#64748b;font-size:0.7rem'>auto-triggered retraining</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # MTTD
        mttd_val = _first_value(mttd_raw)
        if mttd_val != "–":
            secs = float(mttd_val)
            mttd_display = f"{secs / 3600:.1f}h" if secs >= 3600 else f"{secs:.0f}s"
        else:
            mttd_display = "–"
        mttd_color = "#22c55e" if mttd_val != "–" and float(mttd_val) < 86400 else "#f59e0b"
        kc3.markdown(
            f"<div style='background:#1e293b;border-left:4px solid {mttd_color};"
            f"border-radius:8px;padding:0.8rem 1rem'>"
            f"<div style='color:#94a3b8;font-size:0.75rem;font-weight:600'>MEAN TIME TO DETECT</div>"
            f"<div style='color:#e2e8f0;font-size:1.6rem;font-weight:700;margin-top:0.3rem'>{mttd_display}</div>"
            f"<div style='color:#64748b;font-size:0.7rem'>drift detection latency</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _render_alerts() -> None:
    """Show active Prometheus alerts with summary, severity grouping, and silence via Alertmanager."""
    st.markdown(
        '<div class="section-header">🔔 Active Alerts</div>',
        unsafe_allow_html=True,
    )
    url = f"{_prometheus_url()}/api/v1/alerts"
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, OSError):
        mode = _detect_mode()
        if mode == "k8s":
            st.warning(
                "\u26a0\ufe0f Could not fetch alerts from Prometheus. "
                "Make sure the K8s stack is running and port-forwarded (`make k8s-ports`)."
            )
        else:
            st.warning(
                "\u26a0\ufe0f Could not fetch alerts from Prometheus. Ensure the Docker stack is running."
            )
        return

    alerts = data.get("data", {}).get("alerts", [])

    # ── Summary banner ──────────────────────────────────────────
    firing = [a for a in alerts if a.get("state") == "firing"]
    pending = [a for a in alerts if a.get("state") == "pending"]
    n_critical = sum(1 for a in firing if a.get("labels", {}).get("severity") == "critical")
    n_warning = sum(1 for a in firing if a.get("labels", {}).get("severity") == "warning")
    n_info = sum(1 for a in firing if a.get("labels", {}).get("severity") == "info")

    sc1, sc2, sc3, sc4 = st.columns(4)
    if not alerts:
        sc1.markdown(
            "<div style='background:#166534;border-left:4px solid #22c55e;"
            "border-radius:8px;padding:0.6rem 1rem;text-align:center'>"
            "<strong style='font-size:1.4rem;color:#22c55e'>✅ All Clear</strong><br>"
            "<span style='color:#86efac;font-size:0.85rem'>No active alerts</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    for col, label, value, bg, accent in [
        (sc1, "🔴 Critical", n_critical, "#450a0a", "#ef4444"),
        (sc2, "🟡 Warning", n_warning, "#422006", "#f59e0b"),
        (sc3, "🔵 Info", n_info, "#0c1a4e", "#6366f1"),
        (sc4, "🟠 Pending", len(pending), "#1a1a2e", "#f97316"),
    ]:
        color = accent if value > 0 else "#64748b"
        col.markdown(
            f"<div style='background:{bg};border-left:4px solid {color};"
            f"border-radius:8px;padding:0.6rem 1rem;text-align:center'>"
            f"<strong style='font-size:1.5rem;color:{color}'>{value}</strong><br>"
            f"<span style='color:#cbd5e1;font-size:0.8rem'>{label}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<p style='color:#94a3b8;font-size:0.85rem;margin-top:0.5rem'>"
        f"<strong style='color:#e2e8f0'>{len(firing)}</strong> firing · "
        f"<strong style='color:#e2e8f0'>{len(pending)}</strong> pending — "
        f"Source: <a href='{_prometheus_url()}/alerts' target='_blank' "
        f"style='color:#6366f1'>Prometheus alerts page ↗</a></p>",
        unsafe_allow_html=True,
    )

    # ── Severity order and colours ──────────────────────────────
    _SEV_ORDER = {"critical": 0, "warning": 1, "info": 2}
    _SEV_COLOUR = {"critical": "#ef4444", "warning": "#f59e0b", "info": "#6366f1"}
    _SEV_BG = {"critical": "#450a0a", "warning": "#422006", "info": "#0c1a4e"}

    # Sort: firing before pending, then by severity, then by name
    def _sort_key(a: dict) -> tuple:
        state_ord = 0 if a.get("state") == "firing" else 1
        sev = a.get("labels", {}).get("severity", "info")
        sev_ord = _SEV_ORDER.get(sev, 9)
        name = a.get("labels", {}).get("alertname", "")
        return (state_ord, sev_ord, name)

    alerts_sorted = sorted(alerts, key=_sort_key)

    st.markdown("---")

    for idx, alert in enumerate(alerts_sorted):
        state = alert.get("state", "unknown")
        labels = alert.get("labels", {})
        name = labels.get("alertname", "unknown")
        severity = labels.get("severity", "info")
        component = labels.get("component", "")
        uc = labels.get("uc", "")

        sev_color = _SEV_COLOUR.get(severity, "#64748b")
        sev_bg = _SEV_BG.get(severity, "#1e293b")
        state_icon = "🔴" if state == "firing" else "🟡" if state == "pending" else "⚪"

        # Active-since timestamp
        active_at = alert.get("activeAt", "")
        active_since = ""
        if active_at:
            try:
                from datetime import datetime

                dt = datetime.fromisoformat(active_at.replace("Z", "+00:00"))
                now = datetime.now(dt.tzinfo)
                delta = now - dt
                mins = int(delta.total_seconds() // 60)
                active_since = f" · active {mins}m"
            except Exception:
                pass

        meta_parts = [f"severity={severity}"]
        if component:
            meta_parts.append(f"component={component}")
        if uc:
            meta_parts.append(uc)
        meta_str = " · ".join(meta_parts)

        summary = alert.get("annotations", {}).get("summary", "—")
        description = alert.get("annotations", {}).get("description", "")

        header = f"{state_icon} **{name}**  ·  {state}{active_since}  ·  {meta_str}"
        with st.expander(header, expanded=(severity == "critical" and state == "firing")):
            st.markdown(
                f"<div style='background:{sev_bg};border-left:4px solid {sev_color};"
                f"border-radius:6px;padding:0.6rem 1rem;margin-bottom:0.6rem'>"
                f"<strong>Summary:</strong> {summary}"
                f"</div>",
                unsafe_allow_html=True,
            )
            if description:
                st.markdown(f"_{description}_")
            with st.expander("🏷️ Raw labels", expanded=False):
                st.json(labels)
            # Silence button
            if st.button("🔕 Silence for 2h", key=f"silence_{idx}_{name}_{state}"):
                _silence_alert(labels)


def _silence_alert(labels: dict) -> None:
    """Create a silence in Alertmanager for the given alert labels."""
    from datetime import datetime, timedelta

    now = datetime.now(UTC)
    end = now + timedelta(hours=2)
    matchers = [
        {"name": k, "value": v, "isRegex": False}
        for k, v in labels.items()
        if k in ("alertname", "severity")
    ]
    payload = json.dumps(
        {
            "matchers": matchers,
            "startsAt": now.isoformat(),
            "endsAt": end.isoformat(),
            "createdBy": "streamlit-dashboard",
            "comment": "Silenced from Streamlit monitoring dashboard",
        }
    ).encode()
    req = urllib.request.Request(
        f"{_alertmanager_url()}/api/v2/silences",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as _resp:  # noqa: F841
            st.success("🔕 Alert silenced for 2 hours.")
    except (urllib.error.URLError, TimeoutError, OSError):
        st.error("Could not reach Alertmanager to create silence.")


# ── Grafana dashboards ──────────────────────────────────────────

_GRAFANA_DASHBOARDS: dict[str, str] = {
    "System Health": "mlops-system-health",
    "Model Performance": "mlops-model-performance",
    "Data Quality": "mlops-data-quality",
    "Business KPIs": "mlops-business-kpis",
    "Alerts Overview": "mlops-alerts-overview",
    "Container Infrastructure": "mlops-container-infra",
    "Model A/B Testing": "mlops-model-ab",
    "Retraining Pipeline": "mlops-retraining-pipeline",
    "Kubernetes Cluster": "mlops-k8s-cluster",
}

_GRAFANA_TIME_RANGES: dict[str, str] = {
    "Last 5 minutes": "now-5m",
    "Last 15 minutes": "now-15m",
    "Last 30 minutes": "now-30m",
    "Last 1 hour": "now-1h",
    "Last 3 hours": "now-3h",
    "Last 6 hours": "now-6h",
    "Last 12 hours": "now-12h",
    "Last 24 hours": "now-24h",
    "Last 7 days": "now-7d",
}


def _render_grafana_embed() -> None:
    """Embed Grafana dashboard via iframe with dashboard and time range selectors."""
    st.markdown(
        '<div class="section-header">📈 Grafana Dashboards</div>',
        unsafe_allow_html=True,
    )
    grafana = _grafana_url()

    # Dashboard + time range selectors side by side
    sel_col, time_col, height_col = st.columns([2, 2, 1])
    with sel_col:
        selected = st.selectbox(
            "Dashboard",
            list(_GRAFANA_DASHBOARDS.keys()),
            key="grafana_dash_select",
        )
    with time_col:
        time_label = st.selectbox(
            "Time range",
            list(_GRAFANA_TIME_RANGES.keys()),
            index=3,  # default: Last 1 hour
            key="grafana_time_range",
        )
    with height_col:
        height = st.slider("Height", 300, 900, 600, 50, key="grafana_height")

    uid = _GRAFANA_DASHBOARDS[selected]
    time_from = _GRAFANA_TIME_RANGES[time_label]

    embed_url = f"{grafana}/d/{uid}/{uid}?orgId=1&refresh=10s&from={time_from}&to=now&kiosk"
    direct_url = f"{grafana}/d/{uid}/{uid}?orgId=1&refresh=10s&from={time_from}&to=now"

    st.markdown(
        f"[Open **{selected}** in Grafana \u2197]({direct_url})  \u00b7  [Full Grafana UI \u2197]({grafana})"
    )

    st.info(
        "\U0001f4a1 The embedded Grafana panel below requires Grafana's "
        "`allow_embedding = true` and anonymous auth or an auth proxy.  "
        "If the panel is blank, use the direct link above."
    )

    import streamlit.components.v1 as components

    iframe_html = (
        f'<iframe src="{embed_url}" width="100%" height="{height}px" '
        f'style="border:none;border-radius:12px;background:#1e293b" '
        f'loading="lazy"></iframe>'
    )
    components.html(iframe_html, height=height + 20)


# ── Auto-refreshing fragments ───────────────────────────────────

_REFRESH_INTERVAL = 20  # seconds
_ALERTS_REFRESH_INTERVAL = 20  # longer interval to prevent expander collapse


@st.fragment(run_every=_REFRESH_INTERVAL)
def _live_infrastructure() -> None:
    """Auto-refreshing infrastructure panel."""
    _render_infrastructure()
    st.caption(f"🔄 Auto-refreshes every {_REFRESH_INTERVAL}s")


@st.fragment(run_every=_REFRESH_INTERVAL)
def _live_api_metrics() -> None:
    """Auto-refreshing API metrics panel."""
    _render_api_metrics()
    st.caption(f"🔄 Auto-refreshes every {_REFRESH_INTERVAL}s")


@st.fragment(run_every=_ALERTS_REFRESH_INTERVAL)
def _live_alerts() -> None:
    """Auto-refreshing alerts panel."""
    _render_alerts()
    st.caption(f"🔄 Auto-refreshes every {_ALERTS_REFRESH_INTERVAL}s")


@st.fragment
def _grafana_section() -> None:
    """Grafana tab — wrapped in a fragment so dashboard/range selector changes
    do NOT trigger a full-page rerun (which would reset the active tab to
    Infrastructure). Interactive widgets inside a fragment rerun only the
    fragment, leaving the tab selection unchanged."""
    _render_grafana_embed()


# ── Main render ─────────────────────────────────────────────────


def render() -> None:
    """Render the Monitoring Dashboard page."""
    try:
        _render_content()
    except Exception:
        _logger.exception("Unhandled exception in Monitoring.render()")
        raise


def _render_content() -> None:
    """Inner render — separated so exceptions are caught and logged by render()."""
    st.markdown(get_global_css(), unsafe_allow_html=True)
    st.markdown(
        hero_section(
            "📊 Monitoring Dashboard",
            "Live system health via Prometheus metrics and Grafana dashboards — "
            "CPU, memory, container resources, API throughput, and active alerts.",
        ),
        unsafe_allow_html=True,
    )

    prom_ok = _prom_up()
    if not prom_ok:
        mode = _detect_mode()
        if mode == "k8s":
            st.warning(
                "\u26a0\ufe0f Prometheus is unreachable.  "
                "Make sure the K8s stack is running and port-forwarded: "
                "`make k8s-up` then `make k8s-ports`.  \n"
                f"**URL:** `{_prometheus_url()}`"
            )
        else:
            st.warning(
                "\u26a0\ufe0f Prometheus is unreachable.  "
                "Make sure the Docker stack is running (`make up` or `docker compose up -d`).  \n"
                f"**URL:** `{_prometheus_url()}`"
            )

    # Use st.radio (keyed) instead of st.tabs() \u2014 Grafana dashboard selectbox was
    # causing the page to jump back to "Infrastructure" tab on every rerun.
    _MON_TABS = [
        "\U0001f3d7\ufe0f Infrastructure",
        "\u26a1 API Metrics",
        "\U0001f514 Alerts",
        "\U0001f4c8 Grafana",
    ]
    active_mon = st.radio(
        "Monitoring tab",
        _MON_TABS,
        horizontal=True,
        key="_mon_tab",
        label_visibility="collapsed",
    )
    st.markdown(
        "<hr style='margin:0 0 1rem 0;border-color:#334155;'>",
        unsafe_allow_html=True,
    )

    if active_mon == _MON_TABS[0]:
        _live_infrastructure()
    elif active_mon == _MON_TABS[1]:
        _live_api_metrics()
    elif active_mon == _MON_TABS[2]:
        _live_alerts()
    else:
        _grafana_section()

    # ── Footer ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f"[Prometheus ↗]({_prometheus_url()}) &nbsp;·&nbsp; "
        f"[Grafana ↗]({_grafana_url()}) &nbsp;·&nbsp; "
        f"[Alertmanager ↗]({_alertmanager_url()})"
    )
