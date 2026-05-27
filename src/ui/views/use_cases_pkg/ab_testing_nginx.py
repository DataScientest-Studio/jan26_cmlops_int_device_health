"""Tab — Nginx Traffic Split A/B Testing.

True production-grade A/B testing: Nginx weighted upstream routes live
traffic percentage-wise between champion (api:8000) and challenger
(api_challenger:8000).  Predictions are stored in the database with their
respective model_version, and the Grafana A/B dashboard reflects the split
in real time.

Architecture
~~~~~~~~~~~~
                         ┌─────────────────────────────────┐
                         │        Nginx upstream            │
  /predict ──────────►  │  server api:8000  weight=X       │──► Champion
                         │  server api_challenger weight=Y  │──► Challenger
                         └─────────────────────────────────┘
                              X + Y = 100  (integer weights)

This is the *production-grade* approach.  Nginx decides which backend
serves each request; application code and the client are unaware of the
routing.  Predictions land in the same PostgreSQL table with distinct
model_version values, making Grafana A/B dashboards meaningful.

Compare to the ``A/B Testing`` tab (side-by-side evaluation), where
Streamlit calls *both* containers with the *identical* payload.  That gives
a deterministic offline comparison but does not simulate real traffic split.
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

import numpy as np
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
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_NGINX_CONF = _PROJECT_ROOT / "docker" / "nginx" / "nginx.conf"

# Sentinel comment to identify the champion-only default upstream
_UPSTREAM_PATTERN = re.compile(
    r"(\s+upstream api \{)[^}]+(keepalive \d+;\s+\})",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Nginx helpers
# ---------------------------------------------------------------------------


def _current_upstream_state() -> str:
    """Read the current upstream and split_clients blocks from nginx.conf.

    When A/B split is active, both the upstream block and the split_clients
    block are shown together so operators can see the full routing picture.
    """
    try:
        content = _NGINX_CONF.read_text(encoding="utf-8")
        upstream_m = re.search(r"upstream api \{[^}]+\}", content, re.DOTALL)
        split_m = re.search(
            r"split_clients \$request_id \$ab_backend \{[^}]+\}", content, re.DOTALL
        )
        parts: list[str] = []
        if upstream_m:
            parts.append(upstream_m.group(0).strip())
        # Only show split_clients when A/B is actually active (not the champion-only stub)
        if split_m:
            split_text = split_m.group(0).strip()
            if "challenger" in split_text:
                parts.append(split_text)
        return "\n\n".join(parts) if parts else "(could not parse)"
    except Exception as exc:
        return f"(read error: {exc})"


def _apply_nginx_split(challenger_pct: int) -> tuple[str, int]:
    """
    Rewrite nginx.conf upstream block to split traffic by challenger_pct %.
    Then reload Nginx (zero-downtime).

    Returns (output_message, return_code).
    """
    try:
        content = _NGINX_CONF.read_text(encoding="utf-8")
    except Exception as exc:
        return f"Cannot read nginx.conf: {exc}", 1

    if challenger_pct == 0:
        new_upstream = (
            "    upstream api {\n"
            "        # Champion only — 100 % of traffic\n"
            "        server api:8000;\n"
            "        keepalive 32;\n"
            "    }"
        )
        new_split = "    split_clients $request_id $ab_backend {\n        *     champion;\n    }"
    elif challenger_pct == 100:
        new_upstream = (
            "    upstream api {\n"
            "        # Challenger only — 100 % of traffic\n"
            "        server api_challenger:8000;\n"
            "        keepalive 32;\n"
            "    }"
        )
        new_split = "    split_clients $request_id $ab_backend {\n        *     challenger;\n    }"
    else:
        champ_w = 100 - challenger_pct
        chall_w = challenger_pct
        # The upstream api block MUST only contain the champion server.
        # The challenger routing is handled exclusively via split_clients +
        # the if ($ab_backend = "challenger") block in /predict.
        # Adding api_challenger:8000 to the upstream block causes nginx
        # round-robin to route some "champion" requests to the challenger,
        # resulting in inverted split percentages (e.g. set 40% challenger →
        # observed ~63% challenger because both mechanisms fire simultaneously).
        new_upstream = (
            "    upstream api {\n"
            f"        # A/B split active: champion {champ_w}% / challenger {chall_w}%\n"
            f"        # Challenger routing is handled by split_clients below.\n"
            f"        server api:8000;\n"
            "        keepalive 32;\n"
            "    }"
        )
        new_split = (
            "    split_clients $request_id $ab_backend {\n"
            f"        {chall_w}%   challenger;\n"
            "        *     champion;\n"
            "    }"
        )

    # Replace the upstream block (between "upstream api {" and the matching "}")
    new_content = re.sub(
        r"    upstream api \{[^}]+\}",
        new_upstream,
        content,
        flags=re.DOTALL,
    )

    if new_content == content:
        return "ERROR: upstream api block not found in nginx.conf — no change made.", 1

    # Also update split_clients to match the new split percentage
    new_content = re.sub(
        r"    split_clients \$request_id \$ab_backend \{[^}]+\}",
        new_split,
        new_content,
        flags=re.DOTALL,
    )

    try:
        _NGINX_CONF.write_text(new_content, encoding="utf-8")
    except Exception as exc:
        return f"Cannot write nginx.conf: {exc}", 1

    # Reload Nginx — zero-downtime, no container restart needed
    result = subprocess.run(
        ["docker", "exec", "mlops_nginx", "nginx", "-s", "reload"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
    )
    out = (result.stdout or "").strip()
    err = (result.stderr or "").strip()
    combined = "\n".join(filter(None, [out, err])) or "Nginx reloaded."
    return combined, result.returncode


def _restore_champion_only() -> tuple[str, int]:
    """Restore nginx.conf to champion-only upstream and reload."""
    return _apply_nginx_split(0)


def _nginx_running() -> bool:
    """Return True if mlops_nginx container is running."""
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Running}}", "mlops_nginx"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=5,
    )
    return (result.stdout or "").strip() == "true"


def _challenger_running() -> bool:
    """Return True if mlops_api_challenger container is running."""
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Running}}", "mlops_api_challenger"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=5,
    )
    return (result.stdout or "").strip() == "true"


def _start_challenger(mode: str) -> tuple[str, int]:
    """Start api_challenger via docker-compose ab_testing profile."""
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
            "api_challenger",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        cwd=str(_PROJECT_ROOT),
    )
    out = (result.stdout or "").strip()
    err = (result.stderr or "").strip()
    return "\n".join(filter(None, [out, err])), result.returncode


def _stop_challenger(mode: str) -> tuple[str, int]:
    """Stop api_challenger container."""
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
            "stop",
            "api_challenger",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        cwd=str(_PROJECT_ROOT),
    )
    out = (result.stdout or "").strip()
    err = (result.stderr or "").strip()
    return "\n".join(filter(None, [out, err])), result.returncode


# ---------------------------------------------------------------------------
# Traffic helpers
# ---------------------------------------------------------------------------


def _send_via_nginx(
    nginx_url: str,
    time_values: list[float],
    amplitude_values: list[float],
) -> tuple[dict | None, float, str]:
    """
    Send one prediction through Nginx.
    Returns (response_dict, latency_s, model_version_str).
    """
    t0 = time.monotonic()
    try:
        resp = requests.post(
            f"{nginx_url}/predict",
            json={
                "device_id": "ab_test",
                "time_values": time_values,
                "amplitude_values": amplitude_values,
            },
            headers={"X-API-Key": "dev-key-12345"},
            timeout=15,
        )
        elapsed = time.monotonic() - t0
        resp.raise_for_status()
        data = resp.json()
        model_version = str(
            data.get("model_version") or data.get("registry_version") or data.get("version") or "?"
        )
        return data, elapsed, model_version
    except Exception:
        return None, time.monotonic() - t0, "error"


def _generate_test_signals(n: int, seed: int) -> list[dict]:
    """Generate n mixed healthy/unhealthy signals."""
    from src.signal_processing.signal_generator import generate_signal

    rng = np.random.RandomState(seed)
    signals: list[dict] = []
    for i in range(n):
        if rng.random() < 0.3:
            sig = generate_signal(
                shape_type="lorentzian",
                mu=float(rng.uniform(42, 58)),
                width_param=float(rng.uniform(3.8, 5.1) * GAMMA_SIGMA_FACTOR),
                height=float(rng.uniform(1.0, 1.5)),
                noise_level=float(rng.uniform(0.06, 0.10)),
                seed=seed + i,
            )
            expected = 1
        else:
            sig = generate_signal(
                shape_type="gaussian",
                mu=float(rng.uniform(48, 52)),
                width_param=float(rng.uniform(2.0, 3.0)),
                height=float(rng.uniform(2.5, 3.0)),
                noise_level=float(rng.uniform(0.01, 0.02)),
                seed=seed + i,
            )
            expected = 0
        signals.append(
            {
                "time_values": list(sig.signal.time),
                "amplitude": list(sig.signal.amplitude),
                "expected_label": expected,
            }
        )
    return signals


def _run_nginx_traffic_test(
    nginx_url: str,
    n_signals: int,
    seed: int,
    progress_container,
) -> dict:
    """Send n_signals through Nginx and collect model_version distribution."""
    signals = _generate_test_signals(n_signals, seed)
    bar = progress_container.progress(0, text="Sending signals through Nginx…")

    results_by_version: dict[str, list] = {}  # model_version → list of latencies
    errors = 0

    for i, sig in enumerate(signals):
        bar.progress((i + 1) / n_signals, text=f"Signal {i + 1}/{n_signals}…")
        data, latency_s, model_version = _send_via_nginx(
            nginx_url,
            sig["time_values"],
            sig["amplitude"],
        )
        if data is None:
            errors += 1
            model_version = "error"
        results_by_version.setdefault(model_version, []).append(latency_s * 1000)

    bar.empty()

    total_ok = n_signals - errors
    summary = []
    for mv, lats in sorted(results_by_version.items()):
        if mv == "error":
            continue
        summary.append(
            {
                "model_version": mv,
                "requests": len(lats),
                "share_pct": f"{len(lats) / n_signals * 100:.1f}%",
                "mean_latency_ms": f"{np.mean(lats):.1f}",
                "p95_latency_ms": f"{np.percentile(lats, 95):.1f}",
            }
        )

    return {
        "total": n_signals,
        "errors": errors,
        "ok": total_ok,
        "summary": summary,
        "raw": results_by_version,
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def render_ab_testing_nginx_tab(mode: str) -> None:
    """Render the Nginx Traffic Split A/B Testing use case tab."""
    import os as _os

    st.markdown(SECTION_CSS, unsafe_allow_html=True)

    # In K8s mode the Docker nginx container doesn't exist — nginx runs as a
    # K8s deployment and its config is managed via ConfigMap (not a local file).
    # The live‑rewrite‑and‑reload workflow is Docker‑only.
    if _os.environ.get("DEPLOYMENT_MODE", "") == "k8s":
        st.info(
            "⚙️ **Nginx Traffic Split is not available in K8s mode.**\n\n"
            "This feature rewrites the local `docker/nginx/nginx.conf` and reloads "
            "the Docker nginx container. In K8s, nginx is configured via a ConfigMap "
            "(`k8s/base/nginx/configmap.yaml`). Edit the ConfigMap and run "
            "`kubectl apply -k k8s/base/` to update the K8s nginx configuration.",
            icon="ℹ️",
        )
        return

    # ── Hero description ────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#1e1b4b,#312e81);
                    border-radius:12px;padding:1.25rem 1.5rem;margin-bottom:1rem;
                    border-left:4px solid #818cf8;">
          <h3 style="color:#e0e7ff;margin:0 0 0.5rem 0;">⚖️ Nginx Traffic Split</h3>
          <p style="color:#c7d2fe;margin:0;line-height:1.6;">
            True production-grade A/B testing — Nginx routes live traffic
            <strong style="color:#fff">percentage-wise</strong> to the champion
            and challenger API.  Each model serves a real share of requests and
            stores predictions (with its own <code>model_version</code>) in the
            database, making the Grafana A/B dashboard statistically meaningful.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Concept explanation ─────────────────────────────────────────────────
    with st.expander("📖 How this differs from the A/B Testing tab", expanded=False):
        st.markdown("""
## Side-by-Side Evaluation vs. Traffic Split

| Aspect | A/B Testing tab | **This tab (Nginx Traffic Split)** |
|---|---|---|
| **Routing** | Streamlit calls *both* containers with the same payload | Nginx distributes individual requests to champion **or** challenger |
| **Traffic share** | Always 50/50 (identical batch to each) | Configurable 0–100 % per backend |
| **Client awareness** | Streamlit knows which container answered | Client (Streamlit) does not know which backend served a request |
| **Grafana A/B dashboard** | Both model versions always appear equally | Shows *actual* traffic proportion — meaningful split |
| **Use for project defense** | Fast offline comparison on labeled data | Realistic production A/B scenario |
| **Analogous to production** | Canary testing offline / shadow mode | Real canary deployment |

---

## How Nginx Weighted Routing Works

Nginx's `upstream` block supports integer weights:

```nginx
upstream api {
    server api:8000             weight=80;  # champion  — 80 %
    server api_challenger:8000  weight=20;  # challenger — 20 %
    keepalive 32;
}
```

Nginx uses weighted round-robin: for every 100 requests (approximately),
80 go to the champion and 20 to the challenger.  The routing is done at the
load balancer — neither container nor client knows about the other backend.

Reloading Nginx (`nginx -s reload`) applies the new weights **with zero
downtime**: active connections are served by the old worker while new
connections use the updated config.

---

## Typical A/B Testing in Practice

In production MLOps, a traffic split deployment usually follows these steps:

1. **Train challenger** — a new model version is trained and evaluated in MLflow.
2. **Assign challenger alias** — the new version is tagged as `@challenger` in the registry.
3. **Canary release** — 5–10 % of live traffic is routed to the challenger.
4. **Monitor** — Grafana/Prometheus tracks accuracy drift, latency, error rate **per model_version**.
5. **Promote or rollback** — if the challenger outperforms the champion (or at least matches it), it is promoted; otherwise traffic is rolled back to 0 % for the challenger.

This tab lets you demonstrate **steps 3–5** with a full-spectrum slider (0–100 %)
and a live Grafana view.

---

## Connection to Grafana A/B Dashboard

The Grafana **A/B Testing** dashboard queries the `predictions` table and groups
panels by `model_version`.  With true Nginx routing, each prediction row carries
the model version that actually served it, so the panels accurately reflect:

- Prediction count per model version over time
- Accuracy (F1/accuracy) split
- Latency percentiles per backend
- Error rates per backend

The dashboard becomes a **live leaderboard** that updates as more traffic flows.
        """)

    # ── MLflow model info ────────────────────────────────────────────────────
    try:
        client, uri = get_mlflow_client()
    except Exception as exc:
        st.error(f"Cannot connect to MLflow: {exc}")
        return

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
            "Both **champion** and **challenger** must exist in the MLflow registry "
            "before starting a traffic-split test."
        )
        return

    st.markdown("---")

    # ── Infrastructure status ────────────────────────────────────────────────
    st.markdown("#### 🔌 Infrastructure Status")
    inf1, inf2 = st.columns(2)
    with inf1:
        if _nginx_running():
            st.success("**Nginx** — running ✓")
        else:
            st.error("**Nginx** — not running.  Start the full stack first.")
    with inf2:
        if _challenger_running():
            st.success("**Challenger API** — running ✓")
        else:
            st.warning("**Challenger API** — not running.  Deploy it below.")

    st.markdown("---")

    # ── Traffic split slider ─────────────────────────────────────────────────
    st.markdown("#### 🎚️ Traffic Split Configuration")

    challenger_pct = st.slider(
        "Challenger traffic share (%)",
        min_value=0,
        max_value=100,
        value=st.session_state.get("_ab_nginx_challenger_pct", 20),
        step=5,
        key="_ab_nginx_slider",
        help=(
            "0 % = all traffic to champion only.  "
            "100 % = all traffic to challenger only.  "
            "50 % = equal split."
        ),
    )
    champion_pct = 100 - challenger_pct

    # Visual split indicator
    bar_html = f"""
    <div style="display:flex;border-radius:8px;overflow:hidden;height:28px;margin:0.5rem 0 1rem 0;
                font-size:0.8rem;font-weight:600;">
      <div style="width:{champion_pct}%;background:#4f46e5;display:flex;align-items:center;
                  justify-content:center;color:white;min-width:0;overflow:hidden;
                  transition:width 0.3s;">
        {"🏆 " + str(champion_pct) + "%" if champion_pct >= 15 else ""}
      </div>
      <div style="width:{challenger_pct}%;background:#dc2626;display:flex;align-items:center;
                  justify-content:center;color:white;min-width:0;overflow:hidden;
                  transition:width 0.3s;">
        {"🥊 " + str(challenger_pct) + "%" if challenger_pct >= 15 else ""}
      </div>
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)
    st.caption(
        f"Champion (api): **{champion_pct}%**  •  "
        f"Challenger (api_challenger): **{challenger_pct}%**  •  "
        f"Nginx weights: champion={champion_pct}, challenger={challenger_pct}"
    )

    st.markdown("---")

    # ── Deploy / teardown controls ───────────────────────────────────────────
    st.markdown("#### 🚀 Deploy Controls")

    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
    with btn_col1:
        deploy_btn = st.button(
            "🚀 Apply Traffic Split",
            key="_ab_nginx_deploy",
            type="primary",
            help=(
                "1. Start challenger container if needed\n"
                "2. Patch nginx.conf with selected weights\n"
                "3. Reload Nginx (zero-downtime)"
            ),
        )
    with btn_col2:
        stop_btn = st.button(
            "⛔ Restore Champion-Only",
            key="_ab_nginx_stop",
            type="secondary",
            help="Remove challenger from nginx upstream and stop challenger container.",
        )

    if deploy_btn:
        if not _nginx_running():
            st.error("Nginx is not running. Start the full Docker stack first.")
        else:
            # Step 1: start challenger if not running
            if not _challenger_running():
                with st.spinner("Starting challenger container…"):
                    out, rc = _start_challenger(mode)
                if rc != 0:
                    st.error(f"Failed to start challenger:\n```\n{out}\n```")
                    st.stop()
                else:
                    st.success("Challenger container started.")
                    time.sleep(3)  # brief wait for healthcheck

            # Step 2: apply nginx split
            with st.spinner("Patching nginx.conf and reloading Nginx…"):
                out, rc = _apply_nginx_split(challenger_pct)
            if rc == 0:
                st.session_state["_ab_nginx_active"] = True
                st.session_state["_ab_nginx_challenger_pct"] = challenger_pct
                st.success(
                    f"Traffic split applied: **Champion {champion_pct}%** / "
                    f"**Challenger {challenger_pct}%**.  "
                    f"Nginx reloaded.  {out}"
                )
            else:
                st.error(f"Nginx reload failed (rc={rc}):\n```\n{out}\n```")

    if stop_btn:
        with st.spinner("Restoring champion-only routing…"):
            out_nginx, rc_nginx = _restore_champion_only()
        with st.spinner("Stopping challenger container…"):
            out_chall, rc_chall = _stop_challenger(mode)
        st.session_state.pop("_ab_nginx_active", None)
        st.session_state.pop("_ab_nginx_results", None)
        if rc_nginx == 0:
            st.success("Champion-only routing restored.  Nginx reloaded.")
        else:
            st.warning(f"Nginx reload output: {out_nginx}")

    st.markdown("---")

    # ── Current nginx upstream state ─────────────────────────────────────────
    with st.expander("🔍 Current nginx.conf upstream block", expanded=False):
        st.code(_current_upstream_state(), language="nginx")

    st.markdown("---")

    # ── Send test traffic ────────────────────────────────────────────────────
    active = st.session_state.get("_ab_nginx_active", False)

    if not active:
        st.info(
            "Apply a traffic split above before sending test traffic.  "
            "The challenger container must be running for the split to work."
        )
        return

    st.markdown("#### 📡 Send Test Traffic Through Nginx")
    st.markdown(
        "Signals are sent **exclusively through Nginx** (port 80).  "
        "Nginx routes each request to champion or challenger according to the "
        "configured weights.  Both backends store their predictions in the "
        "database with their own `model_version` label."
    )

    tc1, tc2 = st.columns(2)
    with tc1:
        n_signals = st.slider(
            "Number of test signals",
            min_value=10,
            max_value=500,
            value=150,
            step=10,
            key="_ab_nginx_n",
        )
    with tc2:
        seed = st.number_input("Random seed", value=42, step=1, key="_ab_nginx_seed")

    if n_signals < 100:
        import math as _math

        _p = st.session_state.get("_ab_nginx_challenger_pct", 40) / 100.0
        _std_pct = _math.sqrt(_p * (1 - _p) / n_signals) * 100
        st.warning(
            f"**Small sample warning:** With {n_signals} signals, the observed challenger "
            f"share can deviate \u00b1{2 * _std_pct:.0f}% (2\u03c3) from the configured split — "
            "even with a perfect nginx configuration.  "
            "Use **150+ signals** for reliable distribution statistics."
        )

    from src.ui.components.docker_utils import get_host, get_service_url

    nginx_url = get_service_url("mlops_nginx", 80) or f"http://{get_host()}:80"

    rc1, rc2, rc3 = st.columns([1, 1, 2])
    with rc1:
        run_btn = st.button("▶️ Send Traffic", key="_ab_nginx_run", type="primary")
    with rc2:
        clear_btn = st.button("🔄 Clear Results", key="_ab_nginx_clear")

    if run_btn:
        prog = st.container()
        results = _run_nginx_traffic_test(nginx_url, n_signals, int(seed), prog)
        results["configured_challenger_pct"] = st.session_state.get(
            "_ab_nginx_challenger_pct", None
        )
        st.session_state["_ab_nginx_results"] = results

    if clear_btn:
        st.session_state.pop("_ab_nginx_results", None)
        st.rerun()

    # ── Results ──────────────────────────────────────────────────────────────
    cached: dict | None = st.session_state.get("_ab_nginx_results")
    if cached:
        _render_traffic_results(cached, n_signals=cached["total"])


def _render_traffic_results(results: dict, n_signals: int) -> None:
    """Render the traffic split results."""
    import math as _math

    import plotly.graph_objects as go

    st.markdown("---")
    st.markdown("#### 📊 Traffic Distribution Results")

    total = results["total"]
    errors = results["errors"]
    ok = results["ok"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Total signals sent", total)
    m2.metric("Successfully routed", ok)
    m3.metric("Errors", errors)

    # Statistical note: even a correct nginx split will show variance with small N
    configured_pct = results.get("configured_challenger_pct")
    if configured_pct is not None and total >= 10:
        p = configured_pct / 100.0
        std_pct = _math.sqrt(p * (1 - p) / total) * 100 if total > 0 else 0
        ci_lo = max(0, configured_pct - 2 * std_pct)
        ci_hi = min(100, configured_pct + 2 * std_pct)
        st.info(
            f"**Configured challenger share:** {configured_pct}%  \n"
            f"**Expected 95% range** (statistical noise for n={total}): "
            f"{ci_lo:.0f}% – {ci_hi:.0f}% challenger  \n"
            "Observed share outside this range indicates a configuration issue; "
            "within the range is normal statistical variance."
        )

    summary = results["summary"]
    if not summary:
        if errors == total:
            st.error(
                "All requests failed.  Check that Nginx is running and that the "
                "traffic split has been applied.  Also verify that both API containers "
                "are healthy before sending traffic."
            )
        else:
            st.warning(
                "Requests succeeded but no `model_version` was found in the responses.  "
                "The distribution cannot be determined from the response payload.  "
                "Check the **Grafana A/B dashboard** to see model_version-split data "
                "from the `predictions` table."
            )
        return

    # ── Distribution table ───────────────────────────────────────────────────
    import pandas as pd

    st.markdown("##### Distribution by model_version")
    st.dataframe(
        pd.DataFrame(summary).rename(
            columns={
                "model_version": "Model Version",
                "requests": "Requests",
                "share_pct": "Share",
                "mean_latency_ms": "Mean Latency (ms)",
                "p95_latency_ms": "p95 Latency (ms)",
            }
        ),
        hide_index=True,
    )

    # ── Pie chart ────────────────────────────────────────────────────────────
    raw = {k: len(v) for k, v in results["raw"].items() if k != "error"}
    if raw:
        labels = list(raw.keys())
        values = list(raw.values())
        colors = ["#4f46e5", "#dc2626", "#16a34a", "#ca8a04"]

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.45,
                    marker_colors=colors[: len(labels)],
                    textinfo="label+percent",
                    textfont_size=13,
                )
            ]
        )
        fig.update_layout(
            title="Nginx Routing Distribution",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            height=350,
            margin={"t": 50, "b": 20, "l": 20, "r": 20},
            showlegend=True,
            legend={"orientation": "h", "yanchor": "bottom", "y": -0.15},
        )
        st.plotly_chart(fig)

    # ── Grafana link ─────────────────────────────────────────────────────────
    st.info(
        "📈 **Open the Grafana A/B Dashboard** to see latency, accuracy, and "
        "prediction count split by `model_version` in real time.  "
        "The dashboard updates as predictions land in the database."
    )
