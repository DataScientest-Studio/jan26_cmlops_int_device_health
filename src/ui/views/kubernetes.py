"""☸️ Kubernetes — cluster control, pod management, scaling, and resilience dashboard."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from src.ui.logging_ui import get_ui_logger
from src.ui.styles import get_global_css, hero_section

_logger = get_ui_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_K8S_NAMESPACE = os.environ.get("K8S_NAMESPACE", "mlops")
_PROJECT_ROOT_PATH = Path(__file__).resolve().parents[3]

# Deployments that can be scaled via the UI
_SCALABLE_DEPLOYMENTS = ["api", "streamlit", "airflow", "mlflow", "prometheus", "grafana"]

# Overlay options (maps display label → Makefile K8S_OVERLAY value)
_OVERLAYS = {
    "local — Docker Desktop K8s (recommended for development)": "local",
    "cloud — Cloud overlay (3 API replicas, production-like)": "cloud",
    "ghcr — GHCR images (CI/CD mode, pulls from ghcr.io)": "ghcr",
}

# GitHub repo details for workflow dispatch
_GH_OWNER = os.environ.get("GITHUB_OWNER", "your-github-username")
_GH_REPO = "mlops-device-health"
_DEPLOY_WORKFLOW = "deploy-k8s.yml"


# ── Helpers ──────────────────────────────────────────────────────────────────

# Base URL for internal API calls (K8s operations proxied through FastAPI)
_K8S_API_BASE = os.environ.get("K8S_API_BASE", "http://127.0.0.1:8000")


def _api_get(path: str) -> dict | None:
    """GET ``path`` from the internal API and return parsed JSON, or None on any error."""
    import urllib.error
    import urllib.request

    url = f"{_K8S_API_BASE}{path}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
            return json.loads(resp.read())
    except Exception:
        _logger.warning("API GET %s failed: %s", path, "request error")
        return None


def _api_post(path: str, payload: dict | None = None) -> dict | None:
    """POST to ``path`` on the internal API and return parsed JSON, or None on any error.

    The K8s proxy endpoints accept an empty JSON body ``{}``.
    """
    import urllib.error
    import urllib.request

    url = f"{_K8S_API_BASE}{path}"
    data = b"{}"  # K8s proxy endpoints accept empty body
    req = urllib.request.Request(url, data=data, method="POST")  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            return json.loads(resp.read())
    except Exception:
        _logger.warning("API POST %s failed: %s", path, "request error")
        return None


def _run_make(target: str, extra_vars: dict[str, str] | None = None) -> tuple[int, str]:
    """
    Run a Makefile target and return (returncode, combined stdout+stderr).

    Runs in the project root directory using subprocess so that the terminal
    output is captured and can be displayed in the Streamlit UI.
    """
    cmd = ["make", target]
    if extra_vars:
        cmd += [f"{k}={v}" for k, v in extra_vars.items()]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(_PROJECT_ROOT_PATH),
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = (result.stdout or "") + (result.stderr or "")
        return result.returncode, output
    except subprocess.TimeoutExpired:
        return 1, "⏱ Timed out after 5 minutes."
    except FileNotFoundError:
        return 1, "`make` command not found. Install GNU Make and ensure it is on PATH."
    except Exception as exc:
        return 1, f"Unexpected error: {exc}"


def _kubectl_available() -> bool:
    """Return True if kubectl is on PATH."""
    try:
        subprocess.run(
            ["kubectl", "version", "--client", "--short"], capture_output=True, timeout=5
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _k8s_context() -> str:
    """Return the current kubectl context name."""
    try:
        result = subprocess.run(
            ["kubectl", "config", "current-context"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unavailable"


def _namespace_exists() -> bool:
    """Return True if the mlops namespace exists in the current cluster."""
    try:
        result = subprocess.run(
            ["kubectl", "get", "namespace", _K8S_NAMESPACE],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


# ── Direct kubectl pod operations (no in-cluster API needed) ─────────────────


def _kubectl_get_pods() -> list[dict[str, Any]]:
    """
    Return live pod info from kubectl, parsed from JSON.
    Works from the host machine — no port-forward or in-cluster API required.
    Returns a list of dicts with keys: name, status, ready, restarts, node, labels.
    """
    try:
        result = subprocess.run(
            ["kubectl", "get", "pods", "-n", _K8S_NAMESPACE, "-o", "json"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            _logger.warning("kubectl get pods failed: %s", result.stderr[:200])
            return []
        data = json.loads(result.stdout)
        pods: list[dict[str, Any]] = []
        for item in data.get("items", []):
            meta = item.get("metadata", {})
            status_obj = item.get("status", {})
            phase = status_obj.get("phase", "Unknown")
            cs_list = status_obj.get("containerStatuses") or []
            ready = bool(cs_list and cs_list[0].get("ready", False))
            restarts = cs_list[0].get("restartCount", 0) if cs_list else 0
            pods.append(
                {
                    "name": meta.get("name", "?"),
                    "status": phase,
                    "ready": ready,
                    "restarts": restarts,
                    "node": item.get("spec", {}).get("nodeName", ""),
                    "labels": meta.get("labels", {}),
                    "app": meta.get("labels", {}).get("app", ""),
                }
            )
        return pods
    except Exception as exc:
        _logger.warning("kubectl get pods error: %s", exc)
        return []


def _kubectl_scale(deployment: str, replicas: int) -> tuple[bool, str]:
    """
    Scale a deployment using kubectl.
    Returns (success, message).
    """
    try:
        result = subprocess.run(
            [
                "kubectl",
                "scale",
                f"deployment/{deployment}",
                f"--replicas={replicas}",
                "-n",
                _K8S_NAMESPACE,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return True, f"deployment.apps/{deployment} scaled"
        return False, result.stderr.strip() or result.stdout.strip()
    except subprocess.TimeoutExpired:
        return False, "kubectl scale timed out after 30 s"
    except Exception as exc:
        return False, str(exc)


def _kubectl_delete_pod(pod_name: str) -> tuple[bool, str]:
    """
    Delete (kill) a pod by name using kubectl. The Deployment will recreate it.
    Returns (success, message).
    """
    try:
        result = subprocess.run(
            ["kubectl", "delete", "pod", pod_name, "-n", _K8S_NAMESPACE, "--grace-period=0"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "kubectl delete pod timed out after 30 s"
    except Exception as exc:
        return False, str(exc)


# ── GitHub helpers ────────────────────────────────────────────────────────────


def _gh_token() -> str | None:
    """Read GitHub token from env or .env.secrets file."""
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        return token
    secrets_path = _PROJECT_ROOT_PATH / ".env.secrets"
    if secrets_path.exists():
        for line in secrets_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("GITHUB_TOKEN=") or line.startswith("GH_TOKEN="):
                _, _, val = line.partition("=")
                val = val.strip().strip('"').strip("'")
                if val:
                    return val
    return None


def _trigger_github_workflow(
    overlay: str, reason: str = "Triggered from Streamlit"
) -> tuple[bool, str]:
    """
    Trigger the deploy-k8s.yml workflow via GitHub API workflow_dispatch.
    Returns (success, message).
    """
    import json
    import urllib.error
    import urllib.request

    token = _gh_token()
    if not token:
        return False, (
            "No GitHub token found. Set `GITHUB_TOKEN` in `.env.secrets` "
            "with `workflow` scope to trigger CI/CD deploys."
        )

    url = (
        f"https://api.github.com/repos/{_GH_OWNER}/{_GH_REPO}"
        f"/actions/workflows/{_DEPLOY_WORKFLOW}/dispatches"
    )
    payload = json.dumps(
        {
            "ref": "main",
            "inputs": {
                "overlay": overlay,
                "reason": reason,
            },
        }
    ).encode()

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            # 204 No Content = success
            if resp.status == 204:
                runs_url = (
                    f"https://github.com/{_GH_OWNER}/{_GH_REPO}"
                    f"/actions/workflows/{_DEPLOY_WORKFLOW}"
                )
                return True, (
                    f"✅ Workflow dispatched successfully. [Watch run on GitHub]({runs_url})"
                )
            return False, f"Unexpected status {resp.status}"
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()[:300]
        if exc.code == 404:
            return False, (
                f"Workflow `{_DEPLOY_WORKFLOW}` not found in repo `{_GH_OWNER}/{_GH_REPO}`. "
                "Ensure the workflow file is committed to the `main` branch."
            )
        if exc.code == 422:
            return False, (
                "Workflow dispatch failed (HTTP 422). The `main` branch may not exist "
                f"or the workflow is disabled. Details: {body}"
            )
        return False, f"GitHub API error HTTP {exc.code}: {body}"
    except Exception as exc:
        return False, f"Request failed: {exc}"


def _get_workflow_runs(limit: int = 5) -> list[dict]:
    """Fetch recent runs of the deploy-k8s.yml workflow from GitHub API."""
    import json
    import urllib.error
    import urllib.request

    token = _gh_token()
    url = (
        f"https://api.github.com/repos/{_GH_OWNER}/{_GH_REPO}"
        f"/actions/workflows/{_DEPLOY_WORKFLOW}/runs?per_page={limit}"
    )
    headers: dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return data.get("workflow_runs", [])
    except Exception:
        return []


# ── Status colour helpers ─────────────────────────────────────────────────────

_STATUS_COLOURS: dict[str, str] = {
    "Running": "🟢",
    "Pending": "🟡",
    "Succeeded": "✅",
    "Failed": "🔴",
    "Unknown": "⚪",
}


def _status_icon(phase: str) -> str:
    return _STATUS_COLOURS.get(phase or "Unknown", "⚪")


# ── Main render function ──────────────────────────────────────────────────────


def render() -> None:
    """Render the Kubernetes management page."""
    st.markdown(get_global_css(), unsafe_allow_html=True)
    hero_section(
        title="☸️ Kubernetes",
        subtitle=f"Cluster control, pod management, scaling, and resilience — namespace <code>{_K8S_NAMESPACE}</code>",
    )

    # ── kubectl / cluster status bar ─────────────────────────────────────────
    kubectl_ok = _kubectl_available()
    ns_exists = _namespace_exists() if kubectl_ok else False
    ctx = _k8s_context() if kubectl_ok else "n/a"

    c1, c2, c3 = st.columns(3)
    c1.metric("kubectl", "✅ Available" if kubectl_ok else "❌ Not found")
    c2.metric("Context", ctx)
    c3.metric("mlops namespace", "✅ Running" if ns_exists else "⭕ Not deployed")

    st.info(
        "Pod management, scaling, and resilience controls run directly via **kubectl** — no "
        "port-forward or in-cluster API access required. "
        "Cluster start/stop uses Makefile targets (`make k8s-*`) on the host shell."
    )

    # ── Tab navigation (session-state-backed radio — survives any button rerun) ─
    _TAB_NAMES = [
        "🚀 Cluster Control",
        "📋 Pod List",
        "⚖️ Scale Deployments",
        "💥 Resilience Testing",
    ]
    if "k8s_active_tab" not in st.session_state:
        st.session_state["k8s_active_tab"] = 0

    _active_label = st.radio(
        "##k8s_nav",
        _TAB_NAMES,
        index=st.session_state["k8s_active_tab"],
        horizontal=True,
        key="k8s_tab_radio",
        label_visibility="collapsed",
    )
    st.session_state["k8s_active_tab"] = _TAB_NAMES.index(_active_label)
    st.markdown("---")
    _active_tab = st.session_state["k8s_active_tab"]

    # ── Tab 0: Cluster Control ────────────────────────────────────────────────
    if _active_tab == 0:
        st.markdown("### 🚀 Cluster Control")

        # ── Live running-stack status ────────────────────────────────────────
        if ns_exists:
            live_pods = _kubectl_get_pods()
            running_count = sum(1 for p in live_pods if p["status"] == "Running")
            total_count = len(live_pods)
            if running_count > 0:
                st.success(
                    f"✅ **Stack is deployed and running** — "
                    f"{running_count}/{total_count} pods Running in namespace `{_K8S_NAMESPACE}`"
                )
                # Show a compact pod status table
                cols = st.columns(min(total_count, 5))
                for i, pod in enumerate(live_pods):
                    icon = _status_icon(pod["status"])
                    app_name = pod.get("app") or pod["name"].split("-")[0]
                    rdy = "✅" if pod["ready"] else "⏳"
                    cols[i % len(cols)].markdown(
                        f"**{icon} {app_name}**  \n{rdy} {pod['status']}  \n↩ {pod['restarts']}",
                        help=f"Pod: {pod['name']}",
                    )
                st.caption(
                    "ℹ️ **Access via port-forward** (NodePort is not reachable on Windows):  \n"
                    "`kubectl port-forward -n mlops service/nginx 30080:80`  \n"
                    "Then open → http://localhost:30080 (Streamlit app)  \n"
                    "API: http://localhost:30080/**health** · "
                    "Swagger: http://localhost:30080/**docs** · "
                    "or use the `/api/` prefix: http://localhost:30080/**api**/health"
                )
            else:
                st.warning(
                    f"⚠️ Namespace `{_K8S_NAMESPACE}` exists but {total_count} pod(s) found with "
                    f"no Running pods. The stack may still be initialising — check the Pod List tab."
                )
        else:
            st.info("ℹ️ Namespace `mlops` not found. Deploy the stack with **▶️ Deploy (k8s-up)**.")

        st.markdown("---")
        st.markdown(
            "Start or stop the full MLOps Kubernetes stack using the same Makefile targets "
            "you would run from the terminal. "
            "Images must be built first with **Build Images** before the first deploy."
        )

        st.markdown("#### Select Overlay")
        overlay_label = st.selectbox(
            "Deployment overlay",
            options=list(_OVERLAYS.keys()),
            index=0,
            key="k8s_overlay_select",
            help=(
                "**local** — single replica API, ideal for Docker Desktop K8s. "
                "**cloud** — 3 API replicas + HPA, production-like."
            ),
        )
        overlay = _OVERLAYS[overlay_label]

        st.markdown(
            "> ℹ️ **GHCR mode** is not a separate K8s overlay. "
            "To use GHCR-built images in Kubernetes, first pull them with `make ghcr` in Docker "
            "Compose mode (they are cached to your local Docker daemon), then deploy with "
            "`make k8s-up`. "
            "Tag the images with `docker tag ghcr.io/… mlops-device-health/api:latest` if needed."
        )

        st.markdown("---")
        col_build, col_up, col_down, col_nuke = st.columns(4)

        with col_build:
            if st.button("🔨 Build Images", key="k8s_build", use_container_width=True):
                _logger.info("K8s build images requested")
                with st.spinner("Building Docker images for Kubernetes…"):
                    rc, out = _run_make("k8s-build")
                if rc == 0:
                    _logger.info("K8s build succeeded")
                    st.success("✅ Images built successfully.")
                else:
                    _logger.error("K8s build failed rc={}", rc)
                    st.error(f"❌ Build failed (exit {rc}).")
                st.code(out[-3000:], language="bash")

        with col_up:
            if st.button(
                "▶️ Deploy (k8s-up)", key="k8s_up", type="primary", use_container_width=True
            ):
                _logger.info("K8s deploy requested overlay={}", overlay)
                with st.spinner(f"Deploying stack to K8s (overlay={overlay})…"):
                    rc, out = _run_make("k8s-up", {"K8S_OVERLAY": overlay})
                if rc == 0:
                    _logger.info("K8s deploy succeeded overlay={}", overlay)
                    st.success(
                        "✅ Stack deployed!  \n"
                        "Use `kubectl port-forward -n mlops service/nginx 30080:80` to access. "
                        "Then open http://localhost:30080"
                    )
                else:
                    _logger.error("K8s deploy failed rc={} overlay={}", rc, overlay)
                    st.error(f"❌ Deploy failed (exit {rc}).")
                st.code(out[-3000:], language="bash")

        with col_down:
            if st.button("⏹ Teardown (k8s-down)", key="k8s_down", use_container_width=True):
                _logger.info("K8s teardown requested")
                with st.spinner("Tearing down K8s stack (PVCs kept)…"):
                    rc, out = _run_make("k8s-down", {"K8S_OVERLAY": overlay})
                if rc == 0:
                    _logger.info("K8s teardown succeeded")
                    st.success("✅ Stack removed (data PVCs preserved).")
                else:
                    _logger.error("K8s teardown failed rc={}", rc)
                    st.error(f"❌ Teardown failed (exit {rc}).")
                st.code(out[-3000:], language="bash")

        with col_nuke:
            nuke_confirmed = st.checkbox(
                "Confirm full wipe (deletes PVCs / data)",
                value=False,
                key="k8s_nuke_confirm",
            )
            if st.button(
                "💣 Nuke (k8s-nuke)",
                key="k8s_nuke",
                disabled=not nuke_confirmed,
                use_container_width=True,
            ):
                _logger.warning("K8s nuke requested — deleting all PVCs")
                with st.spinner("Nuking K8s stack including all PVCs…"):
                    rc, out = _run_make("k8s-nuke", {"K8S_OVERLAY": overlay})
                if rc == 0:
                    _logger.info("K8s nuke succeeded")
                    st.success("✅ Stack and all PVCs deleted.")
                else:
                    _logger.error("K8s nuke failed rc={}", rc)
                    st.error(f"❌ Nuke failed (exit {rc}).")
                st.code(out[-3000:], language="bash")

        st.markdown("---")
        st.markdown("#### 📊 Quick Status")
        if st.button("🔄 Refresh Status", key="k8s_ctrl_status"):
            _logger.info("K8s status refresh requested")
            with st.spinner("Running kubectl get pods/deployments/services…"):
                rc, out = _run_make("k8s-status")
            st.code(out[-3000:], language="bash")

        st.markdown("---")
        st.markdown("#### 🤖 CI/CD Deploy (GitHub Actions)")
        st.markdown(
            "Trigger the **`deploy-k8s.yml`** GitHub Actions workflow remotely. "
            "The workflow spins up a Kind cluster in CI, pulls the latest GHCR images, "
            "deploys the K8s manifests, and smoke-tests the API `/health` endpoint. "
            "Requires `GITHUB_TOKEN` with **`workflow`** scope in `.env.secrets`."
        )

        has_token = _gh_token() is not None
        if not has_token:
            st.warning(
                "⚠️ No `GITHUB_TOKEN` found. "
                "Add it to `.env.secrets` with `workflow` scope to enable CI/CD triggering."
            )

        ci_overlay = st.selectbox(
            "CI overlay",
            options=["ghcr", "local", "cloud"],
            index=0,
            key="k8s_ci_overlay",
            help=(
                "**ghcr** — deploys GHCR-pushed images (requires successful build workflow). "
                "**local** — builds images from source inside CI. "
                "**cloud** — cloud production-like settings."
            ),
            disabled=not has_token,
        )
        ci_reason = st.text_input(
            "Reason (optional)",
            value="Triggered from Streamlit",
            key="k8s_ci_reason",
            disabled=not has_token,
        )

        if st.button(
            "🚀 Trigger CI/CD Deploy",
            key="k8s_cicd_trigger",
            type="primary",
            disabled=not has_token,
            use_container_width=True,
        ):
            with st.spinner("Dispatching workflow to GitHub Actions…"):
                ok, msg = _trigger_github_workflow(ci_overlay, ci_reason)
            if ok:
                _logger.info(
                    "CI/CD workflow dispatched overlay={} reason={}", ci_overlay, ci_reason
                )
                st.success(msg)
            else:
                _logger.error("CI/CD workflow dispatch failed: {}", msg)
                st.error(f"❌ {msg}")

        st.markdown("---")
        st.markdown("#### 📜 Recent CI/CD Workflow Runs")
        if st.button("🔄 Refresh Runs", key="k8s_runs_refresh"):
            runs = _get_workflow_runs(limit=8)
            if not runs:
                st.info(
                    "No runs found — or `GITHUB_TOKEN` missing. "
                    f"[View on GitHub](https://github.com/{_GH_OWNER}/{_GH_REPO}"
                    f"/actions/workflows/{_DEPLOY_WORKFLOW})"
                )
            else:
                _STATUS_ICONS = {
                    "success": "✅",
                    "failure": "❌",
                    "cancelled": "🚫",
                    "in_progress": "⏳",
                    "queued": "🟡",
                    "skipped": "⏩",
                }
                for run in runs:
                    concl = run.get("conclusion") or run.get("status") or "unknown"
                    icon = _STATUS_ICONS.get(concl, "⚪")
                    title = f"{icon} #{run['run_number']} — {run['display_title'][:60]}"
                    with st.expander(title, expanded=False):
                        c1, c2, c3 = st.columns(3)
                        c1.caption(f"**Status:** {concl}")
                        c2.caption(f"**Branch:** {run.get('head_branch', '?')}")
                        c3.caption(f"**Triggered:** {run.get('event', '?')}")
                        st.markdown(f"[🔗 Open on GitHub]({run['html_url']})")

    # ── Tab 1: Pod List ───────────────────────────────────────────────────────
    elif _active_tab == 1:
        st.markdown("### 📋 Pod Status")
        st.caption(
            "Live pod list via **kubectl** — no port-forward required. "
            "Click Refresh to reload without leaving this tab."
        )

        # Use a session-state counter as a cache-buster. Incrementing it forces
        # _kubectl_get_pods to be called fresh on the *same* rerun without
        # navigating away from this tab (no st.cache_data.clear() + no st.rerun()).
        if "k8s_pod_refresh_v" not in st.session_state:
            st.session_state["k8s_pod_refresh_v"] = 0

        col_refresh, col_auto = st.columns([1, 4])
        with col_refresh:
            if st.button("🔄 Refresh", key="k8s_refresh"):
                st.session_state["k8s_pod_refresh_v"] += 1

        with col_auto:
            auto_refresh = st.checkbox(
                "Auto-refresh every 15 s",
                value=False,
                key="k8s_auto_refresh",
            )

        if not kubectl_ok:
            st.error("❌ kubectl is not available. Install kubectl and ensure it is on your PATH.")
        elif not ns_exists:
            st.info(
                f"Namespace `{_K8S_NAMESPACE}` does not exist. "
                "Deploy the stack first from the Cluster Control tab."
            )
        else:
            pods = _kubectl_get_pods()
            if not pods:
                st.info(f"No pods found in namespace `{_K8S_NAMESPACE}`.")
            else:
                # Summary metrics
                running = sum(1 for p in pods if p["status"] == "Running")
                pending = sum(1 for p in pods if p["status"] == "Pending")
                failed = sum(1 for p in pods if p["status"] == "Failed")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total pods", len(pods))
                m2.metric("🟢 Running", running)
                m3.metric("🟡 Pending", pending)
                m4.metric("🔴 Failed", failed)

                st.markdown("---")

                # Pod table
                for pod in pods:
                    icon = _status_icon(pod["status"])
                    ready_icon = "✅" if pod["ready"] else "⏳"
                    with st.expander(
                        f"{icon} `{pod['name']}` — {pod['status']} {ready_icon}",
                        expanded=False,
                    ):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Status", pod["status"])
                        c2.metric("Ready", "Yes" if pod["ready"] else "No")
                        c3.metric("Restarts", pod["restarts"])
                        if pod.get("node"):
                            st.caption(f"Node: `{pod['node']}`")
                        if pod.get("labels"):
                            st.caption(
                                "Labels: "
                                + ", ".join(f"`{k}={v}`" for k, v in pod["labels"].items())
                            )

        if auto_refresh:
            time.sleep(15)
            st.rerun()

    # ── Tab 2: Scale Deployments ─────────────────────────────────────────────
    elif _active_tab == 2:
        st.markdown("### ⚖️ Scale Deployments")
        st.caption(
            "Adjust replica count for any deployment via **kubectl** — works from the host. "
            "The HPA will override manual scaling if CPU utilisation exceeds 60 %."
        )

        with st.form("k8s_scale_form"):
            deployment = st.selectbox(
                "Deployment",
                options=_SCALABLE_DEPLOYMENTS,
                index=0,
                key="k8s_scale_deployment",
            )
            replicas = st.slider(
                "Replica count",
                min_value=1,
                max_value=10,
                value=3,
                key="k8s_scale_replicas",
            )
            submitted = st.form_submit_button("⚖️ Scale", type="primary")

        if submitted:
            if not kubectl_ok:
                st.error("❌ kubectl not available.")
            elif not ns_exists:
                st.error(f"❌ Namespace `{_K8S_NAMESPACE}` not found. Is the stack deployed?")
            else:
                _logger.info("Scaling deployment={} replicas={}", deployment, replicas)
                with st.spinner(f"Scaling `{deployment}` to {replicas} replicas…"):
                    ok, msg = _kubectl_scale(deployment, replicas)
                if ok:
                    _logger.info("Scale succeeded deployment={} replicas={}", deployment, replicas)
                    st.success(
                        f"✅ Scaled `{deployment}` to **{replicas}** replica(s) "
                        f"in namespace `{_K8S_NAMESPACE}`.  \n`{msg}`"
                    )
                else:
                    _logger.error("Scale failed deployment={}: {}", deployment, msg)
                    st.error(f"❌ Failed to scale `{deployment}`: {msg}")

    # ── Tab 3: Resilience Testing ─────────────────────────────────────────────
    elif _active_tab == 3:
        st.markdown("### 💥 Resilience Testing")
        st.warning(
            "⚠️ **Pod Kill** terminates a running pod. "
            "The Deployment controller will schedule a replacement immediately. "
            "Use this to demonstrate self-healing and rolling restarts."
        )

        if not kubectl_ok:
            st.error("❌ kubectl not available.")
        elif not ns_exists:
            st.info(
                f"Namespace `{_K8S_NAMESPACE}` not found. "
                "Deploy the stack first from the Cluster Control tab."
            )
        else:
            # Fetch running pods directly via kubectl
            all_pods = _kubectl_get_pods()
            pod_names = [p["name"] for p in all_pods if p["status"] == "Running"]

            if not pod_names:
                st.info(
                    f"No running pods found in namespace `{_K8S_NAMESPACE}`. "
                    "The stack may still be starting up — check the Pod List tab."
                )
            else:
                # Show persistent result from the previous kill (survives st.rerun).
                # Without this, the result message disappears almost instantly because
                # the immediate st.rerun() clears all ephemeral st.success/st.error calls.
                if "k8s_kill_result" in st.session_state:
                    _res = st.session_state["k8s_kill_result"]
                    if _res["ok"]:
                        st.success(_res["msg"])
                    else:
                        st.error(_res["msg"])
                    if st.button("✖ Dismiss", key="k8s_kill_dismiss"):
                        del st.session_state["k8s_kill_result"]
                        st.rerun()

                # confirmed checkbox MUST be outside the form so that checking it
                # triggers an immediate rerun, updating `disabled` on the submit button.
                confirmed = st.checkbox(
                    "I confirm I want to terminate this pod",
                    value=False,
                    key="k8s_kill_confirm",
                )
                with st.form("k8s_kill_form"):
                    pod_to_kill = st.selectbox(
                        "Select pod to terminate",
                        options=pod_names,
                        key="k8s_kill_pod_select",
                    )
                    kill_submitted = st.form_submit_button(
                        "💥 Kill Pod", type="primary", disabled=not confirmed
                    )

                if kill_submitted and confirmed:
                    _logger.info("Killing pod={} ns={}", pod_to_kill, _K8S_NAMESPACE)
                    with st.spinner(f"Deleting pod `{pod_to_kill}`…"):
                        ok, msg = _kubectl_delete_pod(pod_to_kill)
                    if ok:
                        _logger.info("Pod {} deleted successfully", pod_to_kill)
                        st.session_state["k8s_kill_result"] = {
                            "ok": True,
                            "msg": (
                                f"✅ Pod `{pod_to_kill}` deleted from "
                                f"namespace `{_K8S_NAMESPACE}`.\n`{msg}`\n"
                                "The Deployment controller will schedule a replacement pod."
                            ),
                        }
                        # Reset confirmation so the button is disabled again on reload.
                        # Must delete (not assign) the key — Streamlit forbids direct
                        # assignment to session_state keys that are bound to a widget.
                        # Deleting lets the checkbox re-initialise from its default=False.
                        st.session_state.pop("k8s_kill_confirm", None)
                    else:
                        _logger.error("Failed to kill pod {}: {}", pod_to_kill, msg)
                        st.session_state["k8s_kill_result"] = {
                            "ok": False,
                            "msg": f"❌ Failed to delete pod `{pod_to_kill}`: {msg}",
                        }
                    st.rerun()
