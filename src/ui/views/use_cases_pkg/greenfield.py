"""Tab 1 — Greenfield Bootstrap: wipe → generate → train → register → restart API."""

from __future__ import annotations

import contextlib
import os

import streamlit as st

from src.ui.logging_ui import get_ui_logger

_logger = get_ui_logger(__name__)

from ._common import (
    SECTION_CSS,
    get_host_db_url,
    reload_api_model,
    restart_api,
    stop_api,
)

# ── Wipe ────────────────────────────────────────────────────────


def _render_wipe_section(mode: str) -> None:
    """Dedicated wipe-all-data section with double confirmation.

    Two separate buttons are provided:
    - "Wipe Local" — wipes PostgreSQL + local MLflow (container in local mode,
      buffer container in cloud mode). Fast, no network calls.
    - "Wipe DagsHub" — wipes DagsHub MLflow (all experiments/models/runs) and
      runs DVC remote gc. Cloud mode only. Requires network access to DagsHub.
    """
    st.markdown("### 🗑️ Step 1 — Wipe Data & Models")

    # ── Wipe Local ──────────────────────────────────────────────────────────
    st.markdown("#### 🗑️ Wipe Local (PostgreSQL + MLflow)")
    if mode == "local":
        st.markdown(
            "Delete **all local** rows from the PostgreSQL database and "
            "**all experiments/models** from the local MLflow container."
        )
    else:
        st.markdown(
            "Delete **all** rows from the PostgreSQL database (devices, signals, "
            "features, labels, predictions) and **all experiments/models** from "
            "the **MLflow buffer container**. The FastAPI service will be stopped.\n\n"
            "DagsHub is **not touched** by this operation — it retains the archive "
            "of previously synced models. Use **Wipe DagsHub** below to also clear it."
        )

    confirm_key = "_wipe_confirmed"

    if st.button("🗑️ Wipe Local (PostgreSQL + MLflow)", key="wipe_start_btn", type="secondary"):
        st.session_state[confirm_key] = True

    if st.session_state.get(confirm_key, False):
        st.error(
            "**⚠️ SEVERE WARNING — This action is irreversible!**\n\n"
            "This will **permanently delete**:\n"
            "- All rows from every data table "
            "(sparse_labels → features → raw_signals → predictions → devices)\n"
            "- All MLflow experiments, runs, and registered models in the buffer\n"
            "- The FastAPI service will be **stopped** (no model to serve)\n\n"
            "**This cannot be undone.** Are you absolutely sure?"
        )
        c1, c2, _ = st.columns([1, 1, 3])
        with c1:
            if st.button("✅ Yes, wipe local", key="wipe_yes", type="primary"):
                st.session_state[confirm_key] = False
                _execute_wipe(mode)
        with c2:
            if st.button("❌ Cancel", key="wipe_cancel"):
                st.session_state[confirm_key] = False
                st.rerun()

    # ── Wipe DagsHub (cloud mode only) ──────────────────────────────────────
    if mode == "cloud":
        st.markdown("---")
        st.markdown("#### ☁️ Wipe DagsHub (MLflow + DVC Remote)")
        st.markdown(
            "Delete **all experiments, runs, and registered models** from DagsHub MLflow, "
            "and garbage-collect unreferenced objects from the DagsHub DVC S3 remote.\n\n"
            "The local PostgreSQL database and MLflow buffer are **not affected** by this operation. "
            "After this wipe, DagsHub will be empty; use **Sync Buffer → DagsHub** to re-populate it "
            "from the local buffer."
        )
        st.info(
            "ℹ️ This resets the incremental sync state file so the next push "
            "uploads all buffer runs as new entries.",
            icon="ℹ️",
        )

        dh_confirm_key = "_wipe_dagshub_confirmed"
        if st.button("☁️ Wipe DagsHub (MLflow + DVC)", key="wipe_dagshub_btn", type="secondary"):
            st.session_state[dh_confirm_key] = True

        if st.session_state.get(dh_confirm_key, False):
            st.error(
                "**⚠️ SEVERE WARNING — DagsHub data will be permanently deleted!**\n\n"
                "This will **permanently delete**:\n"
                "- All MLflow experiments, runs, and registered models on DagsHub\n"
                "- Unreferenced DVC cache objects from DagsHub S3 storage\n\n"
                "The local buffer and PostgreSQL are NOT affected.\n\n"
                "**This cannot be undone.** Are you absolutely sure?"
            )
            c1, c2, _ = st.columns([1, 1, 3])
            with c1:
                if st.button("✅ Yes, wipe DagsHub", key="wipe_dagshub_yes", type="primary"):
                    st.session_state[dh_confirm_key] = False
                    _execute_wipe_dagshub()
            with c2:
                if st.button("❌ Cancel DagsHub wipe", key="wipe_dagshub_cancel"):
                    st.session_state[dh_confirm_key] = False
                    st.rerun()


def _wipe_dvc_remote() -> None:
    """Run dvc gc to remove unreferenced cache objects from the DagsHub S3 remote.

    Execution order:
    1. dvc gc --cloud --workspace --all-commits  — removes remote objects not referenced
       by any local workspace file or git commit.
    2. git add dvc.lock *.dvc  — stage any updated pointer files.
    3. git commit [skip ci]   — record the gc in git history without triggering CI.

    This is intentionally non-fatal: if DVC is not configured or DagsHub is
    unreachable, we show the error and let the user retry.
    """
    import subprocess

    status = st.status("Cleaning DVC remote...", expanded=True)
    try:
        status.write("Running `dvc gc --cloud --workspace --all-commits` ...")
        result = subprocess.run(
            ["dvc", "gc", "--cloud", "--workspace", "--all-commits", "--force"],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,  # prevent interactive prompt hang
            timeout=300,  # 5-minute timeout for large remotes
        )
        if result.returncode != 0:
            status.update(label="DVC gc failed", state="error")
            st.error(
                f"DVC gc failed (exit {result.returncode}):\n\n"
                f"```\n{result.stderr.strip() or result.stdout.strip()}\n```\n\n"
                "Check that DVC remote credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) "
                "are set and that the DagsHub remote is reachable."
            )
            return

        status.write("DVC remote cleaned.")
        if result.stdout.strip():
            st.code(result.stdout.strip(), language=None)

        # Commit any updated .dvc / dvc.lock files (non-fatal)
        status.write("Staging updated DVC pointer files...")
        subprocess.run(
            ["git", "add", "--ignore-missing", "dvc.lock", "*.dvc", "data/.gitignore"],
            capture_output=True,
        )
        commit_result = subprocess.run(
            ["git", "commit", "-m", "[skip ci] chore(data): dvc gc --cloud cleanup"],
            capture_output=True,
            text=True,
        )
        if commit_result.returncode not in (0, 1):
            status.write(f"⚠️ git commit skipped: {commit_result.stderr.strip()[:100]}")

        status.update(label="DVC Remote Cleaned ✅", state="complete")
        st.success(
            "DVC remote garbage-collected successfully. "
            "Unreferenced cache objects have been removed from DagsHub S3."
        )
    except subprocess.TimeoutExpired:
        status.update(label="DVC gc timed out", state="error")
        st.error(
            "DVC gc timed out after 5 minutes. Run it manually: `dvc gc --cloud --workspace --all-commits --force`"
        )
    except Exception as exc:
        status.update(label="DVC gc failed", state="error")
        st.error(f"DVC gc error: {exc}")


def _execute_wipe(mode: str) -> None:
    """Perform the actual wipe operation with feedback."""
    status = st.status("Wiping all data and models...", expanded=True)
    try:
        status.write("Stopping FastAPI service...")
        stop_api()
        status.write("FastAPI stopped.")

        status.write("Wiping PostgreSQL database...")
        from src.database.database import Database

        with Database(db_url=get_host_db_url() or None) as db:
            deleted = db.wipe_all_data()
        total = sum(deleted.values())
        status.write(f"Database: deleted {total} rows across {len(deleted)} tables.")

        status.write("Cleaning MLflow experiments and models...")
        _wipe_mlflow(mode)
        status.write("MLflow cleaned.")

        # Reset the incremental sync state file so the next sync to DagsHub
        # correctly identifies all future buffer runs as new (not already synced).
        _reset_sync_state_file(status)

        # Mark that a local wipe occurred in this session so the MLflow Explorer
        # "Restore Buffer from DagsHub" button shows an extra confirmation prompt.
        st.session_state["_post_wipe"] = True

        status.update(label="Wipe Complete", state="complete")
        st.success(
            f"Wiped **{total}** database rows and cleaned MLflow. "
            "FastAPI is **stopped** — run Bootstrap to create a new model and restart it."
        )
        for tbl, cnt in deleted.items():
            st.write(f"  - `{tbl}`: {cnt} rows")
    except Exception as exc:
        status.update(label="Wipe Failed", state="error")
        st.error(f"Wipe failed: {exc}")


def _reset_sync_state_file(status=None) -> None:  # noqa: ANN001
    """Delete data/.mlflow_sync_state.json if it exists.

    Resetting this file after a wipe ensures the next incremental sync treats
    all buffer runs as new and uploads them all to DagsHub (which is now either
    empty or holds the pre-wipe state that should be overwritten).
    """
    from pathlib import Path

    state_file = Path(__file__).resolve().parents[4] / "data" / ".mlflow_sync_state.json"
    if state_file.exists():
        state_file.unlink()
        msg = "Sync state file reset (next push will be a full sync)."
    else:
        msg = "No sync state file found (already clean)."
    if status is not None:
        status.write(msg)


def _execute_wipe_dagshub() -> None:
    """Wipe DagsHub MLflow (all experiments/runs/models) and run DVC remote gc.

    This is a destructive remote operation — it permanently removes:
    1. All experiments, runs, and registered models from DagsHub MLflow.
    2. Unreferenced DVC cache objects from DagsHub S3 (via ``dvc gc --cloud``).
    3. The local incremental sync state file so the next push starts fresh.

    The local PostgreSQL database and MLflow buffer are NOT affected.
    """
    from src.ui.views.mlflow_explorer import _dagshub_uri, _env_or_file

    status = st.status("Wiping DagsHub…", expanded=True)
    try:
        user = _env_or_file("DAGSHUB_USER") or ""
        token = _env_or_file("DAGSHUB_TOKEN") or ""
        dh_uri = _dagshub_uri()

        if not dh_uri or not user or not token:
            status.update(label="Missing credentials", state="error")
            st.error(
                "DagsHub credentials not found. Set `DAGSHUB_USER`, `DAGSHUB_TOKEN`, "
                "and `DAGSHUB_REPO` in `.env.secrets`."
            )
            return

        # Step 1: Wipe DagsHub MLflow
        status.write(f"Wiping DagsHub MLflow at `{dh_uri}` …")
        os.environ["MLFLOW_TRACKING_USERNAME"] = user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        _wipe_mlflow_rest(dh_uri)
        status.write("DagsHub MLflow wiped.")

        # Step 2: Restore local env (buffer is primary)
        from src.ui.views.mlflow_explorer import _tracking_uri

        os.environ["MLFLOW_TRACKING_URI"] = _tracking_uri()
        os.environ["MLFLOW_TRACKING_USERNAME"] = ""
        os.environ["MLFLOW_TRACKING_PASSWORD"] = ""

        # Step 3: DVC remote gc
        status.write("Running DVC remote garbage collection …")
        _wipe_dvc_remote_quiet(status)

        # Step 4: Reset sync state file
        _reset_sync_state_file(status)

        # Step 5: Invalidate DagsHub view cache
        st.session_state.pop("_dagshub_view_cache", None)

        status.update(label="DagsHub Wipe Complete ✅", state="complete")
        st.success(
            "DagsHub MLflow wiped and DVC remote garbage-collected. "
            "Use **Sync Buffer → DagsHub** in the MLflow Explorer to re-populate DagsHub "
            "from the local buffer."
        )
    except Exception as exc:
        status.update(label="DagsHub wipe failed", state="error")
        st.error(f"DagsHub wipe failed: {exc}")


def _wipe_dvc_remote_quiet(status) -> None:  # noqa: ANN001
    """Run dvc gc --cloud (non-interactive) and write progress to *status*."""
    import subprocess

    try:
        status.write("Running `dvc gc --cloud --workspace --all-commits --force` …")
        result = subprocess.run(
            ["dvc", "gc", "--cloud", "--workspace", "--all-commits", "--force"],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,  # prevent interactive prompt hang
            timeout=300,
        )
        if result.returncode != 0:
            status.write(
                f"⚠️ DVC gc returned exit {result.returncode} "
                f"(may still have cleaned some objects):\n"
                f"{result.stderr.strip()[:300]}"
            )
        else:
            status.write("DVC remote gc complete.")
            if result.stdout.strip():
                status.write(result.stdout.strip()[:300])
    except subprocess.TimeoutExpired:
        status.write("⚠️ DVC gc timed out (> 5 min). Run manually if needed.")
    except Exception as exc:
        status.write(f"⚠️ DVC gc skipped: {exc}")


def _wipe_mlflow(mode: str) -> None:
    """Delete all experiments and registered models from MLflow.

    In **local** mode the SQLite backend keeps an auto-increment counter for
    model versions even after ``delete_registered_model``.  To guarantee that
    the next bootstrap starts at version 1, we stop the MLflow container,
    physically remove the SQLite DB and artifact store, then restart it.

    In **cloud** mode we use direct REST API calls (DagsHub) to avoid the
    MLflow SDK blocking indefinitely on network I/O.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    from src.ui.views.mlflow_explorer import _tracking_uri

    uri = _tracking_uri()
    os.environ["MLFLOW_TRACKING_URI"] = uri

    if mode == "cloud":
        from src.ui.views.mlflow_explorer import _env_or_file

        user = _env_or_file("DAGSHUB_USER")
        token = _env_or_file("DAGSHUB_TOKEN")
        if user:
            os.environ["MLFLOW_TRACKING_USERNAME"] = user
        if token:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        _wipe_mlflow_rest(uri)
        return

    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)

    # API-level deletion (local mode only)
    for rm in client.search_registered_models():
        for mv in client.search_model_versions(f"name='{rm.name}'"):
            with contextlib.suppress(Exception):
                client.delete_model_version(rm.name, mv.version)
        with contextlib.suppress(Exception):
            client.delete_registered_model(rm.name)

    for exp in client.search_experiments():
        if exp.name != "Default":
            with contextlib.suppress(Exception):
                client.delete_experiment(exp.experiment_id)

    # Local mode: reset SQLite DB so version counter restarts at 1
    _reset_local_mlflow_storage()


def _wipe_mlflow_rest(uri: str) -> None:
    """Delete all models and experiments via MLflow REST API (cloud/DagsHub mode).

    Uses ``urllib.request`` with explicit auth headers so it never blocks the
    Streamlit thread the way the MLflow SDK can in cloud mode.
    """
    import base64
    import json
    import urllib.error
    import urllib.parse
    import urllib.request

    user = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
    pwd = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")
    _auth = ""
    if user and pwd:
        _auth = base64.b64encode(f"{user}:{pwd}".encode()).decode()

    def _req(url: str, body: dict | None = None, method: str = "GET") -> dict:
        """Make a single HTTP request with exponential back-off on HTTP 429.

        Retries up to 3 times.  Waits are 2 s, 4 s, 8 s (doubles each attempt).
        Respects the ``Retry-After`` response header when present.
        All other HTTP errors are re-raised immediately.
        """
        import time

        data = json.dumps(body).encode() if body is not None else None
        _max_retries = 3
        _wait = 2.0
        for attempt in range(_max_retries + 1):
            req = urllib.request.Request(url, data=data, method=method)
            if _auth:
                req.add_header("Authorization", f"Basic {_auth}")
            if data:
                req.add_header("Content-Type", "application/json")
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    raw = resp.read().decode("utf-8")
                    return json.loads(raw) if raw.strip() else {}  # type: ignore[no-any-return]
            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt < _max_retries:
                    # Honour Retry-After if present, otherwise use exponential back-off
                    retry_after_raw = exc.headers.get("Retry-After", "")
                    try:
                        wait_secs = float(retry_after_raw) if retry_after_raw else _wait
                    except ValueError:
                        wait_secs = _wait
                    time.sleep(wait_secs)
                    _wait *= 2  # exponential back-off
                    continue
                raise

        raise RuntimeError("_req: exhausted retries without returning")  # unreachable

    base = uri.rstrip("/")

    # ── Delete all registered models + versions ──────────────────────────
    try:
        data = _req(f"{base}/api/2.0/mlflow/registered-models/search")
        for rm in data.get("registered_models", []):
            name = rm.get("name", "")
            if not name:
                continue
            # Delete each version first
            try:
                params = urllib.parse.urlencode({"filter": f"name='{name}'"})
                vdata = _req(f"{base}/api/2.0/mlflow/model-versions/search?{params}")
                for mv in vdata.get("model_versions", []):
                    ver = mv.get("version", "")
                    if ver:
                        with contextlib.suppress(Exception):
                            _req(
                                f"{base}/api/2.0/mlflow/model-versions/delete",
                                {"name": name, "version": ver},
                                method="DELETE",
                            )
            except Exception:
                pass
            with contextlib.suppress(Exception):
                _req(
                    f"{base}/api/2.0/mlflow/registered-models/delete",
                    {"name": name},
                    method="DELETE",
                )
    except Exception:
        pass

    # ── Delete all experiments + their runs (except Default) ─────────────
    try:
        # Use POST /search to list all experiments (works on DagsHub and MLflow 2.x+)
        # DagsHub requires at least one field in the body (empty {} returns 400).
        edata = _req(
            f"{base}/api/2.0/mlflow/experiments/search", {"max_results": 1000}, method="POST"
        )
        for exp in edata.get("experiments", []):
            # Skip the Default experiment and already-deleted ones
            if exp.get("name") == "Default":
                continue
            if exp.get("lifecycle_stage") == "deleted":
                continue
            eid = exp.get("experiment_id", "")
            if not eid:
                continue

            # Delete all runs in this experiment first.
            # DagsHub/MLflow does NOT automatically delete runs when an
            # experiment is soft-deleted, so runs stay in the leaderboard.
            try:
                rdata = _req(
                    f"{base}/api/2.0/mlflow/runs/search",
                    {"experiment_ids": [eid], "max_results": 5000},
                    method="POST",
                )
                for run in rdata.get("runs", []):
                    rid = (run.get("info") or {}).get("run_id", "")
                    if rid:
                        with contextlib.suppress(Exception):
                            _req(
                                f"{base}/api/2.0/mlflow/runs/delete",
                                {"run_id": rid},
                                method="POST",
                            )
            except Exception:
                pass

            with contextlib.suppress(Exception):
                _req(
                    f"{base}/api/2.0/mlflow/experiments/delete",
                    {"experiment_id": eid},
                    method="POST",
                )
    except Exception:
        pass


def _reset_local_mlflow_storage() -> None:
    """Stop the local MLflow container, purge its SQLite DB + artifacts, restart.

    This resets the version auto-increment counter so the next registered model
    starts at version 1.
    """
    import subprocess

    # 1. Stop MLflow container
    subprocess.run(
        ["docker", "compose", "-f", "docker-compose.yml", "stop", "mlflow"],
        capture_output=True,
        timeout=30,
    )

    # 2. Remove SQLite DB and artifacts inside the named volumes.
    #    The volumes are still mounted on the host; we use a temporary
    #    container with the same volume mounts to do the cleanup.
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            "mlops_device_health_mlflow_db:/mlflow",
            "-v",
            "mlops_device_health_mlflow_artifacts:/mlflow/artifacts",
            "alpine",
            "sh",
            "-c",
            "rm -rf /mlflow/mlflow.db /mlflow/artifacts/*",
        ],
        capture_output=True,
        timeout=30,
    )

    # 3. Start MLflow again — it will create a fresh empty DB on boot
    subprocess.run(
        ["docker", "compose", "-f", "docker-compose.yml", "start", "mlflow"],
        capture_output=True,
        timeout=30,
    )

    # 4. Brief wait for the server to initialise the new DB
    import time

    time.sleep(3)


# ── Signal parameters ───────────────────────────────────────────


def _render_signal_params() -> dict:
    """Render colour-coded signal parameter controls.

    Default ranges match the baseline in signal_generator.py:
      Healthy Gaussian: μ∈[48,52], σ∈[2.0,3.0], H∈[2.5,3.0], noise∈[0.01,0.02]
      Unhealthy Lorentzian (as σ): μ∈[42,58], σ∈[3.8,5.1], H∈[1.0,1.5], noise∈[0.06,0.10]
    """
    st.markdown("### 📡 Step 2 — Signal Generation Parameters")

    # ── Default values ──
    _gf_defaults = {
        "gf_n_samples": 100,
        "gf_gaussian_fraction": 0.7,
        "gf_labeled_fraction": 0.2,
        "gf_seed": 42,
        "gf_g_mu": (48.0, 52.0),
        "gf_g_sigma": (2.0, 3.0),
        "gf_g_height": (2.5, 3.0),
        "gf_g_noise": (0.01, 0.02),
        "gf_l_mu": (42.0, 58.0),
        "gf_l_sigma": (3.8, 5.1),
        "gf_l_height": (1.0, 1.5),
        "gf_l_noise": (0.06, 0.10),
    }
    # Initialise session-state keys once (avoids "widget created with default
    # value but also had its value set via Session State" warning).
    for _k, _v in _gf_defaults.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    # ── Restore defaults button ──
    if st.button("🔄 Restore Defaults", key="gf_restore_defaults"):
        for k, v in _gf_defaults.items():
            st.session_state[k] = v
        st.rerun()

    # ── General parameters card ──
    with st.container(border=True):
        st.markdown(
            '<div class="signal-section-general"><strong>📊 General Parameters</strong></div>',
            unsafe_allow_html=True,
        )
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            n_samples = st.number_input(
                "Number of samples",
                min_value=20,
                max_value=5000,
                step=50,
                help="Total signals to generate per dataset",
                key="gf_n_samples",
            )
        with col2:
            gaussian_fraction = st.slider(
                "Healthy / Unhealthy ratio",
                min_value=0.1,
                max_value=0.9,
                step=0.05,
                help="Fraction of healthy (Gaussian) signals",
                key="gf_gaussian_fraction",
            )
        with col3:
            labeled_fraction = st.slider(
                "Labeled ratio",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                help="0 = no labels, 1.0 = all labeled",
                key="gf_labeled_fraction",
            )
        with col4:
            seed = st.number_input(
                "Random seed",
                min_value=0,
                max_value=99999,
                help="Seed for reproducible generation",
                key="gf_seed",
            )

    # ── Gaussian parameters card ──
    with st.container(border=True):
        st.markdown(
            '<div class="signal-section-healthy">'
            "<strong>🟢 Healthy Signals — Gaussian Parameters</strong></div>",
            unsafe_allow_html=True,
        )
        g1, g2, g3, g4 = st.columns(4)
        with g1:
            g_mu = st.slider(
                "μ range (center)",
                min_value=35.0,
                max_value=60.0,
                step=0.5,
                key="gf_g_mu",
            )
        with g2:
            g_sigma = st.slider(
                "σ range (width)",
                min_value=0.5,
                max_value=6.0,
                step=0.1,
                help="Standard deviation (narrower = healthier)",
                key="gf_g_sigma",
            )
        with g3:
            g_height = st.slider(
                "Height range",
                min_value=1.0,
                max_value=5.0,
                step=0.1,
                key="gf_g_height",
            )
        with g4:
            g_noise = st.slider(
                "Noise range",
                min_value=0.0,
                max_value=0.15,
                step=0.005,
                format="%.3f",
                key="gf_g_noise",
            )

    # ── Lorentzian parameters card ──
    with st.container(border=True):
        st.markdown(
            '<div class="signal-section-unhealthy">'
            "<strong>🔴 Unhealthy Signals — Lorentzian Parameters</strong></div>",
            unsafe_allow_html=True,
        )
        l1, l2, l3, l4 = st.columns(4)
        with l1:
            l_mu = st.slider(
                "μ range (center)",
                min_value=35.0,
                max_value=60.0,
                step=0.5,
                key="gf_l_mu",
            )
        with l2:
            l_sigma = st.slider(
                "σ range (width)",
                min_value=2.0,
                max_value=6.0,
                step=0.1,
                help="Equivalent Gaussian σ — converted to Lorentzian γ via γ\u2009=\u20091.18\u2009σ",
                key="gf_l_sigma",
            )
        with l3:
            l_height = st.slider(
                "Height range",
                min_value=0.5,
                max_value=3.5,
                step=0.1,
                key="gf_l_height",
            )
        with l4:
            l_noise = st.slider(
                "Noise range",
                min_value=0.0,
                max_value=0.15,
                step=0.005,
                format="%.3f",
                key="gf_l_noise",
            )

    return {
        "n_samples": int(n_samples),
        "gaussian_fraction": gaussian_fraction,
        "labeled_fraction": labeled_fraction,
        "seed": int(seed),
        "gauss_mu_range": tuple(g_mu),  # type: ignore[arg-type]
        "gauss_sigma_range": tuple(g_sigma),  # type: ignore[arg-type]
        "gauss_height_range": tuple(g_height),  # type: ignore[arg-type]
        "gauss_noise_range": tuple(g_noise),  # type: ignore[arg-type]
        "lor_mu_range": tuple(l_mu),  # type: ignore[arg-type]
        "lor_sigma_range": tuple(l_sigma),  # type: ignore[arg-type]
        "lor_height_range": tuple(l_height),  # type: ignore[arg-type]
        "lor_noise_range": tuple(l_noise),  # type: ignore[arg-type]
    }


# ── Bootstrap execution ────────────────────────────────────────


def _render_bootstrap_section(signal_params: dict, mode: str) -> None:
    """Classifier selection + bootstrap-run button."""
    st.markdown("### 🚀 Step 3 — Bootstrap (Generate → Train → Register → Restart API)")

    col1, col2, _ = st.columns([2, 2, 1])
    with col1:
        classifier = st.selectbox(
            "Classifier",
            options=["logistic_regression", "decision_tree", "random_forest", "svc"],
            format_func=lambda x: {
                "logistic_regression": "Logistic Regression",
                "decision_tree": "Decision Tree",
                "random_forest": "Random Forest",
                "svc": "Support Vector Classifier (SVC)",
            }[x],
            index=0,
            key="gf_classifier",
        )
    with col2:
        st.write("")
        st.write("")
        promote = st.checkbox(
            "Promote to Production after training",
            value=True,
            key="gf_promote",
        )

    if mode == "local":
        st.info(
            "**Local sandbox mode** — DVC tracking and DagsHub sync are skipped. "
            "Data and models stay in the local MLflow container and PostgreSQL."
        )

    st.markdown("---")

    running_key = "_gf_running"
    if st.session_state.get(running_key, False):
        _run_greenfield_pipeline(signal_params, classifier, promote, mode)
        return

    if st.button("🚀 Start Bootstrap", key="gf_start_btn", type="primary"):
        st.session_state[running_key] = True
        st.rerun()


def _run_greenfield_pipeline(
    signal_params: dict,
    classifier: str,
    promote: bool,
    mode: str,
) -> None:
    """Execute the greenfield bootstrap and display progress."""
    from scripts.greenfield_init import BootstrapConfig, BootstrapResult, run_bootstrap

    from ._common import get_experiment_name, get_model_name

    config = BootstrapConfig(
        n_samples=signal_params["n_samples"],
        gaussian_fraction=signal_params["gaussian_fraction"],
        labeled_fraction=signal_params["labeled_fraction"],
        seed=signal_params["seed"],
        classifier=classifier,
        model_name=get_model_name(),
        experiment_name=get_experiment_name(),
        promote=promote,
        wipe=False,
        gauss_mu_range=signal_params["gauss_mu_range"],
        gauss_sigma_range=signal_params["gauss_sigma_range"],
        gauss_height_range=signal_params["gauss_height_range"],
        gauss_noise_range=signal_params["gauss_noise_range"],
        lor_mu_range=signal_params["lor_mu_range"],
        lor_sigma_range=signal_params["lor_sigma_range"],
        lor_height_range=signal_params["lor_height_range"],
        lor_noise_range=signal_params["lor_noise_range"],
    )

    status_container = st.status("Running Greenfield Bootstrap...", expanded=True)
    progress_bar = st.progress(0.0)
    log_area = st.empty()
    log_lines: list[str] = []

    def _ui_progress(step: str, message: str, fraction: float) -> None:
        log_lines.append(f"[{step}] {message}")
        log_area.code("\n".join(log_lines[-20:]), language="text")
        if 0.0 <= fraction <= 1.0:
            progress_bar.progress(fraction)

    _logger.info(
        "Starting greenfield bootstrap: classifier={} n_samples={} promote={}",
        classifier,
        config.n_samples,
        promote,
    )
    result: BootstrapResult = run_bootstrap(config, progress_callback=_ui_progress)

    if result.success:
        _ui_progress("api", "Reloading model in FastAPI service…", 0.97)
        try:
            ok, msg, _ = reload_api_model()
            if ok:
                _ui_progress("api", f"API model reloaded — {msg}", 0.99)
            else:
                # Hot-reload failed — fall back to container restart
                restart_api()
                _ui_progress("api", f"API hot-reload failed ({msg}), used container restart.", 0.99)
        except Exception as exc:
            _ui_progress("api", f"API reload error: {exc}", 0.99)

    st.session_state["_gf_running"] = False

    if result.success:
        _logger.info(
            "Greenfield bootstrap OK — version={} f1={:.4f} accuracy={:.4f} run_id={}",
            result.model_version,
            result.test_f1,
            result.test_accuracy,
            result.mlflow_run_id,
        )
        status_container.update(label="Greenfield Bootstrap Complete!", state="complete")
        progress_bar.progress(1.0)
        st.success(
            "Bootstrap completed! The API model has been hot-reloaded with the "
            "new model. Switch to the **Predictions** page to test it."
        )

        # DVC timeout note: the dvc add / dvc push commands have a 90s timeout to
        # avoid blocking the UI when DagsHub is unreachable or slow.  A timeout is
        # NOT an error — the model was trained and registered in MLflow successfully.
        # DVC tracking is best-effort: it records the data hash for reproducibility
        # but does NOT affect model availability or prediction quality.
        _gf_dvc_hashes = result.dvc_hashes if result.dvc_hashes else {}
        if mode == "cloud" and not any(_gf_dvc_hashes.values()):
            st.info(
                "⚠️ **DVC tracking skipped** — `dvc add` timed out (90 s) because DagsHub "
                "was unreachable or slow. This is **normal** and does not affect the model. "
                "Training and MLflow registration completed successfully. "
                "DVC data lineage will be recorded on the next successful sync.",
                icon="ℹ️",
            )

        st.markdown("#### Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test F1", f"{result.test_f1:.4f}")
        c2.metric("Test Accuracy", f"{result.test_accuracy:.4f}")
        c3.metric("Model Version", f"v{result.model_version}")
        c4.metric("Promoted", "Yes" if result.promoted else "No")

        st.markdown("#### Lineage")
        lineage_data: dict[str, object] = {
            "MLflow Run ID": result.mlflow_run_id,
            "Git SHA": result.git_sha,
            "Mode": mode,
        }
        if mode == "cloud":
            lineage_data["DVC Data Hashes"] = {
                k: v[:12] + "..." if v else "N/A" for k, v in result.dvc_hashes.items()
            }
        lineage_data["Data Files"] = list(result.data_paths.keys())
        st.json(lineage_data)
    else:
        _logger.warning("Greenfield bootstrap FAILED: {}", result.error)
        status_container.update(label="Bootstrap Failed", state="error")
        st.error(f"**Error:** {result.error}")


# ── Public entry-point ──────────────────────────────────────────


def render_greenfield_tab(mode: str) -> None:
    """Render the full Greenfield Bootstrap use case tab."""
    st.markdown(SECTION_CSS, unsafe_allow_html=True)
    st.markdown(
        "Wipe existing data, generate fresh synthetic signals with custom "
        "parameter ranges, train a bootstrap classifier, promote it, and "
        "restart the API."
    )

    _render_wipe_section(mode)
    st.markdown("---")
    signal_params = _render_signal_params()
    st.markdown("---")
    _render_bootstrap_section(signal_params, mode)
