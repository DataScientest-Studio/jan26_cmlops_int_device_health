"""MLflow Explorer — experiments, runs, model registry, and metrics.

Connects to the MLflow Tracking Server (local Docker or DagsHub cloud)
via the ``mlflow`` Python SDK.  Falls back gracefully when the server
is unreachable.
"""

from __future__ import annotations

import contextlib
import os
import sys
import time as _time
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from src.ui.components.docker_utils import get_host, get_service_url
from src.ui.logging_ui import get_ui_logger
from src.ui.styles import get_global_css, hero_section

# ── Session-state cache keys ────────────────────────────────────
_CACHE_KEY = "_mlflow_client_cache"
_DATA_CACHE_KEY = "_mlflow_data_cache"
_DATA_CACHE_TTL = 120  # seconds — cached results expire after 2 minutes

# ── MLflow connection helpers ───────────────────────────────────


_logger = get_ui_logger(__name__)


def _detect_mode() -> str:
    # Priority: env var (set by make ui) → .current_mode file → default local
    # Prioritise .current_mode file — it is updated on every stack start/stop
    # (even from within Streamlit).  DEPLOYMENT_MODE env var is baked in once
    # at ``make ui`` startup and can become stale if the user switches mode.
    mode_file = Path(_PROJECT_ROOT) / ".current_mode"
    if mode_file.exists():
        val = mode_file.read_text().strip()
        if val in ("local", "cloud", "k8s"):
            return val
    env_mode = os.environ.get("DEPLOYMENT_MODE", "").strip()
    if env_mode in ("local", "cloud", "k8s"):
        return env_mode
    return "local"


def _load_env_file(name: str) -> dict[str, str]:
    """Parse a simple KEY=VALUE env file (no shell variable expansion)."""
    path = Path(_PROJECT_ROOT) / name
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if value and not value.startswith("$"):
                result[key] = value
    return result


def _env_or_file(key: str) -> str:
    """Get env var, falling back to .env.secrets if not in os.environ."""
    val = os.environ.get(key, "")
    if val:
        return val
    secrets = _load_env_file(".env.secrets")
    return secrets.get(key, "")


def _tracking_uri() -> str:
    """Return MLflow tracking URI reachable **from the host**.

    Port values are read from ``MLFLOW_BUFFER_PORT`` / ``MLFLOW_PORT`` env
    vars rather than ``MLFLOW_TRACKING_URI``.  This matters because
    ``MLFLOW_TRACKING_URI`` is baked into the process environment by
    ``make ui`` at startup — if the user starts Streamlit in local mode and
    then switches to cloud (which updates ``.current_mode`` but not env vars),
    ``MLFLOW_TRACKING_URI`` stays at the local value (5001) while mode
    detection correctly returns "cloud".  Using the port-specific vars avoids
    this stale-env-var trap.
    """
    mode = _detect_mode()

    # ---- K8s mode ----------------------------------------------------------
    # MLflow is accessible via kubectl port-forward at localhost:5000.
    # MLFLOW_TRACKING_URI is set by make ui from .env.k8s.
    if mode == "k8s":
        return os.environ.get("MLFLOW_TRACKING_URI", f"http://{get_host()}:5000")

    # ---- Cloud mode --------------------------------------------------------
    if mode == "cloud":
        port = os.environ.get("MLFLOW_BUFFER_PORT", "5002")
        return f"http://{get_host()}:{port}"

    # ---- Local mode --------------------------------------------------------
    # Try docker port first (gives the actual published port even if it differs
    # from MLFLOW_PORT, e.g. because the container was started manually).
    url = get_service_url("mlops_mlflow", 5000)
    if url != f"http://{get_host()}:5000":
        # docker port succeeded and returned a non-default port
        return url
    # Fallback to env var (set by make ui from .env.local)
    port = os.environ.get("MLFLOW_PORT", "5001")
    return f"http://{get_host()}:{port}"


def _dagshub_uri() -> str:
    """Return the DagsHub MLflow URI for the secondary view in cloud mode.

    Reads ``MLFLOW_DAGSHUB_URI`` env var first; falls back to building it
    from ``DAGSHUB_USER`` + ``DAGSHUB_REPO`` (from ``.env.secrets``).
    Returns empty string when credentials are not available.
    """
    uri = os.environ.get("MLFLOW_DAGSHUB_URI", "")
    if uri:
        return uri
    user = _env_or_file("DAGSHUB_USER")
    repo = _env_or_file("DAGSHUB_REPO")
    if user and repo:
        return f"https://dagshub.com/{user}/{repo}.mlflow"
    return ""


def _get_client():  # noqa: ANN202
    """Return a configured MlflowClient (lazy import)."""
    uri = _tracking_uri()

    # Override env var BEFORE importing mlflow so the SDK never sees
    # the Docker-internal hostname from .env.local (http://mlflow:5000).
    os.environ["MLFLOW_TRACKING_URI"] = uri

    # Cap all MLflow SDK HTTP requests at 30 s.  Without this, requests uses
    # no socket timeout; on Windows this causes indefinite hangs when the
    # server is slow to respond.
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "30")

    # Cloud auth — set before import so SDK picks them up immediately.
    # K8s uses local MLflow (no DagsHub auth needed).
    mode = _detect_mode()
    if mode == "cloud":
        user = _env_or_file("DAGSHUB_USER")
        token = _env_or_file("DAGSHUB_TOKEN")
        if user:
            os.environ["MLFLOW_TRACKING_USERNAME"] = user
        if token:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    elif mode == "k8s":
        # Clear any stale DagsHub credentials so the local K8s MLflow is used directly
        os.environ.pop("MLFLOW_TRACKING_USERNAME", None)
        os.environ.pop("MLFLOW_TRACKING_PASSWORD", None)

    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(uri)
    _logger.debug("MLflow client → tracking_uri={}", uri)
    return MlflowClient(tracking_uri=uri)


# ── Data-level caching (avoids re-querying MLflow on every render) ──


def _get_cached(sub_key: str):  # noqa: ANN202
    """Return cached data for *sub_key*, or ``None`` if stale / absent."""
    cache = st.session_state.get(_DATA_CACHE_KEY)
    if not isinstance(cache, dict):
        return None
    entry = cache.get(sub_key)
    if entry and (_time.time() - entry.get("ts", 0)) < _DATA_CACHE_TTL:
        return entry.get("data")
    return None


def _set_cached(sub_key: str, data: object) -> None:
    """Store *data* under *sub_key* with the current timestamp."""
    if _DATA_CACHE_KEY not in st.session_state:
        st.session_state[_DATA_CACHE_KEY] = {}
    st.session_state[_DATA_CACHE_KEY][sub_key] = {"data": data, "ts": _time.time()}


# ── Section renderers ───────────────────────────────────────────


def _render_experiments(client) -> None:  # noqa: ANN001
    """Show all experiments."""
    st.markdown(
        '<div class="section-header">🧪 Experiments</div>',
        unsafe_allow_html=True,
    )
    experiments = _get_cached("experiments")
    if experiments is None:
        try:
            _logger.debug("Fetching MLflow experiments")
            experiments = list(client.search_experiments())
            _logger.info("MLflow experiments fetched: {} found", len(experiments))
            _set_cached("experiments", experiments)
        except Exception as exc:
            _logger.warning("Failed to fetch MLflow experiments: {}", exc)
            st.error(f"Failed to fetch experiments: {exc}")
            return

    if not experiments:
        st.info("No experiments found.")
        return

    for exp in experiments:
        name = exp.name
        exp_id = exp.experiment_id
        lc = exp.lifecycle_stage
        icon = "🟢" if lc == "active" else "🗄️"
        with st.expander(f"{icon} {name}  (ID: {exp_id})", expanded=False):
            st.markdown(f"**Lifecycle:** {lc}")
            if exp.artifact_location:
                st.markdown(f"**Artifact Location:** `{exp.artifact_location}`")
            if hasattr(exp, "tags") and exp.tags:
                st.json(exp.tags)

            # Link to experiment in MLflow UI
            uri = _tracking_uri()
            st.markdown(f"[Open experiment in MLflow UI ↗]({uri}/#/experiments/{exp_id})")

            # Show recent runs in this experiment
            try:
                runs = client.search_runs(
                    experiment_ids=[exp_id],
                    max_results=5,
                    order_by=["start_time DESC"],
                )
                if runs:
                    st.markdown(f"**Recent runs:** {len(runs)}")
                    for r in runs:
                        status = r.info.status
                        run_id = r.info.run_id[:12]
                        f1 = r.data.metrics.get("test_f1_score", "—")
                        f1_str = f"{f1:.4f}" if isinstance(f1, float) else f1
                        start_ts = ""
                        if r.info.start_time:
                            from datetime import datetime, timezone

                            start_ts = datetime.fromtimestamp(
                                r.info.start_time / 1000, tz=timezone.utc
                            ).strftime(" · %Y-%m-%d %H:%M UTC")
                        run_link = f"{uri}/#/experiments/{exp_id}/runs/{r.info.run_id}"
                        st.markdown(
                            f"- [`{run_id}`]({run_link}) · {status} · F1={f1_str}{start_ts}"
                        )
            except Exception:
                pass


def _render_run_leaderboard(client) -> None:  # noqa: ANN001
    """Model leaderboard — best runs sorted by F1."""
    st.markdown(
        '<div class="section-header">🏅 Run Leaderboard</div>',
        unsafe_allow_html=True,
    )
    try:
        all_exps = _get_cached("experiments") or list(client.search_experiments())
        exp_ids = [e.experiment_id for e in all_exps]
        if not exp_ids:
            st.info("No experiments found.  Train a model first (UC-04/UC-05).")
            return

        runs = _get_cached("leaderboard_runs")
        if runs is None:
            runs = client.search_runs(
                experiment_ids=exp_ids,
                order_by=["attribute.start_time DESC"],
                max_results=30,
            )
            _set_cached("leaderboard_runs", runs)
    except Exception as exc:
        st.error(f"Failed to fetch runs: {exc}")
        return

    if not runs:
        st.info("No runs found.  Train a model first (UC-04/UC-05).")
        return

    from datetime import datetime, timezone

    uri = _tracking_uri()

    # Build markdown table with clickable run links
    lines = [
        "| Run ID | Exp | Status | F1 | Accuracy | Model | Started | Duration |",
        "|--------|-----|--------|-----|----------|-------|---------|----------|",
    ]
    for r in runs:
        start_ts = "—"
        if r.info.start_time:
            start_ts = datetime.fromtimestamp(r.info.start_time / 1000, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M"
            )
        run_id_short = r.info.run_id[:12]
        exp_id = r.info.experiment_id
        run_link = f"{uri}/#/experiments/{exp_id}/runs/{r.info.run_id}"
        f1 = r.data.metrics.get("test_f1_score", "—")
        f1_str = f"{f1:.4f}" if isinstance(f1, float) else f1
        acc = r.data.metrics.get("test_accuracy", "—")
        acc_str = f"{acc:.4f}" if isinstance(acc, float) else acc
        model_type = r.data.params.get("model_type") or r.data.params.get("classifier_type", "—")
        dur = "—"
        if r.info.end_time and r.info.start_time:
            dur = f"{(r.info.end_time - r.info.start_time) / 1000:.1f}s"
        lines.append(
            f"| [{run_id_short}]({run_link}) | {exp_id} | {r.info.status} "
            f"| {f1_str} | {acc_str} | {model_type} | {start_ts} | {dur} |"
        )

    st.caption(
        "Latest 30 runs across all experiments, sorted chronologically (newest first). "
        "Click a Run ID for full metrics, parameters, and artifacts."
    )
    st.markdown("\n".join(lines), unsafe_allow_html=True)

    # Link to MLflow UI
    st.markdown(f"[Open MLflow UI ↗]({uri})")


# ── REST API fallback for model registry ────────────────────────


def _rest_search_registered_models(client):  # noqa: ANN001, ANN202
    """Query registered models via REST API when SDK returns empty.

    MLflow SDK v3.x may silently return [] when talking to a v2.x server.
    This fallback calls the REST endpoint directly and wraps results in
    lightweight namespace objects so the rest of the rendering code works.
    """
    import json
    import urllib.error
    import urllib.request
    from types import SimpleNamespace

    uri = _tracking_uri()
    url = f"{uri}/api/2.0/mlflow/registered-models/search"
    try:
        req = urllib.request.Request(url, method="GET")
        # Add auth headers for cloud mode (DagsHub)
        user = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
        pwd = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")
        if user and pwd:
            import base64

            creds = base64.b64encode(f"{user}:{pwd}".encode()).decode()
            req.add_header("Authorization", f"Basic {creds}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, Exception):
        return []

    raw_models = data.get("registered_models", [])
    models = []
    for rm in raw_models:
        m = SimpleNamespace(
            name=rm.get("name", ""),
            description=rm.get("description", ""),
            creation_timestamp=rm.get("creation_timestamp"),
            last_updated_timestamp=rm.get("last_updated_timestamp"),
            tags=rm.get("tags", {}),
        )
        models.append(m)
    return models


def _rest_search_model_versions(model_name: str) -> list:
    """Query model versions via REST API fallback."""
    import json
    import urllib.error
    import urllib.parse
    import urllib.request
    from types import SimpleNamespace

    uri = _tracking_uri()
    params = urllib.parse.urlencode({"filter": f"name='{model_name}'"})
    url = f"{uri}/api/2.0/mlflow/model-versions/search?{params}"
    try:
        req = urllib.request.Request(url, method="GET")
        user = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
        pwd = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")
        if user and pwd:
            import base64

            creds = base64.b64encode(f"{user}:{pwd}".encode()).decode()
            req.add_header("Authorization", f"Basic {creds}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, Exception):
        return []

    raw_versions = data.get("model_versions", [])
    versions = []
    for rv in raw_versions:
        v = SimpleNamespace(
            version=rv.get("version", "0"),
            current_stage=rv.get("current_stage", "None"),
            run_id=rv.get("run_id", ""),
            source=rv.get("source", ""),
            creation_timestamp=rv.get("creation_timestamp"),
            aliases=rv.get("aliases", []),
        )
        versions.append(v)
    return versions


def _rest_get_model_aliases(model_name: str) -> dict[str, str]:
    """Fetch aliases for a registered model via REST API.

    Calls ``GET /api/2.0/mlflow/registered-models/get`` and parses the
    ``aliases`` list.  Works with both MLflow v2 and v3 servers.

    Returns:
        Mapping of ``{version_str: alias_name}`` (e.g. ``{"2": "champion"}``).
    """
    import json
    import urllib.error
    import urllib.parse
    import urllib.request

    uri = _tracking_uri()
    params = urllib.parse.urlencode({"name": model_name})
    url = f"{uri}/api/2.0/mlflow/registered-models/get?{params}"
    try:
        req = urllib.request.Request(url, method="GET")
        user = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
        pwd = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")
        if user and pwd:
            import base64

            creds = base64.b64encode(f"{user}:{pwd}".encode()).decode()
            req.add_header("Authorization", f"Basic {creds}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return {}

    alias_map: dict[str, str] = {}
    rm = data.get("registered_model", {})
    for a in rm.get("aliases", []):
        # REST API returns list of {"alias": "champion", "version": "2"} dicts
        if isinstance(a, dict):
            ver = str(a.get("version", ""))
            aname = str(a.get("alias", ""))
        else:
            # Unexpected format — skip
            continue
        if ver and aname:
            alias_map[ver] = aname
    return alias_map


def _render_model_registry(client) -> None:  # noqa: ANN001
    """Show registered models and their version stages."""
    st.markdown(
        '<div class="section-header">📦 Model Registry</div>',
        unsafe_allow_html=True,
    )
    try:
        models = _get_cached("registry_models")
        if models is None:
            # Materialise paginated results into a plain list so
            # truthiness / caching / iteration all work reliably.
            try:
                models = list(client.search_registered_models())
            except Exception:
                models = []
            # Fallback: SDK v3.x talking to server v2.x may return empty
            # or raise — query REST API directly.
            if not models:
                models = _rest_search_registered_models(client)
            # Only cache non-empty results so a subsequent page refresh
            # re-checks MLflow once models have been registered.
            if models:
                _set_cached("registry_models", models)
    except Exception as exc:
        st.error(f"Failed to fetch registered models: {exc}")
        return

    if not models:
        st.info("No models registered yet.  Run a Greenfield Bootstrap first.")
        # Add a manual refresh button
        if st.button("🔄 Refresh Registry", key="mlflow_refresh_registry"):
            _set_cached("registry_models", None)
            st.rerun()
        return

    stage_icons = {
        "Production": "✅",
        "Staging": "🧪",
        "Archived": "🗄️",
        "None": "⚪",
    }

    for model in models:
        name = model.name
        st.subheader(f"📦 {name}")
        desc = getattr(model, "description", None)
        if desc:
            st.markdown(desc)

        uri = _tracking_uri()

        try:
            cache_key_versions = f"registry_versions_{name}"
            versions = _get_cached(cache_key_versions)
            if versions is None:
                versions = list(client.search_model_versions(f"name='{name}'"))
                # REST fallback for SDK v3.x → server v2.x mismatch
                if not versions:
                    versions = _rest_search_model_versions(name)
                _set_cached(cache_key_versions, versions)
        except Exception:
            versions = _rest_search_model_versions(name)

        # Build version→alias map via direct REST call to registered-models/get.
        # The SDK alias objects differ across MLflow versions; REST is canonical.
        cache_key_aliases = f"registry_aliases_{name}"
        _alias_map: dict[str, str] = _get_cached(cache_key_aliases) or {}
        if not _alias_map:
            _alias_map = _rest_get_model_aliases(name)
            if _alias_map:
                _set_cached(cache_key_aliases, _alias_map)

        if not versions:
            st.info("No versions found.")
            continue

        for v in sorted(versions, key=lambda x: int(x.version), reverse=True):
            # Resolve alias from the pre-fetched REST map first, then fall back
            # to the version object's own .aliases attribute.
            _v_alias = _alias_map.get(str(v.version), "")
            if not _v_alias:
                _obj_aliases = getattr(v, "aliases", []) or []
                for _a in _obj_aliases:
                    _v_alias = _a.get("alias", "") if isinstance(_a, dict) else str(_a)
                    if _v_alias:
                        break
            # Map alias to display stage.
            # Any version with no alias is "Archived" (replaced or never deployed).
            if _v_alias == "champion":
                stage = "Production"
            elif _v_alias == "challenger":
                stage = "Staging"
            else:
                stage = "Archived"
            icon = stage_icons.get(stage, "🗄️")
            run_link = v.run_id[:12] if v.run_id else "—"
            with st.expander(
                f"{icon} Version {v.version} — **{stage}**  ·  run `{run_link}`",
                expanded=(stage == "Production"),
            ):
                c1, c2, c3 = st.columns(3)
                c1.metric("Version", v.version)
                c2.metric("Stage", stage)
                c3.metric("Run ID", run_link)
                if v.source:
                    st.markdown(f"**Source:** `{v.source}`")
                if hasattr(v, "creation_timestamp") and v.creation_timestamp:
                    from datetime import datetime, timezone

                    ts = datetime.fromtimestamp(v.creation_timestamp / 1000, tz=timezone.utc)
                    st.markdown(f"**Created:** {ts.strftime('%Y-%m-%d %H:%M:%S UTC')}")

                # Fetch run metrics for this version
                if v.run_id:
                    try:
                        cache_key_run = f"registry_run_{v.run_id}"
                        run = _get_cached(cache_key_run)
                        if run is None:
                            run = client.get_run(v.run_id)
                            _set_cached(cache_key_run, run)
                        metrics = run.data.metrics
                        if metrics:
                            st.markdown("**Metrics:**")
                            metric_cols = st.columns(min(len(metrics), 4))
                            for i, (k, val) in enumerate(sorted(metrics.items())):
                                metric_cols[i % len(metric_cols)].metric(
                                    k, f"{val:.4f}" if isinstance(val, float) else val
                                )
                    except Exception:
                        pass

                # Links to MLflow UI
                base = uri.rstrip("/")
                model_url = f"{base}/#/models/{name}/versions/{v.version}"
                st.markdown(f"[Open model version in MLflow UI ↗]({model_url})")
                if v.run_id:
                    run_url = f"{base}/#/experiments/0/runs/{v.run_id}"
                    st.markdown(f"[Open training run ↗]({run_url})")


def _render_approval_queue() -> None:
    """Model Approval Queue — Task 5 Human Review Gate.

    Reads pending model_approvals rows from the database and lets a human
    approve or reject each challenger model before promotion proceeds.
    """
    import contextlib
    import os
    import sqlite3
    from datetime import datetime, timezone
    from pathlib import Path

    st.markdown(
        '<div class="section-header">📋 Model Approval Queue</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "This queue shows challenger models waiting for human review before production promotion. "
        "The Airflow `automated_retraining` DAG pauses at the **wait_for_human_approval** task "
        "until a reviewer approves or rejects the model here."
    )

    pg_url = os.environ.get("DATABASE_URL", "")
    if not pg_url:
        # Fallback: build from POSTGRES_HOST / DB_PORT / POSTGRES_USER etc.
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

    def _load_approvals() -> list[dict]:
        rows = []
        if pg_url:
            try:
                import psycopg2

                conn = psycopg2.connect(pg_url)
                with conn, conn.cursor() as cur:
                    cur.execute(
                        "SELECT id, model_version, mlflow_run_id, challenger_f1, "
                        "champion_f1, champion_f1_on_challenger_test, status, "
                        "created_at, decided_at, decided_by "
                        "FROM model_approvals ORDER BY id DESC LIMIT 50"
                    )
                    cols = [d[0] for d in cur.description]
                    rows = [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]
                conn.close()
                return rows
            except Exception as exc:
                st.warning(f"PostgreSQL unavailable: {exc} — trying local SQLite…")

        _project_root = Path(__file__).resolve().parents[3]
        db_path = _project_root / "data" / "mlops.db"
        if not db_path.exists():
            return []
        with contextlib.closing(sqlite3.connect(str(db_path))) as con:
            con.row_factory = sqlite3.Row
            rows = [
                dict(r)
                for r in con.execute(
                    "SELECT id, model_version, mlflow_run_id, challenger_f1, "
                    "champion_f1, champion_f1_on_challenger_test, status, "
                    "created_at, decided_at, decided_by "
                    "FROM model_approvals ORDER BY id DESC LIMIT 50"
                ).fetchall()
            ]
        return rows

    def _update_status(approval_id: int, new_status: str, decided_by: str) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        if pg_url:
            try:
                import psycopg2

                conn = psycopg2.connect(pg_url)
                with conn, conn.cursor() as cur:
                    cur.execute(
                        "UPDATE model_approvals SET status=%s, decided_at=%s, decided_by=%s WHERE id=%s",
                        (new_status, now_iso, decided_by, approval_id),
                    )
                conn.close()
                return
            except Exception as exc:
                st.warning(f"PostgreSQL update failed: {exc} — using SQLite fallback")

        _project_root = Path(__file__).resolve().parents[3]
        db_path = _project_root / "data" / "mlops.db"
        with contextlib.closing(sqlite3.connect(str(db_path))) as con:
            con.execute(
                "UPDATE model_approvals SET status=?, decided_at=?, decided_by=? WHERE id=?",
                (new_status, now_iso, decided_by, approval_id),
            )
            con.commit()

    approvals = _load_approvals()
    if not approvals:
        st.success("No approval requests — nothing pending.")
        return

    pending = [a for a in approvals if a["status"] == "pending"]
    history = [a for a in approvals if a["status"] != "pending"]

    if pending:
        st.markdown(f"**{len(pending)} pending** approval(s):")
        for appr in pending:
            _chall_f1 = appr.get("challenger_f1") or 0.0
            with st.expander(
                f"🔔 #{appr['id']} — {appr['model_version']}  (challenger F1 {_chall_f1:.4f})",
                expanded=True,
            ):
                # Use champion_f1_on_challenger_test (fair) if available, else champion_f1
                _champ_f1_fair = appr.get("champion_f1_on_challenger_test")
                _champ_f1_display = (
                    _champ_f1_fair
                    if _champ_f1_fair is not None
                    else (appr.get("champion_f1") or 0.0)
                )
                _is_fair = _champ_f1_fair is not None
                _champ_label = "Champion F1†" if _is_fair else "Champion F1"
                c1, c2, c3 = st.columns(3)
                c1.metric("Challenger F1", f"{_chall_f1:.4f}")
                c2.metric(_champ_label, f"{_champ_f1_display:.4f}")
                delta = _chall_f1 - _champ_f1_display
                c3.metric("Delta", f"{delta:+.4f}", delta_color="normal")
                if _is_fair:
                    st.caption(
                        "† Champion re-evaluated on the same test signals as challenger (fair comparison)."
                    )
                st.caption(
                    f"MLflow run: `{appr.get('mlflow_run_id') or 'n/a'}` · "
                    f"Requested: {appr.get('created_at', '')}"
                )
                reviewer = st.text_input(
                    "Reviewer name / team", value="", key=f"reviewer_{appr['id']}"
                )
                btn_col1, btn_col2 = st.columns(2)
                if btn_col1.button("✅ Approve", key=f"approve_{appr['id']}", type="primary"):
                    _update_status(appr["id"], "approved", reviewer or "streamlit_user")
                    st.success(f"Model {appr['model_version']} approved!")
                    st.rerun()
                if btn_col2.button("🚫 Reject", key=f"reject_{appr['id']}"):
                    _update_status(appr["id"], "rejected", reviewer or "streamlit_user")
                    st.warning(f"Model {appr['model_version']} rejected.")
                    st.rerun()

    if history:
        with st.expander(f"Decision history ({len(history)} records)", expanded=False):
            import pandas as pd

            st.dataframe(pd.DataFrame(history), width="stretch")


def _render_connection_info() -> None:
    """Show connection details with full URI and clickable link."""
    uri = _tracking_uri()
    mode = _detect_mode()

    if mode == "cloud":
        dh_uri = _dagshub_uri()
        st.markdown(
            f"""
            **Mode:** ☁️ Cloud (Local-First Buffer) &nbsp;·&nbsp;
            **Buffer:** [`{uri}`]({uri})
            {f"&nbsp;·&nbsp; **DagsHub:** [`{dh_uri}`]({dh_uri})" if dh_uri else ""}
            """,
            unsafe_allow_html=True,
        )
        if dh_uri:
            st.info(
                "📊 **Local-first mode active.** All live operations use the buffer. "
                "DagsHub is updated only by the scheduled sync DAG. "
                "Use the **DagsHub View** tab below to inspect the remote state.",
            )
    else:
        mode_label = "☸️ K8s (Port-Forward)" if mode == "k8s" else "🐳 Local (Docker)"
        st.markdown(
            f"**Mode:** {mode_label} &nbsp;·&nbsp; **Tracking URI:** [`{uri}`]({uri})",
            unsafe_allow_html=True,
        )


def _render_pull_from_dagshub() -> None:
    """Render the Pull from DagsHub controls (local mode only)."""
    st.markdown(
        '<div class="section-header">⬇️ Pull MLflow Data from DagsHub</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Sync experiments, runs, metrics, and artifacts from DagsHub cloud "
        "into this local MLflow instance. The sync is **incremental** — "
        "only new or updated runs are transferred."
    )

    # Check prerequisites
    user = _env_or_file("DAGSHUB_USER")
    token = _env_or_file("DAGSHUB_TOKEN")
    repo = _env_or_file("DAGSHUB_REPO")
    if not all([user, token, repo]):
        st.warning(
            "Missing DagsHub credentials. Set `DAGSHUB_USER`, `DAGSHUB_TOKEN`, "
            "and `DAGSHUB_REPO` in `.env.secrets` or as environment variables."
        )
        return

    st.info(f"**DagsHub source:** `{user}/{repo}`")

    # Pull state
    pull_key = "_mlflow_pull_running"
    pull_result_key = "_mlflow_pull_result"

    col_pull, col_reset = st.columns([1, 1])
    with col_pull:
        if st.button(
            "⬇️ Start Pull",
            key="mlflow_pull_btn",
            type="primary",
            disabled=st.session_state.get(pull_key, False),
        ):
            st.session_state[pull_key] = True
            st.rerun()

    with col_reset:
        if st.button("🔄 Reset sync state", key="mlflow_reset_sync_btn"):
            state_file = Path(_PROJECT_ROOT) / "data" / ".mlflow_sync_state.json"
            if state_file.exists():
                state_file.unlink()
                st.success("Sync state reset — next pull will be a full sync.")
            else:
                st.info("No sync state file found (already clean).")

    if st.session_state.get(pull_key, False):
        status = st.status("Pulling from DagsHub…", expanded=True)

        try:
            from src.training.mlflow_sync import (
                build_dagshub_uri,
                pull_from_dagshub,
            )

            dagshub_uri = build_dagshub_uri(user, repo)
            local_uri = _tracking_uri()
            status.write(f"Source: `{dagshub_uri}`  →  Target: `{local_uri}`")

            summary = pull_from_dagshub(
                local_mlflow_uri=local_uri,
                dagshub_uri=dagshub_uri,
                dagshub_user=user,
                dagshub_token=token,
                sync_artifacts=True,
            )
            st.session_state[pull_key] = False
            st.session_state[pull_result_key] = summary

            exps = summary.get("experiments_synced", 0)
            runs = summary.get("runs_synced", 0)
            if runs > 0:
                status.update(
                    label=f"Pulled {runs} run(s) from {exps} experiment(s)!",
                    state="complete",
                )
            else:
                status.update(label="No new runs to sync", state="complete")
                st.info(
                    "No new runs found. If you expected data, click "
                    "**Reset sync state** and try again."
                )
        except Exception as exc:
            st.session_state[pull_key] = False
            status.update(label="Pull failed", state="error")
            st.error(str(exc))

    # Show previous result
    if pull_result_key in st.session_state:
        st.json(st.session_state[pull_result_key])


def _render_dagshub_view() -> None:
    """Render the DagsHub secondary view tab (cloud/local-first mode only).

    Shows experiments and model registry from DagsHub with clear [DagsHub] labels.
    Data is cached in ``st.session_state["_dagshub_view_cache"]`` so re-visiting
    the page does NOT trigger another DagsHub API call. The cache is invalidated:
      - After a successful push (Sync Buffer → DagsHub)
      - After a successful restore (Restore Buffer from DagsHub)
      - When the user clicks "Refresh DagsHub Data"
    """
    import time as _time_mod

    st.markdown(
        '<div class="section-header">☁️ DagsHub Remote State</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "**[DagsHub Remote]** This view shows a snapshot of DagsHub MLflow — "
        "separate from the live buffer. Use this to verify what has been synced "
        "and what is still pending. Cached in session memory; click Refresh for latest data."
    )

    dh_uri = _dagshub_uri()
    if not dh_uri:
        st.warning(
            "DagsHub URI not configured. Set `DAGSHUB_USER` and `DAGSHUB_REPO` in `.env.secrets`."
        )
        return

    user = _env_or_file("DAGSHUB_USER")
    token = _env_or_file("DAGSHUB_TOKEN")
    if not user or not token:
        st.warning("Missing DagsHub credentials (`DAGSHUB_USER` / `DAGSHUB_TOKEN`).")
        return

    st.markdown(f"**DagsHub URI:** `{dh_uri}`")

    _cache_key = "_dagshub_view_cache"
    cached = st.session_state.get(_cache_key)

    # Show "Last fetched" badge and Refresh button when cached data is available
    if cached and cached.get("source_uri") == dh_uri:
        fetched_at = cached.get("fetched_at_ts", 0)
        elapsed = int(_time_mod.time() - fetched_at)
        if elapsed < 60:
            age_str = f"{elapsed}s ago"
        elif elapsed < 3600:
            age_str = f"{elapsed // 60}m ago"
        else:
            age_str = f"{elapsed // 3600}h ago"
        col_info, col_btn = st.columns([3, 1])
        with col_info:
            st.caption(f"📷 Snapshot from DagsHub — last fetched: **{age_str}**")
        with col_btn:
            if st.button("🔄 Refresh", key="dh_refresh_btn"):
                st.session_state.pop(_cache_key, None)
                st.rerun()
    else:
        # No cache — show Connect button
        if not st.button("🔌 Connect to DagsHub", key="dh_connect_btn"):
            st.caption("Click **Connect to DagsHub** to fetch remote state.")
            return
        cached = None  # force fetch

    # Fetch if needed (no cache, or forced refresh)
    if cached is None or cached.get("source_uri") != dh_uri:
        try:
            import mlflow as _mlflow_dh
            from mlflow.tracking import MlflowClient as _DhClient

            os.environ["MLFLOW_TRACKING_USERNAME"] = user
            os.environ["MLFLOW_TRACKING_PASSWORD"] = token
            os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "30")
            _mlflow_dh.set_tracking_uri(dh_uri)
            dh_client = _DhClient(tracking_uri=dh_uri)

            with st.spinner("Fetching DagsHub data…"):
                exps = list(dh_client.search_experiments())
                try:
                    models = list(dh_client.search_registered_models())
                except Exception:
                    models = []

                # Build lightweight cache (no individual run metrics)
                exp_list = [
                    {
                        "name": e.name,
                        "experiment_id": e.experiment_id,
                        "lifecycle_stage": e.lifecycle_stage,
                    }
                    for e in exps
                ]
                model_list = []
                for m in models:
                    versions = list(dh_client.search_model_versions(f"name='{m.name}'"))
                    model_list.append(
                        {
                            "name": m.name,
                            "versions": [
                                {
                                    "version": v.version,
                                    "aliases": getattr(v, "aliases", []) or [],
                                    "run_id": v.run_id,
                                }
                                for v in versions[:10]
                            ],
                        }
                    )

            cached = {
                "experiments": exp_list,
                "models": model_list,
                "fetched_at_ts": _time_mod.time(),
                "fetched_at_iso": _time_mod.strftime("%Y-%m-%dT%H:%M:%SZ", _time_mod.gmtime()),
                "source_uri": dh_uri,
            }
            st.session_state[_cache_key] = cached

        except Exception as exc:
            error_msg = str(exc)
            if "429" in error_msg or "Too Many" in error_msg.lower():
                st.warning("⚠️ DagsHub rate-limiting (HTTP 429). Wait 60 s and try again.")
            else:
                st.error(f"DagsHub connection error: {exc}")
            return
        finally:
            # Always restore buffer as active tracking URI
            buffer_uri = _tracking_uri()
            os.environ["MLFLOW_TRACKING_URI"] = buffer_uri
            os.environ["MLFLOW_TRACKING_USERNAME"] = ""
            os.environ["MLFLOW_TRACKING_PASSWORD"] = ""
            import mlflow as _mlflow_restore

            _mlflow_restore.set_tracking_uri(buffer_uri)

    # ── Render cached data ───────────────────────────────────────────────
    st.markdown("#### 🧪 Experiments [DagsHub]")
    exp_list = cached.get("experiments", [])
    if exp_list:
        for exp in exp_list:
            st.markdown(
                f"- `{exp['name']}` (ID: {exp['experiment_id']}, stage: {exp['lifecycle_stage']})"
            )
    else:
        st.info("No experiments found on DagsHub.")

    st.markdown("#### 📦 Model Registry [DagsHub]")
    model_list = cached.get("models", [])
    if model_list:
        for m in model_list:
            st.markdown(f"**{m['name']}** — {len(m['versions'])} version(s)")
            for v in m["versions"]:
                aliases = v.get("aliases", [])
                alias_str = f" `{'`, `'.join(aliases)}`" if aliases else ""
                st.markdown(f"  - v{v['version']}{alias_str} · run `{v['run_id'][:8]}...`")
    else:
        st.info("No registered models on DagsHub.")


def _render_push_to_dagshub() -> None:
    """Render the Sync Buffer → DagsHub tab (cloud/local-first mode only)."""
    st.markdown(
        '<div class="section-header">⬆️ Sync Buffer → DagsHub</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Upload new experiments and runs from the local MLflow buffer to DagsHub. "
        "The sync is **incremental** — only runs not yet on DagsHub are transferred. "
        "Runs are identified by the `mlflow_sync.source_run_id` tag."
    )

    user = _env_or_file("DAGSHUB_USER")
    token = _env_or_file("DAGSHUB_TOKEN")
    repo = _env_or_file("DAGSHUB_REPO")
    if not all([user, token, repo]):
        st.warning(
            "Missing DagsHub credentials. Set `DAGSHUB_USER`, `DAGSHUB_TOKEN`, "
            "and `DAGSHUB_REPO` in `.env.secrets`."
        )
        return

    dh_uri = _dagshub_uri()
    buffer_uri = _tracking_uri()
    st.info(f"**Source (buffer):** `{buffer_uri}`  →  **Target (DagsHub):** `{dh_uri}`")

    col_sync, col_restore = st.columns([1, 1])

    with col_sync:
        sync_arts = st.checkbox("Sync artifacts (model .pkl files)", value=True, key="sync_arts_cb")
        if st.button("⬆️ Sync Buffer → DagsHub", key="push_to_dagshub_btn", type="primary"):
            with st.status("Syncing buffer → DagsHub…", expanded=True) as status:
                try:
                    from src.training.mlflow_sync import push_to_dagshub

                    summary = push_to_dagshub(
                        local_mlflow_uri=buffer_uri,
                        dagshub_uri=dh_uri,
                        dagshub_user=user,
                        dagshub_token=token,
                        sync_artifacts=sync_arts,
                    )
                    runs = summary.get("runs_synced", 0)
                    exps = summary.get("experiments_synced", 0)
                    if runs > 0:
                        status.update(
                            label=f"✅ Synced {runs} run(s) from {exps} experiment(s) to DagsHub!",
                            state="complete",
                        )
                    else:
                        status.update(
                            label="✅ No new runs to sync — DagsHub is up to date.",
                            state="complete",
                        )
                    # DagsHub content changed — invalidate the view cache
                    st.session_state.pop("_dagshub_view_cache", None)
                    st.json(summary)
                except Exception as exc:
                    status.update(label="Sync failed", state="error")
                    st.error(str(exc))

    with col_restore:
        st.markdown("**Restore from DagsHub** (download DagsHub → buffer)")
        st.caption("Use when buffer is empty or after a `docker volume rm`.")

        # Wipe-restore guard: if a local wipe was performed in this session,
        # require explicit confirmation before restoring from DagsHub so the
        # wipe is not accidentally undone.
        _post_wipe = st.session_state.get("_post_wipe", False)
        _restore_confirmed_key = "_restore_after_wipe_confirmed"
        if _post_wipe:
            st.warning(
                "⚠️ **A local wipe was performed in this session.** "
                "Restoring from DagsHub will re-populate the buffer with the "
                "previously wiped models and experiments, effectively undoing the wipe. "
                "Only proceed if this is intentional."
            )
            st.checkbox(
                "I understand — restore anyway",
                key=_restore_confirmed_key,
            )
            restore_disabled = not st.session_state.get(_restore_confirmed_key, False)
        else:
            restore_disabled = False

        if st.button(
            "⬇️ Restore Buffer from DagsHub",
            key="restore_from_dagshub_btn",
            disabled=restore_disabled,
        ):
            with st.status("Restoring buffer from DagsHub…", expanded=True) as status:
                try:
                    from src.training.mlflow_sync import pull_from_dagshub

                    summary = pull_from_dagshub(
                        local_mlflow_uri=buffer_uri,
                        dagshub_uri=dh_uri,
                        dagshub_user=user,
                        dagshub_token=token,
                        sync_artifacts=True,
                    )
                    # Clear the wipe guard after a successful restore
                    st.session_state.pop("_post_wipe", None)
                    st.session_state.pop(_restore_confirmed_key, None)
                    # Also invalidate the DagsHub view cache (user may want a fresh view)
                    st.session_state.pop("_dagshub_view_cache", None)
                    runs = summary.get("runs_synced", 0)
                    if runs > 0:
                        status.update(
                            label=f"✅ Restored {runs} run(s) from DagsHub!", state="complete"
                        )
                    else:
                        status.update(
                            label="✅ No new runs to restore — buffer is current.", state="complete"
                        )
                    st.json(summary)
                except Exception as exc:
                    status.update(label="Restore failed", state="error")
                    st.error(str(exc))


# ── Main render ─────────────────────────────────────────────────


def _show_mode_help(mode: str) -> None:
    """Show contextual help based on deployment mode."""
    if mode == "cloud":
        st.info(
            "**Cloud mode** — ensure `DAGSHUB_USER`, `DAGSHUB_TOKEN`, "
            "and `DAGSHUB_REPO` are set in `.env.secrets`."
        )
    elif mode == "k8s":
        st.info(
            "**K8s mode** — ensure `make k8s-ports` port-forwards are active "
            "(MLflow should be at `localhost:5000`). Run `make k8s-ports` to set them up."
        )
    else:
        st.info(
            "**Local mode** — ensure the Docker stack is running "
            "(`make local` or `docker compose up -d`)."
        )


def render() -> None:
    """Render the MLflow Explorer page."""
    try:
        _render_content()
    except Exception:
        _logger.exception("Unhandled exception in mlflow_explorer.render()")
        raise


def _render_content() -> None:
    """Inner render — separated so exceptions are caught and logged by render()."""
    # NOTE: Do NOT clear the data cache here on every render — this was the
    # cause of HTTP 429 rate-limit errors from DagsHub (every page load
    # triggered fresh API calls regardless of the 2-minute TTL).
    # Users can force a refresh with the "🔄 Refresh connection" button below.

    st.markdown(get_global_css(), unsafe_allow_html=True)
    st.markdown(
        hero_section(
            "🔬 MLflow Explorer",
            "Browse experiments, compare runs, and inspect the model registry — "
            "works with both local Docker MLflow and DagsHub cloud.",
        ),
        unsafe_allow_html=True,
    )

    # ── Connection info (non-critical) ──────────────
    with contextlib.suppress(Exception):
        _render_connection_info()

    # ── Refresh button to force reconnection and data cache clear ───────────
    # Also auto-clear the data cache when the deployment mode has changed
    # (e.g. user wiped data in Greenfield then switched to cloud mode).
    _cur_mode_for_cache = _detect_mode()
    _cached_mode = st.session_state.get("_mlflow_cache_mode", "")
    if _cached_mode != _cur_mode_for_cache:
        # Mode changed — evict stale experiment/run/model lists immediately
        st.session_state.pop(_DATA_CACHE_KEY, None)
        st.session_state["_mlflow_cache_mode"] = _cur_mode_for_cache

    if st.button(
        "🔄 Refresh All (clear cache)",
        key="mlflow_refresh_btn",
        help="Clear all cached experiment, run and model data and reload fresh from MLflow.",
    ):
        with contextlib.suppress(Exception):
            st.session_state.pop(_CACHE_KEY, None)
            st.session_state.pop(_DATA_CACHE_KEY, None)
            st.session_state.pop("_mlflow_cache_mode", None)
        st.rerun()

    # ── Client acquisition with session-state caching ──
    uri = ""
    with contextlib.suppress(Exception):
        uri = _tracking_uri()

    client = None
    need_connectivity_test = True

    # Reuse cached client when the URI hasn't changed
    try:
        cache = st.session_state.get(_CACHE_KEY)
        if isinstance(cache, dict) and cache.get("uri") == uri:
            client = cache.get("client")
            if client is not None:
                need_connectivity_test = False
    except Exception:
        pass

    if client is None:
        try:
            client = _get_client()
        except Exception as exc:
            mode = _detect_mode()
            st.error(f"Cannot connect to MLflow at `{uri}`. Error: `{exc}`")
            _show_mode_help(mode)
            return

    # ── Connectivity test (only for new/uncached clients) ──
    # Use requests directly with a socket-level timeout instead of
    # ThreadPoolExecutor + future.result(timeout=N).  The Future approach only
    # stops *waiting* for the thread; the underlying requests call inside the
    # MLflow SDK has no socket timeout and hangs indefinitely on Windows,
    # causing every subsequent render to time out as well.
    if need_connectivity_test:
        import requests as _requests
        from requests.auth import HTTPBasicAuth as _HTTPBasicAuth

        _PROBE_TIMEOUT = 12  # seconds — socket-level; enforced on all platforms

        # Build the probe URL: MLflow REST endpoint that returns quickly
        _probe_url = uri.rstrip("/") + "/api/2.0/mlflow/experiments/search"

        # Credentials for cloud (DagsHub) — empty strings for local
        _probe_user = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
        _probe_pass = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")
        _auth = _HTTPBasicAuth(_probe_user, _probe_pass) if _probe_user else None

        def _do_probe(verify: bool = True) -> _requests.Response:
            return _requests.post(
                _probe_url,
                json={"max_results": 1},
                auth=_auth,
                timeout=_PROBE_TIMEOUT,
                verify=verify,
            )

        _ssl_bypassed = False
        try:
            _resp = _do_probe(verify=True)
        except _requests.exceptions.SSLError:
            # Corporate TLS-inspection proxy presents a self-signed cert that
            # certifi doesn't know about; the browser works because the corporate
            # CA is in the OS trust store. Fall back to unverified for dev use.
            try:
                _requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]
                _resp = _do_probe(verify=False)
                _ssl_bypassed = True
                # Tell the MLflow SDK to also skip TLS verification
                os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
            except Exception as exc:
                mode = _detect_mode()
                from src.ui.views.use_cases_pkg._common import reset_mlflow_client_cache

                reset_mlflow_client_cache()
                st.session_state.pop(_CACHE_KEY, None)
                st.error(f"Cannot connect to MLflow at `{uri}`. SSL error: `{exc}`")
                _show_mode_help(mode)
                return
        except _requests.exceptions.Timeout:
            from src.ui.views.use_cases_pkg._common import reset_mlflow_client_cache

            reset_mlflow_client_cache()
            st.session_state.pop(_CACHE_KEY, None)
            st.error(
                f"MLflow at `{uri}` is not responding (timed out after {_PROBE_TIMEOUT} s).  "
                "The server may be starting up — try refreshing in a few seconds."
            )
            return
        except _requests.exceptions.ConnectionError as exc:
            mode = _detect_mode()
            from src.ui.views.use_cases_pkg._common import reset_mlflow_client_cache

            reset_mlflow_client_cache()
            st.session_state.pop(_CACHE_KEY, None)
            # "Remote end closed connection without response" = server warming up
            exc_str = str(exc)
            if "RemoteDisconnected" in exc_str or "Connection aborted" in exc_str:
                st.warning(
                    f"MLflow at `{uri}` is warming up — connection was closed before "
                    "a response was received.  This is normal right after K8s/Docker "
                    "startup.  **Please refresh in a few seconds.**"
                )
            else:
                st.error(f"Cannot connect to MLflow at `{uri}`. Connection error: `{exc}`")
                _show_mode_help(mode)
            return
        except Exception as exc:
            mode = _detect_mode()
            st.error(f"Cannot connect to MLflow at `{uri}`. Error: `{exc}`")
            _show_mode_help(mode)
            return

        # 200 = success; 400 = bad request body (server alive); 429 = rate limit; anything else is error
        if _resp.status_code == 429:
            st.warning(
                f"⚠️ MLflow at `{uri}` is rate-limiting requests (HTTP 429).  "
                "DagsHub limits the number of MLflow API calls per minute.  "
                "Wait 60 seconds and refresh, or use the cached data below."
            )
            # Don't return — try to use cached data below even if the probe was rate-limited
        elif _resp.status_code not in (200, 400):
            mode = _detect_mode()
            st.error(
                f"MLflow at `{uri}` returned HTTP {_resp.status_code}.  "
                "Check your credentials in `.env.secrets`."
            )
            _show_mode_help(mode)
            return

        if _ssl_bypassed:
            st.warning(
                "SSL certificate verification was bypassed (corporate TLS proxy detected).  "
                "Connection is active but unverified.  "
                "To fix permanently, set `REQUESTS_CA_BUNDLE` to your corporate CA bundle path."
            )

        # Cache successful client for subsequent renders
        with contextlib.suppress(Exception):
            st.session_state[_CACHE_KEY] = {"client": client, "uri": uri}

    # ── FU-5: Buffer empty detection (cloud mode only) ──────────
    # If the buffer has no experiments it is likely empty after a fresh
    # `make cloud` start or after `docker volume rm mlflow_buffer_db`.
    # Show a prominent orange banner so the user knows how to populate it.
    if _detect_mode() == "cloud":
        try:
            _exps_check = list(client.search_experiments()) if client else []
            if not _exps_check:
                st.warning(
                    "⚠️ **MLflow buffer is empty.** No experiments found in the buffer.  \n"
                    "This typically means the buffer was just started fresh (new install, "
                    "volume reset, or first `make cloud`).  \n"
                    "**To populate the buffer:**  \n"
                    "- Run **Greenfield Bootstrap** (UC-04) to create the first model, or  \n"
                    "- Use the **⬆️ Sync to DagsHub** tab → **Restore Buffer from DagsHub** "
                    "to download previously archived experiments and models."
                )
        except Exception:
            pass  # Don't let a check failure block the page

    mode = _detect_mode()
    try:
        # Use st.radio (keyed) instead of st.tabs() to prevent tab-jump on rerun.
        if mode == "cloud":
            _MLF_TABS = [
                "\U0001f4e6 Model Registry [Buffer]",
                "\U0001f3c5 Leaderboard [Buffer]",
                "\U0001f9ea Experiments [Buffer]",
                "\u2601\ufe0f DagsHub View",
                "\u2b06\ufe0f Sync to DagsHub",
                "\U0001f4cb Approval Queue",
            ]
        elif mode == "k8s":
            _MLF_TABS = [
                "\U0001f4e6 Model Registry [K8s]",
                "\U0001f3c5 Leaderboard [K8s]",
                "\U0001f9ea Experiments [K8s]",
                "\U0001f4cb Approval Queue",
            ]
        else:
            _MLF_TABS = [
                "\U0001f4e6 Model Registry",
                "\U0001f3c5 Leaderboard",
                "\U0001f9ea Experiments",
                "\U0001f4cb Approval Queue",
            ]

        active_mlf = st.radio(
            "MLflow tab",
            _MLF_TABS,
            horizontal=True,
            key="_mlf_tab",
            label_visibility="collapsed",
        )
        st.markdown(
            "<hr style='margin:0 0 1rem 0;border-color:#334155;'>",
            unsafe_allow_html=True,
        )

        if active_mlf == _MLF_TABS[0]:
            _render_model_registry(client)
        elif active_mlf == _MLF_TABS[1]:
            _render_run_leaderboard(client)
        elif active_mlf == _MLF_TABS[2]:
            _render_experiments(client)
        elif mode == "cloud" and active_mlf == _MLF_TABS[3]:
            _render_dagshub_view()
        elif mode == "cloud" and active_mlf == _MLF_TABS[4]:
            _render_push_to_dagshub()
        elif active_mlf.startswith("\U0001f4cb"):
            _render_approval_queue()

    except Exception as exc:
        st.error(f"Error rendering MLflow data: {exc}")

    # ── Footer ──────────────────────────────
    with contextlib.suppress(Exception):
        st.markdown("---")
        footer_uri = uri or _tracking_uri()
        st.markdown(f"[Open MLflow UI ↗]({footer_uri})")
