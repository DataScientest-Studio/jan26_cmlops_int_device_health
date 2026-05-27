"""
MLflow sync between local Docker MLflow and DagsHub cloud.

Uses the MLflow REST API directly (via ``requests``) so that it works in
environments where the ``mlflow`` Python package is not installed (e.g. the
Airflow worker container).

Two sync directions are supported:

* **push** – copy experiments / runs / artifacts from local → DagsHub
* **pull** – copy experiments / runs / artifacts from DagsHub → local

Both directions are *incremental*: a JSON state file records the last-synced
timestamp per direction so only new/updated runs are transferred.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_SYNC_STATE_FILE = "data/.mlflow_sync_state.json"
_BATCH_SIZE = 100  # max runs per search_runs page (MLflow REST API limit: 1000)


# ---------------------------------------------------------------------------
# REST response normalisation helpers
# ---------------------------------------------------------------------------


def _tags_to_dict(tags: list[dict[str, str]] | dict[str, str] | None) -> dict[str, str]:
    """Convert REST API tag/param format (list of {key, value}) to flat dict."""
    if tags is None:
        return {}
    if isinstance(tags, dict):
        return tags
    return {t["key"]: t["value"] for t in tags if "key" in t and "value" in t}


def _metrics_to_dict(metrics: list[dict[str, Any]] | dict[str, Any] | None) -> dict[str, Any]:
    """Convert REST API metrics list to flat {key: value} dict."""
    if metrics is None:
        return {}
    if isinstance(metrics, dict):
        return metrics
    return {m["key"]: m["value"] for m in metrics if "key" in m and "value" in m}


# ---------------------------------------------------------------------------
# Helpers – REST wrappers
# ---------------------------------------------------------------------------


class MLflowRESTClient:
    """Thin wrapper around the MLflow 2.x REST API."""

    def __init__(
        self,
        base_url: str,
        username: str = "",
        token: str = "",
        verify_ssl: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers["Content-Type"] = "application/json"
        if username and token:
            self.session.auth = (username, token)
        # Allow callers to disable SSL verification when a corporate TLS proxy
        # inserts a self-signed certificate (common on Windows / VPN setups).
        if not verify_ssl:
            self.session.verify = False
            import urllib3

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # -- experiments --------------------------------------------------------

    def search_experiments(self) -> list[dict[str, Any]]:
        """Return all experiments (paginated)."""
        experiments: list[dict[str, Any]] = []
        page_token: str | None = None
        while True:
            body: dict[str, Any] = {"max_results": 200}
            if page_token:
                body["page_token"] = page_token
            resp = self._post("/api/2.0/mlflow/experiments/search", body)
            experiments.extend(resp.get("experiments", []))
            page_token = resp.get("next_page_token")
            if not page_token:
                break
        return experiments

    def create_experiment(self, name: str, tags: list[dict[str, str]] | None = None) -> str:
        """Create experiment; return experiment_id."""
        body: dict[str, Any] = {"name": name}
        if tags:
            body["tags"] = tags
        resp = self._post("/api/2.0/mlflow/experiments/create", body)
        return resp["experiment_id"]

    def get_experiment_by_name(self, name: str) -> dict[str, Any] | None:
        resp = self._get(
            "/api/2.0/mlflow/experiments/get-by-name", params={"experiment_name": name}
        )
        return resp.get("experiment")

    def restore_experiment(self, experiment_id: str) -> None:
        """Restore a soft-deleted experiment."""
        self._post("/api/2.0/mlflow/experiments/restore", {"experiment_id": experiment_id})

    # -- runs ---------------------------------------------------------------

    def search_runs(
        self,
        experiment_ids: list[str],
        filter_string: str = "",
        order_by: list[str] | None = None,
        max_results: int = _BATCH_SIZE,
        page_token: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "experiment_ids": experiment_ids,
            "max_results": max_results,
        }
        if filter_string:
            body["filter"] = filter_string
        if order_by:
            body["order_by"] = order_by
        if page_token:
            body["page_token"] = page_token
        return self._post("/api/2.0/mlflow/runs/search", body)

    def create_run(self, experiment_id: str, start_time: int, tags: list[dict[str, str]]) -> str:
        """Create a new run; return run_id."""
        resp = self._post(
            "/api/2.0/mlflow/runs/create",
            {"experiment_id": experiment_id, "start_time": start_time, "tags": tags},
        )
        return resp["run"]["info"]["run_id"]

    def update_run(self, run_id: str, status: str, end_time: int) -> None:
        self._post(
            "/api/2.0/mlflow/runs/update",
            {"run_id": run_id, "status": status, "end_time": end_time},
        )

    def log_batch(
        self,
        run_id: str,
        metrics: list[dict[str, Any]] | None = None,
        params: list[dict[str, Any]] | None = None,
        tags: list[dict[str, Any]] | None = None,
    ) -> None:
        body: dict[str, Any] = {"run_id": run_id}
        if metrics:
            body["metrics"] = metrics
        if params:
            body["params"] = params
        if tags:
            body["tags"] = tags
        self._post("/api/2.0/mlflow/runs/log-batch", body)

    def get_run(self, run_id: str) -> dict[str, Any]:
        resp = self._get("/api/2.0/mlflow/runs/get", params={"run_id": run_id})
        return resp["run"]

    # -- artifacts ----------------------------------------------------------

    def list_artifacts(self, run_id: str, path: str = "") -> list[dict[str, Any]]:
        params: dict[str, str] = {"run_id": run_id}
        if path:
            params["path"] = path
        resp = self._get("/api/2.0/mlflow/artifacts/list", params=params)
        return resp.get("files", [])

    def download_artifact(self, run_id: str, path: str) -> bytes:
        """Download a single artifact file and return raw bytes."""
        url = f"{self.base_url}/get-artifact"
        resp = self.session.get(url, params={"run_id": run_id, "path": path}, timeout=120)
        resp.raise_for_status()
        return resp.content

    def upload_artifact(self, run_id: str, local_path: str, artifact_path: str = "") -> None:
        """Upload a single artifact file via the MLflow artifacts API."""
        url = f"{self.base_url}/api/2.0/mlflow-artifacts/artifacts"
        if artifact_path:
            url += f"/{artifact_path}"
        filename = os.path.basename(local_path)
        url += f"/{filename}"
        with open(local_path, "rb") as f:
            data = f.read()
        resp = self.session.put(
            url,
            data=data,
            headers={"Content-Type": "application/octet-stream"},
            params={"run_id": run_id},
            timeout=120,
        )
        resp.raise_for_status()

    # -- internal -----------------------------------------------------------

    def _post(self, endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        resp = self.session.post(url, json=body, timeout=60)
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    def _get(self, endpoint: str, params: dict[str, str] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        resp = self.session.get(url, params=params, timeout=60)
        resp.raise_for_status()
        return resp.json() if resp.content else {}


# ---------------------------------------------------------------------------
# Sync state persistence
# ---------------------------------------------------------------------------


def _load_sync_state(state_file: str = _SYNC_STATE_FILE) -> dict[str, Any]:
    p = Path(state_file)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _save_sync_state(state: dict[str, Any], state_file: str = _SYNC_STATE_FILE) -> None:
    p = Path(state_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Core sync logic
# ---------------------------------------------------------------------------


def _ensure_experiment(
    target: MLflowRESTClient,
    experiment_name: str,
) -> str:
    """Get or create experiment on target; return experiment_id.

    Handles the case where a previous experiment with the same name was
    soft-deleted (``lifecycle_stage == "deleted"``).  The MLflow REST API
    refuses to create a new experiment when a deleted one with the same name
    exists, so we restore it first.
    """
    existing = target.get_experiment_by_name(experiment_name)
    if existing:
        lifecycle = existing.get("lifecycle_stage", "active")
        if lifecycle == "deleted":
            target.restore_experiment(existing["experiment_id"])
            logger.info(
                "Restored soft-deleted experiment %r (id=%s)",
                experiment_name,
                existing["experiment_id"],
            )
        return existing["experiment_id"]

    try:
        return target.create_experiment(experiment_name)
    except requests.HTTPError as exc:
        # Race condition or name collision with deleted experiment not
        # returned by get-by-name on some MLflow versions — retry once.
        if exc.response is not None and exc.response.status_code == 400:
            logger.warning(
                "create_experiment(%r) returned 400; retrying after lookup: %s",
                experiment_name,
                exc,
            )
            existing = target.get_experiment_by_name(experiment_name)
            if existing:
                if existing.get("lifecycle_stage") == "deleted":
                    target.restore_experiment(existing["experiment_id"])
                return existing["experiment_id"]
        raise


def _sync_run(
    source: MLflowRESTClient,
    target: MLflowRESTClient,
    target_experiment_id: str,
    run: dict[str, Any],
    sync_artifacts: bool = True,
) -> str:
    """Copy a single run (metrics, params, tags, artifacts) to target.

    Returns the new run_id on the target.
    """
    info = run["info"]
    data = run.get("data", {})

    # Normalise REST API list-of-dicts → flat dict
    tags_dict = _tags_to_dict(data.get("tags"))
    params_dict = _tags_to_dict(data.get("params"))
    metrics_dict = _metrics_to_dict(data.get("metrics"))

    # Create run on target
    tags = [{"key": k, "value": v} for k, v in tags_dict.items()]
    # Add a provenance tag
    tags.append({"key": "mlflow_sync.source_run_id", "value": info["run_id"]})
    tags.append({"key": "mlflow_sync.source_uri", "value": source.base_url})

    new_run_id = target.create_run(
        experiment_id=target_experiment_id,
        start_time=info.get("start_time", 0),
        tags=tags,
    )

    # Log params (in batches of 100)
    params = [{"key": k, "value": v} for k, v in params_dict.items()]
    for i in range(0, len(params), 100):
        target.log_batch(new_run_id, params=params[i : i + 100])

    # Log metrics (in batches of 1000)
    metrics = [
        {"key": k, "value": v, "timestamp": info.get("start_time", 0), "step": 0}
        for k, v in metrics_dict.items()
    ]
    for i in range(0, len(metrics), 1000):
        target.log_batch(new_run_id, metrics=metrics[i : i + 1000])

    # Sync artifacts
    if sync_artifacts:
        _sync_artifacts_recursive(source, target, info["run_id"], new_run_id, "")

    # Finalize run status
    target.update_run(
        new_run_id,
        status=info.get("status", "FINISHED"),
        end_time=info.get("end_time", info.get("start_time", 0)),
    )

    return new_run_id


def _sync_artifacts_recursive(
    source: MLflowRESTClient,
    target: MLflowRESTClient,
    source_run_id: str,
    target_run_id: str,
    path: str,
) -> int:
    """Recursively download artifacts from source and upload to target.

    Returns the number of files synced.
    """
    count = 0
    try:
        files = source.list_artifacts(source_run_id, path)
    except requests.HTTPError:
        logger.debug("No artifacts at path=%r for run %s", path, source_run_id)
        return 0

    for f in files:
        file_path = f.get("path", "")
        is_dir = f.get("is_dir", False)
        if is_dir:
            count += _sync_artifacts_recursive(
                source, target, source_run_id, target_run_id, file_path
            )
        else:
            try:
                content = source.download_artifact(source_run_id, file_path)
                # Write to temp file, then upload
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.basename(file_path)
                ) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                # artifact_path is the directory part, filename is the basename
                parent = str(Path(file_path).parent) if "/" in file_path else ""
                target.upload_artifact(target_run_id, tmp_path, artifact_path=parent)
                os.unlink(tmp_path)
                count += 1
            except Exception:
                logger.warning(
                    "Failed to sync artifact %s for run %s", file_path, source_run_id, exc_info=True
                )

    return count


def _get_synced_run_ids(
    target: MLflowRESTClient,
    target_experiment_ids: list[str],
) -> set[str]:
    """Collect source run IDs that have already been synced to the target."""
    synced: set[str] = set()
    for exp_id in target_experiment_ids:
        page_token: str | None = None
        while True:
            resp = target.search_runs([exp_id], max_results=_BATCH_SIZE, page_token=page_token)
            for run in resp.get("runs", []):
                tags = _tags_to_dict(run.get("data", {}).get("tags"))
                src_id = tags.get("mlflow_sync.source_run_id")
                if src_id:
                    synced.add(src_id)
            page_token = resp.get("next_page_token")
            if not page_token:
                break
    return synced


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_dagshub_uri(dagshub_user: str, dagshub_repo: str) -> str:
    """Construct the DagsHub MLflow tracking URI."""
    return f"https://dagshub.com/{dagshub_user}/{dagshub_repo}.mlflow"


def get_dagshub_credentials() -> tuple[str, str, str]:
    """Read DagsHub credentials from env vars.

    Returns (dagshub_user, dagshub_token, dagshub_repo).
    Raises ValueError if required vars are missing.
    """
    user = os.environ.get("DAGSHUB_USER", "")
    token = os.environ.get("DAGSHUB_TOKEN", "")
    repo = os.environ.get("DAGSHUB_REPO", "")
    if not user or not token or not repo:
        msg = (
            "Missing DagsHub credentials. Set DAGSHUB_USER, DAGSHUB_TOKEN, "
            "and DAGSHUB_REPO environment variables."
        )
        raise ValueError(msg)
    return user, token, repo


def pull_from_dagshub(
    local_mlflow_uri: str | None = None,
    dagshub_uri: str | None = None,
    dagshub_user: str | None = None,
    dagshub_token: str | None = None,
    experiment_names: list[str] | None = None,
    sync_artifacts: bool = True,
    state_file: str = _SYNC_STATE_FILE,
) -> dict[str, Any]:
    """Pull experiments/runs from DagsHub into local MLflow (incremental).

    Args:
        local_mlflow_uri: Local MLflow tracking URI (default: from env/``http://localhost:5001``)
        dagshub_uri: DagsHub MLflow URI (default: built from env)
        dagshub_user: DagsHub username (default: from env)
        dagshub_token: DagsHub token (default: from env)
        experiment_names: Only pull these experiments (default: all)
        sync_artifacts: Whether to sync artifact files
        state_file: Path to the sync state JSON file

    Returns:
        Summary dict with counts of synced experiments/runs.
    """
    # Resolve credentials
    if not dagshub_user or not dagshub_token:
        env_user, env_token, env_repo = get_dagshub_credentials()
        dagshub_user = dagshub_user or env_user
        dagshub_token = dagshub_token or env_token
        if not dagshub_uri:
            dagshub_uri = build_dagshub_uri(env_user, env_repo)

    if not dagshub_uri:
        msg = "dagshub_uri is required (or set DAGSHUB_USER + DAGSHUB_REPO env vars)"
        raise ValueError(msg)

    local_mlflow_uri = local_mlflow_uri or os.environ.get(
        "MLFLOW_TRACKING_URI", "http://localhost:5001"
    )

    logger.info("PULL: DagsHub (%s) → local (%s)", dagshub_uri, local_mlflow_uri)

    # Auto-detect SSL issues with DagsHub (corporate TLS proxy / self-signed cert).
    _dagshub_verify_ssl = True
    try:
        _probe = MLflowRESTClient(dagshub_uri, username=dagshub_user, token=dagshub_token)
        _probe.search_experiments()
    except Exception as _ssl_probe_err:
        _err_str = str(_ssl_probe_err).lower()
        if "ssl" in _err_str or "certificate" in _err_str:
            logger.warning(
                "DagsHub SSL verification failed (%s) — retrying without SSL verification. "
                "This is expected on corporate networks with TLS inspection proxies.",
                _ssl_probe_err,
            )
            _dagshub_verify_ssl = False
    remote = MLflowRESTClient(
        dagshub_uri,
        username=dagshub_user,
        token=dagshub_token,
        verify_ssl=_dagshub_verify_ssl,
    )
    local = MLflowRESTClient(local_mlflow_uri)

    # Load state
    state = _load_sync_state(state_file)
    last_pull_ts = state.get("last_pull_timestamp_ms", 0)

    # Get remote experiments
    experiments = remote.search_experiments()
    if not experiments:
        logger.warning(
            "PULL: Remote DagsHub MLflow returned 0 experiments.  "
            "This usually means no experiments have been logged to the DagsHub "
            "MLflow tracking URI (%s).  DagsHub's 'Experiments' tab may show "
            "DagsHub-native experiments that are separate from the MLflow API.",
            dagshub_uri,
        )
    if experiment_names:
        name_set = set(experiment_names)
        experiments = [e for e in experiments if e.get("name") in name_set]

    total_runs_synced = 0
    experiment_count = 0

    for exp in experiments:
        exp_name = exp["name"]

        # Ensure experiment exists locally
        target_exp_id = _ensure_experiment(local, exp_name)

        # Find runs already synced locally
        synced_ids = _get_synced_run_ids(local, [target_exp_id])

        page_token: str | None = None
        exp_runs_synced = 0

        while True:
            filter_str = ""
            if last_pull_ts > 0:
                filter_str = f"attributes.end_time > {last_pull_ts}"

            resp = remote.search_runs(
                [exp["experiment_id"]],
                filter_string=filter_str,
                order_by=["attributes.end_time ASC"],
                page_token=page_token,
            )
            runs = resp.get("runs", [])
            if not runs:
                break

            for run in runs:
                src_run_id = run["info"]["run_id"]
                if src_run_id in synced_ids:
                    logger.debug("Skipping already-synced run %s", src_run_id)
                    continue
                if run["info"].get("status") not in ("FINISHED", "COMPLETED"):
                    continue

                logger.info("Pulling run %s from DagsHub", src_run_id)
                _sync_run(remote, local, target_exp_id, run, sync_artifacts=sync_artifacts)
                exp_runs_synced += 1

            page_token = resp.get("next_page_token")
            if not page_token:
                break

        if exp_runs_synced > 0:
            experiment_count += 1
            total_runs_synced += exp_runs_synced
            logger.info("Experiment '%s': pulled %d runs", exp_name, exp_runs_synced)

    # Only update state timestamp when runs were actually synced,
    # to avoid poisoning the incremental filter with a future timestamp
    # that would skip all runs on subsequent pulls.
    now_iso = datetime.now(tz=timezone.utc).isoformat()
    if total_runs_synced > 0:
        now_ms = int(time.time() * 1000)
        state["last_pull_timestamp_ms"] = now_ms
        state["last_pull_iso"] = now_iso
    state["last_pull_runs_synced"] = total_runs_synced
    _save_sync_state(state, state_file)

    summary = {
        "direction": "pull",
        "source": dagshub_uri,
        "target": local_mlflow_uri,
        "experiments_synced": experiment_count,
        "runs_synced": total_runs_synced,
        "incremental_from_ms": last_pull_ts,
        "timestamp_iso": state.get("last_pull_iso", now_iso),
    }
    logger.info("PULL complete: %s", summary)
    return summary


def push_to_dagshub(
    local_mlflow_uri: str | None = None,
    dagshub_uri: str | None = None,
    dagshub_user: str | None = None,
    dagshub_token: str | None = None,
    experiment_names: list[str] | None = None,
    sync_artifacts: bool = True,
    state_file: str = _SYNC_STATE_FILE,
) -> dict[str, Any]:
    """Push experiments/runs from local MLflow buffer to DagsHub (incremental).

    Symmetric to :func:`pull_from_dagshub` but in the upload direction.
    Runs that have already been synced to DagsHub — identified by the
    ``mlflow_sync.source_run_id`` tag present on the DagsHub side — are
    skipped to prevent duplicates.

    This is called by the ``sync_mlflow_to_dagshub`` Airflow DAG and by the
    Streamlit "Sync Buffer → DagsHub" button.

    Args:
        local_mlflow_uri: Local MLflow buffer URI (default: from env or
            ``http://localhost:5002``)
        dagshub_uri: DagsHub MLflow URI (default: built from env)
        dagshub_user: DagsHub username (default: from env)
        dagshub_token: DagsHub token (default: from env)
        experiment_names: Only push these experiments (default: all non-Default)
        sync_artifacts: Whether to sync artifact files (model PKL, plots, etc.)
        state_file: Path to the sync state JSON file

    Returns:
        Summary dict with counts of synced experiments/runs.
    """
    # Resolve credentials
    if not dagshub_user or not dagshub_token:
        env_user, env_token, env_repo = get_dagshub_credentials()
        dagshub_user = dagshub_user or env_user
        dagshub_token = dagshub_token or env_token
        if not dagshub_uri:
            dagshub_uri = build_dagshub_uri(env_user, env_repo)

    if not dagshub_uri:
        msg = "dagshub_uri is required (or set DAGSHUB_USER + DAGSHUB_REPO env vars)"
        raise ValueError(msg)

    local_mlflow_uri = local_mlflow_uri or os.environ.get(
        "MLFLOW_TRACKING_URI", "http://localhost:5002"
    )

    logger.info("PUSH: local buffer (%s) → DagsHub (%s)", local_mlflow_uri, dagshub_uri)

    local = MLflowRESTClient(local_mlflow_uri)
    # Auto-detect SSL issues with DagsHub (corporate TLS proxy / self-signed cert).
    # First attempt with SSL verification; if it fails with an SSL error, retry
    # without verification and log a warning.
    _dagshub_verify_ssl = True
    try:
        _probe = MLflowRESTClient(dagshub_uri, username=dagshub_user, token=dagshub_token)
        _probe.search_experiments()
    except Exception as _ssl_probe_err:
        _err_str = str(_ssl_probe_err).lower()
        if "ssl" in _err_str or "certificate" in _err_str:
            logger.warning(
                "DagsHub SSL verification failed (%s) — retrying without SSL verification. "
                "This is expected on corporate networks with TLS inspection proxies.",
                _ssl_probe_err,
            )
            _dagshub_verify_ssl = False
    remote = MLflowRESTClient(
        dagshub_uri,
        username=dagshub_user,
        token=dagshub_token,
        verify_ssl=_dagshub_verify_ssl,
    )

    # Load state
    state = _load_sync_state(state_file)
    last_push_ts = state.get("last_push_timestamp_ms", 0)

    # Get local experiments
    experiments = local.search_experiments()
    if experiment_names:
        name_set = set(experiment_names)
        experiments = [e for e in experiments if e.get("name") in name_set]
    else:
        # Skip the MLflow Default experiment (experiment_id "0") — it is
        # auto-created and usually empty; pushing it to DagsHub creates noise.
        experiments = [e for e in experiments if e.get("name") != "Default"]

    total_runs_synced = 0
    experiment_count = 0

    for exp in experiments:
        exp_name = exp["name"]

        # Ensure experiment exists on DagsHub
        target_exp_id = _ensure_experiment(remote, exp_name)

        # Find buffer runs already present on DagsHub (avoid duplicates)
        synced_ids = _get_synced_run_ids(remote, [target_exp_id])

        page_token: str | None = None
        exp_runs_synced = 0

        while True:
            filter_str = ""
            if last_push_ts > 0:
                filter_str = f"attributes.end_time > {last_push_ts}"

            resp = local.search_runs(
                [exp["experiment_id"]],
                filter_string=filter_str,
                order_by=["attributes.end_time ASC"],
                page_token=page_token,
            )
            runs = resp.get("runs", [])
            if not runs:
                break

            for run in runs:
                src_run_id = run["info"]["run_id"]
                if src_run_id in synced_ids:
                    logger.debug("Skipping already-synced run %s", src_run_id)
                    continue
                if run["info"].get("status") not in ("FINISHED", "COMPLETED"):
                    logger.debug("Skipping non-finished run %s", src_run_id)
                    continue

                logger.info("Pushing run %s to DagsHub", src_run_id)
                try:
                    _sync_run(local, remote, target_exp_id, run, sync_artifacts=sync_artifacts)
                    exp_runs_synced += 1
                except requests.HTTPError as exc:
                    if exc.response is not None and exc.response.status_code in (403, 404):
                        logger.warning(
                            "Skipping run %s — DagsHub returned HTTP %s for experiment %r "
                            "(experiment may be read-only or was restored from a soft-delete). "
                            "This run will NOT be retried.",
                            src_run_id,
                            exc.response.status_code,
                            exp_name,
                        )
                    else:
                        raise

            page_token = resp.get("next_page_token")
            if not page_token:
                break

        if exp_runs_synced > 0:
            experiment_count += 1
            total_runs_synced += exp_runs_synced
            logger.info("Experiment '%s': pushed %d runs to DagsHub", exp_name, exp_runs_synced)

    now_iso = datetime.now(tz=timezone.utc).isoformat()
    if total_runs_synced > 0:
        state["last_push_timestamp_ms"] = int(time.time() * 1000)
        state["last_push_iso"] = now_iso
    state["last_push_runs_synced"] = total_runs_synced
    _save_sync_state(state, state_file)

    summary = {
        "direction": "push",
        "source": local_mlflow_uri,
        "target": dagshub_uri,
        "experiments_synced": experiment_count,
        "runs_synced": total_runs_synced,
        "incremental_from_ms": last_push_ts,
        "timestamp_iso": now_iso,
    }
    logger.info("PUSH complete: %s", summary)
    return summary
