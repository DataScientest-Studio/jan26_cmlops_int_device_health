"""Docker management utilities for the Streamlit dashboard."""

from __future__ import annotations

import os
import queue as _queue
import re
import subprocess
import threading as _threading
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from src.ui.logging_ui import get_ui_logger  # noqa: E402

_logger = get_ui_logger(__name__)

# Container definitions: (compose_service, container_name, display_name, default_port, icon)
# default_port is the *internal* container port (used as fallback if Docker is unavailable).
# The actual *host* port is resolved dynamically via get_host_port().
#
# ORDER: Core Networking → Core Application → Databases → ML Platform → Monitoring
# This canonical order is used everywhere containers are listed (Docker Control,
# Services, Monitoring, etc.).
SERVICES = [
    # ── Core Networking ──────────────────────────────────────────
    ("nginx", "mlops_nginx", "Nginx Reverse Proxy", 80, "🌐"),
    # ── Core Application ─────────────────────────────────────────
    ("api", "mlops_api", "FastAPI Prediction", 8000, "⚡"),
    # ── Databases ────────────────────────────────────────────────
    ("postgres", "mlops_postgres", "PostgreSQL (App)", 5432, "🗄️"),
    # ── ML Platform ──────────────────────────────────────────────
    ("mlflow", "mlops_mlflow", "MLflow Tracking", 5001, "🔬"),
    ("airflow-webserver", "mlops_airflow", "Airflow Webserver", 8081, "🔄"),
    # ── Monitoring ───────────────────────────────────────────────
    ("prometheus", "mlops_prometheus", "Prometheus", 9090, "📊"),
    ("grafana", "mlops_grafana", "Grafana", 3000, "📈"),
    ("alertmanager", "mlops_alertmanager", "Alertmanager", 9093, "🔔"),
    ("cadvisor", "mlops_cadvisor", "cAdvisor", 8080, "📦"),
    ("node-exporter", "mlops_node_exporter", "Node Exporter", 9100, "💻"),
    ("postgres_exporter", "mlops_postgres_exporter", "PG Exporter (App DB)", 9187, "🐘"),
    ("blackbox_exporter", "mlops_blackbox_exporter", "Blackbox Exporter", 9115, "🔍"),
]

# In cloud mode the local mlflow container is not running; the buffer is used instead.
# _get_active_services() returns SERVICES with this substitution applied.
_CLOUD_OVERRIDES: dict[str, tuple[str, str, str, int, str]] = {
    "mlops_mlflow": ("mlflow_buffer", "mlops_mlflow_buffer", "MLflow Buffer", 5000, "🔬"),
}

# Extra services that only exist in cloud mode (not in SERVICES base list).
# These are inserted at the correct canonical position by _get_active_services().
# Tuple: (compose_service, container_name, display_name, default_port, icon)
_CLOUD_DB_EXTRA: tuple[str, str, str, int, str] = (
    "postgres_mlflow",
    "mlops_postgres_mlflow",
    "PostgreSQL (MLflow)",
    5432,
    "🗄️",
)
_CLOUD_MONITOR_EXTRA: tuple[str, str, str, int, str] = (
    "postgres_mlflow_exporter",
    "mlops_postgres_mlflow_exporter",
    "PG Exporter (MLflow DB)",
    9187,
    "🐘",
)


def _get_active_services(mode: str | None = None) -> list[tuple]:
    """Return mode-appropriate service list in canonical order.

    Canonical order: Networking → App → Databases → ML Platform → Monitoring

    In cloud mode:
      - ``mlops_mlflow`` (not running) is replaced with ``mlops_mlflow_buffer``
      - ``mlops_postgres_mlflow`` (dedicated MLflow PostgreSQL) is inserted
        right after ``mlops_postgres`` (Databases section)
      - ``mlops_postgres_mlflow_exporter`` is appended at the end of Monitoring
    """
    if mode is None:
        mode = detect_current_mode()
    result = []
    for entry in SERVICES:
        _svc, cname, _display, _port, _icon = entry
        if mode == "cloud" and cname in _CLOUD_OVERRIDES:
            result.append(_CLOUD_OVERRIDES[cname])
        else:
            result.append(entry)
        # In cloud mode, insert postgres_mlflow RIGHT AFTER postgres (App)
        if mode == "cloud" and cname == "mlops_postgres":
            result.append(_CLOUD_DB_EXTRA)
    if mode == "cloud":
        result.append(_CLOUD_MONITOR_EXTRA)
    return result


# ── Dynamic port detection ──────────────────────────────────────

# Cache of container_name -> {internal_port: host_port}
# Uses module-level dict — survives Streamlit reruns within the same
# Python process because Streamlit imports modules once.
_port_cache: dict[str, dict[int, int]] = {}
_port_cache_ts: float = 0.0  # time.monotonic() when cache was populated
_PORT_CACHE_TTL: float = 60.0  # seconds before re-probing Docker

# Env-var fallbacks used when Docker port discovery is unavailable (e.g. K8s mode).
# Maps (container_name, internal_port) → env var name whose value is the host port.
_ENVPORT_FALLBACK: dict[tuple[str, int], str] = {
    ("mlops_nginx", 80): "NGINX_HTTP_PORT",
    ("mlops_airflow", 8080): "AIRFLOW_PORT",
    ("mlops_postgres", 5432): "DB_PORT",
    ("mlops_api", 8000): "API_PORT",
    ("mlops_mlflow", 5000): "MLFLOW_PORT",
    ("mlops_prometheus", 9090): "PROMETHEUS_PORT",
    ("mlops_grafana", 3000): "GRAFANA_PORT",
}


def _parse_docker_port_output(output: str) -> dict[int, int]:
    """Parse ``docker port`` output into {internal_port: host_port} map.

    Example line: ``8080/tcp -> 0.0.0.0:8082``
    """
    mapping: dict[int, int] = {}
    for line in output.strip().splitlines():
        m = re.match(r"(\d+)/\w+\s+->\s+[\d.:]+:(\d+)", line)
        if m:
            internal = int(m.group(1))
            host = int(m.group(2))
            if internal not in mapping:
                mapping[internal] = host
    return mapping


def get_host_port(container_name: str, internal_port: int) -> int:
    """Return the actual host port for a container's internal port.

    Uses ``docker port <container>`` and caches the result with a TTL.
    Falls back to env-var overrides (e.g. ``NGINX_HTTP_PORT=8888`` set by
    ``.env.k8s`` via ``make ui``) when Docker is unavailable.
    Returns *internal_port* unchanged only as last resort.
    """
    import time as _time

    global _port_cache_ts  # noqa: PLW0603

    # Always check env-var overrides FIRST so K8s port-forward values take
    # priority over stale Docker cache or default internal ports.
    env_key = _ENVPORT_FALLBACK.get((container_name, internal_port))
    if env_key:
        env_val = os.environ.get(env_key, "")
        if env_val.isdigit():
            return int(env_val)

    # Expire stale cache
    now = _time.monotonic()
    if now - _port_cache_ts > _PORT_CACHE_TTL:
        _port_cache.clear()
        _port_cache_ts = now

    if container_name not in _port_cache:
        out, rc = _run(["docker", "port", container_name], timeout=5, silent=True)
        if rc == 0 and out:
            _port_cache[container_name] = _parse_docker_port_output(out)
        else:
            # Cache the miss as empty dict so we don't re-probe every render.
            _port_cache[container_name] = {}
    return _port_cache.get(container_name, {}).get(internal_port, internal_port)


def invalidate_port_cache() -> None:
    """Clear the cached port mappings (call after compose up/down)."""
    global _port_cache_ts  # noqa: PLW0603
    _port_cache.clear()
    _port_cache_ts = 0.0


# On Windows, Docker Desktop WSL2 has a broken IPv6 relay path.
# localhost resolves to ::1 → connection reset.  SERVICE_HOST=127.0.0.1 is set
# in .env.windows.local so that all URL helpers use the IPv4 loopback instead.
_SERVICE_HOST: str = os.environ.get("SERVICE_HOST", "localhost")


def get_host() -> str:
    """Return the host name to use for all service URLs.

    Returns ``127.0.0.1`` on Windows (set via ``SERVICE_HOST`` env var in
    ``.env.windows.local``) and ``localhost`` on Linux / macOS.
    """
    return _SERVICE_HOST


def get_service_url(container_name: str, internal_port: int, path: str = "") -> str:
    """Build a ``http://<host>:<host_port><path>`` URL for a service.

    If the container's port is not published to the host (e.g. FastAPI
    which is only reachable through Nginx), returns an empty string.
    """
    host_port = get_host_port(container_name, internal_port)
    return f"http://{_SERVICE_HOST}:{host_port}{path}"


def get_all_service_urls() -> dict[str, str]:
    """Return {display_name: url} for every service in SERVICES.

    Resolves host ports dynamically via ``docker port``.
    Special cases:
      - FastAPI: not published directly; accessed via Nginx on port 80.
    """
    urls: dict[str, str] = {}
    for _svc, cname, display, internal_port, _icon in _get_active_services():
        # FastAPI has no host port — it sits behind Nginx
        if cname == "mlops_api":
            nginx_port = get_host_port("mlops_nginx", 80)
            urls[display] = f"http://{_SERVICE_HOST}:{nginx_port}"
            continue
        urls[display] = get_service_url(cname, internal_port)
    return urls


@dataclass
class ContainerStatus:
    """Status of a single Docker container."""

    name: str
    display_name: str
    icon: str
    port: int
    state: str  # "running", "healthy", "unhealthy", "exited", "not_found"
    health: str  # "healthy", "unhealthy", "starting", "none", "unknown"
    image: str
    uptime: str

    @property
    def is_up(self) -> bool:
        return self.state in ("running",)

    @property
    def status_emoji(self) -> str:
        if self.health == "healthy":
            return "🟢"
        if self.state == "running":
            return "🟡"
        if self.state == "exited":
            return "🔴"
        return "⚫"


def _get_subprocess_env() -> dict[str, str]:
    """Build a subprocess environment that includes .env.windows.local overrides.

    Root cause: ``make ui`` sources ``.env.local`` / ``.env.cloud`` with ``set -a``
    which exports ``NGINX_HTTP_PORT=80`` (and similar) into the Streamlit process
    environment.  Docker Compose v2 gives shell environment variables HIGHER priority
    than ``--env-file`` arguments, so the ``--env-file .env.windows.local`` override
    (which sets ``NGINX_HTTP_PORT=8080``) is silently ignored — Docker then tries to
    bind to host port 80, which Windows HTTP.sys (PID 4) already holds, causing the
    ``ports are not available`` error that prevents nginx and grafana from starting.

    Fix: explicitly merge ``.env.windows.local`` values on top of ``os.environ``
    so that the correct Windows port mappings reach every docker compose call.
    """
    env = dict(os.environ)
    windows_local = PROJECT_ROOT / ".env.windows.local"
    if windows_local.exists():
        for raw_line in windows_local.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            if not key:
                continue
            val = val.strip()
            if len(val) >= 2 and val[0] in ('"', "'") and val[0] == val[-1]:
                val = val[1:-1]
            env[key] = val
    return env


def _run(cmd: list[str], *, timeout: int = 10, silent: bool = False) -> tuple[str, int]:
    """Run a command and return (combined stdout+stderr, returncode).

    When ``silent=True`` the WARNING log for non-zero return codes is
    suppressed.  Use this for probing commands (``docker inspect``,
    ``docker port``) where a non-zero result is expected when containers
    are not running.
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",  # never crash on non-UTF-8 Docker output (e.g. Windows cp1252)
            timeout=timeout,
            cwd=str(PROJECT_ROOT),
            env=_get_subprocess_env(),  # ensures .env.windows.local overrides are active
        )
        # Combine stdout and stderr so callers always see full output
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        combined = stdout
        if stderr:
            combined = (combined + "\n" + stderr) if combined else stderr
        if result.returncode != 0 and not silent:
            _logger.warning(
                "_run rc={} cmd={!r}\n{}",
                result.returncode,
                " ".join(cmd),
                combined[-500:] if len(combined) > 500 else combined,
            )
        return combined, result.returncode
    except subprocess.TimeoutExpired:
        return "(command timed out)", 1
    except FileNotFoundError:
        return "(docker not found)", 1


def _read_secret(key: str) -> str:
    """Read a single key from .env.secrets and return its stripped value.

    Returns an empty string when the file does not exist or the key is absent.
    Handles both ``KEY=value`` and ``KEY="value"`` formats.
    """
    secrets_file = PROJECT_ROOT / ".env.secrets"
    if not secrets_file.exists():
        return ""
    for line in secrets_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith(f"{key}="):
            value = line[len(key) + 1 :]
            # Strip surrounding quotes if present
            if len(value) >= 2 and value[0] in ('"', "'") and value[0] == value[-1]:
                value = value[1:-1]
            return value.strip()
    return ""


def get_container_statuses() -> list[ContainerStatus]:
    """Query Docker for the status of all project containers."""
    statuses: list[ContainerStatus] = []
    for _svc, cname, display, internal_port, icon in _get_active_services():
        fmt = "{{.State.Status}}|{{.State.Health.Status}}|{{.Config.Image}}|{{.State.StartedAt}}"
        out, rc = _run(["docker", "inspect", "--format", fmt, cname], silent=True)
        host_port = get_host_port(cname, internal_port)
        if rc != 0:
            statuses.append(
                ContainerStatus(
                    name=cname,
                    display_name=display,
                    icon=icon,
                    port=host_port,
                    state="not_found",
                    health="unknown",
                    image="—",
                    uptime="—",
                )
            )
            continue
        parts = out.split("|")
        state = parts[0] if len(parts) > 0 else "unknown"
        health = parts[1] if len(parts) > 1 and parts[1] else "none"
        image = parts[2].split("/")[-1][:40] if len(parts) > 2 else "—"
        started = parts[3][:19].replace("T", " ") if len(parts) > 3 else "—"
        statuses.append(
            ContainerStatus(
                name=cname,
                display_name=display,
                icon=icon,
                port=host_port,
                state=state,
                health=health,
                image=image,
                uptime=started,
            )
        )
    return statuses


def get_container_logs(container_name: str, tail: int = 100) -> str:
    """Fetch the last N lines of logs from a container."""
    try:
        result = subprocess.run(
            ["docker", "logs", "--tail", str(tail), "--timestamps", container_name],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            cwd=str(PROJECT_ROOT),
        )
        # Docker may output to stdout, stderr, or both
        combined = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
        return combined or "(no logs available)"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "(no logs available)"


# K8s pod label → (display_name, icon)
_K8S_POD_META: dict[str, tuple[str, str]] = {
    "api": ("FastAPI Prediction", "⚡"),
    "airflow": ("Airflow Webserver", "🌀"),
    "grafana": ("Grafana Dashboards", "📊"),
    "mlflow": ("MLflow Tracking", "📈"),
    "nginx": ("Nginx Reverse Proxy", "🌐"),
    "postgres": ("PostgreSQL (App)", "🐘"),
    "prometheus": ("Prometheus", "🔥"),
    "streamlit": ("Streamlit UI", "🖥"),
}

# K8s service name → host port env var
_K8S_SVC_PORT: dict[str, str] = {
    "api": "API_PORT",
    "airflow": "AIRFLOW_PORT",
    "grafana": "GRAFANA_PORT",
    "mlflow": "MLFLOW_PORT",
    "nginx": "NGINX_HTTP_PORT",
    "postgres": "DB_PORT",
    "prometheus": "PROMETHEUS_PORT",
}


def get_k8s_pod_statuses() -> list[ContainerStatus]:
    """Return ContainerStatus for each running K8s pod in the mlops namespace."""
    import re

    statuses: list[ContainerStatus] = []
    out, rc = _run(
        [
            "kubectl",
            "get",
            "pods",
            "-n",
            "mlops",
            "--no-headers",
            "-o",
            "custom-columns=NAME:.metadata.name,STATUS:.status.phase,"
            "READY:.status.containerStatuses[0].ready,"
            "IMAGE:.status.containerStatuses[0].image,"
            "STARTED:.status.startTime",
        ],
        silent=True,
    )
    if rc != 0:
        return statuses

    for line in out.splitlines():
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) < 3:
            continue
        full_name = parts[0]
        phase = parts[1] if len(parts) > 1 else "Unknown"
        ready = (parts[2] if len(parts) > 2 else "false").lower() == "true"
        image = parts[3].split("/")[-1][:40] if len(parts) > 3 else "—"
        started = parts[4][:19].replace("T", " ") if len(parts) > 4 else "—"

        # Derive service label from pod name (e.g. "api-685fb96499-kcmxs" → "api")
        label = full_name.split("-")[0]
        display, icon = _K8S_POD_META.get(label, (full_name, "📦"))

        # Map phase + ready to a Docker-like state string
        if phase == "Running" and ready:
            state, health = "running", "healthy"
        elif phase == "Running" and not ready:
            state, health = "running", "starting"
        elif phase == "Pending":
            state, health = "created", "none"
        else:
            state, health = phase.lower(), "none"

        # Resolve host port from env var
        env_key = _K8S_SVC_PORT.get(label, "")
        port_str = os.environ.get(env_key, "")
        port = int(port_str) if port_str.isdigit() else 0

        statuses.append(
            ContainerStatus(
                name=full_name,
                display_name=display,
                icon=icon,
                port=port,
                state=state,
                health=health,
                image=image,
                uptime=started,
            )
        )
    return statuses


def detect_current_mode() -> str:
    """Detect whether the stack is running in local or cloud mode.

    Priority order:
      1. ``.current_mode`` file — written by ``compose_up()`` and ``make local``/``make cloud``.
         Always current: updated whenever the mode changes, including mode switches done
         from within Streamlit.  Takes priority over the env var because ``make ui`` bakes
         ``DEPLOYMENT_MODE`` once at startup and it becomes stale on an in-session switch.
      2. ``DEPLOYMENT_MODE`` env var — set by ``make ui`` from ``.current_mode`` at startup.
         Useful when ``.current_mode`` is absent (first run, CI, etc.).
      3. Docker container inspection — fallback for Makefile-only starts without ``.current_mode``.
         IMPORTANT: only ``mlops_mlflow`` (local-mode container) triggers ``local``.
         ``mlops_mlflow_buffer`` is the cloud-mode buffer and must NOT trigger ``local``.
      4. ``"unknown"`` when nothing is detected.
    """
    # 1. Authoritative file written by compose_up() / Makefile / mode-switch in Streamlit.
    #    Must take priority over the env var because make ui reads .current_mode ONCE at
    #    startup and bakes it into DEPLOYMENT_MODE; if the user then switches mode from
    #    within Streamlit (which updates .current_mode), the env var stays stale.
    mode_file = PROJECT_ROOT / ".current_mode"
    if mode_file.exists():
        mode = mode_file.read_text().strip()
        if mode in ("local", "cloud", "k8s"):
            return mode

    # 2. DEPLOYMENT_MODE env var fallback (useful when .current_mode is absent)
    env_mode = os.environ.get("DEPLOYMENT_MODE", "").strip()
    if env_mode in ("local", "cloud", "k8s"):
        return env_mode

    # 3. Fallback: check if local MLflow container (mlops_mlflow) is running.
    # mlops_mlflow is ONLY started in local mode via docker-compose.local.yml.
    # mlops_mlflow_buffer is the cloud-mode primary MLflow — it must NOT trigger local.
    out, rc = _run(
        ["docker", "inspect", "--format", "{{.State.Status}}", "mlops_mlflow"], silent=True
    )
    if rc == 0 and "running" in out:
        return "local"

    # 4. Fallback for cloud mode: buffer or API running → cloud
    api_out, api_rc = _run(
        ["docker", "inspect", "--format", "{{.State.Status}}", "mlops_api"], silent=True
    )
    buf_out, buf_rc = _run(
        ["docker", "inspect", "--format", "{{.State.Status}}", "mlops_mlflow_buffer"],
        silent=True,
    )
    if (api_rc == 0 and "running" in api_out) or (buf_rc == 0 and "running" in buf_out):
        return "cloud"

    return "unknown"


def _env_file_flags(mode: str) -> list[str]:
    """Build --env-file flags for the given mode, mirroring the Makefile exactly.

    Load order matches ``LOCAL_ENV`` / ``CLOUD_ENV`` in the Makefile:
    1. Mode-specific file (``.env.local`` or ``.env.cloud``)
    2. ``.env.secrets`` — may override mode-specific values; may be absent on
       fresh clones or CI.
    3. ``.env.windows.local`` — Windows port/TLS overrides (``NGINX_HTTP_PORT``,
       ``MLFLOW_TRACKING_INSECURE_TLS``, etc.); included only when present.
       **Critical on Windows**: without this file the API container cannot reach
       DagsHub (self-signed cert rejection), causing health-check failures that
       prevent ``nginx`` and ``grafana`` from starting (they depend on
       ``api: condition: service_healthy``).
    """
    env_file = ".env.local" if mode == "local" else ".env.cloud"
    flags: list[str] = ["--env-file", env_file]
    secrets = PROJECT_ROOT / ".env.secrets"
    if secrets.exists():
        flags += ["--env-file", ".env.secrets"]
    # Windows-specific overrides (ports 8080/8443, TLS insecure flag, etc.)
    windows_local = PROJECT_ROOT / ".env.windows.local"
    if windows_local.exists():
        flags += ["--env-file", ".env.windows.local"]
    return flags


def make_streaming(
    target: str,
    extra_vars: dict[str, str] | None = None,
    *,
    timeout: int = 900,
) -> Generator[str, None, int]:
    """Run ``make <target>`` and yield output lines as they arrive.

    Merges stdout and stderr so all Docker build/pull progress is captured.
    The process exit code is returned via ``StopIteration.value``::

        gen = make_streaming("cloud-rebuild")
        lines: list[str] = []
        try:
            while True:
                lines.append(next(gen))
        except StopIteration as exc:
            rc = exc.value   # 0 = success, non-zero = failure

    Args:
        target:     Makefile target (e.g. ``"cloud-rebuild"``, ``"ghcr"``).
        extra_vars: Optional ``{KEY: VALUE}`` Make variable overrides.
        timeout:    Kill the process after this many seconds (default 900).
    """
    cmd: list[str] = ["make", "--no-print-directory"]
    if extra_vars:
        for k, v in extra_vars.items():
            cmd.append(f"{k}={v}")
    cmd.append(target)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge stderr → single stream
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(PROJECT_ROOT),
        env=_get_subprocess_env(),
    )

    q: _queue.Queue[str | None] = _queue.Queue()

    def _reader() -> None:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                q.put(line)
        finally:
            q.put(None)  # sentinel

    reader_thread = _threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    import time as _time

    deadline = _time.monotonic() + timeout

    while True:
        remaining = deadline - _time.monotonic()
        if remaining <= 0:
            proc.kill()
            yield "\n[make_streaming] Process killed: timeout exceeded\n"
            break
        try:
            item = q.get(timeout=min(remaining, 1.0))
        except _queue.Empty:
            continue
        if item is None:
            break
        yield item

    reader_thread.join(timeout=5)
    proc.wait()
    return proc.returncode


def compose_up(mode: str, rebuild: bool = True) -> tuple[str, int]:
    """Start the Docker stack by delegating to the Makefile target.

    Calls ``make local`` / ``make local-rebuild`` / ``make cloud`` /
    ``make cloud-rebuild`` directly so behaviour is byte-for-byte identical
    to running the command from a terminal.  This avoids reimplementing the
    env-file, profile, and multi-step logic that lives in the Makefile — and
    crucially avoids the subprocess timeout issues that caused Docker builds to
    be killed mid-way when Python enforced a hard 600 s cap.

    Args:
        mode: "local" or "cloud"
        rebuild: If True, no-cache rebuild (``make *-rebuild``).
                 If False, incremental start (``make local`` / ``make cloud``).
    """
    invalidate_port_cache()
    target_map = {
        ("local", False): "local",
        ("local", True): "local-rebuild",
        ("cloud", False): "cloud",
        ("cloud", True): "cloud-rebuild",
    }
    target = target_map[(mode, rebuild)]
    # Generous timeouts: rebuild pulls/builds images (can take 15+ min on first run);
    # plain start is faster but still needs time for pg_ensure_passwords etc.
    timeout = 900 if rebuild else 300
    out, rc = _run(["make", "--no-print-directory", target], timeout=timeout)
    if rc == 0:
        invalidate_port_cache()
    return out, rc


def compose_pull_and_start() -> tuple[str, int]:
    """Pull images from GHCR then start the stack via ``make ghcr``.

    Delegates to ``make ghcr GHCR_TAG=<tag>`` so authentication, pull order,
    and startup sequence are identical to running the command from a terminal.

    The image tag is read from ``GHCR_TAG`` in ``.env.secrets``
    (default: ``main``).  Images are named::

        ghcr.io/<GITHUB_OWNER>/mlops-device-health-api:<GHCR_TAG>
        ghcr.io/<GITHUB_OWNER>/mlops-device-health-airflow:<GHCR_TAG>
        ghcr.io/<GITHUB_OWNER>/mlops-device-health-streamlit:<GHCR_TAG>

    Requires ``GITHUB_TOKEN`` (read:packages scope) in ``.env.secrets`` for
    private packages.  Set ``GITHUB_OWNER`` in ``.env.secrets`` to override
    the default registry owner (``your-github-username``).
    """
    invalidate_port_cache()
    ghcr_tag = _read_secret("GHCR_TAG") or "main"
    # Pass GHCR_TAG as a Make variable override so it propagates to docker compose
    # substitution in docker-compose.ghcr.yml without requiring shell export.
    out, rc = _run(
        ["make", "--no-print-directory", f"GHCR_TAG={ghcr_tag}", "ghcr"],
        timeout=600,
    )
    if rc == 0:
        invalidate_port_cache()
    return out, rc


def compose_down() -> tuple[str, int]:
    """Stop all containers and return the combined docker compose output.

    Delegates to ``make down`` so the teardown sequence is identical to
    running ``make down`` from a terminal.  Returns ``("", 0)`` only as a
    last-resort fallback — the actual make output is always captured.
    """
    invalidate_port_cache()
    out, rc = _run(["make", "--no-print-directory", "down"], timeout=180)
    invalidate_port_cache()
    return out, rc


def compose_restart(service: str) -> tuple[str, int]:
    """Restart a single service."""
    return _run(
        ["docker", "compose", "-f", "docker-compose.yml", "restart", service],
        timeout=60,
    )


def compose_stop_service(service: str) -> tuple[str, int]:
    """Stop a single service."""
    return _run(
        ["docker", "compose", "-f", "docker-compose.yml", "stop", service],
        timeout=60,
    )


def compose_start_service(service: str) -> tuple[str, int]:
    """Start a single (already created) service."""
    return _run(
        ["docker", "compose", "-f", "docker-compose.yml", "start", service],
        timeout=60,
    )


def get_stack_health_summary() -> dict[str, int]:
    """Return a quick {running, healthy, stopped, total} count for the stack.

    Useful for displaying a status line after compose_up/compose_down without
    rebuilding the full per-container table.
    """
    statuses = get_container_statuses()
    running = sum(1 for s in statuses if s.is_up)
    healthy = sum(1 for s in statuses if s.health == "healthy")
    stopped = len(statuses) - running
    return {
        "running": running,
        "healthy": healthy,
        "stopped": stopped,
        "total": len(statuses),
    }
