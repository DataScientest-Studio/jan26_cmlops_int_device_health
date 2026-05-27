"""GitHub CI/CD Dashboard — workflow runs, GHCR images, local git info.

Uses the GitHub REST API to expose CI/CD status, container registry
images, and workflow dispatch capabilities.  Falls back to local git
info (recent commits, branches) when no API token is available.

Token resolution order:
1. ``GITHUB_TOKEN`` or ``GH_TOKEN`` environment variable
2. ``gh auth token`` CLI (if GitHub CLI is installed)
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    import streamlit.components.v1 as components  # noqa: F401 — needed for Mermaid
except ImportError:  # pragma: no cover
    components = None  # type: ignore[assignment]

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from src.ui.logging_ui import get_ui_logger
from src.ui.styles import get_global_css, hero_section

_logger = get_ui_logger(__name__)

# ── Mermaid rendering ────────────────────────────────────────────


def _render_mermaid(diagram: str, height: int = 320) -> None:
    """Render a Mermaid diagram via embedded Mermaid.js."""
    if components is None:
        st.code(diagram, language="text")
        return
    html_content = f"""
    <html><head>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    </head><body style="background:transparent;margin:0;overflow:auto">
    <div id="diagram" class="mermaid" style="background:#0f172a;border-radius:12px;padding:1rem;
         min-height:200px;overflow:visible;margin-bottom:1rem;">
{diagram.strip()}
    </div>
    <script>
      (function tryRender(n) {{
        if (typeof mermaid !== 'undefined') {{
          mermaid.initialize({{startOnLoad:false,theme:'dark',securityLevel:'loose'}});
          mermaid.run({{nodes:[document.getElementById('diagram')]}});
          var _a = 0;
          var _p = setInterval(function() {{
            _a++;
            var svg = document.querySelector('#diagram svg');
            if (svg || _a > 100) {{
              clearInterval(_p);
              var el = document.getElementById('diagram');
              if (el && window.frameElement) {{
                window.frameElement.style.height = (el.scrollHeight + 32) + 'px';
              }}
            }}
          }}, 100);
        }} else if (n < 100) {{
          setTimeout(function() {{ tryRender(n + 1); }}, 100);
        }}
      }})(0);
    </script>
    </body></html>
    """
    components.html(html_content, height=height, scrolling=True)


# ── Per-workflow Mermaid diagrams ────────────────────────────────

_WF_DIAGRAMS: list[dict] = [
    {
        "title": "lint.yml — Lint & Format",
        "description": (
            "Runs on every push and pull request. Enforces code style (ruff), format "
            "compliance (ruff format --check), and type safety (mypy). Fast — no Docker, "
            "no network calls. Skips data/docs commits."
        ),
        "notes": [
            "Trigger: push to any branch, PRs, workflow_dispatch",
            "Tools: ruff (lint + format) + mypy (type check)",
            "Python: 3.12 via uv; paths-ignore: data/**, doc/**, *.md",
        ],
        "diagram": (
            "flowchart LR\n"
            '  A["\U0001f4e5 push / PR"] --> B[checkout]\n'
            "  B --> C[uv sync --dev]\n"
            "  C --> D[ruff check src/ tests/]\n"
            "  D --> E[ruff format --check]\n"
            "  E --> F[mypy src/]\n"
            '  F --> G{"all pass?"}\n'
            '  G -->|yes| H["\u2705 CI pass"]\n'
            '  G -->|no| I["\u274c CI fail"]'
        ),
    },
    {
        "title": "test.yml \u2014 Unit & Integration Tests",
        "description": (
            "Runs the full non-live test suite on every push and PR. "
            "Generates a bootstrap model before tests if absent. "
            "Uploads coverage to Codecov. No Docker stack required."
        ),
        "notes": [
            "Trigger: push to any branch, PRs, workflow_dispatch",
            'Command: pytest tests/ -v -m "not live" --timeout=60',
            "Excludes: @pytest.mark.live (those need Docker stack \u2192 live-tests.yml)",
            "Coverage: pyproject.toml addopts inject --cov automatically",
        ],
        "diagram": (
            "flowchart LR\n"
            '  A["\U0001f4e5 push / PR"] --> B[checkout]\n'
            "  B --> C[uv sync --dev]\n"
            "  C --> D[generate bootstrap model]\n"
            "  D --> E[\"pytest -m 'not live'\"]\n"
            "  E --> F[upload coverage to Codecov]\n"
            '  F --> G{"tests pass?"}\n'
            '  G -->|yes| H["\u2705 CI pass"]\n'
            '  G -->|no| I["\u274c CI fail"]'
        ),
    },
    {
        "title": "build.yml \u2014 Build & Push Docker Image",
        "description": (
            "Builds the API Docker image on every push and PR. "
            "Runs Trivy container vulnerability scan and uploads SARIF to GitHub Security. "
            "Pushes the image to GHCR only on main branch or version tag pushes. "
            "PRs get build + scan but no push."
        ),
        "notes": [
            "Push to GHCR: main branch and v* tags only (not feature branches)",
            "Scan: Trivy SARIF \u2192 GitHub Code Scanning (continue-on-error)",
            "Cache: Docker BuildKit layer cache via GitHub Actions cache",
            "Tags: branch, PR number, semver (on v* push), SHA prefix",
        ],
        "diagram": (
            "flowchart LR\n"
            '  A["\U0001f4e5 push / PR"] --> B[checkout]\n'
            "  B --> C[login GHCR]\n"
            "  C --> D[docker buildx build]\n"
            "  D --> E[trivy scan]\n"
            "  E --> F[upload SARIF]\n"
            '  F --> G{"main or tag?"}\n'
            '  G -->|yes| H["push image to GHCR \U0001f4e6"]\n'
            '  G -->|no| I["scan only - no push"]\n'
            '  H --> J["\u2705 done"]\n'
            "  I --> J"
        ),
    },
    {
        "title": "code-quality.yml \u2014 Security & Dependency Audit",
        "description": (
            "Runs on PRs to main and weekly (Monday 06:00 UTC) to catch new CVE publications. "
            "Performs Python SAST with bandit and PyPI dependency CVE scan with pip-audit. "
            "Reports are uploaded as artifacts. Does not fail CI by default "
            "(|| true) to allow triage of pre-existing findings."
        ),
        "notes": [
            "Trigger: PRs to main, weekly schedule (Mon 06:00 UTC), workflow_dispatch",
            "bandit -r src/ -ll: medium+high severity Python SAST only",
            "pip-audit: PyPI advisory database CVE scan of requirements.txt",
            "Reports: JSON artifacts (30-day retention) \u2014 download from Actions UI",
            "|| true: soft failure (reports uploaded but CI not blocked)",
        ],
        "diagram": (
            "flowchart LR\n"
            '  A["\U0001f4e5 PR to main / weekly"] --> B[checkout]\n'
            "  B --> C[pip install bandit pip-audit]\n"
            "  C --> D[bandit -r src/ -ll]\n"
            "  C --> E[pip-audit requirements.txt]\n"
            "  D --> F[bandit-report.json]\n"
            "  E --> G[pip-audit-report.json]\n"
            "  F --> H[upload artifacts]\n"
            "  G --> H\n"
            '  H --> I["\u2705 done - review reports"]'
        ),
    },
    {
        "title": "live-tests.yml \u2014 Live Docker Stack Tests",
        "description": (
            "Brings up the full Docker Compose stack and runs @pytest.mark.live tests. "
            "Runs nightly at 02:00 UTC and on PRs touching Docker/Airflow/API paths. "
            "The conftest.py fixture auto-starts and (optionally) tears down the stack. "
            "Container logs are dumped on failure for debugging."
        ),
        "notes": [
            "Trigger: nightly 02:00 UTC, workflow_dispatch, PRs touching docker/airflow/api",
            "Command: pytest tests/ -m live --timeout=900 --no-cov",
            "DOCKER_KEEP_UP=1: skip teardown (CI runner is ephemeral)",
            "DagsHub creds: fake values (ci_test_user) \u2014 not used in live test stack",
            "Cost: ~15 min on GitHub-hosted runners \u2014 most expensive workflow",
        ],
        "diagram": (
            "flowchart LR\n"
            '  A["\u23f0 nightly / PR / manual"] --> B[checkout]\n'
            "  B --> C[uv sync]\n"
            "  C --> D[.env.secrets]\n"
            "  D --> E[bootstrap model]\n"
            '  E --> F["pytest -m live"]\n'
            '  F --> G{"pass?"}\n'
            '  G -->|yes| H["\u2705 passed"]\n'
            "  G -->|no| I[dump logs]\n"
            '  I --> J["\u274c failed"]'
        ),
    },
    {
        "title": "deploy.yml \u2014 Deploy",
        "description": (
            "Manually triggered or on version tag pushes. Pulls the Docker image from GHCR, "
            "runs a PostgreSQL smoke test, and executes the deploy step. "
            "The deploy step is currently a placeholder (no real SSH target). "
            "Suitable for project demo; real deployment requires a target host."
        ),
        "notes": [
            "Trigger: workflow_dispatch (choose staging/production + tag), push v* tags",
            "Smoke test: starts API + postgres:15-alpine, hits /health endpoint",
            "Deploy step: placeholder echo statements (no real target configured)",
            "Verify: prints deployment summary + image SHA",
        ],
        "diagram": (
            "flowchart LR\n"
            '  A["\U0001f680 tag push / manual"] --> B[checkout]\n'
            "  B --> C[login GHCR]\n"
            "  C --> D[pull image]\n"
            "  D --> E[start postgres:15-alpine]\n"
            "  E --> F[start API container]\n"
            "  F --> G[wait for /health]\n"
            '  G --> H{"healthy?"}\n'
            '  H -->|yes| I["deploy placeholder"]\n'
            '  H -->|no| J["\u274c smoke test failed"]\n'
            "  I --> K[verify + summary]\n"
            '  K --> L["\u2705 deployment done"]'
        ),
    },
    {
        "title": "model-quality-gate.yml — Model Quality Gate",
        "description": (
            "Runs on PRs to main that touch training code, feature extraction, or params.yaml. "
            "Trains a fresh LogisticRegression model on git-committed golden reference signals "
            "(data/ci/quality_gate_signals.csv) and fails the PR if accuracy or F1 falls below "
            "the thresholds in params.yaml quality_gate section. No Docker, no DagsHub, no MLflow. "
            "Pure Python — runs in under 60 seconds. See doc/GitHub_Actions_Reference.md Section 15."
        ),
        "notes": [
            "Status: Active — runs on PRs to main touching training code",
            "Trigger: PRs to main touching src/signal_processing, src/training, params.yaml, "
            "data/ci/quality_gate_signals.csv",
            "Data: data/ci/quality_gate_signals.csv (git-tracked plain CSV, NO .dvc sidecar)",
            "Thresholds: params.yaml quality_gate.min_accuracy (0.80) + min_f1 (0.75)",
            "Script: scripts/ci_quality_gate.py | Regenerate: scripts/generate_ci_quality_gate_data.py",
        ],
        "diagram": (
            "flowchart LR\n"
            '  A["\U0001f4e5 PR to main"] --> B[checkout]\n'
            "  B --> C[uv sync --dev]\n"
            "  C --> D[verify quality_gate_signals.csv]\n"
            "  D --> E[load raw signals]\n"
            "  E --> F[extract features]\n"
            "  F --> G[train LogReg 80/20]\n"
            "  G --> H[evaluate accuracy + F1]\n"
            "  H --> I[upload metrics artifact]\n"
            '  I --> J{"meets thresholds?"}\n'
            '  J -->|yes| K["\u2705 quality gate passed"]\n'
            '  J -->|no| L["\u274c quality gate failed"]'
        ),
    },
    {
        "title": "deploy-k8s.yml \u2014 K8s CI/CD Deploy",
        "description": (
            "Deploys the application to a Kind (Kubernetes in Docker) cluster using Kustomize "
            "overlays and GHCR images. Triggered automatically after `build.yml` succeeds on "
            "main, or manually via workflow_dispatch. Creates an ephemeral Kind cluster, loads "
            "images, applies manifests, waits for pods, and smoke-tests the /health endpoint. "
            "The cluster is torn down after the run."
        ),
        "notes": [
            "Trigger: After build.yml succeeds on main (workflow_run), workflow_dispatch",
            "Images: pulled from ghcr.io/${GITHUB_OWNER}/mlops-* (GHCR overlay)",
            "Overlay: k8s/overlays/ghcr/ — applies Kustomize patches over k8s/base/",
            "Smoke test: curl -sf http://localhost:30080/api/health",
            "Teardown: kind delete cluster (cluster is ephemeral per run)",
        ],
        "diagram": (
            "flowchart LR\n"
            '  A["\U0001f4e5 main push (build.yml) / manual"] --> B[checkout]\n'
            "  B --> C[install kubectl + kind]\n"
            "  C --> D[create Kind cluster]\n"
            "  D --> E[pull GHCR images]\n"
            "  E --> F[kubectl apply -k overlays/ghcr]\n"
            "  F --> G[wait pods: Ready]\n"
            '  G --> H{"GET /health"}\n'
            '  H -->|"200 OK"| I["\u2705 smoke test passed"]\n'
            '  H -->|"fail"| J["\u274c workflow fails"]\n'
            "  I --> K[kind delete cluster]"
        ),
    },
]


def _render_workflow_diagrams() -> None:
    """Per-workflow CI/CD diagrams with Mermaid flowcharts."""
    st.markdown(
        '<div class="section-header">\U0001f4ca CI/CD Workflow Diagrams</div>',
        unsafe_allow_html=True,
    )

    wf_titles = [wf["title"] for wf in _WF_DIAGRAMS]
    sel_title = st.selectbox(
        "Select workflow",
        wf_titles,
        key="_wf_diagram_sel",
        label_visibility="visible",
    )
    st.markdown("<hr style='margin:0.5rem 0 1rem 0;border-color:#334155;'>", unsafe_allow_html=True)

    for wf in _WF_DIAGRAMS:
        if wf["title"] == sel_title:
            st.markdown(f"**{wf['title']}**")
            st.markdown(wf["description"])
            _render_mermaid(wf["diagram"], height=340)
            if wf["notes"]:
                st.markdown("**Configuration notes:**")
                for note in wf["notes"]:
                    st.markdown(f"- {note}")
            break


# ── GitHub connection helpers ───────────────────────────────────

_DEFAULT_OWNER = os.environ.get("GITHUB_OWNER", "your-github-username")
_DEFAULT_REPO = "mlops-device-health"


def _gh_owner() -> str:
    return os.environ.get("GITHUB_OWNER", _DEFAULT_OWNER)


def _gh_repo() -> str:
    return os.environ.get("GITHUB_REPO", _DEFAULT_REPO)


@st.cache_data(ttl=300)
def _gh_token_from_cli() -> str | None:
    """Try to get token from ``gh auth token`` CLI."""
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def _gh_token() -> str | None:
    env_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if env_token:
        return env_token
    # Fallback: read from .env.secrets directly
    secrets_path = Path(_PROJECT_ROOT) / ".env.secrets"
    if secrets_path.exists():
        for line in secrets_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("GITHUB_TOKEN=") or line.startswith("GH_TOKEN="):
                _, _, val = line.partition("=")
                val = val.strip().strip('"').strip("'")
                if val:
                    return val
    return _gh_token_from_cli()


def _gh_get(path: str, *, timeout: int = 12) -> dict | list | None:
    """GET from GitHub REST API.  Returns parsed JSON or *None*."""
    url = f"https://api.github.com{path}"
    headers: dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = _gh_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        _logger.warning("GitHub GET {} failed: {}", path, exc)
        return None


def _gh_post(path: str, payload: dict | None = None, *, timeout: int = 15) -> dict | None:
    """POST to GitHub REST API."""
    url = f"https://api.github.com{path}"
    headers: dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = _gh_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if payload is not None:
        headers["Content-Type"] = "application/json"

    data = json.dumps(payload).encode() if payload is not None else b"{}"
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode()
            return json.loads(body) if body.strip() else {"status": resp.status}
    except urllib.error.HTTPError as exc:
        if exc.code == 204:
            return {"status": 204, "ok": True}
        body = exc.read().decode(errors="replace")[:500]
        _logger.warning("GitHub POST {} HTTP {}: {}", path, exc.code, body[:200])
        return {"error": True, "status": exc.code, "message": body}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        _logger.warning("GitHub POST {} connection error: {}", path, exc)
        return None


# ── State helpers ───────────────────────────────────────────────

_CONCLUSION_ICONS = {
    "success": "\U0001f7e2",
    "failure": "\U0001f534",
    "cancelled": "\u26aa",
    "skipped": "\u23ed\ufe0f",
    "timed_out": "\u23f0",
    "action_required": "\U0001f7e1",
    "in_progress": "\U0001f535",
    "queued": "\U0001f7e1",
    "neutral": "\u26aa",
}


def _conclusion_icon(conclusion: str | None, status: str | None = None) -> str:
    if conclusion:
        return _CONCLUSION_ICONS.get(conclusion, "\u26ab")
    if status:
        return _CONCLUSION_ICONS.get(status, "\u26ab")
    return "\u26ab"


def _fmt_ts(ts: str | None) -> str:
    if not ts:
        return "\u2014"
    try:
        from datetime import datetime

        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except (ValueError, AttributeError):
        return ts or "\u2014"


# ── Local git fallback ──────────────────────────────────────────


def _git_cmd(*args: str) -> str | None:
    """Run a git command and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=_PROJECT_ROOT,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def _render_local_git_info() -> None:
    """Show local git information as a fallback."""
    st.markdown(
        '<div class="section-header">\U0001f4bb Local Git Info</div>',
        unsafe_allow_html=True,
    )

    branch = _git_cmd("rev-parse", "--abbrev-ref", "HEAD")
    last_commit = _git_cmd("log", "-1", "--format=%h %s (%an, %ar)")
    remote_url = _git_cmd("remote", "get-url", "origin")

    st.metric("Current Branch", branch or "\u2014")
    if remote_url:
        st.code(remote_url, language=None)
    if last_commit:
        st.markdown(f"**Last commit:** `{last_commit}`")

    # Recent commits
    log_output = _git_cmd("log", "--oneline", "-15", "--format=%h|%s|%an|%ar")
    if log_output:
        st.markdown("**Recent commits:**")
        for line in log_output.splitlines():
            parts = line.split("|", 3)
            if len(parts) == 4:
                sha, msg, author, when = parts
                st.markdown(f"- `{sha}` {msg} \u2014 *{author}*, {when}")

    # Branches
    branches_output = _git_cmd("branch", "-a", "--format=%(refname:short)")
    if branches_output:
        branches = [b.strip() for b in branches_output.splitlines() if b.strip()]
        st.markdown(f"**Branches ({len(branches)}):** {', '.join(f'`{b}`' for b in branches[:15])}")


# ── Section renderers ───────────────────────────────────────────


def _render_workflow_runs() -> None:
    """Show recent workflow runs."""
    st.markdown(
        '<div class="section-header">\U0001f3c3 Recent Workflow Runs</div>',
        unsafe_allow_html=True,
    )
    owner, repo = _gh_owner(), _gh_repo()
    data = _gh_get(f"/repos/{owner}/{repo}/actions/runs?per_page=15")
    if data is None:
        return

    runs = data.get("workflow_runs", []) if isinstance(data, dict) else []
    if not runs:
        st.info("No workflow runs found.")
        return

    for run in runs:
        conclusion = run.get("conclusion")
        status = run.get("status")
        icon = _conclusion_icon(conclusion, status)
        name = run.get("name", "\u2014")
        branch = run.get("head_branch", "\u2014")
        sha = run.get("head_sha", "")[:8]
        created = _fmt_ts(run.get("created_at"))
        html_url = run.get("html_url", "")

        with st.expander(
            f"{icon} {name}  \u00b7  {conclusion or status}  \u00b7  `{branch}` \u00b7 {sha}",
            expanded=False,
        ):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Status", conclusion or status or "\u2014")
            c2.metric("Branch", branch)
            c3.metric("Commit", sha)
            c4.metric("Created", created)
            if html_url:
                st.markdown(f"[View on GitHub \u2197]({html_url})")


def _render_workflows() -> None:
    """List all repository workflows."""
    st.markdown(
        '<div class="section-header">\U0001f4cb Workflows</div>',
        unsafe_allow_html=True,
    )
    owner, repo = _gh_owner(), _gh_repo()
    data = _gh_get(f"/repos/{owner}/{repo}/actions/workflows?per_page=20")
    if data is None:
        return

    workflows = data.get("workflows", []) if isinstance(data, dict) else []
    if not workflows:
        st.info("No workflows found.")
        return

    for wf in workflows:
        name = wf.get("name", "\u2014")
        state = wf.get("state", "unknown")
        path = wf.get("path", "\u2014")
        wf_id = wf.get("id", "\u2014")
        html_url = wf.get("html_url", "")
        icon = "\U0001f7e2" if state == "active" else "\u23f8\ufe0f"
        with st.expander(f"{icon} **{name}** \u2014 `{path}` (state: {state})"):
            st.markdown(
                f"**ID:** `{wf_id}` &nbsp;·&nbsp; **State:** {state} "
                f"&nbsp;·&nbsp; **Path:** `{path}`"
            )
            if html_url:
                st.markdown(f"[View on GitHub \u2197]({html_url})")
            # Fetch and display workflow YAML
            owner, repo = _gh_owner(), _gh_repo()
            yaml_content = _gh_get(
                f"/repos/{owner}/{repo}/contents/{path}",
            )
            if yaml_content and isinstance(yaml_content, dict) and yaml_content.get("content"):
                import base64

                try:
                    decoded = base64.b64decode(yaml_content["content"]).decode()
                    st.code(decoded, language="yaml")
                except Exception:
                    pass


@st.fragment
def _render_trigger_section() -> None:
    """Trigger a workflow_dispatch event."""
    st.markdown(
        '<div class="section-header">\U0001f680 Trigger Workflow</div>',
        unsafe_allow_html=True,
    )
    if not _gh_token():
        st.info("Set `GITHUB_TOKEN` to enable workflow triggers.")
        return

    owner, repo = _gh_owner(), _gh_repo()

    # Reuse cached workflows to avoid slow re-fetch on every interaction
    wf_cache_key = "_gh_trigger_workflows"
    workflows = None
    try:
        cache = st.session_state.get(wf_cache_key)
        if isinstance(cache, list):
            workflows = cache
    except Exception:
        pass

    if workflows is None:
        data = _gh_get(f"/repos/{owner}/{repo}/actions/workflows?per_page=20")
        if not data:
            return
        workflows = data.get("workflows", []) if isinstance(data, dict) else []
        with contextlib.suppress(Exception):
            st.session_state[wf_cache_key] = workflows

    dispatchable = [wf for wf in workflows if wf.get("state") == "active"]
    if not dispatchable:
        st.info("No active workflows available to trigger.")
        return

    # GitHub API only returns workflows from the default branch.  Build a
    # name→id map using numeric IDs for known workflows, then supplement with
    # any extra .yml files found on the currently selected branch (using the
    # filename as the dispatch id — GitHub accepts both numeric ids and
    # filenames in the workflow dispatch endpoint).
    wf_names: dict[str, str | int] = {wf["name"]: wf["id"] for wf in dispatchable}

    # Branch dropdown — populated from local git branches (quick & always available)
    @st.cache_data(ttl=60, show_spinner=False)
    def _local_branches() -> list[str]:
        try:
            result = subprocess.run(
                ["git", "branch", "-a", "--format=%(refname:short)"],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=_PROJECT_ROOT,
            )
            if result.returncode == 0:
                branches: list[str] = []
                for b in result.stdout.splitlines():
                    b = b.strip().lstrip("* ")
                    if b and not b.startswith("origin/HEAD"):
                        if b.startswith("origin/"):
                            b = b[len("origin/") :]
                        if b not in branches:
                            branches.append(b)
                branches.sort(key=lambda x: (x != "main", x))
                return branches or ["main"]
        except Exception:
            pass
        return ["main"]

    known_branches = _local_branches()
    branch = st.selectbox(
        "Branch (ref)",
        known_branches,
        index=0,
        key="gh_trigger_branch",
        help="Select the branch to run the workflow on.",
    )

    # Supplement the workflow list with any .yml files present on the selected
    # branch that aren't already in the API-returned list (e.g., deploy-k8s.yml
    # on a feature branch before merging to main).
    @st.cache_data(ttl=120, show_spinner=False)
    def _branch_workflows(ref: str) -> dict[str, str]:
        """Return {workflow_name: filename} for workflows on a given branch."""
        import base64

        result: dict[str, str] = {}
        contents = _gh_get(f"/repos/{owner}/{repo}/contents/.github/workflows?ref={ref}")
        if not isinstance(contents, list):
            return result
        for item in contents:
            if not str(item.get("name", "")).endswith(".yml"):
                continue
            fname = item["name"]
            file_data = _gh_get(
                f"/repos/{owner}/{repo}/contents/.github/workflows/{fname}?ref={ref}"
            )
            if isinstance(file_data, dict) and file_data.get("content"):
                try:
                    raw = base64.b64decode(file_data["content"]).decode()
                    for line in raw.splitlines():
                        stripped = line.strip()
                        if stripped.startswith("name:"):
                            wf_name = stripped.split(":", 1)[1].strip().strip("'\"")
                            result[wf_name] = fname
                            break
                except Exception:
                    pass
        return result

    branch_wfs = _branch_workflows(branch)
    existing_names = set(wf_names.keys())
    for wf_nm, wf_file in branch_wfs.items():
        if wf_nm not in existing_names:
            wf_names[wf_nm] = wf_file  # use filename as dispatch id

    # Single selectbox showing all workflows (API + branch-specific)
    selected_name = st.selectbox(
        "Select workflow",
        list(wf_names.keys()),
        key="gh_trigger_wf2",
    )

    # ── Workflow-specific inputs ─────────────────────────────────────────────
    # Known workflows that require additional inputs on workflow_dispatch.
    # Maps workflow name to a list of (key, label, widget_type, options/default).
    _WORKFLOW_INPUTS: dict[str, list[dict]] = {
        "Deploy": [
            {
                "key": "environment",
                "label": "Target environment",
                "type": "select",
                "options": ["staging", "production"],
                "required": True,
            },
            {
                "key": "image_tag",
                "label": "Image tag (leave blank for latest)",
                "type": "text",
                "default": "",
                "required": False,
            },
        ],
        "Deploy K8s": [
            {
                "key": "overlay",
                "label": "K8s overlay",
                "type": "select",
                "options": ["local", "cloud", "ghcr"],
                "required": False,
            },
        ],
    }

    dispatch_inputs: dict[str, str] = {}
    spec = _WORKFLOW_INPUTS.get(selected_name, [])
    if spec:
        st.markdown("**Workflow inputs**")
        for field in spec:
            fkey = f"gh_input_{field['key']}"
            if field["type"] == "select":
                val = st.selectbox(
                    field["label"],
                    field["options"],
                    key=fkey,
                )
            else:
                val = st.text_input(
                    field["label"],
                    value=field.get("default", ""),
                    key=fkey,
                )
            if val:
                dispatch_inputs[field["key"]] = str(val)

    if st.button("\u25b6\ufe0f Trigger Workflow", key="gh_trigger_btn"):
        # Validate required inputs
        missing = [
            f["label"] for f in spec if f.get("required") and not dispatch_inputs.get(f["key"])
        ]
        if missing:
            st.error(f"\u274c Required input(s) missing: {', '.join(missing)}")
        else:
            wf_id = wf_names[selected_name]
            _logger.info(
                "GitHub workflow dispatch: workflow='{}' id={} branch='{}' inputs={}",
                selected_name,
                wf_id,
                branch,
                dispatch_inputs,
            )
            payload: dict[str, object] = {"ref": branch}
            if dispatch_inputs:
                payload["inputs"] = dispatch_inputs
            result = _gh_post(
                f"/repos/{owner}/{repo}/actions/workflows/{wf_id}/dispatches",
                payload,
            )
            if result is None:
                _logger.warning("GitHub workflow dispatch '{}' failed: no response", selected_name)
                st.error("\u274c Failed to trigger workflow — no response from GitHub API.")
            elif result.get("error"):
                _logger.warning(
                    "GitHub workflow dispatch '{}' HTTP {}: {}",
                    selected_name,
                    result.get("status", "?"),
                    result.get("message", "")[:200],
                )
                st.error(
                    f"\u274c Failed to trigger workflow (HTTP {result.get('status', '?')}).  \n"
                    f"Ensure the workflow YAML has `on: workflow_dispatch` and your "
                    f"`GITHUB_TOKEN` has `actions:write` scope.  \n"
                    f"```\n{result.get('message', '')}\n```"
                )
            else:
                _logger.info(
                    "GitHub workflow dispatch '{}' on branch '{}' queued", selected_name, branch
                )
                st.success(
                    f"\u2705 Triggered `{selected_name}` on branch `{branch}`.  \n"
                    "Checking for run below (may take a few seconds for GitHub to queue it)\u2026"
                )
                _poll_triggered_run(owner, repo, branch, selected_name)

    # Re-display last triggered run result (survives tab switches)
    last = st.session_state.get("_gh_last_trigger")
    if last:
        run_id = last.get("run_id")
        icon = _conclusion_icon(last.get("conclusion"), last.get("status", ""))
        run_label = f"Run #{run_id}" if run_id else "(queued, not yet found)"
        st.markdown(
            f"**Last trigger:** {icon} **{last['workflow']}** — "
            f"{run_label} — {last.get('conclusion') or last.get('status', '?')}"
            + (f"  \n[View on GitHub ↗]({last['html_url']})" if last.get("html_url") else "")
        )

        # Allow manual refresh if the run is not yet completed
        if (last.get("status") != "completed" or not last.get("conclusion")) and st.button(
            "🔄 Check run status", key="gh_refresh_trigger"
        ):
            if run_id is None:
                # Run not found yet — re-search by branch
                data = _gh_get(
                    f"/repos/{owner}/{repo}/actions/runs?branch={last.get('branch', 'main')}&per_page=5",
                    timeout=10,
                )
                if isinstance(data, dict):
                    for run in data.get("workflow_runs", []):
                        if run.get("name") == last["workflow"]:
                            run_id = run["id"]
                            last = {
                                **last,
                                "run_id": run_id,
                                "html_url": run.get("html_url", last.get("html_url", "")),
                                "status": run.get("status", "queued"),
                                "conclusion": run.get("conclusion"),
                            }
                            st.session_state["_gh_last_trigger"] = last
                            break
                if run_id is None:
                    st.info("Run not found yet. GitHub may take a few more seconds to queue it.")
                    return

            fresh = _gh_get(
                f"/repos/{owner}/{repo}/actions/runs/{run_id}",
                timeout=10,
            )
            if isinstance(fresh, dict) and "id" in fresh:
                new_status = fresh.get("status", "unknown")
                new_conclusion = fresh.get("conclusion")
                st.session_state["_gh_last_trigger"] = {
                    **last,
                    "run_id": run_id,
                    "html_url": fresh.get("html_url", last.get("html_url", "")),
                    "status": new_status,
                    "conclusion": new_conclusion,
                }
                new_icon = _conclusion_icon(new_conclusion, new_status)
                st.markdown(
                    f"**Updated:** {new_icon} **{last['workflow']}** — "
                    f"Run #{run_id} — {new_conclusion or new_status}"
                )
                if new_status == "completed":
                    _render_run_jobs(owner, repo, run_id)
                else:
                    st.info("Run is still in progress. Click 🔄 again to refresh.")
            else:
                st.warning("Could not fetch updated status.")


def _poll_triggered_run(owner: str, repo: str, branch: str, workflow_name: str) -> None:
    """Single-probe check for the newly triggered run (non-blocking).

    Waits 4 seconds for GitHub to queue the run, makes ONE API call to find it,
    then returns. Further status checks are done via the manual refresh button.
    This replaces the previous blocking loop (up to 90 s of time.sleep).
    """
    placeholder = st.empty()
    placeholder.info("⏳ Waiting for GitHub to queue the run…")
    time.sleep(4)  # minimum wait — GitHub typically queues within 2–3 s

    data = _gh_get(
        f"/repos/{owner}/{repo}/actions/runs?branch={branch}&per_page=5",
        timeout=10,
    )

    run_found = None
    if isinstance(data, dict):
        for run in data.get("workflow_runs", []):
            if run.get("name") == workflow_name and run.get("status") in (
                "queued",
                "in_progress",
                "completed",
            ):
                run_found = run
                break

    if not run_found:
        # Not found yet — persist pending state so the refresh button appears
        placeholder.info(
            "Run not queued yet — GitHub may take a few more seconds.  \n"
            f"Use the **🔄 Check run status** button below, or "
            f"[view on GitHub ↗](https://github.com/{owner}/{repo}/actions)."
        )
        st.session_state["_gh_last_trigger"] = {
            "run_id": None,
            "html_url": f"https://github.com/{owner}/{repo}/actions",
            "status": "queued",
            "conclusion": None,
            "workflow": workflow_name,
            "branch": branch,
        }
        return

    run_id = run_found["id"]
    html_url = run_found.get("html_url", "")
    status = run_found.get("status", "unknown")
    conclusion = run_found.get("conclusion")
    icon = _conclusion_icon(conclusion, status)

    if status == "completed":
        placeholder.markdown(
            f"{icon} **Run #{run_id}** — **{conclusion or status}**"
            + (f"  \n[View on GitHub ↗]({html_url})" if html_url else "")
        )
    else:
        placeholder.info(
            f"{icon} **Run #{run_id}** — **{status}**  \n"
            "Use **🔄 Check run status** below to check for completion."
            + (f"  \n[View on GitHub ↗]({html_url})" if html_url else "")
        )

    # Persist result so it survives tab navigation
    st.session_state["_gh_last_trigger"] = {
        "run_id": run_id,
        "html_url": html_url,
        "status": status,
        "conclusion": conclusion,
        "workflow": workflow_name,
        "branch": branch,
    }

    if status == "completed":
        _render_run_jobs(owner, repo, run_id)

    # Fetch and display job details / console output
    _render_run_jobs(owner, repo, run_id)


def _render_run_jobs(owner: str, repo: str, run_id: int) -> None:
    """Fetch and display jobs + console output for a workflow run."""
    data = _gh_get(f"/repos/{owner}/{repo}/actions/runs/{run_id}/jobs", timeout=15)
    if data is None or not isinstance(data, dict):
        return

    jobs = data.get("jobs", [])
    if not jobs:
        st.info("No jobs found for this run yet (may still be queued).")
        return

    for job in jobs:
        name = job.get("name", "\u2014")
        status = job.get("status", "unknown")
        conclusion = job.get("conclusion")
        icon = _conclusion_icon(conclusion, status)
        html_url = job.get("html_url", "")

        with st.expander(f"{icon} Job: **{name}** — {conclusion or status}", expanded=True):
            # Show steps
            steps = job.get("steps", [])
            if steps:
                for step in steps:
                    s_name = step.get("name", "\u2014")
                    s_conclusion = step.get("conclusion")
                    s_status = step.get("status")
                    s_icon = _conclusion_icon(s_conclusion, s_status)
                    st.markdown(f"&nbsp;&nbsp;{s_icon} {s_name}")

            # Fetch and display logs for this job
            job_id = job.get("id")
            if job_id and conclusion:
                logs = _gh_get_logs(owner, repo, job_id)
                if logs:
                    st.code(logs, language="text")
                else:
                    if html_url:
                        st.markdown(f"[View full logs on GitHub \u2197]({html_url})")


def _gh_get_logs(owner: str, repo: str, job_id: int) -> str | None:
    """Fetch plain-text logs for a GitHub Actions job.

    Returns the first 8000 characters of log text, or *None* on failure.

    The GitHub /logs endpoint responds with HTTP 302 → presigned S3 URL → text.
    ``urllib.request.urlopen`` follows redirects automatically, so we just read
    the final response.  A ``Content-Encoding: gzip`` response is also handled.
    """
    import gzip as _gzip

    url = f"https://api.github.com/repos/{owner}/{repo}/actions/jobs/{job_id}/logs"
    headers: dict[str, str] = {
        # text/plain preferred; without this header GitHub returns a zip archive
        "Accept": "text/plain",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = _gh_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read(300_000)  # cap at ~300 KB
            # GitHub may return gzip-encoded body even when Accept: text/plain
            encoding = resp.headers.get("Content-Encoding", "")
            if encoding == "gzip" or (raw[:2] == b"\x1f\x8b"):
                with contextlib.suppress(Exception):
                    raw = _gzip.decompress(raw)
            try:
                text = raw.decode("utf-8", errors="replace")
            except Exception:
                text = raw.decode("latin-1", errors="replace")
            # Trim to a reasonable display size
            if len(text) > 8000:
                text = text[:8000] + "\n\n… (truncated — view full log on GitHub)"
            return text
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
        return None


def _render_ghcr_images() -> None:
    """Show container images from GHCR."""
    st.markdown(
        '<div class="section-header">\U0001f433 Container Images (GHCR)</div>',
        unsafe_allow_html=True,
    )
    owner = _gh_owner()
    data = _gh_get(f"/users/{owner}/packages?package_type=container&per_page=10")
    if data is None:
        return

    packages = data if isinstance(data, list) else []
    if not packages:
        st.info("No container images found in GHCR.")
        return

    for pkg in packages:
        name = pkg.get("name", "\u2014")
        visibility = pkg.get("visibility", "\u2014")
        url = pkg.get("html_url", "")
        updated = _fmt_ts(pkg.get("updated_at"))

        with st.expander(f"\U0001f4e6 {name}  \u00b7  {visibility}", expanded=True):
            c1, c2 = st.columns(2)
            c1.metric("Package", name)
            c2.metric("Updated", updated[:10])
            if url:
                st.markdown(f"[View on GitHub \u2197]({url})")

            versions_data = _gh_get(
                f"/users/{owner}/packages/container/{name}/versions?per_page=10"
            )
            if versions_data and isinstance(versions_data, list):
                st.markdown("**Recent Tags:**")
                for v in versions_data[:8]:
                    tags = v.get("metadata", {}).get("container", {}).get("tags", [])
                    tag_str = ", ".join(tags) if tags else "untagged"
                    vid = str(v.get("id", "\u2014"))[:12]
                    created = _fmt_ts(v.get("created_at"))
                    st.markdown(f"- `{tag_str}` \u2014 ID: `{vid}` \u2014 {created}")


def _render_repo_info() -> None:
    """Quick repo overview."""
    owner, repo = _gh_owner(), _gh_repo()
    data = _gh_get(f"/repos/{owner}/{repo}")
    if data is None:
        return

    if isinstance(data, dict):
        c1, c2, c3 = st.columns(3)
        full_name = data.get("full_name", "\u2014")
        branch = data.get("default_branch", "—")
        visibility = data.get("visibility", "—")
        c1.markdown(
            f"<span style='font-size:1.0rem'><strong>Repository</strong><br>"
            f"<code>{full_name}</code></span>",
            unsafe_allow_html=True,
        )
        c2.markdown(
            f"<span style='font-size:1.0rem'><strong>Default Branch</strong><br>"
            f"<code>{branch}</code></span>",
            unsafe_allow_html=True,
        )
        c3.markdown(
            f"<span style='font-size:1.0rem'><strong>Visibility</strong><br>"
            f"<code>{visibility}</code></span>",
            unsafe_allow_html=True,
        )


# ── Main render ─────────────────────────────────────────────────


def render() -> None:
    """Render the GitHub CI/CD Dashboard page."""
    try:
        _render_content()
    except Exception:
        _logger.exception("Unhandled exception in github_dashboard.render()")
        raise


def _render_content() -> None:
    """Inner render — separated so exceptions are caught and logged by render()."""
    st.markdown(get_global_css(), unsafe_allow_html=True)
    st.markdown(
        hero_section(
            "\U0001f419 GitHub CI/CD Dashboard",
            "Workflow status, container images, and manual triggers \u2014 "
            "your DevOps control plane inside Streamlit.",
        ),
        unsafe_allow_html=True,
    )

    # ── Token check ───────────────
    has_token = bool(_gh_token())
    if not has_token:
        st.warning(
            "\U0001f511 **GitHub token not found.**  This repository is private and requires "
            "authentication for API access.  \n\n"
            "**Setup options:**\n"
            "1. Add `GITHUB_TOKEN=ghp_...` to `.env.secrets`\n"
            "2. Install [GitHub CLI](https://cli.github.com/) and run `gh auth login`\n"
            "3. Set the `GITHUB_TOKEN` environment variable\n\n"
            "Showing local git information below as a fallback."
        )
        _render_local_git_info()

        # Footer
        st.markdown("---")
        owner, repo = _gh_owner(), _gh_repo()
        st.markdown(
            f"[Repository \u2197](https://github.com/{owner}/{repo}) &nbsp;\u00b7&nbsp; "
            f"[Actions \u2197](https://github.com/{owner}/{repo}/actions)"
        )
        return

    # ── Repo info ─────────────────
    _render_repo_info()

    # ── Tabs ──────────────────────
    # Use st.radio (keyed) instead of st.tabs() \u2014 st.tabs() resets to tab 0
    # on every full rerun (e.g. selectbox inside Diagrams tab triggers rerun \u2192 jumps to Runs).
    _GH_TABS = [
        "\U0001f3c3 Runs",
        "\U0001f4cb Workflows",
        "\U0001f4ca Diagrams",
        "\U0001f680 Trigger",
        "\U0001f433 GHCR Images",
    ]
    active_gh = st.radio(
        "GitHub tab",
        _GH_TABS,
        horizontal=True,
        key="_gh_tab",
        label_visibility="collapsed",
    )
    st.markdown(
        "<hr style='margin:0 0 1rem 0;border-color:#334155;'>",
        unsafe_allow_html=True,
    )

    if active_gh == _GH_TABS[0]:
        _render_workflow_runs()
    elif active_gh == _GH_TABS[1]:
        _render_workflows()
    elif active_gh == _GH_TABS[2]:
        _render_workflow_diagrams()
    elif active_gh == _GH_TABS[3]:
        _render_trigger_section()
    else:
        _render_ghcr_images()

    # ── Footer ────────────────────
    st.markdown("---")
    owner, repo = _gh_owner(), _gh_repo()
    st.markdown(
        f"[Repository \u2197](https://github.com/{owner}/{repo}) &nbsp;\u00b7&nbsp; "
        f"[Actions \u2197](https://github.com/{owner}/{repo}/actions) &nbsp;\u00b7&nbsp; "
        f"[Packages \u2197](https://github.com/{owner}/{repo}/pkgs/container/{repo})"
    )
