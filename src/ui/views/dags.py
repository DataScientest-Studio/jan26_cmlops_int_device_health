"""DAGs & Pipelines — Airflow DAG and DVC pipeline visualisation."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from src.ui.logging_ui import get_ui_logger
from src.ui.styles import get_global_css, hero_section

_logger = get_ui_logger(__name__)

# ── Mermaid renderer ─────────────────────────────────────────────────────────


def _render_mermaid(diagram: str, height: int = 500) -> None:
    """Render a Mermaid diagram via embedded Mermaid.js with scrollable container."""
    import streamlit.components.v1 as components

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
      mermaid.initialize({{startOnLoad:true,theme:'dark',securityLevel:'loose'}});
    </script>
    </body></html>
    """
    components.html(html_content, height=height, scrolling=False)


# ── Per-DAG Mermaid diagrams ──────────────────────────────────────────────────

_DAG_DIAGRAMS: dict[str, str] = {
    "automated_retraining": r"""
flowchart LR
    A[validate_data] --> B[extract_features]
    B --> C[train_challenger\nfrom_db=True]
    C --> D[record_training_split\nmodel_training_data]
    D --> E[compare_models\nchampion re-eval\non challenger test]
    E --> F{human\napproval?}
    F -->|required| G[wait_for_human\napproval]
    F -->|auto| H[promote_if_better]
    G -->|approved| H2[promote\nunconditional]
    G -->|rejected| I[skip_promotion]
    H --> J[cleanup_training\nsplits]
    H2 --> J
    I --> J
    J --> K[send_notification]
    classDef task fill:#1e293b,stroke:#6366f1,color:#e2e8f0
    classDef dec fill:#2d1b69,stroke:#a78bfa,color:#e2e8f0
    classDef db fill:#1e3a2f,stroke:#10b981,color:#e2e8f0
    class A,B,C,H,H2,I,K task
    class D,J db
    class E,G task
    class F dec
""",
    "drift_triggered_retraining": r"""
flowchart LR
    A{check_drift\nconditions} -->|drift+labels OK| B[record_trigger]
    A -->|conditions not met| Z([short-circuit\nskip])
    B --> C[trigger_retraining\nautomated_retraining DAG]
    C --> D[notify_team]
    classDef task fill:#1e293b,stroke:#6366f1,color:#e2e8f0
    classDef dec fill:#2d1b69,stroke:#a78bfa,color:#e2e8f0
    classDef skip fill:#374151,stroke:#6b7280,color:#9ca3af
    class B,C,D task
    class A dec
    class Z skip
""",
    "evidently_drift_detection": r"""
flowchart LR
    A{check_prerequisites} -->|ok| B[load_reference_data]
    A -->|skip| Z([short-circuit\nskip])
    A --> C[load_current_data]
    B --> D[run_drift_detection]
    C --> D
    D --> E{check_should\ntrigger_retraining}
    E -->|drift threshold\nbreached| F[trigger_automated\nretraining DAG]
    E -->|no drift| G([short-circuit\nskip])
    classDef task fill:#1e293b,stroke:#6366f1,color:#e2e8f0
    classDef dec fill:#2d1b69,stroke:#a78bfa,color:#e2e8f0
    classDef skip fill:#374151,stroke:#6b7280,color:#9ca3af
    class B,C,D,F task
    class A,E dec
    class G,Z skip
""",
    "model_promotion": r"""
flowchart LR
    A[get_current_info] --> B{branch_action}
    B -->|promote| C[promote_model]
    B -->|rollback| D[rollback_model]
    B -->|archive| E[archive_models]
    C --> F[validate_production]
    D --> F
    E --> F
    F --> G[send_notification]
    classDef task fill:#1e293b,stroke:#6366f1,color:#e2e8f0
    classDef dec fill:#2d1b69,stroke:#a78bfa,color:#e2e8f0
    class A,C,D,E,F,G task
    class B dec
""",
    "sync_production_data": r"""
flowchart LR
    A[export_predictions_db] --> D[dvc_add_exports]
    B[export_features] --> D
    C[export_raw_signals] --> D
    D --> E[dvc_push_to_dagshub]
    E --> F[log_sync_metadata]
    classDef task fill:#1e3a5f,stroke:#60a5fa,color:#e2e8f0
    class A,B,C,D,E,F task
""",
    "database_backup": r"""
flowchart LR
    A[backup_database] --> B[cleanup_old_backups]
    B --> C[log_backup_result]
    classDef task fill:#1e3a2f,stroke:#10b981,color:#e2e8f0
    class A,B,C task
""",
    "sync_mlflow_to_dagshub": r"""
flowchart LR
    A[check_prerequisites] --> B[push_to_dagshub]
    B --> C[log_result]
    classDef task fill:#1e293b,stroke:#a78bfa,color:#e2e8f0
    class A,B,C task
""",
    "batch_rescoring": r"""
flowchart LR
    A[load_champion] --> B[fetch_predictions]
    B --> C[rescore]
    C --> D[write_audit]
    classDef task fill:#1e293b,stroke:#f59e0b,color:#e2e8f0
    class A,B,C,D task
""",
}

# ── Airflow DAG catalogue ─────────────────────────────────────────────────────

AIRFLOW_DAGS = [
    {
        "name": "automated_retraining",
        "schedule": "Weekly (Sunday 02:00 UTC)",
        "icon": "\U0001f504",
        "description": (
            "Full automated retraining pipeline. Reads labeled signals directly from "
            "PostgreSQL (no DVC pull required — the database is the authoritative source). "
            "Extracts features, trains a challenger model with a 90-day sliding window, "
            "logs metrics to MLflow Buffer, compares with the current champion, and "
            "auto-promotes if the challenger improves by the configured threshold. "
            "Triggered weekly and can also be triggered by the `drift_triggered_retraining` DAG."
        ),
        "tasks": [
            "validate_data",
            "extract_features",
            "train_challenger",
            "compare_models",
            "promote_if_better",
            "send_notification",
        ],
    },
    {
        "name": "drift_triggered_retraining",
        "schedule": "Hourly (or manual via Streamlit drift provocation)",
        "icon": "\U0001f4c9",
        "description": (
            "Sole trigger for model retraining on drift. Runs hourly to check the `drift_batches` "
            "table; also triggerable manually from the Streamlit Drift Provocation tab. "
            "If recent drift batches exceed the threshold AND sufficient labeled signals exist, "
            "triggers `automated_retraining` with the `require_human_approval` flag forwarded "
            "from the caller."
        ),
        "tasks": [
            "check_drift",
            "record_trigger",
            "trigger_retraining",
            "notify_team",
        ],
    },
    {
        "name": "evidently_drift_detection",
        "schedule": "Daily (06:00 UTC)",
        "icon": "\U0001f50d",
        "description": (
            "Runs EvidentlyAI data drift and model performance reports daily. "
            "Compares current feature distributions against a reference baseline. "
            "Publishes drift scores to Prometheus and writes rows to `drift_batches`. "
            "Metrics-only — retraining is handled exclusively by `drift_triggered_retraining`."
        ),
        "tasks": [
            "check_prerequisites",
            "load_reference_data",
            "load_current_data",
            "run_drift_detection",
        ],
    },
    {
        "name": "model_promotion",
        "schedule": "Manual trigger (workflow_dispatch)",
        "icon": "\U0001f3c6",
        "description": (
            "Champion/Challenger evaluation pipeline. Loads both the current Production "
            "champion and a Staging challenger from MLflow, evaluates both on a held-out "
            "test set, and promotes the challenger if it wins by the configured margin. "
            "Old champion is archived. Run manually after a new Staging candidate is registered."
        ),
        "tasks": [
            "get_current_info",
            "branch_action",
            "promote_model",
            "rollback_model",
            "archive_models",
            "validate_production",
            "send_notification",
        ],
    },
    {
        "name": "sync_production_data",
        "schedule": "Daily (04:00 UTC)",
        "icon": "\u2601\ufe0f",
        "description": (
            "Exports production predictions, extracted features, and raw signals from "
            "PostgreSQL to DVC-tracked files. Runs `dvc add` + `dvc push` to DagsHub. "
            "The 10% sample rate keeps the exported volume manageable. "
            "Cloud mode only \u2014 requires DagsHub credentials."
        ),
        "tasks": [
            "export_predictions_db",
            "export_features",
            "export_raw_signals",
            "dvc_add_exports",
            "dvc_push_to_dagshub",
            "log_sync_metadata",
        ],
    },
    {
        "name": "database_backup",
        "schedule": "Daily (02:00 UTC) + manual trigger",
        "icon": "\U0001f5c4\ufe0f",
        "description": (
            "Scheduled PostgreSQL backup using `pg_dump` (custom binary format). "
            "Writes timestamped `.dump` files to `data/backups/`. Rotates old backups "
            "keeping the 7 newest. PGPASSWORD passed via env var \u2014 never hardcoded. "
            "Retries up to 2 times with 5-minute delay on failure."
        ),
        "tasks": ["backup_database", "cleanup_old_backups", "log_backup_result"],
    },
    {
        "name": "sync_mlflow_to_dagshub",
        "schedule": "Manual trigger (workflow_dispatch)",
        "icon": "\u2b06\ufe0f",
        "description": (
            "Pushes MLflow data (experiments, runs, metrics, params, tags, artifacts) "
            "from the local MLflow Buffer to DagsHub MLflow incrementally. "
            "A state file tracks the last push timestamp so only new/updated runs are "
            "transferred. Requires `DAGSHUB_USER`, `DAGSHUB_TOKEN`, `DAGSHUB_REPO` env vars."
        ),
        "tasks": ["check_prerequisites", "push_to_dagshub", "log_result"],
    },
    {
        "name": "batch_rescoring",
        "schedule": "Manual trigger (workflow_dispatch)",
        "icon": "\u23ee\ufe0f",
        "description": (
            "Re-scores historical predictions using the current champion model. "
            "Fetches predictions from PostgreSQL (or SQLite fallback) for a configurable "
            "lookback window, runs them through the champion, and writes updated labels and "
            "confidence scores back to the database. Supports dry-run mode for preview. "
            "An audit record is written to the `rescoring_runs` table for traceability. "
            "Triggered manually from the Batch Re-Scoring use case in Streamlit or the Airflow UI."
        ),
        "tasks": [
            "load_champion",
            "fetch_predictions",
            "rescore",
            "write_audit",
        ],
    },
]

# ── DVC pipeline data ─────────────────────────────────────────────────────────

DVC_STAGES = [
    {
        "name": "generate_data",
        "cmd": (
            "python scripts/generate_data.py generate\n"
            "  --n-samples ${generate_data.n_samples}\n"
            "  --gaussian-fraction ${generate_data.gaussian_fraction}\n"
            "  --output-dir data/raw"
        ),
        "deps": [
            "scripts/generate_data.py",
            "src/signal_processing/signal_generator.py",
            "src/signal_processing/signal_models.py",
        ],
        "outputs": [
            "data/raw/dataset_baseline_full.json",
            "data/raw/dataset_baseline_train.json",
            "data/raw/dataset_baseline_test.json",
        ],
        "params": ["generate_data.n_samples", "generate_data.gaussian_fraction"],
    },
    {
        "name": "extract_features",
        "cmd": (
            "python -m src.signal_processing.preprocess\n"
            "  --train-data data/raw/dataset_baseline_train.json\n"
            "  --test-data  data/raw/dataset_baseline_test.json\n"
            "  --output-dir data/processed"
        ),
        "deps": [
            "data/raw/dataset_baseline_train.json",
            "data/raw/dataset_baseline_test.json",
            "src/signal_processing/preprocess.py",
            "src/signal_processing/feature_extractor.py",
        ],
        "outputs": [
            "data/processed/features_train.csv",
            "data/processed/labels_train.csv",
            "data/processed/features_test.csv",
            "data/processed/labels_test.csv",
        ],
        "params": ["preprocess.window_length", "preprocess.polyorder"],
    },
    {
        "name": "train",
        "cmd": (
            "python -m src.training.train\n"
            "  --train-data data/raw/dataset_baseline_train.json\n"
            "  --test-data  data/raw/dataset_baseline_test.json\n"
            "  --model-output models/champion_model.pkl"
        ),
        "deps": [
            "data/raw/dataset_baseline_train.json",
            "data/raw/dataset_baseline_test.json",
            "src/training/train.py",
        ],
        "outputs": ["models/champion_model.pkl"],
        "params": ["train.max_iter", "train.test_size"],
    },
    {
        "name": "evaluate",
        "cmd": (
            "python -m src.training.evaluate\n"
            "  --model models/champion_model.pkl\n"
            "  --features-test data/processed/features_test.csv\n"
            "  --labels-test   data/processed/labels_test.csv\n"
            "  --output metrics/eval_metrics.json"
        ),
        "deps": [
            "models/champion_model.pkl",
            "data/processed/features_test.csv",
            "data/processed/labels_test.csv",
            "src/training/evaluate.py",
        ],
        "metrics": ["metrics/eval_metrics.json"],
    },
    {
        "name": "sync_production_data  [optional]",
        "cmd": (
            "python scripts/sync_production_data.py\n"
            "  --sample-rate 0.1\n"
            "  --dvc-add\n"
            "  # Reads DATABASE_URL (production PostgreSQL)"
        ),
        "deps": ["scripts/sync_production_data.py"],
        "outputs": [
            "data/sync",
            "data/raw_signals",
        ],
        "note": (
            "Optional stage. NOT run via `dvc repro`. "
            "Runs as the `sync_production_data` Airflow DAG on a daily schedule (cloud mode only)."
        ),
    },
]

DVC_MERMAID = r"""
flowchart LR
    P1["params.yaml\ngenerate_data.*"] --> A
    A["generate_data\nscripts/generate_data.py"] --> J1["data/raw/\ndataset_baseline_*.json"]

    J1 --> B["extract_features\npreprocess.py"]
    P2["params.yaml\npreprocess.*"] --> B
    B --> K1["data/processed/\nfeatures_*.csv\nlabels_*.csv"]

    J1 --> C["train\ntrain.py"]
    P3["params.yaml\ntrain.*"] --> C
    C --> L1["models/\nchampion_model.pkl"]

    K1 --> D["evaluate\nevaluate.py"]
    L1 --> D
    D --> M1["metrics/\neval_metrics.json"]

    L1 -.-> E["sync_production_data\n(optional, cloud only)"]

    classDef stage fill:#1e293b,stroke:#10b981,color:#e2e8f0
    classDef artefact fill:#0f172a,stroke:#334155,color:#94a3b8
    classDef param fill:#1e3a5f,stroke:#60a5fa,color:#94a3b8
    classDef optional fill:#1e293b,stroke:#334155,color:#64748b
    class A,B,C,D stage
    class J1,K1,L1,M1 artefact
    class P1,P2,P3 param
    class E optional
"""


# ── Page render ───────────────────────────────────────────────────────────────


def render() -> None:
    """Render the DAGs & Pipelines page."""
    try:
        _render_content()
    except Exception:
        _logger.exception("Unhandled exception in dags.render()")
        raise


def _render_content() -> None:
    """Inner render — separated so exceptions are caught and logged by render()."""
    st.markdown(get_global_css(), unsafe_allow_html=True)
    st.markdown(
        hero_section(
            "DAGs & Pipelines",
            "Airflow orchestration DAGs and DVC reproducible data/ML pipelines.",
        ),
        unsafe_allow_html=True,
    )

    # Use st.radio (keyed) instead of st.tabs() to prevent tab-jump on rerun.
    _DAGS_TABS = ["\U0001f504 Airflow DAGs", "\U0001f4e6 DVC Pipeline"]
    active_dags = st.radio(
        "DAGs page tab",
        _DAGS_TABS,
        horizontal=True,
        key="_dags_tab",
        label_visibility="collapsed",
    )
    st.markdown(
        "<hr style='margin:0 0 1rem 0;border-color:#334155;'>",
        unsafe_allow_html=True,
    )

    if active_dags == _DAGS_TABS[0]:
        _airflow_tab()
    else:
        _dvc_tab()


# ── Airflow tab ───────────────────────────────────────────────────────────────


def _airflow_tab() -> None:
    """Render the Airflow DAGs tab with per-DAG radio navigation."""
    st.markdown(
        '<div class="section-header">\U0001f504 Airflow DAG Overview</div>'
        '<div class="section-subheader">7 orchestration DAGs managing the ML lifecycle (cloud mode)</div>',
        unsafe_allow_html=True,
    )

    dag_labels = [f"{d['icon']}  {d['name']}" for d in AIRFLOW_DAGS]
    sel_label = st.radio(
        "Select DAG to inspect",
        dag_labels,
        key="_airflow_dag_sel",
        label_visibility="visible",
    )
    sel_idx = dag_labels.index(sel_label)
    dag = AIRFLOW_DAGS[sel_idx]

    st.markdown("<hr style='margin:0.5rem 0 1rem 0;border-color:#334155;'>", unsafe_allow_html=True)
    st.markdown(f"### {dag['icon']} `{dag['name']}`")
    st.markdown(f"**Schedule:** {dag['schedule']}")
    st.markdown(dag["description"])

    if dag["name"] in _DAG_DIAGRAMS:
        _render_mermaid(_DAG_DIAGRAMS[str(dag["name"])], height=280)

    st.markdown("**Task sequence:**")
    st.markdown(" \u2192 ".join(f"`{t}`" for t in dag["tasks"]))

    st.markdown("---")
    with st.expander("\U0001f4cb All DAGs — Quick Reference", expanded=False):
        cols = st.columns(3)
        for i, d in enumerate(AIRFLOW_DAGS):
            with cols[i % 3]:
                st.markdown(
                    f'<div class="info-card" style="min-height:80px">'
                    f"<h3>{d['icon']} {d['name']}</h3>"
                    f"<p style='font-size:0.8rem'>\u23f0 {d['schedule']}</p></div>",
                    unsafe_allow_html=True,
                )

    from src.ui.components.docker_utils import get_service_url

    _airflow_url = get_service_url("mlops_airflow", 8081)
    st.markdown(
        '<div class="info-card">'
        "<h3>\U0001f517 Open Airflow UI</h3>"
        f'<p><a href="{_airflow_url}" target="_blank" '
        f'style="color:#818cf8;font-weight:600">{_airflow_url}</a> \u2014 '
        "View DAG runs, task logs, and trigger manual executions.</p></div>",
        unsafe_allow_html=True,
    )


# ── DVC tab ───────────────────────────────────────────────────────────────────


def _dvc_tab() -> None:
    """Render the DVC Pipeline tab."""
    st.markdown(
        '<div class="section-header">\U0001f4e6 DVC Reproducible Pipeline</div>'
        '<div class="section-subheader">4 tracked stages from data generation through evaluation</div>',
        unsafe_allow_html=True,
    )

    _DVC_VIEWS = [
        "\U0001f5fa\ufe0f Pipeline Diagram",
        "\U0001f4cb Stage Details",
        "\U0001f914 What is dvc repro?",
    ]
    sel = st.radio(
        "DVC view",
        _DVC_VIEWS,
        horizontal=True,
        key="_dvc_view_sel",
        label_visibility="collapsed",
    )
    st.markdown("<hr style='margin:0 0 1rem 0;border-color:#334155;'>", unsafe_allow_html=True)

    if sel == _DVC_VIEWS[0]:
        st.markdown(
            "The DVC pipeline tracks 4 core reproducible stages. "
            "`sync_production_data` is defined in `dvc.yaml` but is **not** run via `dvc repro` "
            "\u2014 it runs as an Airflow DAG in cloud mode."
        )
        _render_mermaid(DVC_MERMAID, height=400)

        st.markdown("---")
        st.info(
            "**Where is `dvc repro` used?** \u2014 `dvc repro` is a **development utility** "
            "for local reproducibility. It is **NOT** called in production, "
            "**NOT** by Airflow, **NOT** by Streamlit.  \n\n"
            "In production the pipeline is driven by FastAPI endpoints and Airflow DAGs.  \n\n"
            "\u2139\ufe0f See the **DAGs & Pipelines \u2192 DVC Pipeline \u2192 What is dvc repro?** "
            "tab for a full explanation."
        )
        st.code(
            "# dvc.yaml \u2014 4 reproducible stages (local development only)\n"
            "stages:\n"
            "  generate_data       \u2192 data/raw/dataset_baseline_*.json\n"
            "  extract_features    \u2192 data/processed/features_*.csv + labels_*.csv\n"
            "  train               \u2192 models/champion_model.pkl + metrics/train_metrics.json\n"
            "  evaluate            \u2192 metrics/eval_metrics.json\n"
            "\n"
            "# Reproduce all 4 stages locally (no Docker required):\n"
            "$ dvc repro\n"
            "\n"
            "# sync_production_data is NOT a dvc repro stage:\n"
            "# It runs via the Airflow DAG on a daily schedule (cloud mode only).",
            language="yaml",
        )

    elif sel == _DVC_VIEWS[1]:
        _render_stage_details()

    else:
        _render_dvc_explanation()


def _render_stage_details() -> None:
    st.markdown("#### Stage Details")
    for stage in DVC_STAGES:
        is_optional = "note" in stage
        icon = "\u2600\ufe0f" if is_optional else "\U0001f4cc"
        with st.expander(f"{icon} {stage['name']}", expanded=not is_optional):
            if is_optional:
                st.warning(stage["note"])
            st.code(stage["cmd"], language="bash")
            if "deps" in stage:
                st.markdown("**Dependencies:** " + ", ".join(f"`{d}`" for d in stage["deps"]))
            if "outputs" in stage:
                st.markdown("**Outputs:** " + ", ".join(f"`{o}`" for o in stage["outputs"]))
            if "metrics" in stage:
                st.markdown("**Metrics:** " + ", ".join(f"`{m}`" for m in stage["metrics"]))
            if "params" in stage:
                st.markdown("**Parameters:** " + ", ".join(f"`{p}`" for p in stage["params"]))

    st.markdown("---")
    st.markdown("#### DVC Commands")
    st.code(
        "# Reproduce stages 1-4 (generates data, extracts features, trains, evaluates)\n"
        "dvc repro\n\n"
        "# Run a specific stage only\n"
        "dvc repro train\n\n"
        "# Show pipeline DAG\n"
        "dvc dag\n\n"
        "# Show pipeline status (which stages are stale?)\n"
        "dvc status\n\n"
        "# Push DVC-tracked artefacts to DagsHub remote\n"
        "dvc push\n\n"
        "# Pull DVC-tracked artefacts from DagsHub remote\n"
        "dvc pull",
        language="bash",
    )


def _render_dvc_explanation() -> None:
    st.markdown("## \U0001f914 What is `dvc repro` and why do we have this pipeline?")
    st.markdown(
        "> **TL;DR:** `dvc repro` is a *developer utility* for local experimentation and "
        "reproducibility checks. It is **NOT** used in production, **NOT** called by Airflow, "
        "**NOT** called by Streamlit. The Streamlit/Airflow workflows serve a completely "
        "different purpose."
    )

    st.markdown("---")
    st.markdown("### Two separate workflows for training")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            '<div class="info-card">'
            "<h3>\U0001f4e6 DVC Pipeline + dvc repro</h3>"
            "<p><b>Purpose:</b> Developer tool for local reproducibility</p>"
            "<ul>"
            "<li>Runs entirely on your local machine (no Docker needed)</li>"
            "<li>Tracks which inputs produced which outputs</li>"
            "<li>Caches expensive stages (skips unchanged stages)</li>"
            "<li>Lets anyone check out the repo and get the same results</li>"
            "<li>Good for: verifying baseline, onboarding, parameter exploration</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="info-card">'
            "<h3>\U0001f504 Airflow DAGs + Streamlit</h3>"
            "<p><b>Purpose:</b> Production orchestration with real data</p>"
            "<ul>"
            "<li>Trains on production PostgreSQL data (real device signals)</li>"
            "<li>Scheduled or triggered (drift, manual)</li>"
            "<li>Logs every run to MLflow (experiments, metrics, artifacts)</li>"
            "<li>Champion/Challenger promotion via model_promotion DAG</li>"
            "<li>Good for: ongoing production retraining, monitoring, deployment</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### What does `dvc repro` actually do?")
    st.markdown(
        """
`dvc repro` reads `dvc.yaml` and executes each stage **only if its inputs have changed**
(similar to GNU Make, but for data/ML pipelines). It tracks:

- **Stage command** (`cmd:`) — the script to run
- **Dependencies** (`deps:`) — files/scripts whose changes make the stage stale
- **Outputs** (`outs:`) — files that DVC caches after the stage runs
- **Metrics** (`metrics:`) — JSON files with model performance numbers
- **Parameters** (`params:`) — values from `params.yaml` that influence the run

When you run `dvc repro`:
1. DVC checks if any dependency or parameter has changed since the last run
2. If yes → re-runs that stage (and all downstream stages that depend on it)
3. If no → skips the stage entirely (uses the cached output)
4. After all stages finish, DVC updates `dvc.lock` with new checksums
        """
    )

    st.markdown("---")
    st.markdown("### What does *reproduce* mean here?")
    st.markdown(
        """
The word **"reproduce"** in `dvc repro` means:
> *Reproduce the exact pipeline run that created the tracked artifacts.*

If you commit `dvc.lock` to git alongside the code, anyone who checks out
that commit and runs `dvc pull && dvc repro` will get **exactly the same model,
same metrics, same outputs** — because they're using the same scripts, same parameters,
and the same input data checksums.

This is the core value proposition of DVC: **experiment reproducibility**.

In ML, this matters because:
- The same code + different random seed → different model
- DVC pins the exact parameters and data checksums in `dvc.lock`
- `git log` tells you *what code changed*; `dvc diff` tells you *what data/metrics changed*
        """
    )

    st.markdown("---")
    st.markdown("### Why keep this pipeline when Streamlit + Airflow are more powerful?")
    st.markdown(
        """
| Scenario | DVC pipeline | Airflow / Streamlit |
|:---------|:-------------|:--------------------|
| Local development (no Docker) | ✅ Works | ❌ Needs full stack |
| Quick parameter experiment | ✅ Change params.yaml → dvc repro | 🔶 Need UI |
| Verify new developer gets same baseline | ✅ dvc repro + dvc pull | ❌ Not designed |
| Scheduled production retraining | ❌ Not its job | ✅ Airflow DAG |
| Train on real PostgreSQL data | ❌ Synthetic baseline only | ✅ Both |
| Log runs to MLflow | 🔶 Possible with extra code | ✅ Built in |
        """
    )
    st.markdown(
        """
**We keep the DVC pipeline for:**
1. **Onboarding** — new developers verify their environment works with `dvc repro`
2. **Data versioning** — `dvc push/pull` to DagsHub is how we version training data
3. **Reproducibility audit** — `dvc.lock` + `git log` lets us trace any model back to exact data

**We do NOT use `dvc repro` for:**
- Any production training (that's Airflow)
- Any interactive training (that's Streamlit Use Cases)
- The CI quality gate (uses `data/ci/quality_gate_signals.csv` directly, not DVC)
        """
    )
    st.success(
        "\U0001f4a1 **Practical takeaway:** For day-to-day work, use the Streamlit **Use Cases** "
        "page or the **Airflow DAGs** to retrain models. Use `dvc repro` only when you want to "
        "verify the full pipeline works from scratch on your local machine."
    )
