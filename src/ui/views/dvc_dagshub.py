"""DVC & DagsHub — Data versioning and remote backup page (static, presentation-oriented)."""

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

# ── Mermaid renderer (same pattern as architecture.py / dags.py) ─────────────


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


# ── Diagram definitions ───────────────────────────────────────────────────────

# The canonical "local-first MLflow with DagsHub as backup" diagram
# from presentation_guide.md Section 6.2.
LOCAL_FIRST_DIAGRAM = r"""
graph LR
  subgraph primary["Primary (zero rate limits)"]
    MLflowBuf[mlflow_buffer container\nPostgreSQL backend\nAll live operations]
  end
  subgraph backup["Backup (scheduled sync only)"]
    DagsHub[DagsHub\nRemote ledger]
  end
  API[FastAPI] -->|log runs| MLflowBuf
  Airflow[Airflow DAGs] -->|log runs| MLflowBuf
  Streamlit[Streamlit UI] -->|query| MLflowBuf
  MLflowBuf -.->|sync_mlflow_to_dagshub DAG\nscheduled upload| DagsHub
  DagsHub -.->|dvc pull\nmanual restore| MLflowBuf

  classDef callers fill:#1e293b,stroke:#6366f1,color:#e2e8f0
  classDef primary fill:#064e3b,stroke:#10b981,color:#e2e8f0
  classDef backup fill:#1e1b4b,stroke:#818cf8,color:#e2e8f0

  class API,Airflow,Streamlit callers
  class MLflowBuf primary
  class DagsHub backup
"""

DVC_PIPELINE_DIAGRAM = r"""
flowchart LR
  subgraph DB["PostgreSQL — Application DB"]
    PG[predictions\nraw_signals\nfeatures\nsparse_labels]
  end
  subgraph EXPORT["Airflow DAG: sync_production_data"]
    EXP[Export to\nCSV / JSON]
    DVC_ADD[dvc add\nSHA-256 hash\npointer file]
    GIT[git commit\n.dvc pointer]
    DVC_PUSH[dvc push\nto DagsHub S3 remote]
  end
  subgraph REMOTE["DagsHub — DVC Remote"]
    REMOTE_STORE[S3-compatible\nversioned data store]
  end
  subgraph RESTORE["Reproduce on any machine"]
    GIT_PULL[git clone\n+ git pull]
    DVC_PULL[dvc pull\nfetches exact dataset]
    TRAIN[Training run\nidentical results]
  end

  PG --> EXP
  EXP --> DVC_ADD
  DVC_ADD --> GIT
  GIT --> DVC_PUSH
  DVC_PUSH --> REMOTE_STORE
  REMOTE_STORE --> GIT_PULL
  GIT_PULL --> DVC_PULL
  DVC_PULL --> TRAIN

  classDef db fill:#1e3a5f,stroke:#60a5fa,color:#e2e8f0
  classDef pipeline fill:#1e293b,stroke:#6366f1,color:#e2e8f0
  classDef remote fill:#1e1b4b,stroke:#818cf8,color:#e2e8f0
  classDef restore fill:#064e3b,stroke:#10b981,color:#e2e8f0

  class PG db
  class EXP,DVC_ADD,GIT,DVC_PUSH pipeline
  class REMOTE_STORE remote
  class GIT_PULL,DVC_PULL,TRAIN restore
"""

MODES_DIAGRAM = r"""
graph LR
  subgraph local["Local mode  sandbox  isolated"]
    L1[MLflow\nlocalhost:5001\nSQLite backend]
    L2[DVC cache\nlocal filesystem only\nno remote]
    L3[No DagsHub\nconnection]
  end
  subgraph cloud["Cloud / K8s mode  connected  with backup"]
    C1[mlflow_buffer container\nPostgreSQL backend\nPRIMARY]
    C2[DagsHub MLflow mirror\nBACKUP via scheduled DAG]
    C3[DagsHub DVC remote\nS3-compatible BACKUP]
  end

  classDef local fill:#064e3b,stroke:#10b981,color:#e2e8f0
  classDef cloud fill:#1e1b4b,stroke:#818cf8,color:#e2e8f0

  class L1,L2,L3 local
  class C1,C2,C3 cloud
"""

REPRODUCIBILITY_DIAGRAM = r"""
flowchart LR
  subgraph MLFLOW["MLflow run record"]
    TAG[Tag: dvc_data_hash\ne.g. a3f9c1b7]
  end
  subgraph DVC["DVC pointer file"]
    PTR[training_data.dvc\nmd5: a3f9c1b7]
  end
  subgraph DAGSHUB["DagsHub DVC Remote"]
    BLOB[Exact training dataset\nbytes stored by hash]
  end
  subgraph REPRODUCE["Reproduce anywhere"]
    CMD1[git checkout commit-sha]
    CMD2[dvc pull]
    CMD3[python train.py\nidentical model]
  end

  TAG --> PTR
  PTR --> BLOB
  BLOB --> CMD1
  CMD1 --> CMD2
  CMD2 --> CMD3

  classDef mlflow fill:#713f12,stroke:#f59e0b,color:#e2e8f0
  classDef dvc fill:#1e293b,stroke:#6366f1,color:#e2e8f0
  classDef dagshub fill:#1e1b4b,stroke:#818cf8,color:#e2e8f0
  classDef reproduce fill:#064e3b,stroke:#10b981,color:#e2e8f0

  class TAG mlflow
  class PTR dvc
  class BLOB dagshub
  class CMD1,CMD2,CMD3 reproduce
"""


# ── Render helpers ────────────────────────────────────────────────────────────


def _kv_metric(label: str, value: str, color: str = "#6366f1") -> str:
    return f"""
    <div style="background:#1e293b;border-radius:10px;padding:1rem 1.2rem;
                border-left:4px solid {color};margin-bottom:0.5rem">
      <div style="font-size:0.72rem;color:#94a3b8;text-transform:uppercase;
                  letter-spacing:0.08em;margin-bottom:0.25rem">{label}</div>
      <div style="font-size:1rem;font-weight:600;color:#e2e8f0">{value}</div>
    </div>
    """


def _info_card(title: str, body: str, icon: str = "💡", color: str = "#06b6d4") -> str:
    return f"""
    <div style="background:#1e293b;border-radius:12px;padding:1.2rem 1.4rem;
                border:1px solid #334155;margin-bottom:1rem">
      <div style="font-size:1rem;font-weight:700;color:{color};margin-bottom:0.6rem">
        {icon}&nbsp; {title}
      </div>
      <div style="font-size:0.88rem;color:#cbd5e1;line-height:1.6">{body}</div>
    </div>
    """


def _section_header(text: str) -> None:
    st.markdown(
        f'<div class="section-header">{text}</div>',
        unsafe_allow_html=True,
    )


# ── Tab renderers ─────────────────────────────────────────────────────────────


def _tab_overview() -> None:
    """What are DVC and DagsHub — general introduction."""
    _section_header("📦 DVC & DagsHub — Overview")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            """
            <div style="background:linear-gradient(135deg,#1e293b,#0f1f35);
                        border-radius:14px;padding:1.6rem;border:1px solid #3b82f6;
                        height:100%">
              <div style="font-size:1.2rem;font-weight:700;color:#60a5fa;margin-bottom:0.8rem">
                📦 DVC — Data Version Control
              </div>
              <div style="color:#cbd5e1;font-size:0.9rem;line-height:1.7">
                DVC is an open-source tool that brings <strong style="color:#93c5fd">Git-like version
                control to datasets and ML artifacts</strong>. Because raw data files are too large
                to store in Git, DVC stores only a small <em>pointer file</em>
                (<code>.dvc</code>) in Git while the actual binary content is stored in a
                remote storage backend (S3, GCS, SSH, or DagsHub's S3-compatible store).
                <br><br>
                <strong style="color:#93c5fd">Key guarantee:</strong> every training run is tagged
                with a <code>dvc_data_hash</code> in MLflow. Given that hash, anyone can run
                <code>dvc pull</code> and retrieve the exact bytes used for that training — on any
                machine, at any point in the future.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div style="background:linear-gradient(135deg,#1e293b,#1a1040);
                        border-radius:14px;padding:1.6rem;border:1px solid #818cf8;
                        height:100%">
              <div style="font-size:1.2rem;font-weight:700;color:#a78bfa;margin-bottom:0.8rem">
                🌐 DagsHub — ML Collaboration Platform
              </div>
              <div style="color:#cbd5e1;font-size:0.9rem;line-height:1.7">
                DagsHub is a hosting platform purpose-built for ML projects. It provides:
                <ul style="margin:0.5rem 0 0 1rem;padding:0">
                  <li>A <strong style="color:#c4b5fd">hosted MLflow tracking server</strong>
                      accessible via a standard MLflow URI</li>
                  <li>An <strong style="color:#c4b5fd">S3-compatible DVC remote</strong> for
                      storing versioned datasets and model artifacts</li>
                  <li>A <strong style="color:#c4b5fd">Git repository mirror</strong> with
                      experiment annotations</li>
                </ul>
                <br>
                In this project DagsHub acts as the <em>durable off-site ledger</em> — a
                full backup of all training runs and all versioned datasets, written to only
                by scheduled Airflow DAGs and <strong>never on the critical path</strong> of
                real-time predictions.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Why DVC + DagsHub together
    _section_header("🔗 Why DVC + DagsHub Together?")
    st.markdown(
        """
        <div style="background:#1e293b;border-radius:12px;padding:1.4rem 1.6rem;
                    border:1px solid #334155;margin-bottom:1rem">
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1.2rem">
            <div style="text-align:center">
              <div style="font-size:2rem;margin-bottom:0.4rem">🔄</div>
              <div style="font-weight:700;color:#10b981;margin-bottom:0.3rem">Reproducibility</div>
              <div style="font-size:0.83rem;color:#94a3b8;line-height:1.5">
                Same code + same DVC hash → same model. No "it worked on my machine" surprises.
              </div>
            </div>
            <div style="text-align:center">
              <div style="font-size:2rem;margin-bottom:0.4rem">📜</div>
              <div style="font-weight:700;color:#6366f1;margin-bottom:0.3rem">Auditability</div>
              <div style="font-size:0.83rem;color:#94a3b8;line-height:1.5">
                Every MLflow run carries a <code>dvc_data_hash</code> tag. The data that
                produced any model can be recovered at any time.
              </div>
            </div>
            <div style="text-align:center">
              <div style="font-size:2rem;margin-bottom:0.4rem">☁️</div>
              <div style="font-weight:700;color:#818cf8;margin-bottom:0.3rem">Off-site Backup</div>
              <div style="font-size:0.83rem;color:#94a3b8;line-height:1.5">
                DagsHub stores data and model artifacts durably, independent of the local
                deployment environment.
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _tab_local_first() -> None:
    """The local-first architecture and the DagsHub rate-limiting story."""
    _section_header("⚡ Local-First Architecture — The Problem We Solved")

    st.markdown(
        _info_card(
            "The DagsHub Rate-Limiting Problem",
            """Early in the project every operation — training runs logged by Airflow DAGs,
            Streamlit queries to the MLflow Explorer, API health checks — pointed
            <strong>directly at DagsHub's hosted MLflow server</strong>.
            <br><br>
            DagsHub enforces API rate limits. Under normal operational load the limit was
            exceeded, causing <strong>HTTP 429 errors</strong> that:
            <ul style="margin:0.4rem 0 0 1rem;padding:0">
              <li>Broke automated retraining DAGs mid-run</li>
              <li>Made the Streamlit MLflow Explorer time out</li>
              <li>Blocked the API from logging predictions</li>
            </ul>""",
            icon="⚠️",
            color="#f59e0b",
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        _info_card(
            "The Solution: Local-First MLflow with DagsHub as Backup",
            """All live operations now use a <strong>local <code>mlflow_buffer</code>
            container</strong> backed by a dedicated PostgreSQL instance. DagsHub is demoted
            to a <strong>scheduled-only backup</strong> — never on the critical path of any
            real-time operation.
            <br><br>
            Result: near-instant MLflow responses, zero rate-limit exposure for all live
            workloads, and DagsHub remains the durable off-site ledger for long-term
            reproducibility.""",
            icon="✅",
            color="#10b981",
        ),
        unsafe_allow_html=True,
    )

    _section_header("🏗️ Local-First Architecture Diagram")
    _render_mermaid(LOCAL_FIRST_DIAGRAM, height=500)

    # Detail table
    st.markdown("<br>", unsafe_allow_html=True)
    _section_header("📋 What Runs Where")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div style="background:#064e3b;border-radius:10px;padding:1rem;
                        border:1px solid #10b981;text-align:center">
              <div style="font-size:1.4rem">⚡</div>
              <div style="font-weight:700;color:#10b981;margin:0.4rem 0">Live operations</div>
              <div style="font-size:0.82rem;color:#a7f3d0;line-height:1.6">
                <code>mlflow_buffer</code> container<br>
                PostgreSQL MLflow backend<br>
                FastAPI prediction logging<br>
                Airflow DAG metric logging<br>
                Streamlit MLflow Explorer
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div style="background:#1e1b4b;border-radius:10px;padding:1rem;
                        border:1px solid #818cf8;text-align:center">
              <div style="font-size:1.4rem">📅</div>
              <div style="font-weight:700;color:#a78bfa;margin:0.4rem 0">Scheduled sync</div>
              <div style="font-size:0.82rem;color:#c4b5fd;line-height:1.6">
                <code>sync_mlflow_to_dagshub</code> DAG<br>
                Pushes runs + artifacts<br>
                <code>sync_production_data</code> DAG<br>
                Exports DB → DVC → DagsHub
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div style="background:#1e293b;border-radius:10px;padding:1rem;
                        border:1px solid #334155;text-align:center">
              <div style="font-size:1.4rem">🔙</div>
              <div style="font-weight:700;color:#94a3b8;margin:0.4rem 0">Manual restore</div>
              <div style="font-size:0.82rem;color:#94a3b8;line-height:1.6">
                <code>dvc pull</code><br>
                Retrieves exact dataset<br>
                by hash from DagsHub<br>
                Only when needed
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _tab_data_versioning() -> None:
    """How DVC data versioning works end-to-end."""
    _section_header("🔄 Data Versioning Pipeline — Signals → DVC → DagsHub")

    st.markdown(
        """
        <div style="background:#1e293b;border-radius:12px;padding:1.2rem 1.6rem;
                    border-left:4px solid #6366f1;margin-bottom:1.4rem">
          <div style="color:#e2e8f0;font-size:0.92rem;line-height:1.7">
            Every signal that reaches the system is stored in PostgreSQL.
            When the <code>sync_production_data</code> Airflow DAG runs, it exports the
            database contents to CSV/JSON, tracks the export with DVC (producing a
            SHA-256 content-addressed pointer), commits the pointer to Git, and pushes
            the binary payload to the DagsHub S3-compatible remote.
            <br><br>
            The result: any training run in MLflow carries a <code>dvc_data_hash</code>
            tag that, combined with <code>dvc pull</code>, retrieves the exact training
            dataset — down to the byte.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_mermaid(DVC_PIPELINE_DIAGRAM, height=420)

    st.markdown("<br>", unsafe_allow_html=True)
    _section_header("🗄️ What Gets Versioned")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div style="background:#1e293b;border-radius:10px;padding:1rem 1.2rem;
                        border:1px solid #334155">
              <div style="font-weight:700;color:#60a5fa;margin-bottom:0.6rem">
                📊 Training datasets (DVC-tracked)
              </div>
              <ul style="color:#cbd5e1;font-size:0.85rem;line-height:1.7;
                         margin:0;padding-left:1.2rem">
                <li>Feature vectors exported from PostgreSQL</li>
                <li>Sparse labels joined to predictions</li>
                <li>Train / test split manifests (JSON)</li>
                <li>Reference data for drift detection
                    (<code>data/drift/reference_data.parquet</code>)</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div style="background:#1e293b;border-radius:10px;padding:1rem 1.2rem;
                        border:1px solid #334155">
              <div style="font-weight:700;color:#a78bfa;margin-bottom:0.6rem">
                🤖 Model artifacts (MLflow-tracked, DagsHub backup)
              </div>
              <ul style="color:#cbd5e1;font-size:0.85rem;line-height:1.7;
                         margin:0;padding-left:1.2rem">
                <li>Trained sklearn pipeline (<code>.pkl</code>)</li>
                <li>Confusion matrix PNG</li>
                <li>Feature importance plot</li>
                <li>Run parameters and metrics (all algorithms evaluated)</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _tab_reproducibility() -> None:
    """End-to-end reproducibility: the MLflow ↔ DVC connection."""
    _section_header("🔬 End-to-End Reproducibility")

    st.markdown(
        _info_card(
            "The Key Insight",
            """Without DVC: <em>"The model achieved 94% accuracy."</em>
            <br>
            With DVC: <em>"The model achieved 94% accuracy,
            and here is the exact 2847-row dataset that produced it. Run
            <code>dvc pull a3f9c1…</code> and you can reproduce it byte-for-byte."</em>
            <br><br>
            Every MLflow training run is tagged with the SHA-256 hash of the DVC-versioned
            dataset used for training. This creates a permanent, auditable link between
            any model version and the data it was trained on.""",
            icon="🎯",
            color="#6366f1",
        ),
        unsafe_allow_html=True,
    )

    _render_mermaid(REPRODUCIBILITY_DIAGRAM, height=280)

    st.markdown("<br>", unsafe_allow_html=True)
    _section_header("🖥️ Deployment Modes — Two Independent, Isolated Environments")

    st.markdown(
        """
        <div style="background:#1e293b;border-radius:10px;padding:0.9rem 1.2rem;
                    border-left:4px solid #f59e0b;margin-bottom:1rem">
          <div style="color:#fde68a;font-size:0.88rem;line-height:1.6">
            <strong>Important:</strong> Local and Cloud/K8s are completely isolated environments.
            No data, models, or artifacts flow between them. Local is a self-contained sandbox —
            perfect for development and experimentation without affecting production.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_mermaid(MODES_DIAGRAM, height=280)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            """
            <div style="background:linear-gradient(135deg,#052e16,#064e3b);
                        border-radius:12px;padding:1.2rem;border:1px solid #10b981">
              <div style="font-weight:700;color:#10b981;margin-bottom:0.6rem">
                🖥️ Local mode — Sandbox
              </div>
              <ul style="color:#a7f3d0;font-size:0.85rem;line-height:1.7;
                         margin:0;padding-left:1.2rem">
                <li>MLflow at <code>localhost:5001</code>, SQLite backend</li>
                <li>DVC cache on local filesystem only — no remote</li>
                <li>No DagsHub connection, no cloud dependency</li>
                <li>Completely isolated: nothing leaves, nothing enters</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div style="background:linear-gradient(135deg,#0d0b2e,#1e1b4b);
                        border-radius:12px;padding:1.2rem;border:1px solid #818cf8">
              <div style="font-weight:700;color:#a78bfa;margin-bottom:0.6rem">
                ☁️ Cloud / K8s mode
              </div>
              <ul style="color:#c4b5fd;font-size:0.85rem;line-height:1.7;
                         margin:0;padding-left:1.2rem">
                <li>MLflow Buffer container + PostgreSQL backend (PRIMARY)</li>
                <li>DagsHub MLflow mirror (BACKUP, scheduled sync)</li>
                <li>DagsHub DVC remote (BACKUP, scheduled push)</li>
                <li>Rate-limit safe — DagsHub never on critical path</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Public entry-point ────────────────────────────────────────────────────────


def render() -> None:
    """Render the DVC & DagsHub page."""
    try:
        _render_content()
    except Exception:
        _logger.exception("Unhandled exception in dvc_dagshub.render()")
        raise


def _render_content() -> None:
    st.markdown(get_global_css(), unsafe_allow_html=True)

    st.markdown(
        hero_section(
            "DVC & DagsHub",
            "Data versioning, reproducibility, and local-first backup strategy",
        ),
        unsafe_allow_html=True,
    )

    _TABS = [
        "📦 Overview",
        "⚡ Local-First Architecture",
        "🔄 Data Versioning Pipeline",
        "🔬 Reproducibility",
    ]
    active = st.radio(
        "Section",
        _TABS,
        horizontal=True,
        key="_dvc_tab",
        label_visibility="collapsed",
    )
    st.markdown(
        "<hr style='margin:0 0 1rem 0;border-color:#334155;'>",
        unsafe_allow_html=True,
    )

    if active == _TABS[0]:
        _tab_overview()
    elif active == _TABS[1]:
        _tab_local_first()
    elif active == _TABS[2]:
        _tab_data_versioning()
    else:
        _tab_reproducibility()
