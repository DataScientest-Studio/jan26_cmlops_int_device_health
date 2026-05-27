"""Use Cases — interactive workflow runners.

Tabbed interface providing nine MLOps use cases:
  Tab 1 – Greenfield Bootstrap
  Tab 2 – Retraining Pipeline
  Tab 3 – Drift Provocation
  Tab 4 – Champion / Challenger
  Tab 5 – A/B Testing
  Tab 6 – Nginx Traffic Split (true A/B)
  Tab 7 – Model Promotion
  Tab 8 – Model Lineage Audit
  Tab 9 – Batch Re-Scoring
"""

from __future__ import annotations

import streamlit as st

from src.ui.logging_ui import get_ui_logger
from src.ui.styles import get_global_css, hero_section

# Re-export symbols that tests and legacy code reference
from ._common import (  # noqa: F401
    GAMMA_SIGMA_FACTOR as _GAMMA_SIGMA_FACTOR_UI,
)
from ._common import (
    MODEL_NAME as _MODEL_NAME,  # noqa: F401
)
from ._common import SECTION_CSS, detect_mode
from ._common import (
    fetch_champion_info as _fetch_champion_info,  # noqa: F401
)
from .drift_provocation import (
    DRIFT_TYPES as _DRIFT_TYPES,  # noqa: F401
)
from .drift_provocation import (
    generate_batch as _generate_batch,  # noqa: F401
)
from .drift_provocation import (
    ks_tests as _ks_tests,  # noqa: F401
)

_logger = get_ui_logger(__name__)

_UC_TABS = [
    "🚀 Greenfield Bootstrap",
    "🔄 Retraining Pipeline",
    "📊 Drift Provocation",
    "🥊 Champion / Challenger",
    "🔀 A/B Testing",
    "⚖️ Nginx Traffic Split",
    "🏆 Model Promotion",
    "🔍 Model Lineage Audit",
    "⏮️ Batch Re-Scoring",
]

# CSS to style the radio buttons as a tab bar
_TAB_CSS = """
<style>
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] { padding: 0 !important; }
div[data-baseweb="radio"] > div { gap: 0.25rem; flex-wrap: wrap; row-gap: 0.5rem; }
div[data-baseweb="radio"] > div > label {
    border: 1px solid #e2e8f0;
    border-radius: 8px 8px 0 0;
    padding: 0.45rem 1rem;
    margin-bottom: -1px;
    background: #f8fafc;
    font-size: 0.875rem;
    cursor: pointer;
    transition: background 0.15s, color 0.15s;
}
div[data-baseweb="radio"] > div > label:hover { background: #e0e7ff; }
div[data-baseweb="radio"] > div > label[data-checked="true"],
div[data-baseweb="radio"] > div > label[aria-checked="true"] {
    background: white;
    border-bottom-color: white;
    font-weight: 600;
    color: #4f46e5;
}
</style>
"""


def render() -> None:
    """Render the Use Cases page with six tabs."""
    _logger.debug("Rendering Use Cases page")
    try:
        _render_content()
    except Exception:
        _logger.exception("Unhandled exception in use_cases.render()")
        raise


def _render_content() -> None:
    """Inner render — separated so exceptions are caught and logged by render()."""
    st.markdown(get_global_css(), unsafe_allow_html=True)
    st.markdown(SECTION_CSS, unsafe_allow_html=True)
    st.markdown(_TAB_CSS, unsafe_allow_html=True)
    st.markdown(
        hero_section(
            "🧪 MLOps Use Cases",
            "Interactive workflow runners for key MLOps scenarios — "
            "bootstrap, lineage audit, drift detection, retraining, and promotion.",
        ),
        unsafe_allow_html=True,
    )

    mode = detect_mode()

    # Use st.radio with a key= so the selected tab persists across st.rerun() calls.
    # st.tabs() does NOT preserve its selection through reruns, causing page jumps
    # (e.g. clicking "Start A/B Test" or "Trigger Retraining DAG" resets to tab 0).
    active_tab = st.radio(
        "Use Case",
        _UC_TABS,
        horizontal=True,
        key="_uc_tab",
        label_visibility="collapsed",
    )
    st.markdown(
        "<hr style='margin:0 0 1rem 0;border-color:#e2e8f0;'>",
        unsafe_allow_html=True,
    )

    if active_tab == _UC_TABS[0]:
        from .greenfield import render_greenfield_tab

        render_greenfield_tab(mode)

    elif active_tab == _UC_TABS[1]:
        from .retraining_pipeline import render_retraining_pipeline_tab

        render_retraining_pipeline_tab(mode)

    elif active_tab == _UC_TABS[2]:
        from .drift_provocation import render_drift_provocation_tab

        render_drift_provocation_tab(mode)

    elif active_tab == _UC_TABS[3]:
        from .champion_challenger import render_champion_challenger_tab

        render_champion_challenger_tab(mode)

    elif active_tab == _UC_TABS[4]:
        from .ab_testing import render_ab_testing_tab

        render_ab_testing_tab(mode)

    elif active_tab == _UC_TABS[5]:
        from .ab_testing_nginx import render_ab_testing_nginx_tab

        render_ab_testing_nginx_tab(mode)

    elif active_tab == _UC_TABS[6]:
        from .model_promotion import render_model_promotion_tab

        render_model_promotion_tab(mode)

    elif active_tab == _UC_TABS[7]:
        from .lineage_audit import render_lineage_audit_tab

        render_lineage_audit_tab()

    elif active_tab == _UC_TABS[8]:
        from .batch_rescoring import render_batch_rescoring_tab

        render_batch_rescoring_tab(mode)
