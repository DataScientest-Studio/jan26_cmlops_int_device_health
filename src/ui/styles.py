"""Shared CSS styles for the MLOps Device Health Streamlit dashboard."""


def get_global_css() -> str:
    """Return global CSS for the entire app."""
    return """
<style>
/* ── Google Fonts ───────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root variables ─────────────────────────────────────────── */
:root {
    --primary: #6366f1;
    --primary-light: #818cf8;
    --primary-dark: #4f46e5;
    --accent: #06b6d4;
    --accent-light: #22d3ee;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --bg-dark: #0f172a;
    --bg-card: #1e293b;
    --bg-card-hover: #334155;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --border: #334155;
    --gradient-1: linear-gradient(135deg, #6366f1 0%, #06b6d4 100%);
    --gradient-2: linear-gradient(135deg, #10b981 0%, #06b6d4 100%);
    --gradient-3: linear-gradient(135deg, #f59e0b 0%, #ef4444 100%);
    --shadow: 0 4px 6px -1px rgba(0,0,0,0.3), 0 2px 4px -2px rgba(0,0,0,0.2);
    --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.4), 0 4px 6px -4px rgba(0,0,0,0.3);
    --radius: 12px;
    --radius-sm: 8px;
}

/* ── Global ─────────────────────────────────────────────────── */
.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.stApp code, .stApp pre {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
}

/* Prevent dimmed residue from previously visited pages.
   Force an opaque background on every main content container
   so nothing bleeds through during page transitions. */
[data-testid="stAppViewContainer"] {
    background-color: var(--bg-dark, #0f172a) !important;
}
[data-testid="stAppViewContainer"] > section {
    background-color: var(--bg-dark, #0f172a) !important;
}
[data-testid="stVerticalBlock"] {
    background-color: transparent;
}
.stTabs [data-baseweb="tab-panel"] {
    background-color: var(--bg-dark, #0f172a) !important;
}

/* ── Hero section ───────────────────────────────────────────── */
.hero-container {
    background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 50%, #0c4a6e 100%);
    border-radius: var(--radius);
    padding: 2.5rem 2rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(99, 102, 241, 0.2);
    position: relative;
    overflow: hidden;
}
.hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: var(--gradient-1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    position: relative;
}
.hero-subtitle {
    font-size: 1.1rem;
    color: var(--text-secondary);
    font-weight: 300;
    line-height: 1.6;
    position: relative;
}

/* ── Metric cards ───────────────────────────────────────────── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem;
    text-align: center;
    transition: all 0.3s ease;
    box-shadow: var(--shadow);
}
.metric-card:hover {
    border-color: var(--primary);
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--primary-light);
    line-height: 1.2;
}
.metric-label {
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin-top: 0.25rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.metric-icon {
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
}

/* ── Info cards ─────────────────────────────────────────────── */
.info-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow);
}
.info-card h3 {
    color: var(--text-primary);
    font-weight: 600;
    margin-bottom: 0.75rem;
    font-size: 1.1rem;
}
.info-card p, .info-card li {
    color: var(--text-secondary);
    line-height: 1.7;
    font-size: 0.95rem;
}

/* ── Status badges ──────────────────────────────────────────── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.375rem;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.025em;
}
.status-healthy {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}
.status-unhealthy {
    background: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
}
.status-warning {
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.3);
}
.status-running {
    background: rgba(6, 182, 212, 0.15);
    color: #22d3ee;
    border: 1px solid rgba(6, 182, 212, 0.3);
}

/* ── Dot indicators ─────────────────────────────────────────── */
.dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
}
.dot-green { background: #10b981; box-shadow: 0 0 6px rgba(16,185,129,0.5); }
.dot-red { background: #ef4444; box-shadow: 0 0 6px rgba(239,68,68,0.5); }
.dot-yellow { background: #f59e0b; box-shadow: 0 0 6px rgba(245,158,11,0.5); }
.dot-blue { background: #06b6d4; box-shadow: 0 0 6px rgba(6,182,212,0.5); }

/* ── Section headers ────────────────────────────────────────── */
.section-header {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.25rem;
    padding-bottom: 0.75rem;
    border-bottom: 2px solid var(--border);
}
.section-subheader {
    font-size: 0.9rem;
    color: var(--text-muted);
    margin-bottom: 1.5rem;
}

/* ── Diagram container ──────────────────────────────────────── */
.diagram-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2rem;
    margin: 1rem 0;
    overflow-x: auto;
}

/* ── Log viewer ─────────────────────────────────────────────── */
.log-viewer {
    background: #0d1117;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    line-height: 1.5;
    max-height: 400px;
    overflow-y: auto;
    color: #c9d1d9;
}

/* ── Service link cards ─────────────────────────────────────── */
.service-link {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 0.75rem 1rem;
    text-decoration: none;
    color: var(--text-primary);
    transition: all 0.2s ease;
    margin-bottom: 0.5rem;
}
.service-link:hover {
    border-color: var(--primary);
    background: var(--bg-card-hover);
}

/* ── Use case cards ─────────────────────────────────────────── */
.usecase-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem;
    transition: all 0.2s ease;
}
.usecase-card:hover {
    border-color: var(--accent);
    box-shadow: var(--shadow-lg);
}
.usecase-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 8px;
    background: var(--gradient-1);
    color: white;
    font-weight: 700;
    font-size: 0.85rem;
    margin-right: 0.5rem;
}

/* ── Tab overrides ──────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-sm);
    padding: 0.5rem 1rem;
}

/* ── Tech stack grid ────────────────────────────────────────── */
.tech-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 0.5rem 1rem;
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--text-primary);
}

/* ── Iframe container ───────────────────────────────────────── */
.iframe-container {
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    box-shadow: var(--shadow-lg);
}

/* ── Animations ─────────────────────────────────────────────── */
@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 4px rgba(16,185,129,0.4); }
    50% { box-shadow: 0 0 12px rgba(16,185,129,0.8); }
}
.pulse-green { animation: pulse-green 2s ease-in-out infinite; }

/* ── Hide Streamlit defaults (keep sidebar toggle visible) ─── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
/* Ensure the sidebar collapse/expand button is always accessible */
[data-testid="collapsedControl"] {
    visibility: visible !important;
    display: flex !important;
    z-index: 999999 !important;
    position: fixed !important;
    top: 0.5rem !important;
    left: 0.5rem !important;
}
</style>
"""


def metric_card(icon: str, value: str, label: str) -> str:
    """Generate HTML for a single metric card."""
    return f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


def status_badge(status: str, text: str) -> str:
    """Generate HTML for a status badge."""
    css_class = {
        "healthy": "status-healthy",
        "running": "status-running",
        "unhealthy": "status-unhealthy",
        "warning": "status-warning",
    }.get(status, "status-warning")
    dot_class = {
        "healthy": "dot-green",
        "running": "dot-blue",
        "unhealthy": "dot-red",
        "warning": "dot-yellow",
    }.get(status, "dot-yellow")
    return (
        f'<span class="status-badge {css_class}"><span class="dot {dot_class}"></span>{text}</span>'
    )


def hero_section(title: str, subtitle: str) -> str:
    """Generate HTML for a hero section."""
    return f"""
    <div class="hero-container">
        <div class="hero-title">{title}</div>
        <div class="hero-subtitle">{subtitle}</div>
    </div>
    """
