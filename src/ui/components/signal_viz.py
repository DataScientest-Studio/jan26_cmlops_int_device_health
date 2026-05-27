"""Signal visualisation helpers for the Streamlit dashboard."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _gaussian(t: np.ndarray, mu: float, sigma: float, height: float) -> np.ndarray:
    return height * np.exp(-((t - mu) ** 2) / (2 * sigma**2))


def _lorentzian(t: np.ndarray, mu: float, gamma: float, height: float) -> np.ndarray:
    return height * gamma**2 / ((t - mu) ** 2 + gamma**2)


def create_signal_comparison_chart() -> go.Figure:
    """Create an interactive chart comparing healthy vs unhealthy signals."""
    rng = np.random.default_rng(42)
    t = np.linspace(0, 100, 201)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Healthy Signal (Gaussian, Low Noise)",
            "Unhealthy Signal (Lorentzian, High Noise)",
            "Feature Space Separation",
            "Signal Overlay Comparison",
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # Healthy signal
    s_healthy = _gaussian(t, mu=50, sigma=3.5, height=2.5) + rng.normal(0, 0.02, len(t))
    fig.add_trace(
        go.Scatter(
            x=t,
            y=s_healthy,
            mode="lines",
            name="Healthy (Gaussian)",
            line={"color": "#10b981", "width": 2},
        ),
        row=1,
        col=1,
    )

    # Unhealthy signal
    gamma = 1.1775 * 3.5
    s_unhealthy = _lorentzian(t, mu=38, gamma=gamma, height=2.5) + rng.normal(0, 0.08, len(t))
    fig.add_trace(
        go.Scatter(
            x=t,
            y=s_unhealthy,
            mode="lines",
            name="Unhealthy (Lorentzian)",
            line={"color": "#ef4444", "width": 2},
        ),
        row=1,
        col=2,
    )

    # Feature space
    n_samples = 60
    healthy_snr = rng.normal(48, 5, n_samples)
    healthy_fwhm = rng.normal(8.2, 0.8, n_samples)
    unhealthy_snr = rng.normal(18, 6, n_samples)
    unhealthy_fwhm = rng.normal(10.5, 1.2, n_samples)

    fig.add_trace(
        go.Scatter(
            x=healthy_snr,
            y=healthy_fwhm,
            mode="markers",
            name="Healthy",
            marker={"color": "#10b981", "size": 8, "opacity": 0.7},
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=unhealthy_snr,
            y=unhealthy_fwhm,
            mode="markers",
            name="Unhealthy",
            marker={"color": "#ef4444", "size": 8, "opacity": 0.7},
        ),
        row=2,
        col=1,
    )

    # Overlay
    fig.add_trace(
        go.Scatter(
            x=t,
            y=s_healthy,
            mode="lines",
            name="Healthy",
            line={"color": "#10b981", "width": 2, "dash": "solid"},
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=s_unhealthy,
            mode="lines",
            name="Unhealthy",
            line={"color": "#ef4444", "width": 2, "dash": "dash"},
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=600,
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font={"family": "Inter, sans-serif", "color": "#e2e8f0"},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.15, "xanchor": "center", "x": 0.5},
        margin={"l": 60, "r": 30, "t": 50, "b": 60},
    )
    fig.update_xaxes(title_text="Time (a.u.)", row=1, col=1, gridcolor="#334155")
    fig.update_xaxes(title_text="Time (a.u.)", row=1, col=2, gridcolor="#334155")
    fig.update_xaxes(title_text="SNR (dB)", row=2, col=1, gridcolor="#334155")
    fig.update_xaxes(title_text="Time (a.u.)", row=2, col=2, gridcolor="#334155")
    fig.update_yaxes(title_text="Amplitude", row=1, col=1, gridcolor="#334155")
    fig.update_yaxes(title_text="Amplitude", row=1, col=2, gridcolor="#334155")
    fig.update_yaxes(title_text="FWHM", row=2, col=1, gridcolor="#334155")
    fig.update_yaxes(title_text="Amplitude", row=2, col=2, gridcolor="#334155")
    return fig


def create_drift_scenario_chart() -> go.Figure:
    """Create a chart showing how data drift evolves over time."""
    np.random.default_rng(99)
    days = np.arange(1, 31)

    # Normal -> drifted SNR distribution over 30 days
    snr_means = np.where(days < 15, 48 - days * 0.2, 48 - 14 * 0.2 - (days - 14) * 1.5)
    snr_upper = snr_means + 5
    snr_lower = snr_means - 5

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=days,
            y=snr_upper,
            mode="lines",
            line={"width": 0},
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=days,
            y=snr_lower,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(99,102,241,0.2)",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=days,
            y=snr_means,
            mode="lines+markers",
            name="Mean SNR",
            line={"color": "#818cf8", "width": 2},
            marker={"size": 5},
        )
    )
    # Threshold line
    fig.add_hline(
        y=30,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text="Drift Alert Threshold",
        annotation_position="top left",
        annotation_font_color="#ef4444",
    )
    # Detection point
    detection_day = 18
    fig.add_vline(
        x=detection_day,
        line_dash="dot",
        line_color="#f59e0b",
        annotation_text="Drift Detected (Day 18)",
        annotation_position="top right",
        annotation_font_color="#f59e0b",
    )

    fig.update_layout(
        title="Data Drift Scenario — SNR Degradation Over 30 Days",
        xaxis_title="Day",
        yaxis_title="Mean SNR (dB)",
        height=400,
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font={"family": "Inter, sans-serif", "color": "#e2e8f0"},
        margin={"l": 60, "r": 30, "t": 60, "b": 50},
    )
    fig.update_xaxes(gridcolor="#334155")
    fig.update_yaxes(gridcolor="#334155")
    return fig


def create_feature_extraction_diagram() -> go.Figure:
    """Create a visual pipeline showing feature extraction from raw signal."""
    rng = np.random.default_rng(7)
    t = np.linspace(0, 100, 201)
    raw = _gaussian(t, 50, 3.5, 2.5) + rng.normal(0, 0.03, len(t))

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("1. Raw Signal", "2. Denoised", "3. Feature Vector"),
        horizontal_spacing=0.08,
    )

    fig.add_trace(
        go.Scatter(
            x=t, y=raw, mode="lines", line={"color": "#94a3b8", "width": 1.5}, showlegend=False
        ),
        row=1,
        col=1,
    )

    # Smoothed
    from scipy.ndimage import uniform_filter1d  # noqa: E402

    smoothed = uniform_filter1d(raw, size=7)
    fig.add_trace(
        go.Scatter(
            x=t, y=smoothed, mode="lines", line={"color": "#6366f1", "width": 2}, showlegend=False
        ),
        row=1,
        col=2,
    )

    # Features as bar chart
    features = {
        "peak_height": 2.48,
        "peak_center": 50.1,
        "FWHM": 8.25,
        "SNR": 47.3,
        "peak_area": 21.6,
        "noise_level": 0.031,
    }
    # Normalise to [0,1] for display
    vals = list(features.values())
    max_v = max(vals)
    norm = [v / max_v for v in vals]

    fig.add_trace(
        go.Bar(
            x=list(features.keys()),
            y=norm,
            marker_color=[
                "#6366f1",
                "#06b6d4",
                "#10b981",
                "#f59e0b",
                "#818cf8",
                "#ef4444",
            ],
            showlegend=False,
            text=[f"{v:.2f}" for v in vals],
            textposition="outside",
            textfont={"size": 10, "color": "#e2e8f0"},
        ),
        row=1,
        col=3,
    )

    fig.update_layout(
        height=320,
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font={"family": "Inter, sans-serif", "color": "#e2e8f0", "size": 11},
        margin={"l": 50, "r": 20, "t": 50, "b": 50},
    )
    fig.update_xaxes(gridcolor="#334155")
    fig.update_yaxes(gridcolor="#334155")
    return fig
