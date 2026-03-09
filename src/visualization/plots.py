"""
Visualization utilities for CPD analysis results.

Provides reusable plotting functions for:
- Time series with detected change-points
- Before/After "Snapshot" comparison
- Clutch vs. Core performance breakdown
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_timeseries_with_changepoints(
    dates: pd.DatetimeIndex,
    signal: np.ndarray,
    breakpoints: list[int],
    metric_name: str = "Metric",
    title: str = "Performance Time Series with Change-Points",
) -> go.Figure:
    """
    Plot a time series with vertical lines at detected change-points.

    Args:
        dates: DatetimeIndex for x-axis.
        signal: 1D array of metric values.
        breakpoints: List of breakpoint indices.
        metric_name: Label for y-axis.
        title: Plot title.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=signal,
        mode="lines+markers",
        name=metric_name,
        line=dict(color="#1f77b4"),
        marker=dict(size=3),
    ))

    for bp in breakpoints:
        if bp < len(dates):
            fig.add_vline(
                x=dates[bp],
                line_dash="dash",
                line_color="red",
                annotation_text="CP",
            )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=metric_name,
        template="plotly_white",
    )
    return fig


def plot_before_after_snapshot(
    before_stats: dict,
    after_stats: dict,
    metrics: list[str],
    title: str = "Before vs. After Change-Point Snapshot",
) -> go.Figure:
    """
    Create a grouped bar chart comparing metrics before and after a change-point.

    Args:
        before_stats: Dict mapping metric name to pre-CP average.
        after_stats: Dict mapping metric name to post-CP average.
        metrics: List of metric names to display.
        title: Plot title.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=metrics,
        y=[before_stats.get(m, 0) for m in metrics],
        name="Before",
        marker_color="#636EFA",
    ))

    fig.add_trace(go.Bar(
        x=metrics,
        y=[after_stats.get(m, 0) for m in metrics],
        name="After",
        marker_color="#EF553B",
    ))

    fig.update_layout(
        title=title,
        barmode="group",
        template="plotly_white",
    )
    return fig


def plot_multivariate_dashboard(
    df: pd.DataFrame,
    metrics: list[str],
    breakpoints: list[int],
    player_name: str = "Player",
) -> go.Figure:
    """
    Create a multi-panel dashboard showing multiple metrics with shared change-points.

    Args:
        df: DataFrame indexed by date.
        metrics: List of metric column names.
        breakpoints: Shared breakpoint indices.
        player_name: Player name for title.

    Returns:
        Plotly Figure with subplots.
    """
    n_metrics = len(metrics)
    fig = make_subplots(
        rows=n_metrics, cols=1,
        shared_xaxes=True,
        subplot_titles=metrics,
        vertical_spacing=0.05,
    )

    for i, metric in enumerate(metrics, 1):
        if metric in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[metric],
                    mode="lines",
                    name=metric,
                ),
                row=i, col=1,
            )
            for bp in breakpoints:
                if bp < len(df):
                    fig.add_vline(
                        x=df.index[bp],
                        line_dash="dash",
                        line_color="red",
                        row=i, col=1,
                    )

    fig.update_layout(
        title=f"{player_name} — Multivariate Performance Dashboard",
        height=250 * n_metrics,
        template="plotly_white",
        showlegend=False,
    )
    return fig
