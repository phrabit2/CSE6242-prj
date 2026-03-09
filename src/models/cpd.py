"""
Change-Point Detection (CPD) algorithms for baseball performance analysis.

Implements three CPD approaches:
1. PELT (Pruned Exact Linear Time) — Killick et al. (2012)
2. CUSUM (Cumulative Sum) — via Binary Segmentation
3. Bayesian Online CPD — Adams & MacKay (2007)

Reference framework: Truong, Oudre, and Vayatis (2020)
"""

import numpy as np
import pandas as pd
import ruptures as rpt
from dataclasses import dataclass


@dataclass
class ChangePointResult:
    """Result container for change-point detection."""
    method: str
    breakpoints: list[int]
    signal: np.ndarray
    dates: pd.DatetimeIndex | None = None
    metadata: dict | None = None


def detect_pelt(signal: np.ndarray, model: str = "rbf", pen: float = 3.0) -> list[int]:
    """
    PELT algorithm for optimal change-point detection.

    Args:
        signal: 1D or 2D numpy array (n_samples, n_features).
        model: Cost model — 'l2', 'l1', 'rbf', 'linear', 'normal'.
        pen: Penalty value controlling number of breakpoints.

    Returns:
        List of breakpoint indices.
    """
    algo = rpt.Pelt(model=model).fit(signal)
    breakpoints = algo.predict(pen=pen)
    return breakpoints


def detect_binseg(signal: np.ndarray, model: str = "l2", n_bkps: int = 5) -> list[int]:
    """
    Binary Segmentation for change-point detection (supports CUSUM-like behavior).

    Args:
        signal: 1D or 2D numpy array.
        model: Cost model.
        n_bkps: Maximum number of breakpoints.

    Returns:
        List of breakpoint indices.
    """
    algo = rpt.Binseg(model=model).fit(signal)
    breakpoints = algo.predict(n_bkps=n_bkps)
    return breakpoints


def detect_bayesian_online(signal: np.ndarray, hazard: float = 1 / 200) -> list[int]:
    """
    Bayesian Online Change-Point Detection (Adams & MacKay, 2007).

    Simplified implementation using run-length posterior.

    Args:
        signal: 1D numpy array.
        hazard: Prior probability of a change-point at each time step.

    Returns:
        List of detected change-point indices.
    """
    # TODO: Implement full Bayesian online CPD
    # Placeholder using ruptures' window-based approach as proxy
    algo = rpt.Window(model="l2", width=20).fit(signal)
    breakpoints = algo.predict(pen=3.0)
    return breakpoints


def run_multivariate_cpd(
    df: pd.DataFrame,
    metrics: list[str],
    method: str = "pelt",
    **kwargs,
) -> ChangePointResult:
    """
    Run CPD on multivariate time series data.

    Args:
        df: DataFrame indexed by date with metric columns.
        metrics: List of column names to include.
        method: Detection method — 'pelt', 'binseg', or 'bayesian'.
        **kwargs: Additional arguments passed to the detector.

    Returns:
        ChangePointResult with detected breakpoints.
    """
    available = [m for m in metrics if m in df.columns]
    signal = df[available].values

    # Standardize signal
    signal = (signal - np.nanmean(signal, axis=0)) / (np.nanstd(signal, axis=0) + 1e-8)
    signal = np.nan_to_num(signal, nan=0.0)

    detectors = {
        "pelt": detect_pelt,
        "binseg": detect_binseg,
        "bayesian": detect_bayesian_online,
    }

    detector = detectors.get(method, detect_pelt)
    breakpoints = detector(signal, **kwargs)

    return ChangePointResult(
        method=method,
        breakpoints=breakpoints,
        signal=signal,
        dates=df.index if isinstance(df.index, pd.DatetimeIndex) else None,
    )
