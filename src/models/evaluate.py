"""
Evaluation metrics for Change-Point Detection results.

Measures detection precision and recall against documented
historical events (e.g., player rank changes, known slumps/breakouts).
"""

import numpy as np


def detection_precision(predicted: list[int], actual: list[int], margin: int = 5) -> float:
    """
    Compute precision: fraction of predicted change-points near an actual one.

    Args:
        predicted: Predicted breakpoint indices.
        actual: Ground truth breakpoint indices.
        margin: Tolerance window (in time steps) for a match.

    Returns:
        Precision score in [0, 1].
    """
    if not predicted:
        return 0.0

    true_positives = sum(
        1 for p in predicted if any(abs(p - a) <= margin for a in actual)
    )
    return true_positives / len(predicted)


def detection_recall(predicted: list[int], actual: list[int], margin: int = 5) -> float:
    """
    Compute recall: fraction of actual change-points detected.

    Args:
        predicted: Predicted breakpoint indices.
        actual: Ground truth breakpoint indices.
        margin: Tolerance window (in time steps) for a match.

    Returns:
        Recall score in [0, 1].
    """
    if not actual:
        return 0.0

    detected = sum(
        1 for a in actual if any(abs(p - a) <= margin for p in predicted)
    )
    return detected / len(actual)


def f1_score(precision: float, recall: float) -> float:
    """Compute F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def segment_comparison(signal: np.ndarray, breakpoint: int) -> dict:
    """
    Compare signal statistics before and after a breakpoint.

    Args:
        signal: 1D numpy array.
        breakpoint: Index of the change-point.

    Returns:
        Dict with before/after mean, std, and effect size.
    """
    before = signal[:breakpoint]
    after = signal[breakpoint:]

    before_mean = np.nanmean(before)
    after_mean = np.nanmean(after)
    pooled_std = np.sqrt(
        (np.nanvar(before) * len(before) + np.nanvar(after) * len(after))
        / (len(before) + len(after))
    )
    effect_size = (after_mean - before_mean) / (pooled_std + 1e-8)

    return {
        "before_mean": before_mean,
        "after_mean": after_mean,
        "before_std": np.nanstd(before),
        "after_std": np.nanstd(after),
        "effect_size": effect_size,
        "direction": "improvement" if effect_size > 0 else "decline",
    }
