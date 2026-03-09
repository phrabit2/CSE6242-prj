"""Unit tests for Change-Point Detection algorithms."""

import numpy as np
import pytest

from src.models.cpd import detect_pelt, detect_binseg, ChangePointResult


def _make_signal_with_shift(n: int = 200, shift_at: int = 100, shift_size: float = 3.0) -> np.ndarray:
    """Create a synthetic signal with a known mean shift."""
    rng = np.random.default_rng(42)
    signal = np.concatenate([
        rng.normal(0, 1, shift_at),
        rng.normal(shift_size, 1, n - shift_at),
    ])
    return signal


class TestPELT:
    def test_detects_single_shift(self):
        signal = _make_signal_with_shift()
        bkps = detect_pelt(signal, model="l2", pen=3.0)
        # Should detect a breakpoint near index 100
        assert any(abs(bp - 100) <= 15 for bp in bkps), f"Expected breakpoint near 100, got {bkps}"

    def test_no_change_in_constant_signal(self):
        signal = np.ones(200)
        bkps = detect_pelt(signal, model="l2", pen=10.0)
        # Only the terminal breakpoint (n) should be returned
        assert len(bkps) <= 1


class TestBinarySeg:
    def test_detects_single_shift(self):
        signal = _make_signal_with_shift()
        bkps = detect_binseg(signal, model="l2", n_bkps=1)
        assert any(abs(bp - 100) <= 15 for bp in bkps), f"Expected breakpoint near 100, got {bkps}"
