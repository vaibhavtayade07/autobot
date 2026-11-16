"""Offline math invariant tests.

Runs with ONLY numpy/pandas/scipy installed: it imports the math self-test from
autobot.cli (which imports only the math_* modules) and asserts every SPEC §3 /
DESIGN §5 invariant is green.
"""
from __future__ import annotations

import sys

import pytest

# `autobot.cli` only imports heavy deps lazily, so this import is safe offline.
from autobot.cli import run_math_selftest


def test_all_math_invariants():
    results = run_math_selftest()
    failed = [(n, d) for n, ok, d in results if not ok]
    if failed:
        msg = "; ".join(f"{n}: {d}" for n, d in failed)
        pytest.fail(f"{len(failed)} invariant(s) FAILED -> {msg}")
    # Sanity: we actually exercised a meaningful number of checks.
    assert len(results) >= 25


def test_no_nan_reaches_risk():
    """Spot-check that key invariants never propagate NaN (SPEC edge #5)."""
    import numpy as np

    from autobot.math_stats import ou_half_life, spread_z
    from autobot.math_microstructure import microprice

    assert np.isfinite(microprice(100.0, 1.0, 101.0, 2.0))
    assert np.all(np.isfinite(spread_z(np.ones(50), window=20)))
    assert ou_half_life(np.cumsum(np.random.default_rng(0).normal(0, 1, 400))) is None


if __name__ == "__main__":
    # Allow direct execution: python tests/test_math.py
    res = run_math_selftest()
    failed = [n for n, ok, _ in res if not ok]
    print(f"PASS {sum(1 for _, ok, _ in res if ok)}/{len(res)}")
    sys.exit(1 if failed else 0)
