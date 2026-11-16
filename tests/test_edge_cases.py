"""Edge-case tests for real bugs found during QA review.

These target the math/risk/signal boundaries the live engine depends on, including
regressions for: NaN/Inf ATR in the signal path, constant-series OU, Kelly edges
(0/1), zero-volume VPIN, volume-profile with zero-volume bins, and a defense-in-depth
check that the risk gate rejects a non-finite stop.
"""
import math

import numpy as np
import pytest

from autobot.data.contracts import FeatureBundle, Signal
from autobot.config.settings import RiskConfig
from autobot.signals_volume import evaluate as vol_eval
from autobot.math_stats import ou_half_life, kalman_hedge_ratio
from autobot.math_microstructure import vpin
from autobot.math_vol import volume_profile
from autobot.math_risk import kelly_fraction
from autobot.engine.live_loop import risk_gate


def _fb(atr=2.0, volume_ratio=1.0, **kw) -> FeatureBundle:
    base = dict(
        symbol="BTC/USDT", ts=0, close=100.0, vwap=100.0, sma_fast=100.0,
        sma_slow=100.0, imbalance=0.0, microprice=100.0, weighted_mid=100.0,
        vpin=0.0, hurst=0.5, shannon=0.0, apen=0.0, amihud=0.0, garch_var=0.0,
        garch_unstable=False, vp_poc=100.0, vp_val=99.0, vp_vah=101.0,
        atr=atr, obv=0.0, obv_slope=0.0, kalman_beta=0.0, ou_half_life=None,
        spread_z=0.0, volume_ratio=volume_ratio, regime="range",
        raw_depth=None, raw_ticks=[],
    )
    base.update(kw)
    return FeatureBundle(**base)


# --- signal path must not emit NaN-laden signals --------------------------------

def test_eval_nan_atr_rejected():
    fb = _fb(atr=float("nan"), volume_ratio=5.0, close=100.0, vwap=99.0,
             vp_poc=99.0, sma_fast=99.0, sma_slow=99.0, regime="trend")
    assert vol_eval(fb, RiskConfig(), 100000.0, ["vol_breakout"]) is None


def test_eval_inf_atr_rejected():
    fb = _fb(atr=float("inf"), volume_ratio=5.0, close=100.0, vwap=99.0,
             vp_poc=99.0, sma_fast=99.0, sma_slow=99.0, regime="trend")
    assert vol_eval(fb, RiskConfig(), 100000.0, ["vol_breakout"]) is None


def test_eval_volume_ratio_zero_rejected():
    assert vol_eval(_fb(atr=2.0, volume_ratio=0.0), RiskConfig(), 100000.0,
                    ["vol_breakout"]) is None


def test_eval_volume_ratio_nan_rejected():
    assert vol_eval(_fb(atr=2.0, volume_ratio=float("nan")), RiskConfig(),
                    100000.0, ["vol_breakout"]) is None


# --- risk gate must refuse a non-finite stop (defense in depth) -----------------

def test_risk_gate_rejects_nan_stop():
    sig = Signal(
        id="x", symbol="BTC/USDT", strategy="s", side="long", ts=0,
        price=100.0, confidence=0.5, score=1.0, regime="range",
        stop_price=float("nan"), target_price=None, size_pct_equity=0.01,
        metadata={},
    )
    dec = risk_gate(sig, 100000.0, None)
    assert dec.approved is False
    assert "stop" in dec.reason.lower()


# --- OU half-life on constant / flat series (no crash, no false MR) --------------

def test_ou_constant_series_none():
    assert ou_half_life(np.ones(100)) is None
    assert ou_half_life(np.full(50, 5.0)) is None
    assert ou_half_life(np.zeros(10)) is None  # <3 returns None anyway


# --- Kelly edges 0 / 1 and non-positive payoff -----------------------------------

def test_kelly_edges():
    assert kelly_fraction(0.0, 2.0) == 0.0
    assert kelly_fraction(1.0, 2.0) == 1.0
    assert kelly_fraction(0.5, 0.0) == 0.0
    assert kelly_fraction(0.5, -1.0) == 0.0
    assert kelly_fraction(1.5, 2.0) == 1.0  # win_rate > 1 clamped to 1


# --- VPIN zero volume / all-buy --------------------------------------------------

def test_vpin_zero_volume():
    assert vpin([], bucket_size=10.0) == 0.0
    vp_all_buy = vpin([(100.0, 1.0, "buy"), (100.0, 1.0, "buy")], bucket_size=10.0)
    assert 0.0 <= vp_all_buy <= 1.0
    assert vp_all_buy == 1.0  # fully informed (all one side)


# --- Volume profile with a zero-volume bin (no crash, POC in value area) --------

def test_volume_profile_zero_volume_bin():
    prices = np.array([100.0, 100.0, 105.0, 110.0, 110.0])
    volumes = np.array([0.0, 0.0, 5.0, 0.0, 0.0])
    poc, val, vah = volume_profile(prices, volumes, bins=5)
    assert math.isfinite(poc) and math.isfinite(val) and math.isfinite(vah)
    assert val <= poc <= vah


# --- Kalman numeric stability on a long series -----------------------------------

def test_kalman_long_series_stable():
    n = 100_000
    rng = np.random.default_rng(0)
    x = 100.0 + np.cumsum(rng.normal(0, 1, n))
    y = 0.5 * x + rng.normal(0, 0.1, n)
    beta, P = kalman_hedge_ratio(y, x)
    assert math.isfinite(beta)
    assert abs(beta - 0.5) < 0.05
    assert np.all(np.isfinite(P)) and np.all(P >= 0.0)
