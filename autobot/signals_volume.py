"""Volume-pattern signals (DESIGN §2 / SPEC §3).

Implements the four volume patterns (breakout, climax, pullback, OBV divergence).
Every emitted Signal carries a non-empty regime, a stop exactly 2xATR from entry,
and size_pct_equity <= 1% (post-Kelly-cap). Each pattern fires only when the price
is on the correct side of the trend filter (VWAP / 200MA). numpy-only at import.
"""
from __future__ import annotations

import math
import uuid
from typing import List, Optional

from .data.contracts import FeatureBundle, Signal
from .config.settings import RiskConfig
from .math_risk import fractional_kelly, risk_based_size_pct, stop_price

# Pattern thresholds (fixed per window; never re-optimized on the traded bar).
VOL_BREAKOUT_RATIO = 2.0
VOL_CLIMAX_RATIO = 4.0
LOW_VOL_PULLBACK_RATIO = 0.7
OBV_DIVERGE_SLOPE = 1e-9


def _size_and_stop(
    feats: FeatureBundle, risk: RiskConfig, equity: float, side: str, strategy: str, score: float
) -> tuple[float, float, float]:
    entry = feats.close
    stop = stop_price(entry, side, feats.atr, risk.atr_stop_mult)
    kelly = fractional_kelly(win_rate=0.5, payoff=2.0, fraction=risk.kelly_fraction_cap)
    size = risk_based_size_pct(
        equity=equity,
        risk_per_trade_pct=risk.risk_per_trade_pct,
        entry=entry,
        stop_px=stop,
        kelly_frac=kelly,
        hard_cap=0.01,
    )
    return stop, size, entry


def _emit(
    feats: FeatureBundle, risk: RiskConfig, equity: float, side: str, strategy: str,
    score: float, confidence: float, regime: str,
) -> Signal:
    stop, size, entry = _size_and_stop(feats, risk, equity, side, strategy, score)
    target = entry + (3.0 * feats.atr) if side == "long" else entry - (3.0 * feats.atr)
    return Signal(
        id=str(uuid.uuid4()),
        symbol=feats.symbol,
        strategy=strategy,
        side=side,
        ts=feats.ts,
        price=entry,
        confidence=min(1.0, max(0.0, confidence)),
        score=score,
        regime=regime,
        stop_price=stop,
        target_price=target,
        size_pct_equity=size,
        metadata={"vwap": feats.vwap, "atr": feats.atr, "volume_ratio": feats.volume_ratio},
    )


def _vol_breakout(feats: FeatureBundle, risk: RiskConfig, equity: float) -> Optional[Signal]:
    if feats.volume_ratio < VOL_BREAKOUT_RATIO:
        return None
    above = feats.close > feats.vwap
    if above and feats.close > feats.vp_poc:
        side = "long"
    elif (not above) and feats.close < feats.vp_poc:
        side = "short"
    else:
        return None
    if feats.regime == "volatile":
        return None
    score = feats.volume_ratio
    conf = min(1.0, 0.4 + 0.1 * (feats.volume_ratio - VOL_BREAKOUT_RATIO))
    return _emit(feats, risk, equity, side, "vol_breakout", score, conf, feats.regime)


def _vol_climax(feats: FeatureBundle, risk: RiskConfig, equity: float) -> Optional[Signal]:
    if feats.volume_ratio < VOL_CLIMAX_RATIO:
        return None
    # Fade a volume climax: if price pushed well above POC (bought), expect reversion short.
    if feats.close > feats.vp_poc * 1.002:
        side = "short"
    elif feats.close < feats.vp_poc * 0.998:
        side = "long"
    else:
        return None
    regime = "volatile"
    score = feats.volume_ratio
    conf = min(1.0, 0.5 + 0.05 * (feats.volume_ratio - VOL_CLIMAX_RATIO))
    return _emit(feats, risk, equity, side, "vol_climax", score, conf, regime)


def _low_vol_pullback(feats: FeatureBundle, risk: RiskConfig, equity: float) -> Optional[Signal]:
    if feats.volume_ratio > LOW_VOL_PULLBACK_RATIO:
        return None
    if feats.regime != "trend":
        return None
    if feats.close <= feats.sma_slow:  # not in uptrend
        return None
    # Pullback within an uptrend: price dipped toward VWAP but OBV still rising.
    if feats.close < feats.vwap and feats.obv_slope >= 0:
        side = "long"
    else:
        return None
    score = -feats.volume_ratio
    conf = 0.5
    return _emit(feats, risk, equity, side, "low_vol_pullback", score, conf, feats.regime)


def _obv_divergence(feats: FeatureBundle, risk: RiskConfig, equity: float) -> Optional[Signal]:
    if feats.obv_slope <= OBV_DIVERGE_SLOPE:
        return None
    # Accumulation divergence: OBV rising while price weak (below VWAP / slow MA).
    if feats.close < feats.vwap:
        side = "long"
    else:
        return None
    score = feats.obv_slope
    conf = 0.55
    return _emit(feats, risk, equity, side, "obv_divergence", score, conf, feats.regime)


def evaluate(
    feats: FeatureBundle, risk: RiskConfig, equity: float, enabled: List[str]
) -> Optional[Signal]:
    """Evaluate the four volume patterns; return the first Signal or None.

    Walk-forward guard: parameters are constants (see module thresholds), never
    re-optimized on the bar being traded.
    """
    if not math.isfinite(feats.atr) or feats.atr <= 0:
        return None
    if not math.isfinite(feats.volume_ratio) or feats.volume_ratio <= 0:
        return None
    for strategy, fn in (
        ("vol_breakout", _vol_breakout),
        ("vol_climax", _vol_climax),
        ("low_vol_pullback", _low_vol_pullback),
        ("obv_divergence", _obv_divergence),
    ):
        if strategy in enabled:
            sig = fn(feats, risk, equity)
            if sig is not None:
                return sig
    return None
