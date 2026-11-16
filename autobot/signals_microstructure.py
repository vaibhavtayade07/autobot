"""Microstructure signals for crypto (book-imbalance / VPIN). numpy-only at import.

Fires short-horizon, execution-side signals from the local order book + VPIN. Each
signal still respects the trend filter (price vs VWAP) and the 2xATR stop + 1% cap.
"""
from __future__ import annotations

import uuid
from typing import List, Optional

from .data.contracts import FeatureBundle, Signal
from .config.settings import RiskConfig
from .math_risk import fractional_kelly, risk_based_size_pct, stop_price

IMBALANCE_THRESHOLD = 0.5
VPIN_LO, VPIN_HI = 0.30, 0.90  # informed-flow band; too high => already priced in


def _emit(feats, risk, equity, side, strategy, score, confidence):
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
    target = entry + 3.0 * feats.atr if side == "long" else entry - 3.0 * feats.atr
    return Signal(
        id=str(uuid.uuid4()),
        symbol=feats.symbol,
        strategy=strategy,
        side=side,
        ts=feats.ts,
        price=entry,
        confidence=min(1.0, max(0.0, confidence)),
        score=score,
        regime=feats.regime,
        stop_price=stop,
        target_price=target,
        size_pct_equity=size,
        metadata={"imbalance": feats.imbalance, "vpin": feats.vpin},
    )


def _micro_imbalance(feats: FeatureBundle, risk: RiskConfig, equity: float) -> Optional[Signal]:
    if feats.regime == "volatile":
        return None
    if abs(feats.imbalance) < IMBALANCE_THRESHOLD:
        return None
    above = feats.close > feats.vwap
    # Trade in the direction of book pressure only when the trend filter agrees.
    if feats.imbalance > 0 and above:
        side = "long"
    elif feats.imbalance < 0 and (not above):
        side = "short"
    else:
        return None
    score = feats.imbalance
    conf = min(1.0, 0.4 + abs(feats.imbalance) * 0.5)
    return _emit(feats, risk, equity, side, "micro_imbalance", score, conf)


def _micro_vpin(feats: FeatureBundle, risk: RiskConfig, equity: float) -> Optional[Signal]:
    if not (VPIN_LO <= feats.vpin <= VPIN_HI):
        return None
    above = feats.close > feats.vwap
    # Informed buying (high VPIN) with rising price -> follow; vice versa for selling.
    if feats.imbalance > 0 and above:
        side = "long"
    elif feats.imbalance < 0 and (not above):
        side = "short"
    else:
        return None
    score = feats.vpin
    conf = min(1.0, 0.35 + feats.vpin * 0.4)
    return _emit(feats, risk, equity, side, "micro_vpin", score, conf)


def evaluate(
    feats: FeatureBundle, risk: RiskConfig, equity: float, enabled: List[str]
) -> Optional[Signal]:
    """Return a microstructure Signal or None. Walk-forward: constant thresholds."""
    if feats.atr <= 0:
        return None
    for strategy, fn in (
        ("micro_imbalance", _micro_imbalance),
        ("micro_vpin", _micro_vpin),
    ):
        if strategy in enabled:
            sig = fn(feats, risk, equity)
            if sig is not None:
                return sig
    return None
