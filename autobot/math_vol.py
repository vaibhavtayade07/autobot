"""Volume + volatility math. numpy-only at import; ``arch`` is lazy (EWMA fallback).

Functions: Volume Profile POC / value area, OBV, Chaikin Money Flow, volume ratio,
and GARCH(1,1) with an EWMA fallback when the fit is non-stationary (alpha+beta>=1)
or unavailable. Every return honors the SPEC §3 invariants.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Volume Profile
# ---------------------------------------------------------------------------


def volume_profile(
    prices: np.ndarray, volumes: np.ndarray, bins: int = 100
) -> Tuple[float, float, float]:
    """(poc, val, vah). POC is a price inside [min, max]; [val, vah] holds >= 0.68 of vol."""
    prices = np.asarray(prices, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    if prices.size == 0:
        return (0.0, 0.0, 0.0)
    total = float(volumes.sum())
    if total <= 0.0:
        lo, hi = float(prices.min()), float(prices.max())
        mid = 0.5 * (lo + hi)
        return (mid, mid, mid)

    hist, edges = np.histogram(prices, bins=bins, weights=volumes)
    poc_idx = int(np.argmax(hist))
    poc = 0.5 * (edges[poc_idx] + edges[poc_idx + 1])

    # Value area: add bins from highest volume down until >= 68% of total volume.
    # NOTE (QA #11, accepted): modeling the value area as the top-N volume bins is a
    # v1 simplification — left as-is; not a blocker.
    order = np.argsort(hist)[::-1]
    cum = 0.0
    included = set()
    for idx in order:
        included.add(idx)
        cum += hist[idx]
        if cum >= 0.68 * total:
            break
    idxs = sorted(included)
    val = float(edges[min(idxs)])
    vah = float(edges[max(idxs) + 1])

    lo, hi = float(prices.min()), float(prices.max())
    poc = min(max(poc, lo), hi)
    return (poc, val, vah)


# ---------------------------------------------------------------------------
# Volume-derived indicators
# ---------------------------------------------------------------------------


def obv(closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """On-Balance Volume (cumulative)."""
    closes = np.asarray(closes, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    out = np.zeros(len(closes))
    obv_val = 0.0
    for i in range(len(closes)):
        if i > 0:
            if closes[i] > closes[i - 1]:
                obv_val += volumes[i]
            elif closes[i] < closes[i - 1]:
                obv_val -= volumes[i]
        out[i] = obv_val
    return out


def obv_slope(obv_series: np.ndarray, window: int = 20) -> float:
    """Slope of OBV over the trailing window via least squares (price per bar)."""
    s = np.asarray(obv_series, dtype=float)
    if len(s) < 2:
        return 0.0
    seg = s[-window:] if len(s) >= window else s
    x = np.arange(len(seg), dtype=float)
    if len(seg) < 2:
        return 0.0
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, seg, rcond=None)
    return float(coef[0])


def chaikin_money_flow(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int = 20
) -> np.ndarray:
    """Chaikin Money Flow over a rolling window."""
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    volume = np.asarray(volume, dtype=float)
    n = len(close)
    if n == 0:
        return np.array([])
    denom = np.where((high - low) == 0.0, 1e-12, (high - low))
    mfm = ((close - low) - (high - close)) / denom
    mfv = mfm * volume
    out = np.zeros(n)
    cum = np.cumsum(mfv)
    cumv = np.cumsum(volume)
    for i in range(n):
        lo = max(0, i - period + 1)
        num = cum[i] - (cum[lo - 1] if lo > 0 else 0.0)
        den = cumv[i] - (cumv[lo - 1] if lo > 0 else 0.0)
        out[i] = num / den if den > 1e-12 else 0.0
    return out


def volume_ratio(volumes: np.ndarray, window: int = 20) -> np.ndarray:
    """Ratio of current volume to the trailing mean (>1 = elevated)."""
    volumes = np.asarray(volumes, dtype=float)
    out = np.zeros(len(volumes))
    for i in range(len(volumes)):
        lo = max(0, i - window + 1)
        seg = volumes[lo : i + 1]
        m = seg.mean()
        out[i] = volumes[i] / m if m > 1e-12 else 0.0
    return out


# ---------------------------------------------------------------------------
# GARCH(1,1) with EWMA fallback
# ---------------------------------------------------------------------------


def _ewma_var(returns: np.ndarray, lam: float = 0.94) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    var = float(np.var(r))
    if var <= 0:
        var = 1e-12
    for x in r:
        var = lam * var + (1.0 - lam) * float(x * x)
    return max(var, 1e-12)


def garch_11(returns: np.ndarray, horizon: int = 1) -> Tuple[float, bool]:
    """(forecast_var, unstable_flag). Positive variance always.

    Uses ``arch`` lazily; if alpha+beta >= 1 (non-stationary) or arch is absent,
    falls back to EWMA variance and sets ``unstable_flag=True``.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.size < 2:
        seed = float(np.var(returns)) if returns.size > 0 else 0.0
        return (max(seed, 1e-12), False)

    try:
        from arch import arch_model

        model = arch_model(returns, vol="Garch", p=1, q=1, rescale=False)
        res = model.fit(disp="off")
        params = res.params
        omega = float(params[0])
        alpha = float(params[1])
        beta = float(params[2])
        if alpha + beta >= 1.0:
            return (_ewma_var(returns), True)
        resid = res.resid
        last_r2 = float(resid[-1] ** 2)
        last_sigma2 = float(res.conditional_volatility[-1] ** 2)
        sigma2 = omega + alpha * last_r2 + beta * last_sigma2
        for _ in range(2, horizon + 1):
            sigma2 = omega + (alpha + beta) * sigma2
        if sigma2 <= 0:
            return (_ewma_var(returns), True)
        return (float(sigma2), False)
    except Exception:
        return (_ewma_var(returns), True)
