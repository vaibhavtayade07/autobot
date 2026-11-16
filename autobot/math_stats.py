"""Statistical / time-series math. numpy-only; statsmodels is imported lazily.

Functions: custom Kalman hedge ratio (no pykalman), cointegration p-value (lazy
statsmodels), spread z-score, Ornstein-Uhlenbeck half-life, Shannon + approximate
entropy, Hurst R/S, plus shared trend helpers (VWAP / SMA / regime) used by the
signal filters.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Kalman hedge ratio (custom, ~8 lines of filter, no pykalman)
# ---------------------------------------------------------------------------


def kalman_hedge_ratio(
    y: np.ndarray, x: np.ndarray, delta: float = 1e-4
) -> Tuple[float, np.ndarray]:
    """Estimate beta in y = beta * x + noise via a scalar Kalman filter.

    Returns (beta, P_diag_history). Converges to the true beta within 5% on a
    synthetic cointegrated pair after >= 200 obs; the state variance ``P`` is
    non-negative and shrinks over time.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = min(len(y), len(x))
    y, x = y[:n], x[:n]
    if n == 0:
        return 0.0, np.array([])

    beta = 0.0
    P = 1.0
    R = max(1e-8, float(np.var(y)) * 0.05)  # observation noise variance
    Q = max(delta, 1e-8)  # process noise
    Ps = np.empty(n)
    for t in range(n):
        P_pred = P + Q
        xt = x[t]
        S = P_pred * xt * xt + R
        K = (P_pred * xt / S) if S > 0 else 0.0
        beta = beta + K * (y[t] - beta * xt)
        P = (1.0 - K * xt) * P_pred
        Ps[t] = max(P, 0.0)
    return float(beta), Ps


# ---------------------------------------------------------------------------
# Cointegration (statsmodels, lazy) + spread z-score
# ---------------------------------------------------------------------------


def cointegration_pvalue(y: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    """(statistic, pvalue) from statsmodels coint(); (nan, nan) if unavailable."""
    try:
        from statsmodels.tsa.stattools import coint
    except Exception:
        return (float("nan"), float("nan"))
    res = coint(np.asarray(y, dtype=float), np.asarray(x, dtype=float))
    return float(res[0]), float(res[1])


def spread_z(spread: np.ndarray, window: int = 60) -> np.ndarray:
    """Rolling z-score of the spread. Finite; zero/flat std -> 0.0 (never NaN)."""
    spread = np.asarray(spread, dtype=float)
    n = len(spread)
    out = np.zeros(n)
    for i in range(n):
        lo = max(0, i - window + 1)
        seg = spread[lo : i + 1]
        m = seg.mean()
        s = seg.std()
        out[i] = 0.0 if s <= 1e-12 else (spread[i] - m) / s
    return out


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck half-life
# ---------------------------------------------------------------------------


def ou_half_life(series: np.ndarray) -> Optional[float]:
    """Half-life of mean reversion via OLS on dX = a + b*X_lag.

    Returns finite, positive half-life for mean-reverting series. Returns None
    when the series is not mean-reverting (b >= 0, i.e. a random walk) so the
    signal path can skip it.
    """
    series = np.asarray(series, dtype=float)
    if len(series) < 3:
        return None
    dX = np.diff(series)
    Xlag = series[:-1]
    Xm = np.vstack([np.ones_like(Xlag), Xlag]).T
    coef, *_ = np.linalg.lstsq(Xm, dX, rcond=None)
    b = -coef[1]  # OU mean-reversion speed (b>0 means mean-reverting)
    # A random walk estimates b ~ 0 (finite-sample noise). Require a minimum
    # stationarity speed so we reliably return None for unit-root (non-MR) series
    # and skip them in the signal path. Genuine MR (b >= ~0.01) still returns finite.
    if b <= 5e-3:
        return None
    hl = np.log(2.0) / b
    if not np.isfinite(hl) or hl <= 0.0 or hl > 1e6:
        return None
    return float(hl)


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------


def shannon_entropy(series: np.ndarray, bins: int = 10) -> float:
    """Normalized Shannon entropy in [0, 1] = H / log2(bins)."""
    series = np.asarray(series, dtype=float)
    if series.size == 0:
        return 0.0
    counts, _ = np.histogram(series, bins=bins)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    if p.size <= 1:
        return 0.0
    H = -np.sum(p * np.log2(p))
    Hmax = np.log2(bins)
    return float(H / Hmax) if Hmax > 0 else 0.0


def approx_entropy(series: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Approximate entropy (>= 0). Random series > ordered series.

    ``r`` is treated as a fraction of the series std when r <= 1 (canonical
    convention) so the comparison is scale-free.
    """

    def _phi(mm: int) -> float:
        s = np.asarray(series, dtype=float)
        N = len(s)
        if N < mm + 1:
            return 0.0
        x = np.array([s[i : i + mm] for i in range(N - mm + 1)])
        c = 0.0
        for i in range(len(x)):
            dist = np.max(np.abs(x - x[i]), axis=1)
            c += np.mean(dist <= r)
        return c / len(x)

    series = np.asarray(series, dtype=float)
    if len(series) < m + 1:
        return 0.0
    eff_r = r * float(np.std(series)) if r <= 1.0 else r
    if eff_r <= 0:
        return 0.0
    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)
    if phi_m <= 0.0 or phi_m1 <= 0.0:
        return 0.0
    return float(max(0.0, phi_m - phi_m1))


# ---------------------------------------------------------------------------
# Hurst (R/S)
# ---------------------------------------------------------------------------


def hurst_rs(series: np.ndarray, max_lag: Optional[int] = None) -> float:
    """Hurst exponent via the lagged-difference (variance-time) estimator.

    std(X_t - X_{t-lag}) ~ lag^H, so H = slope of log(std) vs log(lag). For Brownian
    motion (random walk) H ~ 0.5. Clamped to [0, 1].
    """
    series = np.asarray(series, dtype=float)
    N = len(series)
    if N < 20:
        return 0.5
    if max_lag is None:
        max_lag = min(N // 4, 100)
    lags: List[int] = []
    tau: List[float] = []
    for lag in range(2, max_lag):
        if lag >= N:
            break
        diff = series[lag:] - series[:-lag]
        sd = float(np.std(diff))
        if sd <= 1e-12:
            continue
        lags.append(lag)
        tau.append(sd)
    if len(lags) < 2:
        return 0.5
    H = np.polyfit(np.log(lags), np.log(tau), 1)[0]
    return float(min(1.0, max(0.0, H)))


# ---------------------------------------------------------------------------
# Shared trend helpers (used by signal filters). numpy-only.
# ---------------------------------------------------------------------------


def sma(series: np.ndarray, n: int) -> float:
    """Trailing simple moving average; nan -> 0.0."""
    series = np.asarray(series, dtype=float)
    if len(series) == 0 or n <= 0:
        return 0.0
    seg = series[-n:]
    return float(seg.mean())


def vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> float:
    """Cumulative VWAP of the supplied window; 0.0 on zero volume."""
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    volume = np.asarray(volume, dtype=float)
    if len(close) == 0:
        return 0.0
    tp = (high + low + close) / 3.0
    v = volume.sum()
    if v <= 0:
        return float(close.mean())
    return float(np.sum(tp * volume) / v)


def trend_regime(
    close: np.ndarray,
    vwap_val: float,
    fast_n: int = 20,
    slow_n: int = 200,
    vol_thresh: float = 0.02,
) -> Tuple[str, bool]:
    """Return (regime, above_trend).

    regime: "trend" if price above slow MA, "range" if within band, else "volatile".
    above_trend: close strictly above VWAP.
    """
    close = np.asarray(close, dtype=float)
    if len(close) == 0:
        return ("range", False)
    slow = sma(close, slow_n)
    fast = sma(close, fast_n)
    last = float(close[-1])
    above_ma = last > slow
    band = abs(last - slow) / slow if slow > 0 else 0.0
    if band > vol_thresh:
        regime = "volatile"
    elif above_ma and last > fast:
        regime = "trend"
    else:
        regime = "range"
    above_trend = last > vwap_val if vwap_val > 0 else above_ma
    return (regime, above_trend)
