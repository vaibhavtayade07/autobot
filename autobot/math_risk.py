"""Risk math + guards. numpy-only at import (stdlib time for the token bucket).

Functions: Kelly fraction (+ fractional), Wilder ATR (matches pandas-ta within 1e-6),
2xATR stop, risk-based size fraction (1%-equity cap), DrawdownGuard (latching
kill-switch), and TokenBucket (20/s cap for Angel).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np
from collections import deque


# ---------------------------------------------------------------------------
# Kelly
# ---------------------------------------------------------------------------


def kelly_fraction(win_rate: float, payoff: float) -> float:
    """f* = win_rate - (1 - win_rate)/payoff, clamped to [0, 1]. Negative edge -> 0.0."""
    if payoff <= 0.0 or win_rate <= 0.0:
        return 0.0
    if win_rate >= 1.0:
        return 1.0
    f = win_rate - (1.0 - win_rate) / payoff
    return float(min(1.0, max(0.0, f)))


def fractional_kelly(win_rate: float, payoff: float, fraction: float = 0.25) -> float:
    """Kelly scaled by ``fraction`` (e.g. 0.25 cap), still clamped to [0, 1]."""
    f = kelly_fraction(win_rate, payoff)
    return float(min(1.0, max(0.0, f * fraction)))


# ---------------------------------------------------------------------------
# ATR + stop
# ---------------------------------------------------------------------------


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> float:
    """Wilder's ATR (matches pandas_ta.ATR within 1e-6)."""
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    m = len(close)
    if m == 0:
        return 0.0
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce(
        [high - low, np.abs(high - prev_close), np.abs(low - prev_close)]
    )
    if m <= n:
        return float(tr.mean())
    out = np.zeros(m)
    out[n - 1] = float(tr[:n].mean())
    for i in range(n, m):
        out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return float(out[-1])


def stop_price(entry: float, side: str, atr_val: float, mult: float = 2.0) -> float:
    """2xATR stop (default mult=2.0). long -> entry - mult*ATR, short -> entry + mult*ATR."""
    if atr_val < 0:
        atr_val = 0.0
    if side == "long":
        return entry - mult * atr_val
    return entry + mult * atr_val


# ---------------------------------------------------------------------------
# Position sizing (1%-equity risk, Kelly cap, <=1% notional cap)
# ---------------------------------------------------------------------------


def risk_based_size_pct(
    equity: float,
    risk_per_trade_pct: float,
    entry: float,
    stop_px: float,
    kelly_frac: float = 0.25,
    hard_cap: float = 0.01,
) -> float:
    """Fraction of equity to deploy.

    Sized so the stop distance risks exactly ``risk_per_trade_pct`` of equity, then
    capped by the Kelly fraction and a hard 1% notional cap. Never NaN/negative.
    """
    if equity <= 0.0 or entry <= 0.0:
        return 0.0
    stop_dist = abs(entry - stop_px)
    if stop_dist <= 1e-12:
        return 0.0
    risk_amount = equity * risk_per_trade_pct
    notional = (risk_amount / stop_dist) * entry  # units * entry
    size_pct = notional / equity
    size_pct = min(size_pct, kelly_frac, hard_cap)
    return float(max(0.0, size_pct))


# ---------------------------------------------------------------------------
# Drawdown kill-switch
# ---------------------------------------------------------------------------


@dataclass
class DrawdownGuard:
    """Latches True the moment drawdown from peak >= max_dd_pct."""

    max_dd_pct: float
    peak: float = 0.0
    latched: bool = False

    def update(self, equity: float) -> bool:
        if equity > self.peak:
            self.peak = equity
        if self.peak <= 0:
            dd = 0.0
        else:
            dd = (self.peak - equity) / self.peak
        if dd >= self.max_dd_pct:
            self.latched = True
            return True
        return False

    def kill(self) -> bool:
        return self.latched


# ---------------------------------------------------------------------------
# Token bucket (rate limit)
# ---------------------------------------------------------------------------


class TokenBucket:
    """Leaky/token bucket. admit() -> True if a token is available this instant.

    Defaults to 20/s / capacity 20 (Angel cap). Exceeding capacity simply refuses;
    the caller should queue and retry.
    """

    def __init__(self, rate: float = 20.0, capacity: float = 20.0):
        self.rate = rate
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last = time.monotonic()
        self._queue: Deque = deque()

    def admit(self) -> bool:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
        self.last = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

    def queue(self, item: object) -> None:
        self._queue.append(item)

    def pop_queued(self) -> Optional[object]:
        if self._queue:
            return self._queue.popleft()
        return None

    def pending(self) -> int:
        return len(self._queue)
