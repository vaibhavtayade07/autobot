"""Microstructure math. numpy-only, no I/O, no global state.

All functions here are pure: list/tuple/ndarray in, float/ndarray out, and every
return honors the SPEC §3 invariants (imbalance in [-1,1], microprice strictly
inside the spread, VPIN in [0,1], Amihud >= 0).
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

Book = List[Tuple[float, float]]


def _trade_size(t: object) -> float:
    if isinstance(t, (tuple, list)):
        return float(t[1])
    return float(getattr(t, "size"))


def _trade_side(t: object) -> str:
    if isinstance(t, (tuple, list)):
        return str(t[2])
    return str(getattr(t, "side"))


def order_book_imbalance(bids: Book, asks: Book, levels: int = 5) -> float:
    """Size-weighted imbalance in [-1, 1]; 0.0 on empty / zero-size book."""
    bid_vol = float(sum(sz for _, sz in bids[:levels]))
    ask_vol = float(sum(sz for _, sz in asks[:levels]))
    total = bid_vol + ask_vol
    if total <= 0.0:
        return 0.0
    return (bid_vol - ask_vol) / total


def microprice(bid: float, bid_sz: float, ask: float, ask_sz: float) -> float:
    """Strictly inside (bid, ask) when both sizes > 0; returns mid if either is 0."""
    if bid_sz <= 0.0 or ask_sz <= 0.0:
        return 0.5 * (bid + ask)
    # Convex combination -> strictly between the two extremes.
    return (ask_sz * bid + bid_sz * ask) / (bid_sz + ask_sz)


def weighted_mid(bids: Book, asks: Book, levels: int = 5) -> float:
    """Sum(price*size) / Sum(size) over both sides; falls back to top-of-book mid."""
    all_levels = [(p, sz) for p, sz in bids[:levels]] + [(p, sz) for p, sz in asks[:levels]]
    total_sz = float(sum(sz for _, sz in all_levels))
    if total_sz <= 0.0:
        if bids and asks:
            return 0.5 * (bids[0][0] + asks[0][0])
        return 0.0
    return float(sum(p * sz for p, sz in all_levels) / total_sz)


def book_pressure(bids: Book, asks: Book, levels: int = 5) -> float:
    """(bid_vol - ask_vol) / (bid_vol + ask_vol) in [-1, 1]; 0.0 on empty."""
    bid_vol = float(sum(sz for _, sz in bids[:levels]))
    ask_vol = float(sum(sz for _, sz in asks[:levels]))
    total = bid_vol + ask_vol
    if total <= 0.0:
        return 0.0
    return (bid_vol - ask_vol) / total


def vpin(trades: Sequence[object], bucket_size: float, window: int = 50) -> float:
    """Volume-Synchronized Probability of Informed Trading in [0, 1].

    Trades are bucketed by unsigned volume; each bucket's order-flow imbalance is
    |signed_vol| / bucket_vol (which is <= 1). VPIN is the mean over the trailing
    ``window`` buckets. Zero total volume -> 0.0 (no div-by-zero).
    """
    if bucket_size <= 0.0:
        return 0.0

    signed: List[float] = []
    vols: List[float] = []
    for t in trades:
        sz = _trade_size(t)
        side = _trade_side(t)
        s = 1.0 if side in ("buy", "b", "BUY", "Buy") else -1.0
        signed.append(s * sz)
        vols.append(abs(sz))

    total = float(sum(vols))
    if total <= 0.0:
        return 0.0

    ratios: List[float] = []
    idx = 0
    n = len(signed)
    while idx < n:
        bvol = 0.0
        bsigned = 0.0
        while idx < n and bvol < bucket_size:
            remaining = bucket_size - bvol
            v = vols[idx]
            if v <= remaining:
                bvol += v
                bsigned += signed[idx]
                idx += 1
            else:
                frac = remaining / v
                bvol += remaining
                bsigned += signed[idx] * frac
                vols[idx] = v - remaining
                signed[idx] = signed[idx] * (1.0 - frac)
                break
        if bvol > 0.0:
            ratios.append(abs(bsigned) / bvol)

    if not ratios:
        return 0.0
    recent = ratios[-window:]
    return float(np.mean(recent))


def amihud(returns: np.ndarray, volumes: np.ndarray) -> float:
    """Amihud illiquidity = mean(|return| / volume); >= 0. Zero-vol bars contribute 0."""
    returns = np.asarray(returns, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    if returns.size == 0:
        return 0.0
    mask = volumes > 0
    if not np.any(mask):
        return 0.0
    illiq = np.abs(returns[mask]) / volumes[mask]
    return float(np.mean(illiq))
