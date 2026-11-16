"""Keyless local dry-run simulator (CLI ``paper --simulate``).

Synthesizes a random-walk OHLCV + volume tape (and a trivial depth) for each
configured symbol, replays the EXACT live signal path (``compute_features`` +
``signals_*.evaluate``), and synthesizes fills locally:

  * entry  -> at the signal price
  * exit   -> at the 2xATR stop, or on an opposite-side signal
  * cost   -> CostModel.fee() (brokerage + India sell-side STT)

No network, no broker keys. Prints a P&L summary and returns total P&L so the CLI
can exit 0. This is a v1 modeling simplification: exits are signal/stop-based, not
the full live reconcile loop.
"""
from __future__ import annotations

import numpy as np

from ..backtest.cost_model import CostModel
from ..config.settings import Settings, SymbolConfig
from ..data.contracts import Depth
from ..engine.live_loop import compute_features
from ..signals_microstructure import evaluate as micro_evaluate
from ..signals_volume import evaluate as vol_evaluate


def _synth_ohlcv(sym_cfg: SymbolConfig, n_bars: int, rng) -> tuple:
    """Low-volatility random walk + volume tape.

    Kept near its 200-MA so the (solvent) trend filter yields 'trend'/'range' rather
    than 'volatile', letting the existing signal fns actually fire in the dry-run.
    """
    base = 100.0 if sym_cfg.asset_class == "crypto" else 2500.0
    step = base * 0.0008
    level = base + np.cumsum(rng.normal(0, 1, n_bars)) * step
    noise = rng.normal(0, base * 0.0003, n_bars)
    close = level + noise
    atr_like = np.abs(rng.normal(0, 1, n_bars)) * (base * 0.0006) + base * 0.0003
    high = np.maximum.reduce([close, level]) + atr_like
    low = np.minimum.reduce([close, level]) - atr_like
    volume = rng.uniform(1e3, 1e4, n_bars)
    # Occasional volume spikes so vol_breakout / vol_climax can trigger.
    n_spikes = max(1, n_bars // 25)
    idx = rng.integers(0, n_bars, size=n_spikes)
    volume[idx] *= rng.uniform(2.5, 4.0, size=n_spikes)
    return close, high, low, volume


def _synth_depth(sym_cfg: SymbolConfig, close_arr: np.ndarray, i: int, rng) -> Depth:
    """Depth whose order-flow imbalance tracks the local price direction.

    This mirrors the microstructure premise (imbalance leads price) so the trend
    filter aligns and microstructure signals can fire in the dry-run.
    """
    mid = float(close_arr[i])
    spread = max(mid * 0.0001, 1e-8)
    ret = 0.0 if i < 2 else (close_arr[i] - close_arr[i - 2]) / (abs(close_arr[i - 2]) + 1e-9)
    # Scale imbalance with the local return so it clears the 0.5 microstructure
    # threshold during real moves (and stays sub-threshold when flat).
    mag = min(0.85, 0.4 + abs(ret) * 200.0)
    imb = float(np.clip(np.sign(ret) * mag, -0.85, 0.85)) + rng.normal(0, 0.05)
    imb = float(np.clip(imb, -0.9, 0.9))
    bid_sz = 10.0 * (1.0 + imb)
    ask_sz = 10.0 * (1.0 - imb)
    return Depth(
        symbol=sym_cfg.ticker,
        ts=0,
        bids=[(mid - spread, bid_sz)],
        asks=[(mid + spread, ask_sz)],
        is_snapshot=True,
    )


def _replay_symbol(sym_cfg, settings, close, high, low, volume, cost, rng) -> tuple:
    n = len(close)
    pos_side = None  # "long" | "short" | None
    pos_qty = 0.0
    entry = 0.0
    stop = 0.0
    pnl = 0.0
    n_trades = 0

    for i in range(n):
        if i < 200:  # warm-up for slow MA / Hurst
            continue
        depth = _synth_depth(sym_cfg, close, i, rng)
        fb = compute_features(
            symbol=sym_cfg.ticker,
            ts=i * 60_000,
            close=close[: i + 1],
            high=high[: i + 1],
            low=low[: i + 1],
            volume=volume[: i + 1],
            depth=depth,
            ticks=[],
            cfg=settings,
        )
        sig = (
            micro_evaluate(fb, settings.risk, settings.equity_base, sym_cfg.strategies)
            if sym_cfg.asset_class == "crypto"
            else vol_evaluate(fb, settings.risk, settings.equity_base, sym_cfg.strategies)
        )
        price = float(close[i])

        # --- manage open position: stop hit or opposite signal closes it ---
        if pos_side is not None:
            hit_stop = (pos_side == "long" and price <= stop) or (
                pos_side == "short" and price >= stop
            )
            opp = sig is not None and sig.side != pos_side
            if hit_stop or opp:
                exit_px = stop if hit_stop else price
                notional_in = pos_qty * entry
                gross = (exit_px - entry) * pos_qty if pos_side == "long" else (entry - exit_px) * pos_qty
                fee = cost.fee(
                    "sell" if pos_side == "long" else "buy", notional_in, sym_cfg.asset_class
                )
                pnl += gross - fee
                n_trades += 1
                pos_side = None
                pos_qty = 0.0
                if opp:
                    pass  # fall through to open the opposite side below
                else:
                    continue

        # --- open a new position if flat ---
        if sig is not None and pos_side is None:
            side = sig.side
            if sym_cfg.asset_class == "crypto":
                qty = (settings.equity_base * sig.size_pct_equity) / price
            else:
                qty = (settings.equity_base * sig.size_pct_equity) / (price * sym_cfg.lot_size)
            notional = qty * price
            fee = cost.fee(
                "buy" if side == "long" else "sell", notional, sym_cfg.asset_class
            )
            pnl -= fee  # entry cost
            pos_side = side
            pos_qty = qty
            entry = price
            stop = sig.stop_price

    # Close any position still open at the final bar.
    if pos_side is not None:
        exit_px = float(close[-1])
        notional_in = pos_qty * entry
        gross = (exit_px - entry) * pos_qty if pos_side == "long" else (entry - exit_px) * pos_qty
        fee = cost.fee("sell" if pos_side == "long" else "buy", notional_in, sym_cfg.asset_class)
        pnl += gross - fee
        n_trades += 1

    return pnl, n_trades


def run_simulate(settings: Settings, n_bars: int = 500, seed: int = 42) -> float:
    """Run the keyless dry-run and print a P&L summary. Returns total P&L."""
    rng = np.random.default_rng(seed)
    cost = CostModel(settings)
    summary: dict = {}
    total_pnl = 0.0

    for sym_cfg in settings.symbols:
        if not sym_cfg.enabled:
            continue
        close, high, low, volume = _synth_ohlcv(sym_cfg, n_bars, rng)
        pnl, n_trades = _replay_symbol(sym_cfg, settings, close, high, low, volume, cost, rng)
        summary[sym_cfg.ticker] = {"pnl": pnl, "trades": n_trades}
        total_pnl += pnl

    print("\n=== autobot paper --simulate (keyless local dry-run) ===")
    for tk, info in summary.items():
        print(f"  {tk:12s}  trades={info['trades']:4d}  pnl={info['pnl']:.2f}")
    tot_trades = sum(i["trades"] for i in summary.values())
    print(f"  {'TOTAL':12s}  trades={tot_trades:4d}  pnl={total_pnl:.2f}")
    print(
        f"  starting equity={settings.equity_base:.2f}  "
        f"ending equity={settings.equity_base + total_pnl:.2f}"
    )
    return total_pnl
