"""Walk-forward backtest harness (DESIGN §7). Reuses the LIVE signal fns.

The same ``compute_features`` + ``signals_*.evaluate`` used in the live loop are
replayed over historical OHLCV, so there is a single source of truth for signal
logic. vectorbt is imported lazily; if it (or data) is unavailable the harness
reports and exits cleanly.

IMPORTANT: India historical data is rate-limited/sparse via Angel; supply a saved
tick/OHLCV CSV for India names. Crypto uses ccxt.fetchOHLCV lazily.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..config.settings import Settings, SymbolConfig
from ..data.contracts import FeatureBundle
from ..engine.live_loop import compute_features
from ..signals_microstructure import evaluate as micro_evaluate
from ..signals_volume import evaluate as vol_evaluate
from .cost_model import CostModel


def load_ohlcv(symbol: str, sym_cfg: SymbolConfig, settings: Settings) -> Optional[pd.DataFrame]:
    """Load OHLCV. Crypto via ccxt lazily; India from a local CSV path (saved tape)."""
    if sym_cfg.asset_class == "crypto":
        try:
            import ccxt  # lazy

            ex = ccxt.binance({"enableRateLimit": True})
            ohlcv = ex.fetch_ohlcv(
                sym_cfg.ticker, settings.backtest.freq, limit=2000
            )
            df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
            return df
        except Exception as e:
            print(f"[backtest] crypto load failed for {symbol}: {e}")
            return None
    else:
        # India: expect a local CSV at data/<ticker>.csv with OHLCV columns.
        path = f"data/{sym_cfg.ticker}.csv"
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"[backtest] india load failed for {symbol} (expected {path}): {e}")
            return None


def signal_series(
    df: pd.DataFrame, sym_cfg: SymbolConfig, settings: Settings
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Replay the live signal path and return (long_entries, long_exits,
    short_entries, short_exits) boolean arrays.

    Long and short signals are tracked in SEPARATE arrays so vectorbt's
    ``from_signals(..., direction="both", short_entries=..., short_exits=...)``
    prices them correctly (a SHORT is no longer treated as a LONG — QA #5).
    """
    n = len(df)
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    ts0 = int(df["ts"].iloc[0]) if "ts" in df.columns else 0

    signals: list = []
    for i in range(n):
        if i < 200:  # warm-up for slow MA / Hurst
            continue
        fb = compute_features(
            symbol=sym_cfg.ticker,
            ts=ts0 + i * 60_000,
            close=close[: i + 1],
            high=high[: i + 1],
            low=low[: i + 1],
            volume=volume[: i + 1],
            depth=_empty_depth(),
            ticks=[],
            cfg=settings,
        )
        sig = (
            micro_evaluate(fb, settings.risk, settings.equity_base, sym_cfg.strategies)
            if sym_cfg.asset_class == "crypto"
            else vol_evaluate(fb, settings.risk, settings.equity_base, sym_cfg.strategies)
        )
        if sig is None:
            continue
        signals.append((i, sig.side))
    return _build_signal_arrays(n, signals)


def _build_signal_arrays(
    n: int, signals: list
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pure mapping of (bar_index, side) pairs -> long/short entry/exit arrays.

    Exit is forced on the bar after entry (OOS shape), and an opposite signal also
    closes the open position on the same bar. Long and short never overlap per bar.
    """
    long_entries = np.zeros(n, dtype=bool)
    long_exits = np.zeros(n, dtype=bool)
    short_entries = np.zeros(n, dtype=bool)
    short_exits = np.zeros(n, dtype=bool)
    prev = None
    for j, side in signals:
        if side == "long":
            long_entries[j] = True
            if prev == "short":
                short_exits[j] = True
            if j + 1 < n:
                long_exits[j + 1] = True
        else:  # short
            short_entries[j] = True
            if prev == "long":
                long_exits[j] = True
            if j + 1 < n:
                short_exits[j + 1] = True
        prev = side
    return long_entries, long_exits, short_entries, short_exits


def _empty_depth():
    from ..data.contracts import Depth

    return Depth(symbol="", ts=0, bids=[], asks=[], is_snapshot=False)


def run_walkforward(symbol: str, settings: Settings) -> Optional[dict]:
    sym_cfg = next((s for s in settings.symbols if s.ticker == symbol and s.enabled), None)
    if sym_cfg is None:
        print(f"[backtest] symbol {symbol} not enabled")
        return None
    df = load_ohlcv(symbol, sym_cfg, settings)
    if df is None or len(df) < 250:
        print(f"[backtest] insufficient data for {symbol}")
        return None

    long_entries, long_exits, short_entries, short_exits = signal_series(df, sym_cfg, settings)
    try:
        import vectorbt as vbt  # lazy
    except Exception as e:
        print(f"[backtest] vectorbt unavailable: {e}")
        return None

    cost = CostModel(settings)
    # Per-side costs: buy vs sell differ (India sell-side STT). Convert the absolute
    # CostModel.fee() into a fraction of notional for vectorbt's (buy_fee, sell_fee). (#6)
    notional = settings.equity_base
    buy_fee = cost.fee("buy", notional, sym_cfg.asset_class) / notional
    sell_fee = cost.fee("sell", notional, sym_cfg.asset_class) / notional
    close = df["close"].to_numpy(dtype=float)
    pf = vbt.Portfolio.from_signals(
        close,
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        direction="both",  # long AND short priced independently (#5)
        fees=(buy_fee, sell_fee),  # per-side costs (#6)
        freq=settings.backtest.freq,
    )
    metrics = {
        "sharpe": float(pf.sharpe_ratio()),
        "sortino": float(pf.sortino_ratio()),
        "max_drawdown": float(pf.max_drawdown()),
        "total_return": float(pf.total_return()),
    }
    return metrics


def run_backtest(settings: Settings) -> dict:
    results = {}
    for sym in settings.symbols:
        if not sym.enabled:
            continue
        m = run_walkforward(sym.ticker, settings)
        if m is not None:
            results[sym.ticker] = m
            print(f"[backtest] {sym.ticker}: {m}")
    return results
