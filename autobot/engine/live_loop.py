"""Live loop + feature computation + state machine (DESIGN §2/§3).

compute_features() turns a MarketFrame + rolling OHLCV history into a FeatureBundle
using only the local math_* modules. The asyncio state machine INIT -> CONNECTED ->
TRADING/FLAT (orthogonal THROTTLED/PAUSED) -> KILL routes frames through
signals -> risk -> token bucket -> broker.

Heavy deps (ccxt/vectorbt) are imported lazily inside the functions that need them,
so this module is importable with numpy/pandas/scipy only.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..data.contracts import (
    Depth,
    FeatureBundle,
    MarketFrame,
    Order,
    Position,
    RiskDecision,
    Signal,
    Tick,
)
from ..config.settings import Settings, SymbolConfig
from ..math_microstructure import (
    amihud,
    microprice,
    order_book_imbalance,
    vpin,
    weighted_mid,
)
from ..math_stats import (
    approx_entropy,
    hurst_rs,
    shannon_entropy,
    sma,
    trend_regime,
    vwap,
)
from ..math_vol import (
    garch_11,
    obv,
    obv_slope,
    volume_profile,
    volume_ratio,
)
from ..math_risk import atr, DrawdownGuard, TokenBucket
from ..signals_microstructure import evaluate as micro_evaluate
from ..signals_volume import evaluate as vol_evaluate
from .session_gate import IndiaSession

STATE_DIR = Path(os.environ.get("AUTOBO_STATE", "state"))
AUDIT_PATH = STATE_DIR / "audit.jsonl"
KILL_FLAG = STATE_DIR / "kill_flag"


class State(str, Enum):
    INIT = "INIT"
    CONNECTED = "CONNECTED"
    TRADING = "TRADING"
    FLAT = "FLAT"
    THROTTLED = "THROTTLED"
    PAUSED = "PAUSED"
    KILL = "KILL"


# ---------------------------------------------------------------------------
# Feature computation (local math only)
# ---------------------------------------------------------------------------


def compute_features(
    symbol: str,
    ts: int,
    close,
    high,
    low,
    volume,
    depth: Depth,
    ticks: List[Tick],
    cfg: Settings,
) -> FeatureBundle:
    close = np.asarray(close, dtype=float)
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    volume = np.asarray(volume, dtype=float)
    n = len(close)

    returns = np.diff(close) if n > 1 else np.array([0.0])
    vwap_val = vwap(high, low, close, volume) if volume.sum() > 0 else float(close[-1]) if n else 0.0
    sma_fast = sma(close, 20)
    sma_slow = sma(close, 200)

    imb = order_book_imbalance(depth.bids, depth.asks, 5) if depth else 0.0
    if depth and depth.bids and depth.asks:
        mp = microprice(depth.bids[0][0], depth.bids[0][1], depth.asks[0][0], depth.asks[0][1])
    else:
        mp = weighted_mid(depth.bids, depth.asks, 5) if depth else (float(close[-1]) if n else 0.0)
    wmid = weighted_mid(depth.bids, depth.asks, 5) if depth else mp

    if ticks:
        avg_sz = float(np.mean([t.size for t in ticks])) or 1.0
        bucket = max(avg_sz * 10.0, 1e-9)
        vpin_val = vpin(ticks, bucket, window=50)
    else:
        vpin_val = 0.0

    hurst = hurst_rs(close) if n >= 20 else 0.5
    shannon = shannon_entropy(returns) if returns.size else 0.0
    apen = approx_entropy(close) if n >= 3 else 0.0
    amihud_val = amihud(returns, volume[-len(returns):]) if (returns.size and volume.sum() > 0) else 0.0

    if returns.size >= 2:
        gvar, gunstable = garch_11(returns)
    else:
        gvar, gunstable = 0.0, False

    if n > 0:
        poc, val, vah = volume_profile(close[-200:], volume[-200:])
    else:
        poc = val = vah = 0.0

    atr_val = atr(high, low, close, 14) if n >= 2 else 0.0
    obv_series = obv(close, volume) if n else np.array([0.0])
    obv_s = obv_slope(obv_series)
    vr_arr = volume_ratio(volume) if n else np.array([0.0])
    vol_ratio = float(vr_arr[-1]) if vr_arr.size else 0.0

    regime, _ = trend_regime(close, vwap_val) if n else ("range", False)

    return FeatureBundle(
        symbol=symbol,
        ts=ts,
        close=float(close[-1]) if n else 0.0,
        vwap=float(vwap_val),
        sma_fast=sma_fast,
        sma_slow=sma_slow,
        imbalance=imb,
        microprice=mp,
        weighted_mid=wmid,
        vpin=vpin_val,
        hurst=hurst,
        shannon=shannon,
        apen=apen,
        amihud=amihud_val,
        garch_var=gvar,
        garch_unstable=gunstable,
        vp_poc=poc,
        vp_val=val,
        vp_vah=vah,
        atr=atr_val,
        obv=float(obv_series[-1]) if obv_series.size else 0.0,
        obv_slope=obv_s,
        kalman_beta=0.0,
        ou_half_life=None,
        spread_z=0.0,
        volume_ratio=vol_ratio,
        regime=regime,
        raw_depth=depth,
        raw_ticks=ticks,
    )


# ---------------------------------------------------------------------------
# Audit (JSONL) — no heavy deps
# ---------------------------------------------------------------------------


def write_audit(record: dict) -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass


def write_kill_flag(reason: str) -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(KILL_FLAG, "w", encoding="utf-8") as f:
            f.write(f"KILLED {int(time.time()*1000)} {reason}\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Risk gate
# ---------------------------------------------------------------------------


def risk_gate(
    signal: Signal,
    equity: float,
    risk: Settings,
    asset_class: Optional[str] = None,
    lot_size: float = 1.0,
) -> RiskDecision:
    if signal.stop_price is None or not np.isfinite(signal.stop_price) or signal.stop_price == 0.0:
        return RiskDecision(order=None, approved=False, reason="missing stop")
    if signal.size_pct_equity > 0.01:
        return RiskDecision(order=None, approved=False, reason="size exceeds 1% cap")
    if signal.price is None or not np.isfinite(signal.price) or signal.price <= 0:
        return RiskDecision(order=None, approved=False, reason="invalid price")
    ac = asset_class or ("crypto" if signal.symbol.endswith("/USDT") else "india")
    # Size layer: translate size_pct_equity into a real order quantity.
    if ac == "crypto":
        quantity = (equity * signal.size_pct_equity) / signal.price
    else:
        # India notional is priced per lot; scale by config lot_size.
        quantity = (equity * signal.size_pct_equity) / (signal.price * lot_size)
    if not np.isfinite(quantity) or quantity <= 0:
        return RiskDecision(order=None, approved=False, reason="invalid quantity")

    order = Order(
        id=str(uuid.uuid4()),
        signal_id=signal.id,
        symbol=signal.symbol,
        side="buy" if signal.side == "long" else "sell",
        order_type="market",
        quantity=quantity,
        price=signal.price,
        broker="ccxt" if ac == "crypto" else "india",
        status="pending",
        ts_created=signal.ts,
        ts_updated=signal.ts,
    )
    # Estimated entry cost (brokerage + India sell-side STT). Logged so the edge
    # check can subtract it; at minimum it is recorded in the audit trail (#7).
    est_cost = 0.0
    try:
        from ..backtest.cost_model import CostModel

        notional = quantity * signal.price
        est_cost = CostModel(risk).fee(order.side, notional, ac)
    except Exception:
        pass
    return RiskDecision(order=order, approved=True, reason="ok", est_cost=est_cost)


# ---------------------------------------------------------------------------
# State machine / live orchestration
# ---------------------------------------------------------------------------


class LiveLoop:
    def __init__(self, settings: Settings, paper: bool):
        self.settings = settings
        self.paper = paper
        self.state = State.INIT
        self.raw_q: "asyncio.Queue[object]" = asyncio.Queue()
        self.feature_q: "asyncio.Queue[MarketFrame]" = asyncio.Queue()
        self.kill_event = asyncio.Event()
        self.session = IndiaSession(settings)
        self.history: Dict[str, dict] = {}
        self.buckets: Dict[str, TokenBucket] = {}
        self.dd_guards: Dict[str, DrawdownGuard] = {}
        self.equity = settings.equity_base
        self.paused: Dict[str, bool] = {}
        self.last_received_ts: Dict[str, float] = {}
        self.positions: Dict[str, Position] = {}
        self._depth_buf: Dict[str, Optional[Depth]] = {}
        self._tick_buf: Dict[str, List[Tick]] = {}
        self._adapters: Dict[str, object] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._kill_thread: Optional[threading.Thread] = None

    # -- adapters (lazy import so ccxt/smartapi aren't required to import) --
    def _build_adapters(self):
        from ..brokers.ccxt_adapter import CCXTAdapter
        from ..brokers.india_adapter import IndiaAdapter

        adapters = {}
        for sym in self.settings.symbols:
            if not sym.enabled:
                continue
            if sym.asset_class == "crypto":
                adapters[sym.ticker] = CCXTAdapter(self.settings, paper=self.paper)
                self.buckets[sym.ticker] = TokenBucket(rate=100, capacity=100)
            else:
                adapters[sym.ticker] = IndiaAdapter(self.settings, paper=self.paper)
                self.buckets[sym.ticker] = TokenBucket(
                    rate=self.settings.risk.order_rate_limit,
                    capacity=self.settings.risk.order_rate_limit,
                )
            self.dd_guards[sym.ticker] = DrawdownGuard(self.settings.risk.max_drawdown_pct)
        return adapters

    def _push_raw(self, msg: object) -> None:
        self.raw_q.put_nowait(msg)

    def _update_history(self, sym: str, frame: MarketFrame) -> dict:
        h = self.history.setdefault(
            sym, {"close": [], "high": [], "low": [], "volume": []}
        )
        d = frame.depth
        if d and d.bids and d.asks:
            close_p = weighted_mid(d.bids, d.asks, 5)
            high_p = d.asks[0][0]
            low_p = d.bids[0][0]
        elif d and d.bids:
            close_p = high_p = low_p = d.bids[0][0]
        elif d and d.asks:
            close_p = high_p = low_p = d.asks[0][0]
        else:
            close_p = high_p = low_p = 0.0
        vol = float(sum(t.size for t in frame.ticks))
        for k, v in (("close", close_p), ("high", high_p), ("low", low_p), ("volume", vol)):
            h[k].append(v)
            if len(h[k]) > 500:
                h[k].pop(0)
        return h

    async def _marketdata_task(self, adapters):
        while not self.kill_event.is_set():
            try:
                msg = await asyncio.wait_for(self.raw_q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # No market data arrived this second: check staleness. If a symbol
                # has not been heard from in >30s, enter PAUSED and stop producing
                # frames (so the consumer stops sending orders) until data resumes.
                now = time.monotonic()
                for sym in list(self._depth_buf.keys()):
                    if now - self.last_received_ts.get(sym, now) > 30.0:
                        if not self.paused.get(sym):
                            self.paused[sym] = True
                            self.state = State.PAUSED
                            write_audit({"type": "paused", "symbol": sym, "reason": "stale data"})
                    else:
                        if self.paused.get(sym):
                            self.paused[sym] = False
                            if self.state == State.PAUSED:
                                self.state = State.TRADING
                continue

            if isinstance(msg, (Depth, Tick)):
                sym = msg.symbol
                if not self.session.is_open(sym, msg.ts):
                    continue  # India off-hours/holiday -> drop, zero compute
                now = time.monotonic()
                self.last_received_ts[sym] = now  # wall-clock of last real message
                if sym not in self._depth_buf:
                    self._depth_buf[sym] = None
                    self._tick_buf[sym] = []
                if isinstance(msg, Depth):
                    self._depth_buf[sym] = msg
                else:
                    self._tick_buf[sym].append(msg)
                    if len(self._tick_buf[sym]) > 200:
                        self._tick_buf[sym].pop(0)
                depth = self._depth_buf[sym]
                ticks = self._tick_buf[sym]
                if depth is None:
                    continue
                if self.paused.get(sym):
                    # data just resumed -> clear the paused flag
                    self.paused[sym] = False
                    if self.state == State.PAUSED:
                        self.state = State.TRADING
                frame = MarketFrame(
                    symbol=sym,
                    ts=msg.ts,
                    depth=depth,
                    ticks=list(ticks),
                    session_open=True,
                    healthy=True,
                )
                self.feature_q.put_nowait(frame)

    _depth_buf: Dict[str, Optional[Depth]]
    _tick_buf: Dict[str, List[Tick]]

    async def _consumer_task(self):
        while not self.kill_event.is_set():
            # (1) Drain any throttled (queued) orders whose rate-limit has recovered.
            for sym, bucket in list(self.buckets.items()):
                if bucket.pending() > 0:
                    await self._drain_queue(sym, bucket)

            frame = await self.feature_q.get()
            sym = frame.symbol
            h = self._update_history(sym, frame)
            feats = compute_features(
                sym, frame.ts, h["close"], h["high"], h["low"], h["volume"],
                frame.depth, frame.ticks, self.settings,
            )
            sym_cfg = next((s for s in self.settings.symbols if s.ticker == sym), None)
            if sym_cfg is None:
                continue
            signal: Optional[Signal] = None
            if "crypto" == sym_cfg.asset_class:
                signal = micro_evaluate(feats, self.settings.risk, self.equity, sym_cfg.strategies)
            else:
                signal = vol_evaluate(feats, self.settings.risk, self.equity, sym_cfg.strategies)
            if signal is None:
                continue

            # Read live equity from the broker at decision time (fall back to cached).
            equity = await self._read_equity(sym, sym_cfg)

            # Drawdown latch check in the fill/position-reconcile path (#1d).
            self._reconcile_equity(sym, equity)

            decision = risk_gate(
                signal, equity, self.settings,
                asset_class=sym_cfg.asset_class, lot_size=sym_cfg.lot_size,
            )
            write_audit({
                "type": "risk",
                "signal_id": signal.id,
                "approved": decision.approved,
                "reason": decision.reason,
                "est_cost": decision.est_cost,
            })
            if not decision.approved:
                continue

            bucket = self.buckets.get(sym)
            if bucket is not None and not bucket.admit():
                self.state = State.THROTTLED
                write_audit({"type": "throttled", "signal_id": signal.id})
                bucket.queue(decision.order)
                continue
            order = await self._send(sym, decision.order)
            write_audit({"type": "order", "signal_id": signal.id, "order": order.__dict__})
            if order.status == "acked":
                self.state = State.TRADING

    async def _read_equity(self, sym: str, sym_cfg) -> float:
        """Pull current equity from the broker if the adapter supports it."""
        adapter = self._adapters.get(sym)
        if adapter is None:
            return self.equity
        try:
            funds = await asyncio.wait_for(adapter.get_funds(), timeout=2.0)
            eq = self._extract_equity(funds, sym_cfg)
            if eq is not None and eq > 0:
                self.equity = eq
                return eq
        except Exception:
            pass
        return self.equity

    def _extract_equity(self, funds, sym_cfg) -> Optional[float]:
        if not isinstance(funds, dict) or not funds:
            return None
        if sym_cfg.asset_class == "crypto":
            quote = sym_cfg.ticker.split("/")[-1]
            v = funds.get(quote)
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, dict):
                return float(v.get("free", 0) or 0)
        return None

    def _reconcile_equity(self, sym: str, equity: float) -> None:
        """Update the per-symbol drawdown guard; latch -> request kill + flatten (#1d)."""
        guard = self.dd_guards.get(sym)
        if guard is None:
            return
        if guard.update(equity):
            write_audit({"type": "drawdown_kill", "symbol": sym, "equity": equity})
            self._request_kill()

    async def _drain_queue(self, sym: str, bucket) -> None:
        """Re-send queued orders once the token bucket has capacity (#4)."""
        while bucket.pending() > 0:
            if not bucket.admit():
                break  # still throttled -> leave the rest queued for next tick
            order = bucket.pop_queued()
            sent = await self._send(sym, order)
            write_audit({"type": "redeliver", "symbol": sym, "order": sent.__dict__})
            if sent.status == "acked":
                self.state = State.TRADING
            else:
                # Send failed (e.g. transient reject); re-queue and retry next tick.
                bucket.queue(sent)
                break

    async def _send(self, sym: str, order: Order) -> Order:
        # adapter lookup is resolved by the caller; send_order is the only hot-path broker call
        adapter = self._adapters.get(sym)
        if adapter is None:
            order.status = "rejected"
            order.reject_reason = "no adapter"
            return order
        return await adapter.send_order(order)

    # ------------------------------------------------------------------
    # Kill switch wiring (#1a/b/c/e)
    # ------------------------------------------------------------------

    def _request_kill(self) -> None:
        """Thread/loop-safe trigger of the kill event."""
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self.kill_event.set)
        else:
            try:
                self.kill_event.set()
            except RuntimeError:
                pass

    def _install_signal_handlers(self, loop) -> None:
        self._loop = loop
        try:
            loop.add_signal_handler(signal.SIGINT, self._request_kill)
            loop.add_signal_handler(signal.SIGTERM, self._request_kill)
        except (NotImplementedError, RuntimeError, ValueError):
            # Windows / proactor loop may not support add_signal_handler; fall back
            # to the stdlib signal module (handler runs in the main thread).
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    signal.signal(sig, lambda *_a: self._request_kill())
                except (ValueError, OSError, RuntimeError):
                    pass

    def _start_kill_watcher(self, loop) -> None:
        """Background thread that watches state/kill_flag and trips the event (#1b)."""
        self._loop = loop
        self._kill_thread = threading.Thread(
            target=self._kill_watch_loop, args=(loop,), daemon=True
        )
        self._kill_thread.start()

    def _kill_watch_loop(self, loop) -> None:
        while True:
            if KILL_FLAG.exists():
                try:
                    content = KILL_FLAG.read_text(encoding="utf-8").strip()
                except Exception:
                    content = "x"
                if content:
                    loop.call_soon_threadsafe(self.kill_event.set)
                    break
            time.sleep(0.5)

    async def _flatten_all(self) -> None:
        """Flatten every open position and drop queued orders (#1d, #12)."""
        now = int(time.time() * 1000)
        for sym, pos in list(self.positions.items()):
            if pos.qty == 0:
                continue
            side = "sell" if pos.side == "long" else "buy"
            order = Order(
                id=str(uuid.uuid4()),
                signal_id="",
                symbol=sym,
                side=side,
                order_type="market",
                quantity=abs(pos.qty),
                price=None,
                broker="ccxt" if sym.endswith("/USDT") else "india",
                status="pending",
                ts_created=now,
                ts_updated=now,
            )
            ad = self._adapters.get(sym)
            if ad is not None:
                sent = await ad.send_order(order)
                write_audit({"type": "flatten", "symbol": sym, "order": sent.__dict__})
        # Drop any throttled/queued orders so nothing fires after a kill.
        for bucket in self.buckets.values():
            while bucket.pop_queued() is not None:
                pass

    async def run(self):
        if KILL_FLAG.exists():
            write_audit({"type": "start_blocked", "reason": "kill_flag present"})
            return
        self.state = State.INIT
        loop = asyncio.get_running_loop()
        self._install_signal_handlers(loop)
        self._start_kill_watcher(loop)

        adapters = self._build_adapters()
        self._adapters = adapters
        for sym, ad in adapters.items():
            await ad.connect()
            await ad.subscribe([sym])
            ad.on_depth(self._push_raw)
            ad.on_trade(self._push_raw)
        self.state = State.CONNECTED
        tasks = [
            asyncio.create_task(self._marketdata_task(adapters)),
            asyncio.create_task(self._consumer_task()),
        ]
        await self.kill_event.wait()  # #1e: block until SIGINT/SIGTERM/flag/DD latch

        # --- KILL sequence ---
        self.state = State.KILL
        write_kill_flag("manual_or_auto")
        for t in tasks:
            t.cancel()
        for t in tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
        if self.settings.risk.flatten_on_kill:  # #12: read config and act on it
            try:
                await self._flatten_all()
            except Exception:
                pass
        for ad in adapters.values():
            try:
                await ad.close()
            except Exception:
                pass


async def run_live(settings: Settings, paper: bool = False) -> None:
    loop = LiveLoop(settings, paper=paper)
    await loop.run()


def main_live(settings: Settings, paper: bool = False) -> None:
    """Synchronous entry used by the CLI; can be killed via the kill flag / event."""
    try:
        asyncio.run(run_live(settings, paper=paper))
    except KeyboardInterrupt:
        write_kill_flag("keyboard_interrupt")
