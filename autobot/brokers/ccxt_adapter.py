"""CCXT Binance adapter (crypto, 24/7).

ccxt is imported lazily inside connect() so the module is importable offline. The
adapter uses WS watch_order_book / watch_trades (no REST polling except auth) and is
testnet-aware via set_sandbox_mode for paper mode.
"""
from __future__ import annotations

import asyncio
import json
from typing import Callable, List, Optional

from .base import BrokerAdapter
from ..data.contracts import Depth, Order, Position, Tick
from ..math_risk import TokenBucket


class CCXTAdapter(BrokerAdapter):
    def __init__(self, settings, paper: bool = False):
        self.settings = settings
        self.paper = paper
        self.exchange = None
        self._depth_cb: Optional[Callable[[Depth], None]] = None
        self._trade_cb: Optional[Callable[[Tick], None]] = None
        self._fill_cb: Optional[Callable[[Order], None]] = None
        self._order_cb: Optional[Callable[[Order], None]] = None
        self._seq = 0
        self._bucket = TokenBucket(rate=100.0, capacity=100.0)  # crypto: generous

    async def connect(self) -> None:
        import ccxt  # lazy: heavy dep, not required for math/check

        cfg = {
            "apiKey": self.settings.binance_api_key,
            "secret": self.settings.binance_api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }
        ex = ccxt.binance(cfg)
        if self.paper:
            ex.set_sandbox_mode(True)
        # auth (no polling): a lightweight REST call to verify credentials
        await ex.load_markets()
        self.exchange = ex

    async def close(self) -> None:
        # Cancel the watch tasks so a closed exchange doesn't get touched (#10).
        for t in getattr(self, "_tasks", []) or []:
            t.cancel()
        if self.exchange is not None:
            await self.exchange.close()

    async def subscribe(self, symbols: List[str]) -> None:
        if self.exchange is None:
            raise RuntimeError("connect() must be called before subscribe()")
        self._tasks = []
        for sym in symbols:
            self._tasks.append(asyncio.create_task(self._watch_book(sym)))
            self._tasks.append(asyncio.create_task(self._watch_trades(sym)))

    async def _watch_book(self, symbol: str) -> None:
        import ccxt  # for ccxt.NetworkError classification

        while True:
            try:
                ob = await self.exchange.watch_order_book(symbol)
                self._seq += 1
                bids = [(float(p), float(s)) for p, s in ob["bids"][:50]]
                asks = [(float(p), float(s)) for p, s in ob["asks"][:50]]
                depth = Depth(
                    symbol=symbol,
                    ts=int(ob["timestamp"]),
                    bids=bids,
                    asks=asks,
                    is_snapshot=True,
                    seq=self._seq,
                )
                if self._depth_cb:
                    self._depth_cb(depth)
            except ccxt.NetworkError:  # transient network drop -> reconnect (#10)
                await asyncio.sleep(1.0)
                try:
                    await self.connect()
                except Exception:
                    pass
            except Exception:  # any other error -> backoff and retry the watch
                await asyncio.sleep(1.0)
                try:
                    await self.connect()
                except Exception:
                    pass

    async def _watch_trades(self, symbol: str) -> None:
        import ccxt  # for ccxt.NetworkError classification

        while True:
            try:
                trades = await self.exchange.watch_trades(symbol)
                for t in trades:
                    side = "buy" if t.get("side") == "buy" else "sell"
                    tick = Tick(
                        symbol=symbol,
                        ts=int(t["timestamp"]),
                        price=float(t["price"]),
                        size=float(t["amount"]),
                        side=side,
                        source="crypto",
                    )
                    if self._trade_cb:
                        self._trade_cb(tick)
            except ccxt.NetworkError:  # transient network drop -> backoff + retry (#10)
                await asyncio.sleep(1.0)
            except Exception:
                await asyncio.sleep(1.0)

    def on_depth(self, cb: Callable[[Depth], None]) -> None:
        self._depth_cb = cb

    def on_trade(self, cb: Callable[[Tick], None]) -> None:
        self._trade_cb = cb

    def on_fill(self, cb: Callable[[Order], None]) -> None:
        self._fill_cb = cb

    def on_order_update(self, cb: Callable[[Order], None]) -> None:
        self._order_cb = cb

    async def send_order(self, order: Order) -> Order:
        if self.exchange is None:
            order.status = "rejected"
            order.reject_reason = "not connected"
            return order
        if not self._bucket.admit():
            order.status = "rejected"
            order.reject_reason = "rate limit"
            return order
        try:
            side = "buy" if order.side == "buy" else "sell"
            params = {}
            if order.broker == "ccxt" and order.id:
                params["clientOrderId"] = order.id
            # Market orders must pass price=None; only LIMIT carries a price (#11).
            price_arg = None if order.order_type == "market" else order.price
            res = await self.exchange.create_order(
                order.symbol,
                order.order_type,  # market | limit
                side,
                order.quantity,
                price_arg,
                params,
            )
            order.status = "acked"
            order.ts_updated = int(res.get("timestamp", 0)) or order.ts_updated
            order.fill_price = (
                float(res["price"]) if res.get("price") else order.fill_price
            )
        except Exception as e:
            order.status = "rejected"
            order.reject_reason = str(e)
        return order

    async def cancel_order(self, order_id: str) -> Order:
        try:
            await self.exchange.cancel_order(order_id)
        except Exception as e:
            return Order(
                id=order_id,
                signal_id="",
                symbol="",
                side="",
                order_type="",
                quantity=0.0,
                price=None,
                broker="ccxt",
                status="rejected",
                ts_created=0,
                ts_updated=0,
                reject_reason=str(e),
            )
        return Order(
            id=order_id,
            signal_id="",
            symbol="",
            side="",
            order_type="",
            quantity=0.0,
            price=None,
            broker="ccxt",
            status="cancelled",
            ts_created=0,
            ts_updated=0,
        )

    async def get_positions(self) -> List[Position]:
        # Spot has no open derivatives positions; reconcile from balance deltas instead.
        return []

    async def get_funds(self) -> dict:
        if self.exchange is None:
            return {}
        bal = await self.exchange.fetch_balance()
        return bal.get("total", {})

    def is_session_open(self, symbol: str, ts: int) -> bool:
        return True  # crypto trades 24/7
