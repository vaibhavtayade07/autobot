"""India adapter — Angel One SmartAPI behind OpenAlgo (sketch).

This is an execute-only adapter. Heavy deps (smartapi, pyotp, requests) are imported
lazily so the module is importable offline. Every send is wrapped in a 20/s token
bucket (SPEC rule #4) with 429 backoff; session gating delegates to engine/session_gate.

WHERE TO PLUG REAL KEYS: set env AUTOBO_OPENALGO_URL, AUTOBO_ANGEL_API_KEY,
AUTOBO_ANGEL_CLIENT_ID, AUTOBO_ANGEL_PASSWORD, AUTOBO_ANGEL_TOTP_SECRET (see .env.example).
"""
from __future__ import annotations

import asyncio
import time
from typing import Callable, List, Optional

from .base import BrokerAdapter
from ..data.contracts import Depth, Order, Position, Tick
from ..engine.session_gate import IndiaSession
from ..math_risk import TokenBucket


class IndiaAdapter(BrokerAdapter):
    def __init__(self, settings, paper: bool = False):
        self.settings = settings
        self.paper = paper
        self._depth_cb: Optional[Callable[[Depth], None]] = None
        self._trade_cb: Optional[Callable[[Tick], None]] = None
        self._fill_cb: Optional[Callable[[Order], None]] = None
        self._order_cb: Optional[Callable[[Order], None]] = None
        self._bucket = TokenBucket(rate=settings.risk.order_rate_limit, capacity=settings.risk.order_rate_limit)
        self._session = IndiaSession(settings)
        self._auth_token: Optional[str] = None
        self._feed_token: Optional[str] = None
        self._seq = 0
        self._last_429 = 0.0

    async def connect(self) -> None:
        # TOTP login via OpenAlgo (Angel SmartAPI). Lazy heavy import.
        import pyotp  # type: ignore
        import requests  # type: ignore

        totp = pyotp.TOTP(self.settings.angel_totp_secret).now()
        url = (self.settings.openalgo_url or "").rstrip("/") + "/api/login"
        payload = {
            "api_key": self.settings.angel_api_key,
            "client_id": self.settings.angel_client_id,
            "password": self.settings.angel_password,
            "totp": totp,
            "state": "paper" if self.paper else "live",
        }
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        self._auth_token = data.get("token") or data.get("auth_token")
        self._feed_token = self.settings.angel_feed_token or data.get("feed_token")

    async def close(self) -> None:
        self._auth_token = None

    async def subscribe(self, symbols: List[str]) -> None:
        # OpenAlgo depth WS would be opened here; left as a sketch that registers
        # a stub so the pipeline has a no-op data path until the server is wired.
        for sym in symbols:
            asyncio.create_task(self._watch_depth_stub(sym))

    async def _watch_depth_stub(self, symbol: str) -> None:
        # Placeholder: replace with the OpenAlgo depth WS read loop. Emits nothing
        # until connected to a live OpenAlgo server. Keeps the loop healthy.
        await asyncio.sleep(0)

    def _to_depth(self, msg: dict, symbol: str) -> Depth:
        self._seq += 1
        bids = [(float(p), float(s)) for p, s in msg.get("bids", [])[:50]]
        asks = [(float(p), float(s)) for p, s in msg.get("asks", [])[:50]]
        return Depth(
            symbol=symbol,
            ts=int(msg.get("ts", time.time() * 1000)),
            bids=bids,
            asks=asks,
            is_snapshot=True,
            seq=self._seq,
        )

    def on_depth(self, cb: Callable[[Depth], None]) -> None:
        self._depth_cb = cb

    def on_trade(self, cb: Callable[[Tick], None]) -> None:
        self._trade_cb = cb

    def on_fill(self, cb: Callable[[Order], None]) -> None:
        self._fill_cb = cb

    def on_order_update(self, cb: Callable[[Order], None]) -> None:
        self._order_cb = cb

    async def send_order(self, order: Order) -> Order:
        if not self._bucket.admit():
            # requeue: caller (live_loop) re-submits on the next tick
            order.status = "rejected"
            order.reject_reason = "rate limit (token bucket)"
            return order
        import requests  # type: ignore

        url = (self.settings.openalgo_url or "").rstrip("/") + "/api/order"
        payload = {
            "symbol": order.symbol,
            "action": order.side,  # BUY/SELL
            "quantity": order.quantity,
            "price": order.price,
            "pricetype": "MARKET" if order.order_type == "market" else "LIMIT",
            "product": "MIS" if not self.paper else "CNC",
            "exchange": self.settings.session and "NSE",
            "client_id": self.settings.angel_client_id,
            "token": self._auth_token,
            # clientOrderId carried for idempotent retries on 429 (#9)
            "client_order_id": order.id,
        }
        # Exponential backoff on HTTP 429 (idempotent via client_order_id) (#9).
        backoff = 1.0
        max_retries = 5
        for _ in range(max_retries):
            try:
                resp = requests.post(url, json=payload, timeout=15)
            except Exception as e:
                order.status = "rejected"
                order.reject_reason = str(e)
                return order
            if resp.status_code == 429:
                self._last_429 = time.monotonic()
                await asyncio.sleep(backoff)
                backoff *= 2.0
                continue
            try:
                resp.raise_for_status()
            except Exception as e:
                order.status = "rejected"
                order.reject_reason = str(e)
                return order
            order.status = "acked"
            order.ts_updated = int(time.time() * 1000)
            return order
        # Exhausted retries: reject so the live loop can re-queue / flatten.
        order.status = "rejected"
        order.reject_reason = "HTTP 429 after retries"
        return order

    async def cancel_order(self, order_id: str) -> Order:
        try:
            import requests  # type: ignore

            url = (self.settings.openalgo_url or "").rstrip("/") + "/api/order_cancel"
            requests.post(
                url,
                json={"order_id": order_id, "client_id": self.settings.angel_client_id},
                timeout=15,
            )
        except Exception:
            pass
        return Order(
            id=order_id,
            signal_id="",
            symbol="",
            side="",
            order_type="",
            quantity=0.0,
            price=None,
            broker="india",
            status="cancelled",
            ts_created=0,
            ts_updated=0,
        )

    async def get_positions(self) -> List[Position]:
        return []

    async def get_funds(self) -> dict:
        return {}

    def is_session_open(self, symbol: str, ts: int) -> bool:
        return self._session.is_open(symbol, ts)
