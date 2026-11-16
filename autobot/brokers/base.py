"""BrokerAdapter ABC — the execute-only boundary (DESIGN §4).

Adapters normalize market data (push via callbacks) and send/cancel orders; they
never import math_*, signals_*, or risk. All compute happens in the live loop.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from ..data.contracts import Depth, Order, Position, Tick


class BrokerAdapter(ABC):
    # ---- lifecycle ----
    @abstractmethod
    async def connect(self) -> None:
        """Auth + open WS (no market data yet)."""

    @abstractmethod
    async def close(self) -> None:
        """Close WS + cancel subscriptions."""

    # ---- data (normalize to SPEC dataclasses, push via callbacks) ----
    @abstractmethod
    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to depth + trades for each symbol."""

    @abstractmethod
    def on_depth(self, cb: Callable[[Depth], None]) -> None:
        """Register a depth callback."""

    @abstractmethod
    def on_trade(self, cb: Callable[[Tick], None]) -> None:
        """Register a trade callback."""

    def on_fill(self, cb: Callable[[Order], None]) -> None:
        """Register a fill callback (default no-op; subclasses override)."""

    def on_order_update(self, cb: Callable[[Order], None]) -> None:
        """Register an order-update callback (default no-op)."""

    # ---- execution (the ONLY broker network calls in the hot path) ----
    @abstractmethod
    async def send_order(self, order: Order) -> Order:
        """Return the Order with broker ack status + reject_reason."""

    @abstractmethod
    async def cancel_order(self, order_id: str) -> Order:
        """Cancel by client order id."""

    # ---- state (called on fill + every 60s ONLY, never hot path) ----
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Cached/reconciled positions."""

    @abstractmethod
    async def get_funds(self) -> dict:
        """Account funds/balances."""

    # ---- session (default: always open) ----
    def is_session_open(self, symbol: str, ts: int) -> bool:
        """Default: always open. IndiaAdapter delegates to the session gate."""
        return True
