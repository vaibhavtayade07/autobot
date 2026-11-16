"""Cost model shared by live pre-trade checks and the vectorbt backtest (DESIGN §7).

Slippage + brokerage + India sell-side STT. Kept honest and identical across modes.
"""
from __future__ import annotations

from ..config.settings import Settings


class CostModel:
    def __init__(self, settings: Settings):
        self.settings = settings

    def slippage_pct(self, asset_class: str) -> float:
        # Conservative per-side slippage assumption.
        return 0.0005 if asset_class == "crypto" else 0.0010

    def fee(self, side: str, notional: float, asset_class: str) -> float:
        """Brokerage + regulatory. India STT applies to SELL turnover only."""
        if asset_class == "crypto":
            return notional * 0.0010  # Binance taker
        brokerage = min(notional * 0.0003, 20.0)
        stt = notional * 0.00025 if side == "sell" else 0.0
        exchange = notional * 0.00003
        return brokerage + stt + exchange
