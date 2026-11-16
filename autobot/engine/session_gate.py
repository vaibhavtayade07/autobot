"""Session gate (DESIGN §3 / §4). Crypto always open; India gated by IST hours + holidays.

Pure logic, config-only dependency. Uses the stdlib zoneinfo for IST (UTC+5:30).
"""
from __future__ import annotations

from datetime import datetime, time, timezone, timedelta
from pathlib import Path
from typing import Optional

from ..config.settings import Settings

IST = timezone(timedelta(hours=5, minutes=30))


class IndiaSession:
    def __init__(self, settings: Settings):
        self.settings = settings
        s = settings.session.india
        self.open_t = self._parse_hhmm(s.open)
        self.close_t = self._parse_hhmm(s.close)
        self._holidays = self._load_holidays()

    @staticmethod
    def _parse_hhmm(v: str) -> time:
        h, m = v.split(":")
        return time(int(h), int(m))

    def _load_holidays(self) -> set:
        hf = Path(self.settings.session.india.holidays_file)
        if not hf.is_absolute():
            root = Settings._config_path.parent if Settings._config_path else Path.cwd()
            hf = root / hf
        if not hf.exists():
            return set()
        try:
            import json

            data = json.loads(hf.read_text(encoding="utf-8"))
            return set(data.get("holidays", []))
        except Exception:
            return set()

    def _asset_class(self, symbol: str) -> Optional[str]:
        for s in self.settings.symbols:
            if s.ticker == symbol:
                return s.asset_class
        return None

    def is_open(self, symbol: str, ts: int) -> bool:
        asset = self._asset_class(symbol)
        if asset == "crypto" or asset is None and self.settings.session.crypto.always_open:
            return True
        if asset != "india":
            # Unknown symbol: default to crypto (always open) to avoid silently blocking.
            return True
        dt = datetime.fromtimestamp(ts / 1000.0, tz=IST)
        if dt.weekday() >= 5:  # Sat/Sun
            return False
        if dt.strftime("%Y-%m-%d") in self._holidays:
            return False
        return self.open_t <= dt.time() <= self.close_t
