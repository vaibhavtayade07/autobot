"""Pydantic settings + fail-fast validation (DESIGN §6).

Layering: env secrets (never committed) override the YAML non-secret config. The
process fails fast at startup if (a) we don't have >=1 crypto AND >=1 india symbol
enabled, (b) any rate/limit is <= 0, (c) a required secret is missing for a live/paper
broker, or (d) the India holiday calendar is missing while an India symbol is enabled.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RiskConfig(BaseModel):
    risk_per_trade_pct: float = 0.01
    kelly_fraction_cap: float = 0.25
    max_positions: int = 6
    max_drawdown_pct: float = 0.20
    atr_stop_mult: float = 2.0
    order_rate_limit: int = 20
    flatten_on_kill: bool = True


class SymbolConfig(BaseModel):
    ticker: str
    asset_class: Literal["crypto", "india"]
    exchange: str
    timeframes: List[str] = Field(default_factory=lambda: ["1m", "5m"])
    strategies: List[str] = Field(default_factory=list)
    enabled: bool = True
    lot_size: float = 1.0  # contract/trade lot multiplier (India F&O lots, etc.)


class IndiaSessionConfig(BaseModel):
    open: str = "09:15"
    close: str = "15:30"
    timezone: str = "Asia/Kolkata"
    holidays_file: str = "config/india_holidays.json"


class CryptoSessionConfig(BaseModel):
    always_open: bool = True


class SessionConfig(BaseModel):
    india: IndiaSessionConfig = Field(default_factory=IndiaSessionConfig)
    crypto: CryptoSessionConfig = Field(default_factory=CryptoSessionConfig)


class BacktestConfig(BaseModel):
    start: str = "2025-01-01"
    end: str = "2025-06-30"
    initial_capital: float = 100000.0
    freq: str = "1m"
    in_sample: str = "3M"
    out_sample: str = "1M"


class Settings(BaseSettings):
    environment: Literal["live", "paper", "backtest"] = "backtest"
    equity_base: float = 100000.0
    risk: RiskConfig = Field(default_factory=RiskConfig)
    symbols: List[SymbolConfig] = Field(default_factory=list)
    session: SessionConfig = Field(default_factory=SessionConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)

    # secrets (env only)
    binance_api_key: Optional[str] = None
    binance_api_secret: Optional[str] = None
    openalgo_url: Optional[str] = None
    angel_api_key: Optional[str] = None
    angel_client_id: Optional[str] = None
    angel_password: Optional[str] = None
    angel_totp_secret: Optional[str] = None
    angel_feed_token: Optional[str] = None

    model_config = SettingsConfigDict(env_prefix="AUTOBO_", extra="ignore")
    _config_path: Optional[Path] = None  # set by from_config

    @field_validator("symbols")
    @classmethod
    def _non_empty_symbols(cls, v):
        if not v:
            raise ValueError("at least one symbol must be configured")
        return v

    def model_post_init(self, __context) -> None:
        enabled = [s for s in self.symbols if s.enabled]
        if not any(s.asset_class == "crypto" for s in enabled):
            raise ValueError("validation failed: >=1 enabled crypto symbol required")
        if not any(s.asset_class == "india" for s in enabled):
            raise ValueError("validation failed: >=1 enabled india symbol required")

        if self.risk.risk_per_trade_pct <= 0 or self.risk.max_drawdown_pct <= 0:
            raise ValueError("validation failed: risk_per_trade_pct and max_drawdown_pct must be > 0")
        if self.risk.order_rate_limit <= 0:
            raise ValueError("validation failed: order_rate_limit must be > 0")

        if self.environment != "backtest":
            crypto_enabled = any(s.asset_class == "crypto" for s in enabled)
            india_enabled = any(s.asset_class == "india" for s in enabled)
            missing = []
            if crypto_enabled and not (self.binance_api_key and self.binance_api_secret):
                missing.append("BINANCE_API_KEY / BINANCE_API_SECRET")
            if india_enabled and not (
                self.openalgo_url and self.angel_api_key and self.angel_client_id
                and self.angel_password and self.angel_totp_secret
            ):
                missing.append(
                    "OPENALGO_URL / ANGEL_API_KEY / ANGEL_CLIENT_ID / ANGEL_PASSWORD / ANGEL_TOTP_SECRET"
                )
            if missing:
                raise ValueError(
                    "validation failed: missing required secrets for live/paper mode -> "
                    + "; ".join(missing)
                )

        if any(s.asset_class == "india" for s in enabled):
            hf = Path(self.session.india.holidays_file)
            if not hf.is_absolute():
                # resolve relative to the loaded config file, then cwd
                root = self._config_path.parent if self._config_path else Path.cwd()
                hf = root / hf
            if not hf.exists():
                raise ValueError(f"validation failed: india holidays file not found: {hf}")

    @classmethod
    def from_config(
        cls, path: Optional[str] = None, environment: Optional[str] = None
    ) -> "Settings":
        """Load settings from YAML.

        ``environment`` overrides the YAML ``environment`` field. Passing
        ``environment="backtest"`` lets the keyless ``paper --simulate`` dry-run
        skip the live/paper secret validation in ``model_post_init``.
        """
        path = path or os.environ.get("AUTOBO_CONFIG")
        if not path:
            # bundled default config
            here = Path(__file__).resolve().parent
            path = here / "default.yaml"
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {p}")
        cls._config_path = p.resolve()  # used by model_post_init to locate relative files
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if environment is not None:
            data["environment"] = environment
        return cls(**data)
