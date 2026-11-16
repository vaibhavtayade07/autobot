# Autobot

**HFT-style trading engine for crypto and Indian equities.** WebSocket-driven signals computed locally; brokers touched only at execution.

<p>
<img alt="Python 3.13" src="https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white">
<img alt="CCXT" src="https://img.shields.io/badge/CCXT-Binance-F3BA2F">
<img alt="vectorbt" src="https://img.shields.io/badge/backtest-vectorbt-blue">
<img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

Live, math-precise, HFT-style trading bot for **crypto via Binance/CCXT** and **Indian
equities via Angel One SmartAPI behind OpenAlgo**. All signals are computed **locally
from the WebSocket stream**; brokers are touched **only at execution**.

> Honest scope note: true co-located HFT (FPGA / kernel-bypass / exchange colo) is
> **not** possible through retail APIs (Binance WS, Angel SmartAPI/OpenAlgo). `autobot`
> is WS-driven and microsecond-aware at the software level, but it is not matching-engine
> proximate. v1 is a single-process, execute-only bot with hard risk limits, a kill switch,
> and a vectorbt backtest harness.

## Install

```bash
cd D:\micro-saas\vstudio\GitHub\autobot
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
# or: pip install -e .
```

The **math modules** (`autobot/math_*.py`, `autobot/math_risk.py`) depend only on
`numpy`/`pandas`/`scipy` and are unit-testable offline. Heavy deps (`ccxt`, `vectorbt`,
`arch`, `statsmodels`, `pandas-ta`, `smartapi`, `pyotp`) are imported **lazily** inside
the functions that need them, so `import autobot` and the math self-test work without them.

## Configure

1. Copy `.env.example` → `.env` and fill in the real secrets (never commit `.env`).
2. Edit `autobot/config/default.yaml` for symbols, timeframes, risk limits, and session
   hours. Secrets always come from env (`AUTOBO_*` prefix) and override the YAML.

Fail-fast validation runs at startup: you need ≥1 enabled crypto **and** ≥1 enabled India
symbol, all rates > 0, and (for live/paper) the required broker secrets present.

## Run

```bash
# 1) Offline math self-test (numpy only) — should print all PASS
python -m autobot.cli check

# 2) Backtest (needs vectorbt + data; crypto via ccxt, India via saved CSV)
python -m autobot.cli backtest

# 3) Paper mode (Binance testnet / OpenAlgo paper, or simulated fills)
python -m autobot.cli paper

# 4) Live mode (REAL money — requires all secrets)
python -m autobot.cli live

# kill a running process (sets kill flag; live_loop halts + flattens)
python -m autobot.cli kill
```

## Math modules (local, offline-testable)

| Module | Contents |
|---|---|
| `math_microstructure.py` | order-book imbalance, microprice, weighted mid, book pressure, VPIN, Amihud |
| `math_stats.py` | custom Kalman hedge ratio, cointegration p-value (lazy statsmodels), spread z-score, Ornstein-Uhlenbeck half-life, Shannon + approximate entropy, Hurst R/S, trend helpers |
| `math_vol.py` | Volume Profile POC / value area, OBV, Chaikin Money Flow, volume ratio, GARCH(1,1) w/ EWMA fallback |
| `math_risk.py` | Kelly + fractional Kelly, Wilder ATR, 2×ATR stop, 1%-equity risk sizing, DrawdownGuard kill-switch, TokenBucket (20/s for Angel) |

Every function honors the SPEC §3 invariants (imbalance ∈ [-1,1], VPIN ∈ [0,1],
microprice strictly inside spread, Kelly ≤ 1, Hurst ∈ [0,1], GARCH variance > 0 with
EWMA fallback, OU half-life finite/None, POC within range, z-score finite, no NaN to
risk). `tests/test_math.py` asserts all of them.

## Signal path

```
WS adapter -> raw_q -> [session gate + staleness] -> feature_q
  -> compute_features -> signals_* (trend-filtered) -> risk gate
  -> TokenBucket -> broker.send_order (audit JSONL)
```

Both asset classes trade in one asyncio loop: crypto 24/7, India gated to 09:15–15:30
IST + NSE holidays (zero compute cost when closed). State machine: INIT → CONNECTED →
TRADING/FLAT (orthogonal THROTTLED/PAUSED) → KILL.

## Next steps for the founder

1. `pip install -r requirements.txt` (install heavy deps for live/backtest).
2. Add real keys to `.env` (Binance + Angel/OpenAlgo).
3. Run `python -m autobot.cli check` — must be all PASS.
4. Run `python -m autobot.cli paper` for a 30-day paper run; confirm zero unhandled
   exceptions before going live.
