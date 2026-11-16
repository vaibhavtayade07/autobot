# `autobot` — Product / Engineering SPEC

> Live, math-precise, high-frequency-style trading bot for **two asset classes**:
> (1) crypto via **Binance/CCXT**, (2) Indian equities via **Angel One SmartAPI behind OpenAlgo**.
> Repo: `D:\micro-saas\vstudio\GitHub\autobot` · Python 3.11+ · solo founder / one operator.

---

## 0. Goal, Users, Scope

**Goal.** A single Python process that ingests live market data over WebSocket, computes a full
suite of microstructure + statistical math signals **locally**, and routes execution-only orders to
pluggable broker adapters — with hard risk limits, a kill switch, and a vectorbt backtest harness.

**Users.**
- Solo founder / operator (the only human). Configures, launches, monitors, kills.

**Design rules (non-negotiable — from brief + prior volume-trading research):**
1. All signals computed **locally** from the WS stream. Broker API is touched **only at execution**.
2. **Never poll.** Depth, trades, funds, positions come via WS (or cached + refreshed on fill/interval).
3. Custom **Kalman** (no `pykalman`).
4. **Token-bucket** order rate ≤ 20/s for Angel. Exponential backoff on HTTP 429.
5. **Broker-pluggable** via `BrokerAdapter` ABC. `CCXTAdapter` (crypto) + `IndiaAdapter` (SmartAPI/OpenAlgo).
6. Stack (verified Aug 2026): `ccxt`, `vectorbt` (backtest), `arch` (GARCH), `statsmodels` (coint/ADF/Johansen/OLS),
   `pandas-ta` (indicators — **NOT** TA-Lib). Every volume signal ships with a trend filter (VWAP/200MA),
   1%-equity risk, 2×ATR stop, and a walk-forward guard (per prior research: "volume shows intent, not
   confirmation").

**Scope IN.** Both asset classes live; all listed math modules; risk engine; kill switch; backtest harness;
config via env + file; CLI.

**Scope OUT (see §7).** Co-located HFT, ML/RL signals, multi-pair portfolio optimization, derivatives,
tax/audit reporting, mobile UI.

---

## 1. User Stories (solo founder operating the bot)

1. **Configure** — As the operator I set symbols, timeframes, risk limits, and broker creds via env + a config
   file, so the bot starts from one command.
2. **Launch live** — I run `autobot run --live` and watch both WS feeds (crypto + India) connect and stay healthy.
3. **See signals** — I view real-time `Signal` objects (score, regime, side, stop, size) as they are produced.
4. **See the math** — I can inspect the underlying indicators (order-book imbalance, microprice, VPIN, Hurst,
   GARCH vol, Volume Profile POC/VA, Kalman hedge, OU half-life, entropy) per symbol.
5. **Backtest** — I run `autobot backtest` over a historical window with vectorbt and get Sharpe / Sortino /
   max-drawdown per strategy.
6. **Risk is automatic** — The bot sizes every trade via Kelly (capped), enforces 1%-equity risk and a 2×ATR
   stop, and refuses to exceed per-symbol/per-account limits.
7. **Rate-limit safe** — The bot never sends >20 orders/s to Angel; excess is queued by the token bucket.
8. **Kill switch (auto)** — If account drawdown breaches the configured limit, the bot halts, cancels open
   orders, and alerts — without my intervention.
9. **Kill switch (manual)** — I run `autobot kill` (or hit an emergency key) to flatten/halt immediately.
10. **Survive disconnect** — On WS drop the bot reconnects with backoff, detects sequence gaps, and pauses that
    symbol's trading until resynced; no ghost orders.
11. **Session-aware** — India symbols trade only in market hours / non-holiday; crypto trades 24/7. No orders
    fire outside the gate.
12. **Auditable** — Every order, fill, reject, and kill event is written to a JSONL audit log with its source
    `signal_id`, so I can reconcile after the fact.

---

## 2. Functional Modules

| Module | One-line responsibility | Math mapped | Effort |
|---|---|---|---|
| `config/settings.py` | Pydantic settings; load env + YAML; validate (≥1 crypto + ≥1 india, rates>0). | — | S |
| `data/contracts.py` | Dataclasses: `Tick`, `Depth`, `Signal`, `Order`, `Position`. | — | S |
| `math_microstructure.py` | Order-book imbalance, microprice, weighted mid, book pressure. | imbalance, microprice | M |
| `math_stats.py` | VPIN, Hurst R/S, Shannon & approximate entropy, Amihud illiquidity. | VPIN, Hurst, entropy, Amihud | M |
| `math_timeseries.py` | Custom Kalman hedge ratio, OU half-life, ADF/Johansen cointegration, spread + z. | Kalman, OU, coint, Johansen | L |
| `math_vol.py` | GARCH(1,1) forecast, Volume Profile POC/value-area, ATR, Kelly fraction. | GARCH, VP POC/VA, Kelly | M |
| `signals_volume.py` | 4 volume patterns + trend filter + stop + walk-forward gate; regime tag. | vol patterns + stats gating | L |
| `signals_pairs.py` | Cointegrated spread entry/exit via Kalman hedge + OU half-life on z-score. | Kalman, OU, coint | M |
| `risk.py` | Kelly sizer, token bucket, drawdown guard, per-symbol/account limit checks. | Kelly, rate, dd | M |
| `brokers/base.py` | `BrokerAdapter` ABC: connect/subscribe/callbacks/send/cancel/positions/funds/close. | — | S |
| `brokers/ccxt_adapter.py` | CCXT WS (Binance) depth+trades; order send; rate-limited; no polling. | — | M |
| `brokers/india_adapter.py` | OpenAlgo/Angel SmartAPI auth (TOTP), depth WS, order send, session gate. | — | L |
| `engine/marketdata.py` | WS orchestration, depth→`Depth` normalize, positions/funds cache, staleness monitor. | — | M |
| `engine/live_loop.py` | asyncio loop: ingest→buffer→indicators→signals→risk→dispatch→log. | — | L |
| `backtest/run.py` | vectorbt harness; replay saved ticks / fetchOHLCV; per-strategy report. | — | M |
| `logging/audit.py` | JSONL audit of signals/orders/fills/rejects/kills with `signal_id`. | — | S |
| `cli.py` | `run --live/--paper`, `backtest`, `status`, `kill`. | — | S |

---

## 3. Acceptance Criteria (per module, with numerical correctness)

### `math_microstructure.py`
- `order_book_imbalance(bids, asks, levels=5) -> float` ∈ **[-1, 1]** (weighted by size).
- Empty/zero-size book → returns **0.0** (no div-by-zero).
- `microprice(bid, bid_sz, ask, ask_sz) -> float` is **strictly between** bid and ask when both sizes > 0;
  with a zero size returns the mid. Unit test: synthetic book → value inside spread.

### `math_stats.py`
- `vpin(...) -> float` ∈ **[0, 1]**; zero total volume → **0.0**.
- `hurst_rs(series) -> float` ∈ **[0, 1]**; pure Brownian noise ≈ **0.5 ± 0.1** on 2000+ samples.
- `shannon_entropy(series, bins) -> float` normalized ∈ **[0, 1]**.
- `approx_entropy(series, m, r) -> float` ≥ 0; random series > ordered series.
- `amihud(returns, volumes) -> float` ≥ 0; zero-volume bars → contribution **0** (no div-by-zero).

### `math_timeseries.py`
- `kalman_hedge_ratio(y, x)` on a **synthetic cointegrated** pair (y = 0.5·x + noise) converges to β≈0.5
  within **5%** after ≥ 200 obs; state covariance `P` is non-negative definite and shrinks over time.
- `ou_half_life(series) -> float` is **finite and positive** for mean-reverting series; for a random walk
  returns **+∞ / None** (b ≥ 0 guard) and the signal path skips it.
- `adf_test` / `johansen` wrap `statsmodels`; on a known cointegrated pair `coint_pvalue < 0.05`.
- `spread_z(spread, window)` is **finite**; zero/std guard → `0.0` (skip), never `NaN` propagated to risk.

### `math_vol.py`
- `garch_11(returns, horizon)` returns **positive** forecast variance; if fitted `alpha+beta ≥ 1`
  (non-stationary) it **flags unstable** and falls back to EWMA vol instead of emitting garbage.
- `volume_profile(prices, volumes, bins)` → `poc` is a price **within [min, max]** of the input;
  `val`/`vah` bound a region containing **≥ 0.68** (configurable) of total volume.
- `atr(high, low, close, n)` matches `pandas-ta` ATR within 1e-6 on a fixed fixture.
- `kelly_fraction(win_rate, payoff) -> float` is **clamped to [0, 1]**; negative edge → **0.0**.

### `signals_volume.py` & `signals_pairs.py`
- Every emitted `Signal` carries a non-empty `regime`, a `stop_price` exactly **2×ATR** from entry, and
  `size_pct_equity` ≤ **1%** of account equity (post-Kelly-cap).
- Volume breakout fires **only** when price is on the correct side of the trend filter (VWAP or 200MA).
- Pairs signal emits **only after** minimum N observations (Kalman warmed up) and OU half-life is finite.
- Walk-forward: parameters are fixed per window; the live path re-estimates on a rolling schedule, never
  re-optimizes on the bar being traded.

### `risk.py`
- `TokenBucket(rate=20, capacity=20)` admits ≤ **20** sends per second to Angel (measured over 60s burst).
- `DrawdownGuard(max_dd)` returns `True` (kill) the moment **realized+unrealized** DD ≥ limit; latches.
- Sizing rejects any `Order` whose notional risk > 1% equity or whose stop is missing.

### `brokers/*`
- `BrokerAdapter` ABC methods are all implemented by both adapters; `send_order` returns an `Order` with a
  broker ack status; rejects carry `reject_reason`.
- `CCXTAdapter` subscribes via `watchOrderBook`/`watchTrades` (no REST polling except auth).
- `IndiaAdapter` authenticates via Angel TOTP, opens OpenAlgo depth WS, and refuses to trade outside
  09:15–15:30 IST / Indian holidays.

### `engine/live_loop.py` & `engine/marketdata.py`
- On WS disconnect → reconnect with exponential backoff; sequence gap detected → symbol paused + staleness flag.
- No broker REST call in the hot path except `send_order` / `cancel` / fill-driven fund refresh.
- `staleness monitor`: no update for `stale_seconds` (default 30) → suppress that symbol's signals, log WARN.

### Cross-cutting (numerical suite)
A pytest module `tests/test_math_invariants.py` asserts every bound above (imbalance∈[-1,1], VPIN∈[0,1],
Kelly≤1, microprice in spread, Hurst∈[0,1], GARCH positive, OU positive-finite, z finite, POC in range).
**Definition of done for v1 = all invariants green + a 30-day paper run with zero unhandled exceptions.**

---

## 4. Edge Cases / Failure Modes

| # | Failure | Mitigation |
|---|---|---|
| 1 | **WS disconnect** | Exponential backoff reconnect; on resync, gap-detect via `seq`; pause symbol; no orders until healthy. |
| 2 | **HTTP 429 throttle** | Token bucket pre-guards sends; on 429 → exp backoff + requeue; persistent 429 → circuit-break that symbol. |
| 3 | **Broker reject** | Capture `reject_reason`; **never auto-retry a market order**; alert + audit; leave position state untouched. |
| 4 | **Stale data** | Staleness monitor (default 30s no tick/depth) → suppress signals; optional global kill if >1 symbol stale. |
| 5 | **Div-by-zero in z-score** | Zero-std guard → `0.0`/skip; never propagate `NaN` into risk sizing. |
| 6 | **India holiday / session** | Calendar gate (exchange holiday list + 09:15–15:30 IST); out-of-session → no India orders, crypto unaffected. |
| 7 | **Crypto 24/7 vs India session** | Per-asset `session` config; crypto always-on, India gated; loop reads per-symbol gate. |
| 8 | **API key / TOTP missing** | Fail **fast at startup** with explicit error; never enter live loop half-configured. |
| 9 | **Drawdown breach kill-switch** | `DrawdownGuard` latches → halt new entries, cancel open orders, flatten-or-hold (config), alert, write `KILLED`. |
| 10 | **Partial fill** | `Order.status` tracks partial; `Position.avg_price` recomputed from fill deltas; requeue remainder within limits. |
| 11 | **Clock skew / ts disorder** | Monotonic `seq` + `ts`; tolerate minor reorder; drop messages older than last-applied `seq`. |
| 12 | **Exchange maintenance** | Treated as disconnect (§1); auto-pause + reconnect loop. |
| 13 | **Kalman cold start** | Require min N obs before pairs signal; before that, indicator computes but emits no order. |
| 14 | **Microprice zero-size** | Return mid; never divide by zero size. |
| 15 | **GARCH non-stationary fit** | Flag + fall back to EWMA vol; do not emit negative variance. |

---

## 5. Data Contracts

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Tick:                       # trade print
    symbol: str
    ts: int                      # epoch ms
    price: float
    size: float
    side: str                    # "buy" | "sell" (aggressor if known)
    source: str                  # "crypto" | "india"

@dataclass
class Depth:                     # order-book snapshot / update
    symbol: str
    ts: int                      # epoch ms
    bids: list[tuple[float, float]]   # (price, size) descending
    asks: list[tuple[float, float]]   # (price, size) ascending
    is_snapshot: bool
    seq: Optional[int] = None

@dataclass
class Signal:
    id: str                      # uuid4
    symbol: str
    strategy: str                # "vol_breakout" | "vol_climax" | "low_vol_pullback" | "obv_divergence" | "pairs_kalman"
    side: str                    # "long" | "short"
    ts: int
    price: float
    confidence: float            # 0..1
    score: float                 # raw math score
    regime: str                  # "trend" | "range" | "volatile"
    stop_price: float
    target_price: Optional[float]
    size_pct_equity: float       # Kelly-capped, <= 0.01
    metadata: dict = field(default_factory=dict)

@dataclass
class Order:
    id: str                      # client order id (uuid4)
    signal_id: str
    symbol: str
    side: str                    # "buy" | "sell"
    order_type: str              # "market" | "limit"
    quantity: float
    price: Optional[float]
    broker: str                  # "ccxt" | "india"
    status: str                  # "pending"|"sent"|"acked"|"filled"|"partial"|"rejected"|"cancelled"
    ts_created: int
    ts_updated: int
    fill_price: Optional[float] = None
    reject_reason: Optional[str] = None

@dataclass
class Position:
    symbol: str
    side: str                    # "long" | "short" | "flat"
    qty: float
    avg_price: float
    unrealized_pnl: float
    realized_pnl: float
    broker: str
    updated_ts: int
```

---

## 6. Config Schema

`config/settings.py` (pydantic `BaseSettings`); creds come from **env only**, never committed.

```yaml
# autobot.yaml (non-secret)
environment: live                 # live | paper | backtest
equity_base: 100000.0             # reference capital for sizing (per account below)
risk:
  risk_per_trade_pct: 0.01        # 1% equity
  kelly_fraction_cap: 0.25        # cap Kelly at 25% of capital per position
  max_positions: 6
  max_drawdown_pct: 0.20          # KILL SWITCH threshold
  atr_stop_mult: 2.0
  order_rate_limit: 20            # Angel cap; token bucket
  flatten_on_kill: true
symbols:
  - ticker: BTC/USDT
    asset_class: crypto
    exchange: binance
    timeframes: [1m, 5m]
    strategies: [vol_breakout, vol_climax, low_vol_pullback, obv_divergence]
    enabled: true
  - ticker: RELIANCE
    asset_class: india
    exchange: NSE
    timeframes: [1m, 5m]
    strategies: [vol_breakout, pairs_kalman]
    enabled: true
session:
  india:
    open: "09:15"
    close: "15:30"
    timezone: "Asia/Kolkata"
    holidays_file: config/india_holidays.json   # NSE holiday list
  crypto:
    always_open: true
backtest:
  start: "2025-01-01"
  end: "2025-06-30"
  initial_capital: 100000.0
  freq: 1m
```

**Env vars (secret):**
```
BINANCE_API_KEY=...
BINANCE_API_SECRET=...
OPENALGO_URL=http://127.0.0.1:5000        # OpenAlgo local server
ANGEL_API_KEY=...
ANGEL_CLIENT_ID=...
ANGEL_PASSWORD=...
ANGEL_TOTP_SECRET=...                      # for TOTP generation
ANGEL_FEED_TOKEN=...                       # optional; else fetched at login
AUTOBO_CONFIG=D:/micro-saas/vstudio/GitHub/autobot/autobot.yaml
```
Validation at startup: ≥1 crypto + ≥1 india symbol enabled; all `rate>0`; missing required env → **hard fail**.

---

## 7. Out-of-Scope / v2

- **Co-located / true HFT** — no FPGA, kernel-bypass, or exchange colo; v1 is WS-driven, not matching-engine proximate.
- **ML / RL signals** — all v1 signals are classical math; no training pipeline.
- **Multi-pair portfolio optimization** — v1 trades the configured single crypto + single India name; no cross-asset
  risk parity or multi-leg basket.
- **Derivatives** — spot only (no options/futures/perps) in v1.
- **Tax / regulatory / audit reporting exports** — audit log is machine-readable JSONL only, no GST/capital-gains forms.
- **Mobile / web dashboard** — v1 is CLI + log/JSONL observability; a UI is a later phase.
- **Margin lending / short funding** — shorts use broker-supported intraday only; no borrow markets.

---

## 8. Open Questions for Founder (resolve before architect handoff)

1. **Capital model** — single combined equity, or separate crypto (USDT) / India (INR) accounts with independent
   drawdown guards? (Affects `DrawdownGuard` scope and PnL reporting currency.)
2. **Kill-switch behavior** — on breach: flatten all positions, or halt new entries and hold? (config `flatten_on_kill`)
3. **Paper mode** — broker sandbox (Binance testnet / OpenAlgo paper) or fully simulated fills from the WS tape?
4. **Backtest data source** — `ccxt.fetchOHLCV` for crypto is fine; Angel historical API is rate-limited and sparse —
   accept `backtesting.py`/vectorbt on CCXT data + a saved-tick replay for India?
5. **Reconciliation** — how often should cached `Position` be reconciled against broker `get_positions()`? (default: on
   every fill + every 60s)
6. **PnL currency** — report unified in INR (FX-convert crypto) or keep two books?
7. **Alerting channel** — Telegram / email / local log only for kill + reject events?

---

### Build-order recommendation (solo, value-first)
1. `config` + `contracts` + `brokers/base` (S) → 2. `ccxt_adapter` + `marketdata` (M) → 3. `math_*` modules with
invariant tests (M/L) → 4. `risk` + `signals_volume` (M/L) → 5. `live_loop` paper run (L) → 6. `india_adapter` +
session gate (L) → 7. `kill switch` + `audit` (S/M) → 8. `backtest/run` (M). Cut `signals_pairs`/`Johansen` to v1.1
if time-boxed — volume signals alone are shippable.
