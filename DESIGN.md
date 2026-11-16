# `autobot` — SYSTEM DESIGN (Architecture)

> Single-process, asyncio, execute-only trading bot. All math computed locally from the WS
> stream; brokers are touched **only** at execution. Maps 1:1 to the 17 SPEC modules plus two
> internal helpers (`engine/session_gate.py`, `backtest/cost_model.py`).
>
> Stack verified against prior volume-trading research (Aug 2026): `ccxt`, `vectorbt`,
> `arch`, `statsmodels`, `pandas-ta` (NOT TA‑Lib), `pyotp`, `pydantic`, `pyyaml`, `numpy`,
> `pandas`. Rationale: this exact stack is the documented solo-builder default; avoids the
> TA‑Lib C-build pain on Windows and the frozen `backtrader`.

---

## 1. Module Dependency Graph (acyclic)

Import direction = "X imports Y". Leaves (no project imports) are listed first. No cycles:
`risk` is imported by `brokers/*` and `live_loop` but imports nothing back; `audit` is a
leaf that everything writes to.

```
LEAVES (no project imports):
  config/settings.py ── pydantic, yaml, os
  data/contracts.py  ── dataclasses, typing
  math_microstructure.py ── numpy
  math_stats.py      ── numpy
  math_timeseries.py ── numpy, statsmodels
  math_vol.py        ── numpy, arch, pandas_ta
  logging/audit.py   ── json, pathlib   (serializes dicts only → no project imports)

MID (import leaves):
  risk.py            → math_vol, config/settings, data/contracts
  signals_volume.py  → math_microstructure, math_stats, math_vol, config/settings, data/contracts
  signals_pairs.py   → math_timeseries, math_vol, config/settings, data/contracts
  brokers/base.py    → data/contracts, config/settings
  engine/session_gate.py → config/settings

ADAPTERS (import base + risk + leaves):
  brokers/ccxt_adapter.py    → brokers/base, risk, data/contracts, config/settings
  brokers/india_adapter.py   → brokers/base, risk, data/contracts, config/settings, engine/session_gate

ORCHESTRATION (import adapters + signals + risk + leaves):
  engine/marketdata.py → brokers/*, data/contracts, config/settings, logging/audit, engine/session_gate
  engine/live_loop.py  → engine/marketdata, signals_volume, signals_pairs, risk, brokers/*,
                          data/contracts, config/settings, logging/audit

BACKTEST:
  backtest/cost_model.py → config/settings
  backtest/run.py        → signals_volume, signals_pairs, math_*, data/contracts,
                            config/settings, backtest/cost_model

CLI (top of tree):
  cli.py → config/settings, engine/live_loop, engine/marketdata, backtest/run, logging/audit
```

```
                 cli.py
        ┌─────────┬──────────┬──────────┐
   live_loop  marketdata   backtest   audit(leaf)
        │  ┌─────┴─────┐         │
   signals_*  brokers/*  math_*  cost_model
        │        │        │        │
      risk ──► math_vol  base   settings ──► contracts(leaf)
```

---

## 2. Data Flow (per-message pipeline)

Two queues per process: `raw_q` (adapter → marketdata) and `feature_q` (marketdata → live_loop).
Adapters never touch signals/risk; `live_loop` never touches the network except via
`broker.send_order`/`cancel`.

### Stage-by-stage with exact dataclasses

```
[Adapter WS callback]
  raw_binance_msg / raw_angel_msg
        │  normalize (depth→Depth, trade→Tick)
        ▼
  raw_q.put( Depth | Tick )                      # SPEC dataclass
        │
        ▼  engine/marketdata.py  (one task per adapter)
  ── session_gate.is_open(symbol, ts)?  ── no (India off-hours/holiday) → DROP
  ── staleness monitor: no update > stale_seconds(30)? → mark symbol PAUSED, DROP
  ── book_buf[sym].apply(depth); trade_buf[sym].append(tick)   # rolling deques
  frame = MarketFrame(symbol, ts, depth=book_buf.snapshot(),
                      ticks=trade_buf[-K:], session_open=bool, healthy=bool)
        ▼
  feature_q.put(frame)                           # INTERNAL dataclass
        │
        ▼  engine/live_loop.py  (single consumer task)
  feats = compute_features(frame)                # calls math_* → FeatureBundle
  signal = signals_volume.evaluate(feats, sym_cfg)
           or signals_pairs.evaluate(feats, sym_cfg)   # → Signal | None
        │ if None → DEBUG log, continue
        ▼
  decision = risk.gate(signal, portfolio, equity, cfg)   # → RiskDecision
        │ if not approved → audit(reject, signal_id); continue
        ▼
  bucket = throttle[signal.broker]               # TokenBucket (crypto=hi, india=20/s)
  if not bucket.admit(): bucket.queue(decision.order); → THROTTLED; continue
        ▼
  order = broker.send_order(decision.order)      # → Order (acked|rejected|...)
  audit(order, signal_id=signal.id)              # SPEC dataclass, JSONL
  on fill/partial → update Position cache + get_funds() (fill-driven, not hot path)
```

### Dataclasses passed between stages

| Hop | Type | Defined in |
|---|---|---|
| adapter → marketdata | `Depth` / `Tick` | `data/contracts.py` (SPEC) |
| marketdata → live_loop | `MarketFrame(symbol, ts, depth: Depth, ticks: list[Tick], session_open: bool, healthy: bool)` | internal |
| live_loop compute | `FeatureBundle(symbol, ts, imbalance, microprice, weighted_mid, vpin, hurst, shannon, apen, amihud, garch_var, garch_unstable, vp_poc, vp_val, vp_vah, atr, obv, obv_slope, kalman_beta, ou_half_life, spread_z, regime, raw_depth, raw_ticks)` | internal |
| signals → risk | `Signal` | `data/contracts.py` (SPEC) |
| risk → throttle | `RiskDecision(order: Optional[Order], approved: bool, reason: str)` | internal |
| throttle → broker | `Order` | `data/contracts.py` (SPEC) |
| broker → audit | `Order` (with status/fill_price/reject_reason) | `data/contracts.py` (SPEC) |
| fill → cache | `Position` | `data/contracts.py` (SPEC) |

`MarketFrame`, `FeatureBundle`, `RiskDecision` are the only new dataclasses; everything else
is from SPEC §5. Rationale: keep the wire contract identical to SPEC, add thin internal
transport types so stages stay decoupled and unit-testable.

---

## 3. Live Loop State Machine

Single `asyncio.run(main_live(settings))`. `main_live` builds adapters + tasks, then awaits
a `kill_event` (asyncio.Event). All stages above are coroutines/tasks in **one** loop.

### States

| State | Meaning | Entered when |
|---|---|---|
| `INIT` | config load + validation + adapter construction | start |
| `CONNECTED` | WS healthy, no open positions yet | `config_ok` from INIT |
| `TRADING` | ≥1 open position OR actively sending orders | first acked fill / first approved send |
| `FLAT` | all positions zero, still evaluating | `all_flat` (no open pos) |
| `THROTTLED` | token bucket empty / 429 circuit-break for a broker | `throttle_breach` |
| `PAUSED` | symbol-level: WS stale/disconnected; that symbol's signals suppressed, others keep trading | `symbol_unhealthy` |
| `KILL` | global terminal: kill-switch latched; new entries stopped, open orders cancelled, optional flatten | `kill_switch` (manual or auto) |

### Events → transitions

```
INIT --config_ok--> CONNECTED
CONNECTED --first_fill / first_send--> TRADING
TRADING --all_flat--> FLAT
FLAT --approved_signal--> TRADING
TRADING/FLAT --throttle_breach--> THROTTLED
THROTTLED --throttle_recover--> (TRADING if pos else FLAT)
* --symbol_unhealthy--> PAUSED(symbol)   # additive, non-global
PAUSED(symbol) --symbol_healthy--> back to prior state
* --kill_switch--> KILL   # terminal; only restart clears
KILL --restart--> INIT
```

`THROTTLED` and `PAUSED` are **orthogonal** to `TRADING`/`FLAT` (a symbol can be PAUSED while
the bot is globally TRADING on crypto). `KILL` overrides everything.

### How crypto (24/7) and India (session-gated) coexist without blocking

- Each adapter runs its **own coroutine(s)** that `await` its WS read loop
  (`ex.watch_order_book` / `watch_trades` for CCXT; OpenAlgo depth WS for India). They push to
  `raw_q` independently — crypto never awaits India, India never awaits crypto.
- `marketdata` drains `raw_q`; the **session gate is evaluated per-frame**, so India frames
  outside 09:15–15:30 IST / holidays are dropped at the gate and never reach `feature_q`.
  Crypto frames always pass. ⇒ India's off-hours cost the loop *zero* compute.
- `live_loop` is a single consumer of `feature_q`; for the configured 1 crypto + 1 India name
  the per-frame math is sub-millisecond, so no threadpool is needed. (If scaled, offload
  `compute_features` via `loop.run_in_executor` — already isolated behind one call site.)
- All cross-task coordination is `asyncio.Queue` / `asyncio.Event` — **no blocking I/O, no
  `time.sleep`** (reconnect uses `asyncio.sleep` with exponential backoff).

---

## 4. BrokerAdapter Contract

`brokers/base.py` — `BrokerAdapter(ABC)`. **Execute-only boundary:** adapters normalize data
and send/cancel orders + fetch positions/funds. They never import `math_*`, `signals_*`,
or `risk`. All compute happens in `live_loop`.

```python
class BrokerAdapter(ABC):
    # ── lifecycle ──
    async def connect(self) -> None: ...          # auth + open WS (no data yet)
    async def close(self) -> None: ...            # close WS + cancel subscriptions

    # ── data (normalize to SPEC dataclasses, push via callbacks) ──
    async def subscribe(self, symbols: list[str]) -> None: ...
    def on_depth(self, cb: Callable[[Depth], None]) -> None: ...
    def on_trade(self, cb: Callable[[Tick], None]) -> None: ...
    def on_fill(self, cb: Callable[[Order], None]) -> None: ...
    def on_order_update(self, cb: Callable[[Order], None]) -> None: ...

    # ── execution (the ONLY broker network calls in the hot path) ──
    async def send_order(self, order: Order) -> Order: ...     # returns Order w/ status+ack
    async def cancel_order(self, order_id: str) -> Order: ...

    # ── state (called on fill + every 60s ONLY, never in hot path) ──
    async def get_positions(self) -> list[Position]: ...
    async def get_funds(self) -> dict: ...

    # ── session (default: always open) ──
    def is_session_open(self, symbol: str, ts: int) -> bool: ...
```

### CCXTAdapter (crypto)
- `connect`: `ccxt.binance({'apiKey','secret', 'enableRateLimit':True})`; if `paper`, point at
  Binance testnet (`set_sandbox_mode(True)`).
- `subscribe`: per symbol, loop `await ex.watch_order_book(sym)` → normalize to `Depth`
  (rebuild top-N levels, assign monotonic `seq`); `await ex.watch_trades(sym)` → `Tick`.
  **No REST polling** except auth.
- `send_order`: `ex.create_order(sym, type, side, amount, price)` → map result to `Order`
  (`acked`/`rejected` + `reject_reason`).
- `cancel_order`: `ex.cancel_order(client_id, sym)`.
- `get_positions`/`get_funds`: `fetch_positions` / `fetch_balance` (fill/60s only).
- `is_session_open`: **always `True`** (crypto 24/7).

### IndiaAdapter (Angel SmartAPI behind OpenAlgo)
- `connect`: `pyotp.TOTP(ANGEL_TOTP_SECRET).now()` → POST login to `OPENALGO_URL` with
  `ANGEL_API_KEY/CLIENT_ID/PASSWORD/token`; cache `feed_token`.
- `subscribe`: OpenAlgo depth WS for each symbol → normalize `Depth`/`Tick`.
- `send_order`: POST order to OpenAlgo endpoint → `Order` ack/reject. **Wrapped by
  `TokenBucket(rate=20, capacity=20)` + exponential backoff on HTTP 429** (requeue, circuit-break
  the symbol on persistent 429).
- `cancel_order`: OpenAlgo cancel endpoint.
- `get_positions`/`get_funds`: OpenAlgo positions/funds (fill/60s).
- `is_session_open`: delegate to `engine/session_gate.IndiaSession` (09:15–15:30 IST +
  `config/india_holidays.json`; crypto symbols always `True`).

---

## 5. Math Engine Layout

Four files (user's `math_volume.py` ≡ SPEC `math_vol.py`; plus `math_timeseries.py` for
Kalman/OU/coint). All pure functions: `numpy` in, `float`/`tuple`/`ndarray` out, **no I/O, no
global state** → directly reusable in backtest. Every return honors the SPEC §3 invariants.

### `math_microstructure.py`
```python
def order_book_imbalance(bids: list[tuple[float,float]], asks: list[tuple[float,float]],
                         levels: int = 5) -> float:
    """∈ [-1,1]; weighted by size; 0.0 on empty/zero-size book (no div-by-zero)."""

def microprice(bid: float, bid_sz: float, ask: float, ask_sz: float) -> float:
    """Strictly inside (bid,ask) when both sz>0; returns mid if either sz==0."""

def weighted_mid(bids, asks, levels: int = 5) -> float:
    """Σ(p·sz)/Σ(sz) over both sides."""

def book_pressure(bids, asks, levels: int = 5) -> float:
    """∈ [-1,1]; (bid_vol-ask_vol)/(bid_vol+ask_vol), 0.0 on empty."""
```

### `math_stats.py`
```python
def vpin(trades: list[Tick], bucket_size: float, window: int = 50) -> float:
    """∈ [0,1]; (|buy_vol-sell_vol|/total_vol) over rolling buckets; 0.0 on zero total vol."""

def hurst_rs(series: np.ndarray) -> float:
    """∈ [0,1]; R/S exponent. Brownian ~0.5±0.1 on 2000+ samples."""

def shannon_entropy(series: np.ndarray, bins: int = 10) -> float:
    """Normalized ∈ [0,1]; -Σ p·log2(p) / log2(bins)."""

def approx_entropy(series: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """≥ 0; random > ordered series."""

def amihud(returns: np.ndarray, volumes: np.ndarray) -> float:
    """≥ 0; mean(|ret|/vol); zero-volume bars contribute 0 (no div-by-zero)."""
```

### `math_timeseries.py`
```python
def kalman_hedge_ratio(y: np.ndarray, x: np.ndarray, *, delta: float = 1e-4
                       ) -> tuple[float, np.ndarray]:
    """Custom Kalman (NO pykalman). Returns (β, P_diag_history).
       synthetic y=0.5x+noise → β≈0.5 within 5% after ≥200 obs; P non-negative-def, shrinks."""

def ou_half_life(series: np.ndarray) -> Optional[float]:
    """Finite & positive for mean-reverting; None when b≥0 (random walk). Signal path skips None."""

def adf_test(series: np.ndarray) -> tuple[float, float]:
    """(statistic, pvalue); pvalue<0.05 ⇒ cointegrated (statsmodels wrapper)."""

def johansen(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(eigenvalues, trace_stats) via statsmodels."""

def spread_z(spread: np.ndarray, window: int = 60) -> np.ndarray:
    """Finite; zero/std guard → 0.0 (skip), never propagates NaN to risk."""
```

### `math_vol.py`
```python
def garch_11(returns: np.ndarray, horizon: int = 1) -> tuple[float, bool]:
    """(forecast_var, unstable_flag). Positive var always; if alpha+beta≥1 → unstable=True
       and fall back to EWMA vol (never emit negative variance)."""

def volume_profile(prices: np.ndarray, volumes: np.ndarray, bins: int = 100
                   ) -> tuple[float, float, float]:
    """(poc, val, vah). poc ∈ [min,max]; [val,vah] contains ≥0.68 (cfg) of total volume."""

def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> float:
    """Matches pandas_ta.ATR within 1e-6 on fixture."""

def kelly_fraction(win_rate: float, payoff: float) -> float:
    """Clamped [0,1]; negative edge → 0.0."""
```

### Invariant table (asserted in `tests/test_math_invariants.py`)
| Function | Invariant |
|---|---|
| `order_book_imbalance` | ∈ [-1, 1]; empty → 0.0 |
| `microprice` | strictly inside (bid,ask) when sz>0; else mid |
| `vpin` | ∈ [0, 1]; zero vol → 0.0 |
| `hurst_rs` | ∈ [0, 1]; Brownian ≈ 0.5±0.1 |
| `shannon_entropy` | ∈ [0, 1] |
| `approx_entropy` | ≥ 0; random > ordered |
| `amihud` | ≥ 0; zero-vol bar → 0 contribution |
| `kalman_hedge_ratio` | β→0.5±5% after ≥200 obs; P non-neg-def, shrinks |
| `ou_half_life` | finite>0 for MR; None for b≥0 |
| `spread_z` | finite; zero/std → 0.0, no NaN |
| `garch_11` | var>0; unstable→EWMA fallback |
| `volume_profile` | poc∈[min,max]; VA covers ≥0.68 vol |
| `kelly_fraction` | clamped [0,1]; negative edge→0.0 |

---

## 6. Config Loading

`config/settings.py` — `pydantic` `BaseSettings`. Layering: **env secrets** (never committed)
override **YAML** (`AUTOBO_CONFIG` path, default `autobot.yaml`).

```python
class RiskConfig(BaseModel):
    risk_per_trade_pct: float = 0.01
    kelly_fraction_cap: float = 0.25      # cap per-position Kelly at 25%
    max_positions: int = 6
    max_drawdown_pct: float = 0.20       # KILL threshold
    atr_stop_mult: float = 2.0
    order_rate_limit: int = 20
    flatten_on_kill: bool = True

class SymbolConfig(BaseModel):
    ticker: str; asset_class: Literal["crypto","india"]; exchange: str
    timeframes: list[str]; strategies: list[str]; enabled: bool = True

class Settings(BaseSettings):
    environment: Literal["live","paper","backtest"]
    equity_base: float = 100000.0
    risk: RiskConfig
    symbols: list[SymbolConfig]
    session: SessionConfig
    backtest: BacktestConfig
    # secrets (env only)
    binance_api_key: str | None = None
    binance_api_secret: str | None = None
    openalgo_url: str | None = None
    angel_api_key: str | None = None
    angel_client_id: str | None = None
    angel_password: str | None = None
    angel_totp_secret: str | None = None
    angel_feed_token: str | None = None
    model_config = SettingsConfigDict(env_prefix="AUTOBO_", extra="ignore")
```

**Fail-fast validation** (`model_post_init`):
1. ≥1 `crypto` symbol enabled **and** ≥1 `india` symbol enabled → else `ValueError`.
2. All rates/`risk_per_trade_pct`/`max_drawdown_pct` > 0 → else `ValueError`.
3. If `environment != "backtest"`: required secrets for every *enabled* broker present
   (`BINANCE_*` for crypto, `ANGEL_*`+`OPENALGO_URL` for india) → else `ValueError` with the
   exact missing key. Never enter the live loop half-configured (SPEC edge #8).
4. `india_holidays.json` exists if any india symbol enabled.

Rationale: pydantic gives typed config + one validation seam; CLI exits non-zero with the
precise missing field so the operator fixes it in one pass.

---

## 7. Backtest Harness

`backtest/run.py` — **reuses the live signal functions**, does not reimplement them.
`vectorbt.Portfolio.from_signals` consumes boolean `entries`/`exits` per strategy.

### Flow
1. **Load data.** Crypto: `ccxt.fetchOHLCV` (OHLCV for price/ATR/VPIN) + saved trade tape for
   volume features. India: **saved-tick replay** (Angel historical is rate-limited/sparse) —
   `Trade`/`Depth` frames re-fed through the same `compute_features` → `signals_*.evaluate`.
   Rationale: matches SPEC open-question #4 (accept CCXT + saved-tick replay for India).
2. **Walk-forward.** Split `[start,end]` into rolling windows (default 3-month in-sample /
   1-month out-of-sample, step forward). Parameters (Kelly cap, ATR mult, VPIN thresholds) are
   **fixed per window from config** and re-estimated only on the rolling schedule — never on
   the bar being traded (SPEC §3 walk-forward guard). Aggregate metrics (Sharpe / Sortino /
   max-DD) over **out-of-sample** windows only.
3. **Session/holiday mask.** For India symbols, `entries`/`exits` are AND-masked to
   09:15–15:30 IST bars excluding `india_holidays.json`; crypto unmasked.
4. **Costs.** `backtest/cost_model.CostModel` (shared with live pre-trade check):
   - `slippage_pct(asset_class)` → vectorbt `slippage=`.
   - `fee(side, notional, asset_class)` → STT applies to **sell** turnover for India equity;
     modeled by reducing exit proceeds (vectorbt symmetric `fees` can't express sell-only STT,
     so apply STT as a post-adjustment on exits). Brokerage = pct (capped) + flat exchange fee.
   - Binance maker/taker pct for crypto.

```python
class CostModel:
    def fee(self, side: str, notional: float, asset_class: str) -> float: ...
    def slippage_pct(self, asset_class: str) -> float: ...
```

Rationale: one cost model for live + backtest keeps slippage/STT honest and comparable.

---

## 8. Deployment Topology

**Single process, single asyncio event loop, multiple tasks:**

```
autobot run --live
  └─ asyncio.run(main_live(settings))
       ├─ task: ccxt_adapter WS loop      (pushes raw_q)
       ├─ task: india_adapter WS loop     (pushes raw_q)
       ├─ task: marketdata drain         (raw_q → feature_q, session+staleness gate)
       ├─ task: live_loop consumer       (feature_q → signals → risk → throttle → broker)
       ├─ task: kill_monitor             (watches kill_event; on set → cancel+flatten+audit)
       ├─ task: reconcile_timer          (get_positions/get_funds every 60s + on fill)
       └─ awaits kill_event
```

**Run modes** (via `cli.py`):
- `autobot run --paper` → adapters in paper mode (Binance testnet / OpenAlgo paper, or simulated
  fills from the WS tape if broker sandbox unavailable). No real money.
- `autobot run --live` → real adapters.
- `autobot backtest` → offline, no WS, reads saved data.

**Logs / state** (under repo `state/` and `logs/`):
- `logs/audit.jsonl` — every `Signal`/`Order`/`fill`/`reject`/`kill` with `signal_id` (SPEC §1.12).
- `logs/app.log` — rotating structured log (INFO/DEBUG/WARN).
- `state/positions.json` — periodic snapshot of `Position` cache for restart reconciliation.
- `state/kill_flag` — written as `KILLED <ts>` when kill-switch latches; **presence = latched**;
  `main_live` refuses to start trading while the flag exists (operator must delete after review).

**Kill-switch triggers:**
- *Manual:* `autobot kill` (CLI) → sets the shared `kill_event`; also an emergency keypress in
  the running CLI (keyboard-interrupt handler sets `kill_event`).
- *Auto:* `DrawdownGuard` latches the moment realized+unrealized DD ≥ `max_drawdown_pct` → sets
  `kill_event`.
- On trigger: `kill_monitor` cancels all open orders, flattens if `flatten_on_kill`, writes
  `KILLED`, alerts (local log + optional Telegram stub), then stops the loop (terminal → restart
  to recover).

---

## 9. File Tree

```
autobot/
  autobot.yaml                 # non-secret config (secrets come from env only)
  config/
    __init__.py
    settings.py                # pydantic Settings + fail-fast validation (§6)
    india_holidays.json        # NSE holiday list for session gate
  data/
    __init__.py
    contracts.py               # Tick/Depth/Signal/Order/Position dataclasses (SPEC §5)
  math_microstructure.py       # imbalance/microprice/weighted_mid/book_pressure (§5)
  math_stats.py                # vpin/hurst_rs/shannon_entropy/approx_entropy/amihud
  math_timeseries.py           # kalman_hedge_ratio/ou_half_life/adf_test/johansen/spread_z
  math_vol.py                  # garch_11/volume_profile/atr/kelly_fraction
  signals_volume.py            # 4 volume patterns + trend filter + 2×ATR stop + walk-fwd gate
  signals_pairs.py             # kalman spread + OU half-life entry/exit (min-N warmup)
  risk.py                      # Kelly sizer, TokenBucket, DrawdownGuard, per-symbol/account limits
  brokers/
    __init__.py
    base.py                    # BrokerAdapter ABC (§4)
    ccxt_adapter.py            # Binance WS depth/trades + orders (crypto, 24/7)
    india_adapter.py           # OpenAlgo/Angel SmartAPI + TOTP + 20/s bucket + session gate
  engine/
    __init__.py
    marketdata.py              # WS orchestration, normalize, buffers, staleness monitor
    live_loop.py               # asyncio pipeline + state machine (§3)
    session_gate.py            # India 09:15–15:30 IST + holidays; crypto always-open
  backtest/
    __init__.py
    run.py                     # vectorbt harness, walk-forward, OOS split, session mask
    cost_model.py              # slippage/brokerage/STT — shared live + backtest
  logging/
    __init__.py
    audit.py                   # JSONL audit + app logging config
  cli.py                       # run --live/--paper, backtest, status, kill
  tests/
    conftest.py
    test_math_invariants.py    # all §5 bounds (Definition of Done for v1)
    test_contracts.py          # dataclass round-trips
  requirements.txt             # ccxt, vectorbt, arch, statsmodels, pandas-ta, pyotp,
                               #   pydantic, pyyaml, numpy, pandas
  .env.example                 # documented secret keys (template, not real)
  README.md
```

---

## Key Decisions (with rationale)

| # | Decision | Rationale |
|---|---|---|
| D1 | Two queues (`raw_q`, `feature_q`); adapters never import math/signals/risk | Enforces SPEC rule #1 (compute locally, broker execute-only) and keeps stages unit-testable. |
| D2 | Session gate evaluated **per-frame in marketdata**, not in live_loop | India off-hours costs the loop zero compute; crypto never blocked (§3 requirement). |
| D3 | `IndiaAdapter` wraps every send in `TokenBucket(20/s)` + 429 backoff | Satisfies SPEC rule #4 and edge #2 (no >20 orders/s to Angel). |
| D4 | Custom Kalman, no `pykalman` | SPEC rule #3; also one fewer dependency to break on Windows. |
| D5 | `garch_11` falls back to EWMA when `alpha+beta≥1` | SPEC edge #15 — never emit negative variance. |
| D6 | `spread_z`/`imbalance`/`vpin`/`kelly` all guarded → 0.0 / clamp, never `NaN` | SPEC edge #5 and §3 invariants — `NaN` must never reach risk sizing. |
| D7 | Backtest imports `signals_*.evaluate` + `math_*` directly; India = saved-tick replay | Avoids signal/backtest divergence (single source of truth) and matches SPEC open-q #4. |
| D8 | `CostModel` shared by live pre-trade check and vectorbt | Keeps slippage/STT identical across modes. |
| D9 | `kill_flag` file + `kill_event` Event; auto = DrawdownGuard latch | Two independent triggers (manual + auto), terminal until operator reviews. |
| D10 | Separate `equity` / `DrawdownGuard` per account (USDT vs INR) | Resolves open-q #1 default: independent drawdown guards, no cross-currency FX assumption. |
| D11 | Paper mode = broker sandbox OR simulated fills from tape | Resolves open-q #3 with the lowest-effort path that still exercises the full pipeline. |
| D12 | Reconcile `Position` cache on every fill + every 60s | Resolves open-q #5 default; catches drift without hot-path polling. |
| D13 | PnL reported as two books (no FX convert) | Resolves open-q #6 default; avoids inventing an FX source. |
| D14 | Alerting = local log + optional Telegram stub | Resolves open-q #7 minimally; stub kept out of hot path. |
| D15 | Build order per SPEC §8 (config→contracts→base→ccxt→math→risk→signals→live_loop paper→india→kill→backtest) | Value-first; volume signals shippable before pairs; matches solo build-order recommendation. |

**Definition of Done for v1** = all §5 invariants green in `tests/test_math_invariants.py` **and**
a 30-day paper run with zero unhandled exceptions.
