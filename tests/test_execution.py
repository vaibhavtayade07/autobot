"""Execution/safety layer tests (the SHIP-BLOCKER fixes).

These cover the fixes that don't require live brokers / vectorbt / ccxt:
  * kill switch wiring via the state/kill_flag file (#1)
  * real order quantity from size_pct_equity (#2, #7)
  * THROTTLED queue drain/recovery (#4)
  * short backtest direction arrays (#5)
Plus keyless CLI behavior (paper --simulate exits 0, plain paper missing keys exits 2).
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import autobot.engine.live_loop as ll
from autobot.config.settings import Settings
from autobot.data.contracts import Order, Signal
from autobot.engine.live_loop import LiveLoop, State, risk_gate
from autobot.math_risk import TokenBucket


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_signal(symbol="BTC/USDT", side="long", price=100.0, stop=96.0):
    return Signal(
        id="x", symbol=symbol, strategy="s", side=side, ts=0,
        price=price, confidence=0.5, score=1.0, regime="range",
        stop_price=stop, target_price=None, size_pct_equity=0.01, metadata={},
    )


class FakeAdapter:
    def __init__(self):
        self.sent = 0

    async def send_order(self, order):
        order.status = "acked"
        self.sent += 1
        return order


# ---------------------------------------------------------------------------
# #1  kill switch wiring via simulated flag file
# ---------------------------------------------------------------------------


def test_kill_via_flag_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ll, "STATE_DIR", tmp_path)
    monkeypatch.setattr(ll, "KILL_FLAG", tmp_path / "kill_flag")
    settings = Settings.from_config(environment="backtest")
    loop_obj = LiveLoop(settings, paper=True)

    async def runner():
        lp = asyncio.get_running_loop()
        loop_obj._start_kill_watcher(lp)  # the file-watcher thread (#1b)
        # simulate the `kill` CLI subcommand writing the flag file (#1c)
        ll.KILL_FLAG.write_text("KILLED cli\n")
        await asyncio.wait_for(loop_obj.kill_event.wait(), timeout=5.0)

    asyncio.run(runner())
    assert loop_obj.kill_event.is_set()


def test_kill_flag_written_by_cli(tmp_path, monkeypatch):
    """The `kill` subcommand writes the flag the watcher watches (#1c)."""
    monkeypatch.setattr(ll, "STATE_DIR", tmp_path)
    monkeypatch.setattr(ll, "KILL_FLAG", tmp_path / "kill_flag")
    from autobot.cli import _cmd_kill

    assert _cmd_kill(None) == 0
    assert ll.KILL_FLAG.exists()
    assert ll.KILL_FLAG.read_text(encoding="utf-8").strip() != ""


# ---------------------------------------------------------------------------
# #2 + #7  order quantity from size_pct_equity, with cost logged
# ---------------------------------------------------------------------------


def test_quantity_from_size_pct_equity():
    dec = risk_gate(_make_signal(), 100000.0, None)
    assert dec.approved
    # crypto: equity * pct / price
    assert abs(dec.order.quantity - (100000.0 * 0.01) / 100.0) < 1e-9
    assert dec.est_cost > 0  # CostModel fee logged (#7)

    dec2 = risk_gate(
        _make_signal(symbol="RELIANCE", side="short", price=2500.0, stop=2700.0),
        100000.0, None, asset_class="india", lot_size=50.0,
    )
    assert dec2.approved
    # india: equity * pct / (price * lot_size)
    expected = (100000.0 * 0.01) / (2500.0 * 50.0)
    assert abs(dec2.order.quantity - expected) < 1e-12


def test_quantity_zero_or_invalid_rejected():
    # price <= 0 -> reject (no qty=0 order sent)
    bad = _make_signal(price=0.0, stop=1.0)
    dec = risk_gate(bad, 100000.0, None)
    assert not dec.approved
    assert "price" in dec.reason.lower()


# ---------------------------------------------------------------------------
# #4  THROTTLED queue drains once the token bucket recovers
# ---------------------------------------------------------------------------


def test_throttled_drain():
    settings = Settings.from_config(environment="backtest")
    loop_obj = LiveLoop(settings, paper=True)
    bucket = TokenBucket(rate=100, capacity=100)
    bucket.tokens = 0.0  # simulate exhausted capacity -> throttled
    bucket.last = time.monotonic()
    order = Order(
        id="o1", signal_id="s", symbol="BTC/USDT", side="buy", order_type="market",
        quantity=1.0, price=100.0, broker="ccxt", status="pending",
        ts_created=0, ts_updated=0,
    )
    bucket.queue(order)
    loop_obj.buckets = {"BTC/USDT": bucket}
    fake = FakeAdapter()
    loop_obj._adapters = {"BTC/USDT": fake}

    async def run():
        await loop_obj._drain_queue("BTC/USDT", bucket)

    asyncio.run(run())
    # still throttled -> nothing sent, still queued
    assert fake.sent == 0
    assert bucket.pending() == 1

    # capacity restored -> drain re-sends and recovers to TRADING
    bucket.tokens = 5.0
    asyncio.run(run())
    assert fake.sent == 1
    assert bucket.pending() == 0
    assert loop_obj.state == State.TRADING


# ---------------------------------------------------------------------------
# #5  short backtest direction arrays
# ---------------------------------------------------------------------------


def test_short_backtest_direction():
    from autobot.backtest.run import _build_signal_arrays

    n = 5
    le, lx, se, sx = _build_signal_arrays(n, [(1, "short"), (3, "long")])
    assert se[1] and not le[1]            # bar 1 is a SHORT entry only
    assert le[3] and not se[3]            # bar 3 is a LONG entry only
    assert sx[3]                          # short closed when long appears
    assert lx[4]                          # long closed on the next bar
    # long and short never overlap on the same bar
    for i in range(n):
        assert not (le[i] and se[i])


# ---------------------------------------------------------------------------
# CLI: keyless dry-run + clean missing-keys exit
# ---------------------------------------------------------------------------


def _autobot_env():
    env = dict(os.environ)
    for k in list(env):
        if k.startswith("AUTOBO_"):
            del env[k]
    env.pop("AUTOBO_CONFIG", None)
    return env


def test_paper_simulate_keyless():
    env = _autobot_env()
    res = subprocess.run(
        [sys.executable, "-m", "autobot.cli", "paper", "--simulate"],
        capture_output=True, text=True, env=env,
    )
    assert res.returncode == 0, res.stderr
    assert "Traceback" not in res.stderr
    assert "pnl" in res.stdout.lower()


def test_paper_missing_keys_exits_2():
    env = _autobot_env()
    res = subprocess.run(
        [sys.executable, "-m", "autobot.cli", "paper"],
        capture_output=True, text=True, env=env,
    )
    assert res.returncode == 2
    assert "Traceback" not in res.stderr
    assert "requires broker secrets" in res.stderr
