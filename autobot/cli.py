"""Command-line entrypoint.

Subcommands:
  backtest  - run the vectorbt walk-forward harness (needs data + vectorbt)
  paper     - run live pipeline against broker sandboxes / simulated fills
  live      - run live pipeline against real brokers (requires secrets)
  check     - run the math self-test (offline; numpy only, no broker deps)

`check` imports only the math_* modules, so it runs with numpy/pandas/scipy present
and the heavy deps (ccxt/vectorbt/arch/statsmodels) absent.
"""
from __future__ import annotations

import argparse
import sys


def run_math_selftest() -> list[tuple[str, bool, str]]:
    """Run every numerical invariant from SPEC §3 / DESIGN §5.

    Returns a list of (name, passed, detail). Importing happens lazily inside so
    this works offline with only numpy/pandas/scipy.
    """
    import numpy as np

    from autobot.math_microstructure import (
        amihud,
        book_pressure,
        microprice,
        order_book_imbalance,
        vpin,
        weighted_mid,
    )
    from autobot.math_stats import (
        approx_entropy,
        cointegration_pvalue,
        hurst_rs,
        kalman_hedge_ratio,
        ou_half_life,
        shannon_entropy,
        spread_z,
    )
    from autobot.math_vol import (
        chaikin_money_flow,
        garch_11,
        obv,
        obv_slope,
        volume_profile,
        volume_ratio,
    )
    from autobot.math_risk import (
        atr,
        DrawdownGuard,
        fractional_kelly,
        kelly_fraction,
        risk_based_size_pct,
        stop_price,
        TokenBucket,
    )

    results: list[tuple[str, bool, str]] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        results.append((name, bool(cond), detail))

    rng = np.random.default_rng(42)

    # --- order_book_imbalance ---
    try:
        empty = order_book_imbalance([], [])
        b = order_book_imbalance([(100, 10), (99, 5)], [(101, 2), (102, 3)])
        check("imbalance_empty_zero", empty == 0.0, f"empty={empty}")
        check("imbalance_bounds", -1.0 <= b <= 1.0, f"v={b:.4f}")
    except Exception as e:  # pragma: no cover
        check("imbalance", False, f"exception: {e}")

    # --- microprice ---
    try:
        mp = microprice(100.0, 1.0, 101.0, 2.0)
        mid = microprice(100.0, 0.0, 101.0, 0.0)
        check("microprice_in_spread", 100.0 < mp < 101.0, f"mp={mp:.6f}")
        check("microprice_zero_size_mid", abs(mid - 100.5) < 1e-9, f"mid={mid}")
    except Exception as e:  # pragma: no cover
        check("microprice", False, f"exception: {e}")

    # --- weighted_mid / book_pressure ---
    try:
        wm = weighted_mid([(100, 10)], [(101, 5)])
        bp = book_pressure([(100, 10)], [(101, 5)])
        check("weighted_mid_finite", np.isfinite(wm), f"wm={wm}")
        check("book_pressure_bounds", -1.0 <= bp <= 1.0, f"bp={bp:.4f}")
    except Exception as e:  # pragma: no cover
        check("weighted_mid", False, f"exception: {e}")

    # --- vpin / amihud ---
    try:
        trades_all_buy = [(100.0, 1.0, "buy") for _ in range(200)]
        vp = vpin(trades_all_buy, bucket_size=10.0, window=10)
        vp_empty = vpin([], bucket_size=10.0)
        rets = rng.normal(0, 0.01, 100)
        vols = rng.uniform(1, 10, 100)
        am = amihud(rets, vols)
        am_zero = amihud(rets, np.zeros(100))
        check("vpin_bounds", 0.0 <= vp <= 1.0, f"vp={vp:.4f}")
        check("vpin_zero_vol_zero", vp_empty == 0.0, f"vp_empty={vp_empty}")
        check("amihud_nonneg", am >= 0.0, f"am={am:.6e}")
        check("amihud_zero_vol_zero", am_zero == 0.0, f"am0={am_zero}")
    except Exception as e:  # pragma: no cover
        check("vpin_amihud", False, f"exception: {e}")

    # --- kalman hedge ratio ---
    try:
        x = 100.0 + np.cumsum(rng.normal(0, 1, 300))
        y = 0.5 * x + rng.normal(0, 0.1, 300)
        beta, P = kalman_hedge_ratio(y, x)
        shrinks = bool(P[-1] < P[0])
        check("kalman_beta_5pct", abs(beta - 0.5) <= 0.025, f"beta={beta:.4f}")
        check("kalman_P_shrinks", shrinks, f"P0={P[0]:.2f} Pn={P[-1]:.6f}")
        check("kalman_P_nonneg", bool(np.all(P >= 0)), "")
    except Exception as e:  # pragma: no cover
        check("kalman", False, f"exception: {e}")

    # --- cointegration (lazy statsmodels) ---
    try:
        y2 = 0.5 * x + rng.normal(0, 0.1, 300)
        stat, p = cointegration_pvalue(y2, x)
        if np.isnan(p):
            check("coint_statsmodels_absent", True, "statsmodels not installed -> SKIP (lazy)")
        else:
            check("coint_pvalue_lt_05", p < 0.05, f"p={p:.4f}")
    except Exception as e:  # pragma: no cover
        check("coint", False, f"exception: {e}")

    # --- spread_z ---
    try:
        sp = rng.normal(0, 1, 120)
        z = spread_z(sp, window=60)
        z_flat = spread_z(np.ones(50), window=20)
        check("spread_z_finite", bool(np.all(np.isfinite(z))), "")
        check("spread_z_zero_std_zero", bool(np.all(z_flat == 0.0)), f"zflat={z_flat[:3]}")
    except Exception as e:  # pragma: no cover
        check("spread_z", False, f"exception: {e}")

    # --- ou_half_life ---
    try:
        # mean-reverting: X_t = 0.95 X_{t-1} + noise
        mr = np.zeros(400)
        for i in range(1, 400):
            mr[i] = 0.95 * mr[i - 1] + rng.normal(0, 0.1)
        mr = mr[50:]
        hl = ou_half_life(mr)
        rw = np.cumsum(rng.normal(0, 1, 400))
        hl_rw = ou_half_life(rw)
        check("ou_half_life_finite_pos", hl is not None and hl > 0 and np.isfinite(hl), f"hl={hl}")
        check("ou_half_life_rw_none", hl_rw is None, f"hl_rw={hl_rw}")
    except Exception as e:  # pragma: no cover
        check("ou_half_life", False, f"exception: {e}")

    # --- entropy / hurst ---
    try:
        noise = rng.normal(0, 1, 50)
        H = shannon_entropy(noise, bins=10)
        H_const = shannon_entropy(np.ones(50), bins=10)
        check("shannon_bounds", 0.0 <= H <= 1.0, f"H={H:.4f}")
        check("shannon_constant_zero", H_const == 0.0, f"Hc={H_const}")
        ap_rand = approx_entropy(rng.normal(0, 1, 200))
        ap_ord = approx_entropy(np.linspace(0, 1, 200))
        brownian = np.cumsum(rng.normal(0, 1, 3000))
        hurst = hurst_rs(brownian)
        check("ap_en_nonneg", ap_rand >= 0 and ap_ord >= 0, f"rand={ap_rand:.3f} ord={ap_ord:.3f}")
        check("ap_en_random_gt_ordered", ap_rand > ap_ord, f"rand={ap_rand:.3f} ord={ap_ord:.3f}")
        check("hurst_bounds", 0.0 <= hurst <= 1.0, f"hurst={hurst:.4f}")
        check("hurst_brownian_05", 0.4 <= hurst <= 0.6, f"hurst={hurst:.4f}")
    except Exception as e:  # pragma: no cover
        check("entropy_hurst", False, f"exception: {e}")

    # --- volume profile / obv / cmf / vol ratio ---
    try:
        px = rng.uniform(100, 200, 500)
        vv = rng.uniform(1, 10, 500)
        poc, val, vah = volume_profile(px, vv, bins=50)
        covered = vv[(px >= val) & (px <= vah)].sum() / vv.sum()
        check("vp_poc_in_range", val <= poc <= vah, f"poc={poc:.2f} val={val:.2f} vah={vah:.2f}")
        check("vp_value_area_68", covered >= 0.68, f"covered={covered:.3f}")
        closes = np.cumsum(rng.normal(0, 1, 100)) + 100
        vols = rng.uniform(1, 5, 100)
        o = obv(closes, vols)
        oslope = obv_slope(o)
        cmf = chaikin_money_flow(closes, closes, closes, vols)
        vr = volume_ratio(vols)
        check("obv_array", o.shape == closes.shape, "")
        check("obv_slope_finite", np.isfinite(oslope), f"slope={oslope:.3f}")
        check("cmf_array", cmf.shape[0] == len(closes), "")
        check("volume_ratio_pos", bool(np.all(vr >= 0)), "")
    except Exception as e:  # pragma: no cover
        check("volume_profile", False, f"exception: {e}")

    # --- garch (arch lazy; EWMA fallback) ---
    try:
        rets_g = rng.normal(0, 0.02, 300)
        var, unstable = garch_11(rets_g)
        check("garch_var_positive", var > 0.0, f"var={var:.6e} unstable={unstable}")
    except Exception as e:  # pragma: no cover
        check("garch", False, f"exception: {e}")

    # --- kelly / fractional / stop / size / dd / bucket ---
    try:
        k_neg = kelly_fraction(0.3, 1.5)  # negative edge -> 0
        k_pos = kelly_fraction(0.6, 2.0)
        check("kelly_negative_edge_zero", k_neg == 0.0, f"k={k_neg}")
        check("kelly_clamped", 0.0 <= k_pos <= 1.0, f"k={k_pos:.4f}")
        fk = fractional_kelly(0.6, 2.0, fraction=0.25)
        check("fractional_kelly_le_kelly", fk <= k_pos, f"fk={fk:.4f}")
        sp = stop_price(100.0, "long", 2.0, mult=2.0)
        check("stop_2x_atr", abs(sp - 96.0) < 1e-9, f"stop={sp}")
        size = risk_based_size_pct(100000.0, 0.01, 100.0, 96.0, kelly_frac=0.25, hard_cap=0.01)
        check("size_le_1pct", size <= 0.01, f"size={size:.4f}")
        dd = DrawdownGuard(0.20)
        dd.update(100.0)
        triggered = dd.update(79.0)  # 21% drawdown -> kill
        check("drawdown_guard_latch", triggered is True and dd.kill() is True, f"triggered={triggered}")
        tb = TokenBucket(rate=20, capacity=20)
        admitted = 0
        for _ in range(25):
            if tb.admit():
                admitted += 1
        check("token_bucket_cap", admitted <= 20, f"admitted_first_25={admitted}")
    except Exception as e:  # pragma: no cover
        check("kelly_risk", False, f"exception: {e}")

    return results


def _cmd_check(args) -> int:
    results = run_math_selftest()
    passed = sum(1 for _, ok, _ in results if ok)
    width = max((len(n) for n, _, _ in results), default=10)
    print("\n=== autobot math self-test ===")
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        line = f"[{status}] {name.ljust(width)}"
        if detail:
            line += f"  {detail}"
        print(line)
    print(f"\n{passed}/{len(results)} invariants passed.")
    return 0 if passed == len(results) else 1


def _cmd_backtest(args) -> int:
    from autobot.config.settings import Settings
    from autobot.backtest.run import run_backtest

    settings = Settings.from_config()
    run_backtest(settings)
    return 0


def _cmd_paper(args) -> int:
    from pydantic import ValidationError

    from autobot.config.settings import Settings
    from autobot.engine.live_loop import main_live

    if args.simulate:
        # Keyless dry-run: load config as 'backtest' so the live/paper secret check
        # is skipped (no broker keys needed). (#3)
        settings = Settings.from_config(environment="backtest")
        from autobot.engine.simulate import run_simulate

        run_simulate(settings)
        return 0

    try:
        settings = Settings.from_config()
    except ValidationError as e:
        # Plain `paper` with missing keys: clean message, exit(2), no traceback. (#3)
        print(
            "autobot: 'paper' requires broker secrets (set AUTOBO_* env or .env).", file=sys.stderr
        )
        # Pull the concrete reason out of pydantic's error list (not the trailing URL note).
        reasons = [d.get("msg", "") for d in e.errors() if d.get("msg")]
        if reasons:
            print("missing/required: " + "; ".join(dict.fromkeys(reasons)), file=sys.stderr)
        return 2

    main_live(settings, paper=True)
    return 0


def _cmd_kill(args) -> int:
    from autobot.engine.live_loop import write_kill_flag, KILL_FLAG

    write_kill_flag("cli")
    print(f"kill flag written -> {KILL_FLAG}")
    return 0


def _cmd_live(args) -> int:
    from autobot.config.settings import Settings
    from autobot.engine.live_loop import main_live

    settings = Settings.from_config()
    main_live(settings, paper=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="autobot", description="autobot trading bot")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="run the math self-test (offline)")

    bt = sub.add_parser("backtest", help="run vectorbt walk-forward backtest")
    bt.set_defaults(func=_cmd_backtest)

    pp = sub.add_parser("paper", help="run live pipeline in paper/sandbox mode")
    pp.add_argument(
        "--simulate",
        action="store_true",
        help="keyless local dry-run: synthesize tape, replay signals, print P&L",
    )
    pp.set_defaults(func=_cmd_paper)

    lv = sub.add_parser("live", help="run live pipeline against real brokers")
    lv.set_defaults(func=_cmd_live)

    # `kill` writes the state/kill_flag; the running process's watcher thread trips
    # its kill event (#1c).
    kp = sub.add_parser("kill", help="set kill flag for a running process")
    kp.set_defaults(func=_cmd_kill)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "check":
        return _cmd_check(args)
    # lazy import of the heavy path only when actually running a live/backtest command
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
