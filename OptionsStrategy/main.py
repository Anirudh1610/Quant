"""
main.py — Nifty / Bank Nifty Short Straddle-Strangle Strategy
==============================================================

Trend prediction model
----------------------
Each trading day is classified using data from the *previous* close only:

  1. Gift Nifty proxy  — today's opening gap vs yesterday's close
                         (in live trading: replace with actual Gift Nifty level
                         from NSE GIFT City terminal before 09:00 IST)
  2. Nifty Futures     — theoretical cost-of-carry basis (approximated via BS)
                         (in live trading: use your broker's live futures feed)
  3. Previous day      — close-to-close return and candle body direction
  4. RSI(14)           — oversold / overbought momentum indicator
  5. EMA(20)           — price position relative to dynamic support/resistance
  6. ADX(14)           — trend strength filter  →  skip if market is trending
  7. India VIX         — volatility filter       →  skip if extreme volatility

Options positions based on signal
----------------------------------
  NEUTRAL   → sell ATM straddle (both legs at the same ATM strike)
  BULL      → sell strangle skewed upward (call further OTM, put near ATM)
  BEAR      → sell strangle skewed downward (put further OTM, call near ATM)
  STRANGLE  → sell symmetric 1σ-OTM strangle (moderate-ADX environment)
  SKIP      → no trade today

Instruments and expiries
-------------------------
  Nifty 50    : weekly (every Tuesday)
  Bank Nifty  : weekly (every Wednesday) + monthly (last Thursday of month)

Stop-loss
---------
  Position is marked-to-market daily using Black-Scholes + current VIX.
  If the theoretical loss exceeds (stop_loss_multiple × premium received),
  the position is closed at that day's theoretical mid price.
  Set --stop-loss 0 to disable.

Usage
-----
    python main.py
    python main.py --start 2021-01-01 --end 2024-12-31
    python main.py --capital 5000000 --lots-nifty 2 --lots-banknifty 2
    python main.py --max-vix 22 --sigma-multiple 1.2 --stop-loss 2.5
    python main.py --no-banknifty --no-charts
    python main.py --refresh-data       # re-download and overwrite cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — must be set before pyplot import

import pandas as pd

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from strategy.data import load_options_dataset
from strategy.signals import classify_trend
from strategy.backtest import OptionsBacktestConfig, OptionsBacktestEngine
from strategy.metrics import (
    generate_performance_report,
    plot_equity_curve,
    plot_drawdown,
    plot_monthly_heatmap,
    plot_pnl_by_signal,
    plot_pnl_distribution,
    plot_vix_vs_pnl,
)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Nifty / Bank Nifty Short Straddle-Strangle Options Strategy"
    )
    p.add_argument("--start",          default="2020-01-01",  help="Backtest start date YYYY-MM-DD")
    p.add_argument("--end",            default=None,           help="End date (default: today)")
    p.add_argument("--capital",        default=2_000_000.0, type=float,
                   help="Starting capital in ₹ (default: ₹20 lakh)")
    p.add_argument("--data-dir",       default="data/cache",   help="Cache directory for downloaded data")
    p.add_argument("--output",         default="results",      help="Output directory for reports + charts")
    p.add_argument("--no-charts",      action="store_true",    help="Skip chart generation")
    p.add_argument("--no-nifty",       action="store_true",    help="Skip Nifty 50 options")
    p.add_argument("--no-banknifty",   action="store_true",    help="Skip Bank Nifty options")
    p.add_argument("--no-monthly",     action="store_true",    help="Skip Bank Nifty monthly expiry")
    p.add_argument("--entry-days",     default=1, type=int,
                   help="Trading days before expiry to enter (default: 1)")
    p.add_argument("--min-vix",        default=14.0, type=float,
                   help="Skip if India VIX < this — premiums too thin (default: 14)")
    p.add_argument("--max-vix",        default=18.0, type=float,
                   help="Skip if India VIX > this — too dangerous (default: 18)")
    p.add_argument("--max-adx",        default=30.0, type=float,
                   help="Skip if ADX(14) > this (market trending, default: 30)")
    p.add_argument("--max-gap",        default=0.015, type=float,
                   help="Skip if open gap > this fraction (default: 0.015 = 1.5%%)")
    p.add_argument("--stop-loss",      default=2.0, type=float,
                   help="Exit at loss > N × premium received; 0 = disabled (default: 2.0)")
    p.add_argument("--sigma-multiple", default=1.0, type=float,
                   help="OTM distance for strangle legs in daily-IV sigmas (default: 1.0)")
    p.add_argument("--lots-nifty",     default=1, type=int,   help="Nifty lots per trade (default: 1)")
    p.add_argument("--lots-banknifty", default=1, type=int,   help="BankNifty lots per trade (default: 1)")
    p.add_argument("--refresh-data",   action="store_true",   help="Force re-download (ignore cache)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args    = parse_args()
    end     = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("  NIFTY / BANK NIFTY  SHORT STRADDLE-STRANGLE  OPTIONS STRATEGY")
    print("=" * 65)
    print(f"  Period     : {args.start}  →  {end}")
    print(f"  Capital    : ₹{args.capital:,.0f}")
    print(f"  VIX band   : {args.min_vix} – {args.max_vix}   Max ADX: {args.max_adx}   Max gap: {args.max_gap*100:.1f}%")
    print(f"  Stop-loss  : {args.stop_loss}×   σ-multiple: {args.sigma_multiple}")
    print(f"  Entry      : {args.entry_days} day(s) before expiry")
    print("=" * 65 + "\n")

    # ── 1. Download / load data ───────────────────────────────────────────────
    print("[1/4] Loading market data …")
    dataset = load_options_dataset(
        start=args.start, end=end,
        data_dir=args.data_dir,
        refresh=args.refresh_data,
    )
    nifty     = dataset["nifty"]
    banknifty = dataset["banknifty"]
    vix       = dataset["vix"]
    dates     = dataset["dates"]
    print(f"      {len(dates)} trading days  "
          f"({dates[0].date()} → {dates[-1].date()})\n")

    # ── 2. Classify daily trend ───────────────────────────────────────────────
    print("[2/4] Classifying trend signals …")
    trend_nifty = classify_trend(
        nifty["open"], nifty["high"], nifty["low"], nifty["close"], vix,
        min_vix=args.min_vix, max_vix=args.max_vix,
        max_adx=args.max_adx, max_gap=args.max_gap,
    )
    trend_banknifty = classify_trend(
        banknifty["open"], banknifty["high"], banknifty["low"], banknifty["close"], vix,
        min_vix=args.min_vix, max_vix=args.max_vix,
        max_adx=args.max_adx, max_gap=args.max_gap,
    )
    for label, sig in [("Nifty 50 ", trend_nifty), ("BankNifty", trend_banknifty)]:
        counts = sig.value_counts().to_dict()
        print(f"      {label} signals: " +
              "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print()

    # ── 3. Run backtest ───────────────────────────────────────────────────────
    print("[3/4] Running options backtest …")
    cfg = OptionsBacktestConfig(
        initial_capital=args.capital,
        trade_nifty=not args.no_nifty,
        trade_banknifty=not args.no_banknifty,
        entry_days_before=args.entry_days,
        max_vix=args.max_vix,
        max_adx=args.max_adx,
        stop_loss_multiple=args.stop_loss,
        sigma_multiple=args.sigma_multiple,
        n_lots_nifty=args.lots_nifty,
        n_lots_banknifty=args.lots_banknifty,
        trade_banknifty_monthly=not args.no_monthly,
    )
    engine  = OptionsBacktestEngine(cfg)
    results = engine.run(nifty, banknifty, vix, trend_nifty, trend_banknifty)

    trade_log = results["trade_log"]
    equity    = results["equity_curve"]
    returns   = results["daily_returns"]
    print(f"      {len(trade_log)} trades executed\n")

    # ── 4. Report + charts ────────────────────────────────────────────────────
    print("[4/4] Generating performance report …")
    generate_performance_report(equity, returns, trade_log, args.capital, out_dir)

    if not trade_log.empty:
        tl_path = out_dir / "trade_log.csv"
        trade_log.to_csv(tl_path, index=False)
        print(f"  Trade log saved → {tl_path}")

    if not args.no_charts and not trade_log.empty:
        chart_dir = out_dir / "charts"
        chart_dir.mkdir(exist_ok=True)
        print("  Saving charts:")
        plot_equity_curve(equity, chart_dir)
        plot_drawdown(equity, chart_dir)
        plot_monthly_heatmap(trade_log, chart_dir)
        plot_pnl_by_signal(trade_log, chart_dir)
        plot_pnl_distribution(trade_log, chart_dir)
        plot_vix_vs_pnl(trade_log, chart_dir)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
