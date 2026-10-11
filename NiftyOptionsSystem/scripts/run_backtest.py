#!/usr/bin/env python3
"""
scripts/run_backtest.py
--------------------------
Phase 4 CLI: run the backtest engine over cached historical spot + India
VIX (Phase 1 data — run scripts/pull_data.py first if the cache is empty)
and print a performance report.

Does not need a live Kite session — everything here reads from the local
SQLite cache. See backtest/reconstruction.py's module docstring for the
assumptions this implies (flat IV per day, no historical OI, static-carry
forward, synthetic expiry tenor).

Usage:
    python scripts/run_backtest.py --instrument NIFTY --start 2023-01-01 --end 2024-12-31
    python scripts/run_backtest.py --instrument NIFTY --strategy BULL_PUT_SPREAD --tenor-days 7
    python scripts/run_backtest.py --instrument BANKNIFTY --profit-target 0.5 --stop-loss 1.0 --slippage 0.02
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.engine import BacktestConfig, run_backtest
from backtest.metrics import compute_performance_report
from config import INSTRUMENTS
from data.cache import read_spot_ohlc, read_vix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the options-selling backtest over cached history (Phase 4)")
    p.add_argument("--instrument", choices=list(INSTRUMENTS), required=True)
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default=datetime.today().strftime("%Y-%m-%d"))
    p.add_argument("--capital", type=float, default=2_000_000.0)
    p.add_argument("--tenor-days", type=int, default=7)
    p.add_argument("--min-dte-exit", type=int, default=1)
    p.add_argument("--profit-target", type=float, default=0.50)
    p.add_argument("--stop-loss", type=float, default=1.00)
    p.add_argument("--slippage", type=float, default=0.02)
    p.add_argument("--lots", type=int, default=1)
    p.add_argument("--strategy", default=None,
                    choices=["BULL_PUT_SPREAD", "BEAR_CALL_SPREAD", "IRON_CONDOR"],
                    help="Force this strategy on every entry instead of using the regime's recommendation")
    p.add_argument("--assumed-range-score", type=float, default=50.0,
                    help="Backtests can't compute a live OI-based range_score -- this stands in for it (see backtest/engine.py)")
    p.add_argument("--trade-log-csv", default=None, help="Optional path to save the leg-level trade log")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    spot = read_spot_ohlc(args.instrument, args.start, args.end)
    vix = read_vix(args.start, args.end)["close"]
    if spot.empty or vix.empty:
        print("[ERROR] No cached data for that range. Run scripts/pull_data.py first.")
        sys.exit(1)

    cfg = BacktestConfig(
        initial_capital=args.capital,
        tenor_days=args.tenor_days,
        min_dte_exit=args.min_dte_exit,
        profit_target_pct=args.profit_target,
        stop_loss_pct=args.stop_loss,
        slippage_pct=args.slippage,
        n_lots=args.lots,
        strategy_override=args.strategy,
        assumed_range_score=args.assumed_range_score,
    )

    print(f"Running backtest: {args.instrument}  {args.start} -> {args.end}  "
          f"({len(spot)} trading days)")
    print(f"  tenor={cfg.tenor_days}d  min_dte_exit={cfg.min_dte_exit}  "
          f"profit_target={cfg.profit_target_pct:.0%}  stop_loss={cfg.stop_loss_pct:.0%}  "
          f"slippage={cfg.slippage_pct:.1%}/leg  strategy_override={cfg.strategy_override or '(regime-driven)'}\n")

    result = run_backtest(args.instrument, spot, vix, cfg)
    report = compute_performance_report(result["trade_log"], result["equity_curve"], cfg.initial_capital)

    print(json.dumps(report, indent=2, default=str))

    if args.trade_log_csv and not result["trade_log"].empty:
        result["trade_log"].to_csv(args.trade_log_csv, index=False)
        print(f"\nTrade log saved -> {args.trade_log_csv}")


if __name__ == "__main__":
    main()
