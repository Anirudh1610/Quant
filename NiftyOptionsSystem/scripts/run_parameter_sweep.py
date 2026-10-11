#!/usr/bin/env python3
"""
scripts/run_parameter_sweep.py
---------------------------------
Grid-search BacktestConfig/StrikeSelectionConfig parameters on an
in-sample period, then re-validate the top N configs (by IS return) on a
held-out out-of-sample period — the actual check for whether a chosen
config persists on unseen data rather than just being the best-fit noise
on one window. See backtest/sweep.py's module docstring for what this
does and doesn't prove.

Usage:
    python scripts/run_parameter_sweep.py --instrument NIFTY \\
        --is-start 2023-01-01 --is-end 2025-08-11 \\
        --oos-start 2025-08-12 --oos-end 2026-08-11
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.engine import BacktestConfig
from backtest.sweep import evaluate_is_oos, run_grid_search
from config import STRIKE_SELECTION
from data.cache import read_spot_ohlc, read_vix

TENOR_DAYS_GRID = [5, 7, 10]
PROFIT_TARGET_GRID = [0.3, 0.5, 0.75]
STOP_LOSS_GRID = [0.5, 1.0, 1.5]
TARGET_DELTA_GRID = [0.12, 0.175, 0.25]
ASSUMED_RANGE_SCORE_GRID = [50, 70]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid-search backtest parameters with an IS/OOS split")
    p.add_argument("--instrument", default="NIFTY")
    p.add_argument("--is-start", default="2023-01-01")
    p.add_argument("--is-end", default="2025-08-11")
    p.add_argument("--oos-start", default="2025-08-12")
    p.add_argument("--oos-end", default="2026-08-11")
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--min-trades", type=int, default=15)
    p.add_argument("--csv", default=None, help="Optional path to save the full grid results")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    is_spot = read_spot_ohlc(args.instrument, args.is_start, args.is_end)
    is_vix = read_vix(args.is_start, args.is_end)["close"]
    oos_spot = read_spot_ohlc(args.instrument, args.oos_start, args.oos_end)
    oos_vix = read_vix(args.oos_start, args.oos_end)["close"]

    if is_spot.empty or oos_spot.empty:
        print("[ERROR] Missing cached data for the IS or OOS range. Run scripts/pull_data.py first.")
        sys.exit(1)

    print(f"IS:  {args.is_start} -> {args.is_end}  ({len(is_spot)} days)")
    print(f"OOS: {args.oos_start} -> {args.oos_end}  ({len(oos_spot)} days)\n")

    n_combos = (len(TENOR_DAYS_GRID) * len(PROFIT_TARGET_GRID) * len(STOP_LOSS_GRID)
                * len(TARGET_DELTA_GRID) * len(ASSUMED_RANGE_SCORE_GRID))
    print(f"Running {n_combos} combinations on the IS period (this can take a few minutes)...\n")

    grid_results = run_grid_search(
        args.instrument, is_spot, is_vix,
        TENOR_DAYS_GRID, PROFIT_TARGET_GRID, STOP_LOSS_GRID, TARGET_DELTA_GRID, ASSUMED_RANGE_SCORE_GRID,
        min_trades=args.min_trades,
    )

    if args.csv:
        grid_results.to_csv(args.csv, index=False)
        print(f"Full grid saved -> {args.csv}\n")

    reliable = grid_results[grid_results["reliable"]]
    if reliable.empty:
        print(f"No config reached --min-trades {args.min_trades} on the IS period. "
              f"Lower --min-trades or widen the grids.")
        sys.exit(1)

    top = reliable.sort_values("total_return_pct", ascending=False).head(args.top_n)

    print(f"=== Top {len(top)} configs by IS total_return_pct (n_trades >= {args.min_trades}) ===")
    print(top[["tenor_days", "profit_target_pct", "stop_loss_pct", "target_delta", "assumed_range_score",
               "n_trades", "win_rate", "total_return_pct", "max_drawdown_pct", "sharpe_ratio"]]
          .to_string(index=False))

    configs = [
        BacktestConfig(
            tenor_days=int(r["tenor_days"]), profit_target_pct=r["profit_target_pct"],
            stop_loss_pct=r["stop_loss_pct"], assumed_range_score=r["assumed_range_score"],
            strike_selection_cfg=replace(STRIKE_SELECTION, target_delta=r["target_delta"]),
        )
        for _, r in top.iterrows()
    ]

    print(f"\n=== Re-validating top {len(configs)} on OOS ===")
    oos_check = evaluate_is_oos(args.instrument, is_spot, is_vix, oos_spot, oos_vix, configs,
                                 labels=[f"#{i + 1}" for i in range(len(configs))])
    print(oos_check.to_string(index=False))


if __name__ == "__main__":
    main()
