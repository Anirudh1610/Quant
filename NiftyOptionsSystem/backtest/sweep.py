"""
backtest/sweep.py
-------------------
Grid search over BacktestConfig / StrikeSelectionConfig parameters, with an
in-sample/out-of-sample split so a config chosen by IS performance can be
checked for whether it actually holds up on unseen data, rather than just
reporting the best IS number — which, on a large grid, is close to
guaranteed to be at least partly overfit to that one period (the more
combinations you try, the more likely one of them fits IS noise well by
chance alone).

ASSUMPTION FLAG: even the "out-of-sample" check here is a single OOS
window on a single instrument's history — not a walk-forward or
cross-validated evaluation. A config that holds up here should be read as
"not obviously overfit to one lucky window", not as a validated edge.
The underlying backtest itself also carries every Phase 4 assumption
already flagged in backtest/reconstruction.py and backtest/engine.py
(flat IV, no OI, static-carry forward, synthetic expiry tenor) — a
parameter sweep searches within that data quality, it cannot fix it.
"""

from __future__ import annotations

import itertools
from dataclasses import replace

import pandas as pd

from backtest.engine import BacktestConfig, run_backtest
from backtest.metrics import compute_performance_report
from config import STRIKE_SELECTION


def _build_config(tenor_days: int, profit_target_pct: float, stop_loss_pct: float,
                   target_delta: float, assumed_range_score: float,
                   base_cfg: BacktestConfig) -> BacktestConfig:
    strike_cfg = replace(base_cfg.strike_selection_cfg, target_delta=target_delta)
    return replace(base_cfg, tenor_days=tenor_days, profit_target_pct=profit_target_pct,
                   stop_loss_pct=stop_loss_pct, assumed_range_score=assumed_range_score,
                   strike_selection_cfg=strike_cfg)


def run_grid_search(instrument: str, spot_ohlc: pd.DataFrame, vix: pd.Series,
                     tenor_days_grid: list[int], profit_target_grid: list[float],
                     stop_loss_grid: list[float], target_delta_grid: list[float],
                     assumed_range_score_grid: list[float],
                     base_cfg: BacktestConfig | None = None, min_trades: int = 15) -> pd.DataFrame:
    """
    Runs the full cartesian product of the grids, one backtest per
    combination. Returns a DataFrame, one row per combo: the swept params
    plus every compute_performance_report field.

    Configs with fewer than `min_trades` completed trades are still
    included, flagged via a `reliable` column (False) rather than dropped
    — a combo that rarely triggers isn't necessarily wrong, but its return
    figure is noise-dominated and shouldn't be treated the same as one
    backed by dozens of trades. Silently dropping it would also hide that
    it never fired at all, which is itself useful information (e.g. an
    assumed_range_score set higher than any day in the sample achieved).
    """
    base_cfg = base_cfg or BacktestConfig()
    combos = list(itertools.product(tenor_days_grid, profit_target_grid, stop_loss_grid,
                                     target_delta_grid, assumed_range_score_grid))
    rows = []
    for tenor_days, profit_target_pct, stop_loss_pct, target_delta, assumed_range_score in combos:
        cfg = _build_config(tenor_days, profit_target_pct, stop_loss_pct, target_delta,
                             assumed_range_score, base_cfg)
        result = run_backtest(instrument, spot_ohlc, vix, cfg)
        report = compute_performance_report(result["trade_log"], result["equity_curve"], cfg.initial_capital)
        rows.append({
            "tenor_days": tenor_days,
            "profit_target_pct": profit_target_pct,
            "stop_loss_pct": stop_loss_pct,
            "target_delta": target_delta,
            "assumed_range_score": assumed_range_score,
            "reliable": report["n_trades"] >= min_trades,
            **report,
        })
    return pd.DataFrame(rows)


def evaluate_is_oos(instrument: str, is_spot: pd.DataFrame, is_vix: pd.Series,
                     oos_spot: pd.DataFrame, oos_vix: pd.Series,
                     configs: list[BacktestConfig], labels: list[str] | None = None) -> pd.DataFrame:
    """
    Re-runs a shortlist of configs (e.g. the sweep's top N by IS return) on
    both the IS and OOS periods side by side — the actual check for
    whether IS performance persists out of sample.
    """
    labels = labels or [f"config_{i}" for i in range(len(configs))]
    rows = []
    for label, cfg in zip(labels, configs):
        is_result = run_backtest(instrument, is_spot, is_vix, cfg)
        oos_result = run_backtest(instrument, oos_spot, oos_vix, cfg)
        is_report = compute_performance_report(is_result["trade_log"], is_result["equity_curve"], cfg.initial_capital)
        oos_report = compute_performance_report(oos_result["trade_log"], oos_result["equity_curve"], cfg.initial_capital)
        rows.append({
            "label": label,
            "tenor_days": cfg.tenor_days,
            "profit_target_pct": cfg.profit_target_pct,
            "stop_loss_pct": cfg.stop_loss_pct,
            "target_delta": cfg.strike_selection_cfg.target_delta,
            "assumed_range_score": cfg.assumed_range_score,
            "is_n_trades": is_report["n_trades"],
            "is_return_pct": is_report["total_return_pct"],
            "is_win_rate": is_report["win_rate"],
            "is_max_dd_pct": is_report["max_drawdown_pct"],
            "oos_n_trades": oos_report["n_trades"],
            "oos_return_pct": oos_report["total_return_pct"],
            "oos_win_rate": oos_report["win_rate"],
            "oos_max_dd_pct": oos_report["max_drawdown_pct"],
        })
    return pd.DataFrame(rows)
