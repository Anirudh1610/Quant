"""
backtest/metrics.py
---------------------
Performance metrics from a leg-level trade log and a daily equity curve
(both produced by backtest/engine.py).

ASSUMPTION FLAG: Sharpe/Sortino use simple daily returns on the equity
curve (equity_t / equity_{t-1} - 1), not log returns, and the standard
x sqrt(252) annualization. That convention matters somewhat at this short
a horizon (weekly-tenor options trades produce a choppy, jump-heavy equity
curve, not a smooth daily-compounding one) but not materially for
comparing configurations against each other, which is this metric's real
purpose at the Phase 4 skeleton stage — treat the absolute Sharpe/Sortino
numbers as directional, not literal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def trade_level_pnl(trade_log: pd.DataFrame) -> pd.DataFrame:
    """Aggregates leg-level rows (one per leg, see backtest/engine.py's schema) into one row per trade_id."""
    if trade_log.empty:
        return pd.DataFrame(columns=["trade_id", "entry_date", "exit_date", "instrument",
                                      "strategy_type", "exit_reason", "pnl"])
    return trade_log.groupby("trade_id").agg(
        entry_date=("entry_date", "first"),
        exit_date=("exit_date", "first"),
        instrument=("instrument", "first"),
        strategy_type=("strategy_type", "first"),
        exit_reason=("exit_reason", "first"),
        pnl=("leg_pnl", "sum"),
    ).reset_index()


def trade_level_sharpe(trades: pd.DataFrame, initial_capital: float, trading_calendar: pd.DatetimeIndex,
                        risk_free_rate: float = 0.068) -> float:
    """
    Sharpe from per-trade returns (pnl / initial_capital), with each
    trade's risk-free hurdle prorated over that trade's ACTUAL holding
    period in trading days — not a flat rf/trades_per_year applied
    uniformly regardless of duration (the original version of this
    calculation, which barely moved when backtest/engine.py's equity-curve
    accrual fix was applied, for exactly this reason: it was never derived
    from the equity curve, so that fix couldn't touch it).

    `trading_calendar` must be the same trading-day index the backtest ran
    over (e.g. spot_ohlc.index) — holding days are counted against it
    (days strictly after entry_date, up to and including exit_date), using
    the same 252-trading-day/year convention as the equity-curve accrual
    and sharpe_ratio(), so the two are comparable at a matching rf
    treatment. This deliberately does NOT use calendar days: a trade held
    over a weekend has no extra trading-day accrual in the simulation
    itself (the loop only visits trading days), so charging it 2 extra
    calendar days of rf hurdle would be inconsistent with how idle-capital
    accrual actually works elsewhere in this engine.

    Since the rf hurdle now varies per trade (unlike sharpe_ratio()'s
    constant daily_rf), the denominator is std(excess returns), not
    std(raw returns) — the textbook-correct Sharpe definition when the
    benchmark itself varies across observations.
    """
    if trades.empty or len(trades) < 2:
        return float("nan")

    returns = trades["pnl"] / initial_capital

    def _holding_trading_days(row: pd.Series) -> int:
        entry, exit_ = pd.Timestamp(row["entry_date"]), pd.Timestamp(row["exit_date"])
        return int(((trading_calendar > entry) & (trading_calendar <= exit_)).sum())

    holding_days = trades.apply(_holding_trading_days, axis=1)
    rf_hurdle = risk_free_rate * holding_days / TRADING_DAYS_PER_YEAR
    excess = returns - rf_hurdle

    if excess.std() == 0:
        return float("nan")

    n_years = len(trading_calendar) / TRADING_DAYS_PER_YEAR
    trades_per_year = len(trades) / n_years
    return float(excess.mean() / excess.std() * np.sqrt(trades_per_year))


def win_rate(trades: pd.DataFrame) -> float:
    if trades.empty:
        return float("nan")
    return float((trades["pnl"] > 0).mean())


def avg_win(trades: pd.DataFrame) -> float:
    wins = trades.loc[trades["pnl"] > 0, "pnl"]
    return float(wins.mean()) if not wins.empty else 0.0


def avg_loss(trades: pd.DataFrame) -> float:
    losses = trades.loc[trades["pnl"] < 0, "pnl"]
    return float(losses.mean()) if not losses.empty else 0.0


def max_drawdown(equity_curve: pd.Series) -> float:
    """Max peak-to-trough decline, as a positive fraction of the peak (e.g. 0.15 = 15% drawdown)."""
    if equity_curve.empty:
        return float("nan")
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return float(-drawdown.min())


def daily_returns(equity_curve: pd.Series) -> pd.Series:
    return equity_curve.pct_change().dropna()


def sharpe_ratio(equity_curve: pd.Series, risk_free_rate: float = 0.0) -> float:
    rets = daily_returns(equity_curve)
    if rets.empty or rets.std() == 0:
        return float("nan")
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = rets - daily_rf
    return float(excess.mean() / rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(equity_curve: pd.Series, risk_free_rate: float = 0.0) -> float:
    rets = daily_returns(equity_curve)
    if rets.empty:
        return float("nan")
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = rets - daily_rf
    downside = excess[excess < 0]
    downside_std = downside.std() if len(downside) > 1 else 0.0
    if not downside_std:
        return float("nan")
    return float(excess.mean() / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR))


def worst_single_day_loss_pct(equity_curve: pd.Series) -> float:
    """Most negative single-day return, as a positive fraction (e.g. 0.08 = an 8% down day)."""
    rets = daily_returns(equity_curve)
    if rets.empty:
        return float("nan")
    return float(-rets.min())


def compute_performance_report(trade_log: pd.DataFrame, equity_curve: pd.Series,
                                initial_capital: float, risk_free_rate: float = 0.068) -> dict:
    trades = trade_level_pnl(trade_log)
    return {
        "n_trades": len(trades),
        "win_rate": win_rate(trades),
        "avg_win": avg_win(trades),
        "avg_loss": avg_loss(trades),
        "total_pnl": float(trades["pnl"].sum()) if not trades.empty else 0.0,
        "final_equity": float(equity_curve.iloc[-1]) if not equity_curve.empty else initial_capital,
        "total_return_pct": float(equity_curve.iloc[-1] / initial_capital - 1) if not equity_curve.empty else 0.0,
        "max_drawdown_pct": max_drawdown(equity_curve),
        "sharpe_ratio": sharpe_ratio(equity_curve, risk_free_rate),
        "sortino_ratio": sortino_ratio(equity_curve, risk_free_rate),
        "worst_single_day_loss_pct": worst_single_day_loss_pct(equity_curve),
    }
