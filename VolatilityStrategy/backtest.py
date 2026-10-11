"""backtest.py — Event-driven backtesting engine for volatility trading signals.

P&L proxy (vol-swap model)
--------------------------
All vol quantities are annualized fractions, matching the LSTM output units.

    vol_surprise  = clip((actual_rv_ann − implied_vol_ann) / implied_vol_ann, −2, +2)
    trade_pnl     = direction × vol_surprise × risk_per_trade − round_trip_cost

    direction = +1 for LONG VOL, −1 for SHORT VOL

    • LONG VOL profits when actual realized > implied (volatility underpriced)
    • SHORT VOL profits when implied > actual realized (premium collection)
    • Clipping at ±200% of implied bounds the maximum loss to 2× risk_per_trade
    • risk_per_trade = 0.02 → max 2% of capital per trade (realistic for a
      variance-swap / straddle position)

P&L is spread evenly across the hold window for a smooth equity curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from strategy import Signal, StrategyConfig
from utils import get_logger, load_config

logger = get_logger(__name__)

TRADING_DAYS = 252


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    initial_capital: float = 1_000_000.0
    hold_days: int = 5
    transaction_cost: float = 0.0005
    slippage: float = 0.0002
    annual_trading_days: int = 252
    allow_short: bool = True
    # Fraction of capital risked per trade.  The vol-swap P&L is scaled by
    # this so that the maximum possible loss per trade is ~risk_per_trade.
    risk_per_trade: float = 0.02   # 2% of capital per trade

    @classmethod
    def from_cfg(cls, cfg: dict) -> "BacktestConfig":
        s = cfg.get("strategy", {})
        b = cfg.get("backtest", {})
        return cls(
            initial_capital=b.get("initial_capital", 1_000_000.0),
            hold_days=s.get("hold_days", 5),
            transaction_cost=s.get("transaction_cost", 0.0005),
            slippage=s.get("slippage", 0.0002),
            annual_trading_days=b.get("annual_trading_days", 252),
            risk_per_trade=b.get("risk_per_trade", 0.02),
        )

    @property
    def round_trip_cost(self) -> float:
        return 2 * (self.transaction_cost + self.slippage)


# ─────────────────────────────────────────────────────────────────────────────
# Position tracker
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Position:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    signal: int               # Signal.LONG_VOL or Signal.SHORT_VOL
    implied_entry: float
    predicted_rv5: float
    edge: float


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Non-overlapping 5-day hold volatility backtest.

    Rules
    -----
    • A signal is generated on each trading day based on model output.
    • If no position is open, enter a trade on the next available day.
    • Hold for exactly `hold_days` trading days.
    • No overlapping positions.
    • Transaction costs and slippage applied at entry and exit.
    """

    def __init__(self, cfg: BacktestConfig | None = None) -> None:
        self.cfg = cfg or BacktestConfig()

    def run(
        self,
        signals_df: pd.DataFrame,
        actual_rv5: pd.Series,
    ) -> dict:
        """
        Run the backtest.

        Parameters
        ----------
        signals_df : Output of strategy.generate_signals().
                     Required columns: signal, implied_vol_ann, edge, predicted_rv5.
        actual_rv5 : Annualized realized vol for the 5-day hold period.
                     After compute_target fix, actual_rv5[t] = std(r[t+1:t+6])*sqrt(252),
                     i.e. the exact hold-period vol known only in hindsight.

        P&L model (vol-swap proxy)
        --------------------------
        On entry date t:
            actual_ann  = actual_rv5[t]           # annualized realized hold-period vol
            implied_ann = signals_df["implied_vol_ann"][t]  # VIX/100 at entry

            vol_surprise = (actual_ann - implied_ann) / implied_ann
            trade_pnl    = direction * clip(vol_surprise, −2, +2) * risk_per_trade

        This is equivalent to being long/short a variance swap scaled to a
        fixed-risk position (risk_per_trade = max possible loss per trade).
        P&L is then spread evenly across the hold days for a smooth equity curve.

        Returns
        -------
        dict: equity_curve, daily_returns, trade_log, final_capital.
        """
        cfg = self.cfg
        capital = cfg.initial_capital
        hold = cfg.hold_days

        dates = signals_df.index
        equity = pd.Series(np.nan, index=dates)
        daily_returns = pd.Series(0.0, index=dates)
        trade_log: list[dict] = []

        i = 0
        n = len(dates)

        while i < n:
            date = dates[i]
            sig = int(signals_df.at[date, "signal"])
            equity[date] = capital

            if sig == Signal.NEUTRAL or (not cfg.allow_short and sig == Signal.SHORT_VOL):
                i += 1
                continue

            exit_i = min(i + hold, n - 1)
            exit_date = dates[exit_i]
            hold_dates = dates[i + 1: exit_i + 1]

            implied_ann = float(signals_df.at[date, "implied_vol_ann"])
            pred_rv     = float(signals_df.at[date, "predicted_rv5"])
            edge        = float(signals_df.at[date, "edge"])

            # Realized vol DURING the hold period (known only in hindsight)
            actual_ann = float(actual_rv5.get(date, np.nan) or np.nan)
            if np.isnan(actual_ann):
                i = exit_i + 1
                continue

            # Vol-swap P&L: how much better/worse was realized vs implied
            # Clipped at ±200% of implied so one trade can't blow up capital
            implied_safe = max(implied_ann, 0.001)
            vol_surprise = (actual_ann - implied_ann) / implied_safe
            vol_surprise = float(np.clip(vol_surprise, -2.0, 2.0))

            trade_pnl = sig * vol_surprise * cfg.risk_per_trade
            # Subtract round-trip cost (entry + exit) from trade P&L
            trade_pnl -= cfg.round_trip_cost
            daily_pnl = trade_pnl / max(len(hold_dates), 1)

            for hdate in hold_dates:
                capital *= (1.0 + daily_pnl)
                daily_returns[hdate] = daily_pnl
                equity[hdate] = capital

            trade_log.append({
                "entry_date":    date,
                "exit_date":     exit_date,
                "signal":        "LONG VOL" if sig == Signal.LONG_VOL else "SHORT VOL",
                "implied_ann":   round(implied_ann, 6),
                "predicted_rv5": round(pred_rv, 6),
                "actual_rv5":    round(actual_ann, 6),
                "edge":          round(edge, 6),
                "vol_surprise":  round(vol_surprise, 4),
                "net_return":    round(trade_pnl, 6),
                "capital_after": round(capital, 2),
            })

            i = exit_i + 1

        equity = equity.ffill()
        daily_returns = daily_returns.fillna(0.0)
        trade_log_df = pd.DataFrame(trade_log)

        results = {
            "equity_curve": equity,
            "daily_returns": daily_returns,
            "trade_log": trade_log_df,
            "final_capital": capital,
        }
        logger.info(
            "Backtest complete: %d trades  |  final capital: ₹%.0f  |  return: %.1f%%",
            len(trade_log),
            capital,
            (capital / cfg.initial_capital - 1) * 100,
        )
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Performance metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_performance(
    equity: pd.Series,
    daily_returns: pd.Series,
    trade_log: pd.DataFrame,
    initial_capital: float,
    annual_days: int = 252,
) -> dict:
    """Compute full suite of performance statistics."""
    r = daily_returns.replace(0, np.nan).dropna()
    if len(r) == 0:
        return {"error": "No trades"}

    # Returns
    total_return = equity.iloc[-1] / initial_capital - 1
    n_years = len(equity) / annual_days
    cagr = (equity.iloc[-1] / initial_capital) ** (1 / max(n_years, 1e-6)) - 1

    # Volatility & Sharpe
    ann_vol = float(r.std() * np.sqrt(annual_days))
    sharpe = float(r.mean() / r.std() * np.sqrt(annual_days)) if r.std() > 0 else 0.0

    # Sortino
    neg_r = r[r < 0]
    sortino = (
        float(r.mean() / neg_r.std() * np.sqrt(annual_days))
        if len(neg_r) > 0 and neg_r.std() > 0
        else 0.0
    )

    # Drawdown
    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = float(dd.min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0

    # Trade statistics
    stats: dict = {}
    if not trade_log.empty and "net_return" in trade_log.columns:
        wins = trade_log[trade_log["net_return"] > 0]
        losses = trade_log[trade_log["net_return"] <= 0]
        win_rate = len(wins) / len(trade_log)
        gross_profit = wins["net_return"].sum() if len(wins) > 0 else 0.0
        gross_loss = abs(losses["net_return"].sum()) if len(losses) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        stats = {
            "n_trades": len(trade_log),
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4),
        }

    return {
        "total_return": round(total_return, 4),
        "cagr": round(cagr, 4),
        "ann_volatility": round(ann_vol, 4),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "calmar_ratio": round(calmar, 4),
        "max_drawdown": round(max_dd, 4),
        **stats,
    }
