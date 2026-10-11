"""
backtest/engine.py
--------------------
Day-by-day backtest simulation: regime signal -> strategy selection ->
strike selection -> theoretical entry fill (mid, with slippage) -> daily
mark-to-market (full reprice) -> exit on profit target / stop loss / DTE
threshold.

Runs entirely off cached historical spot OHLC + India VIX (Phase 1 data)
via backtest/reconstruction.py's synthetic chain — see that module's
docstring for the assumptions this implies (flat-by-default IV across
strikes, no OI, static-carry forward, synthetic expiry tenor — the tenor
is a fixed calendar-day offset from each entry date, NOT aligned to the
real historical NSE expiry calendar, so `min_dte_exit` counts down in
calendar days from that synthetic date, not real weekly Thursdays).
`BacktestConfig.use_iv_skew=True` switches both chain reconstruction and
daily repricing to the skewed variants (reconstruct_chain_with_skew /
reprice_leg_with_skew) — an illustrative, undoubtedly-not-calibrated skew
shape, not real historical skew (none exists to calibrate against).
Default is False, so the original flat-IV behavior is unchanged unless
explicitly opted into.

Multi-leg fills are modeled as independent legs, not atomic: each leg's
entry/exit fill price is the theoretical price adjusted by `slippage_pct`
independently (worse for the trader on both sides — sold legs fill below
theoretical, bought legs fill above). This is the extent of "leg risk"
modeled here — true non-atomic execution risk (only some legs filling, or
a delay between leg fills during which the market moves) is NOT modeled;
every leg's fill is still computed from the same day's theoretical price,
just individually slipped.

Scope: one instrument per run, no margin-based position sizing (n_lots is
fixed by config, not derived from available capital via strikes/margin.py).
`BacktestConfig.max_concurrent_positions` (default 1) caps how many
positions can be open simultaneously — laddered/overlapping entries. At
most one NEW entry is still considered per day (one regime signal, one
selection), so laddering builds up over consecutive days, not within one.
A slot freed by an exit is not available until the following trading day
in every case — same-day re-entry is never allowed, at any concurrency
cap, matching the original single-position behavior exactly when
max_concurrent_positions=1 (verified by regression test).

Capital is a single shared pool across all concurrently open positions —
there is no per-slot capital partitioning or margin check; N positions
open at once each get the full theoretical fill/reprice treatment against
the same capital base. Risk-free accrual (below) is binary per day (fully
accrued or not at all) regardless of how many of the concurrency cap's
slots are actually filled — a partial-accrual model proportional to unused
slots would be a natural refinement but isn't implemented here.

Risk-free accrual on idle capital
----------------------------------
On any day with zero open positions at the start of that day (no signal, a
NO_TRADE regime, or the entry day itself — which doesn't mark-to-market
until the following day), `capital` compounds at `RISK_FREE_RATE / 252`.
Any day with at least one position already open (including a position's
exit day) is NOT accrued — those days use only the existing full-reprice
mark-to-market / realized-P&L logic, unchanged.

This was added after diagnosing why every Sharpe ratio reported from this
engine came out deeply negative even when total return was positive: the
underlying strategy only holds a position ~26% of trading days (verified
on a real NIFTY run), so the un-accrued equity curve was flat ~74% of the
time — i.e. idle capital was implicitly modeled as earning nothing, while
Sharpe's risk-free hurdle (`compute_performance_report`'s rf=6.8%) was
still charged against every calendar day regardless. That mismatch, not
the strategy's actual risk-adjusted return, was the dominant driver of the
negative Sharpe values seen throughout Phase 4's backtests. Accruing rf
on idle capital fixes the mismatch at its source (the equity curve itself)
rather than patching the Sharpe formula.

Consequence worth knowing: this changes total_return_pct and
max_drawdown_pct too, not just Sharpe — the equity curve now reflects
strategy P&L *plus* risk-free carry on idle cash, not strategy P&L alone.
`total_pnl` (summed straight from the trade log) is unaffected and remains
the pure realized-trade-P&L figure if you need to separate the two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from backtest.metrics import TRADING_DAYS_PER_YEAR
from backtest.reconstruction import (
    DEFAULT_SKEW_SLOPE,
    DEFAULT_SMILE_CURVATURE,
    reconstruct_chain,
    reconstruct_chain_with_skew,
    reprice_leg,
    reprice_leg_with_skew,
)
from config import DIVIDEND_YIELD, INSTRUMENTS, RISK_FREE_RATE, SIGNALS, STRIKE_SELECTION, StrikeSelectionConfig
from signals.regime import NO_TRADE, build_trend_vol_frame, recommend_strategy_type
from strikes.orchestrator import select_spread
from strikes.spreads import SpreadDefinition


@dataclass
class BacktestConfig:
    initial_capital: float = 2_000_000.0
    tenor_days: int = 7                    # synthetic expiry = entry_date + tenor_days calendar days
    profit_target_pct: float = 0.50        # exit when captured profit >= this fraction of net credit received
    stop_loss_pct: float = 1.00            # exit when loss >= this fraction of net credit received
    min_dte_exit: int = 1                  # force exit when remaining (synthetic, calendar-day) DTE <= this
    slippage_pct: float = 0.02             # each leg fills this much worse than theoretical, both entry and exit
    n_lots: int = 1
    strategy_override: str | None = None   # if set, always try this strategy instead of the regime's recommendation
    assumed_range_score: float = 50.0      # backtests can't compute a live OI-based range_score -- see module docstring
    strike_selection_cfg: StrikeSelectionConfig = field(default_factory=lambda: STRIKE_SELECTION)
    use_iv_skew: bool = False              # opt-in cross-strike skew (backtest/reconstruction.py) -- default False
    skew_slope: float = DEFAULT_SKEW_SLOPE              # keeps the flat-IV baseline path fully unchanged for comparison
    smile_curvature: float = DEFAULT_SMILE_CURVATURE
    max_concurrent_positions: int = 1      # laddered/overlapping entries -- 1 reproduces the original single-position engine exactly


@dataclass
class _OpenPosition:
    trade_id: int
    entry_date: date
    expiry_date: date
    spread: SpreadDefinition
    entry_fills: list[float]  # per-leg entry fill price (post-slippage), same order as spread.legs


def _apply_slippage(theoretical_price: float, action: str, slippage_pct: float) -> float:
    """SELL fills below theoretical, BUY fills above — both worse for the trader."""
    return theoretical_price * (1 - slippage_pct) if action == "SELL" else theoretical_price * (1 + slippage_pct)


def _leg_pnl(action: str, entry_fill: float, comparison_price: float, lot_size: int, n_lots: int) -> float:
    """comparison_price is either a mark (mid-backtest valuation) or an actual exit fill — same formula either way."""
    per_share = (entry_fill - comparison_price) if action == "SELL" else (comparison_price - entry_fill)
    return per_share * lot_size * n_lots


def run_backtest(instrument: str, spot_ohlc: pd.DataFrame, vix: pd.Series,
                  cfg: BacktestConfig | None = None) -> dict:
    """
    spot_ohlc: cached historical OHLC (data/spot.py), indexed by date.
    vix: cached India VIX close (data/vix.py), indexed by date.

    Returns {"trade_log": DataFrame (one row per leg), "equity_curve": Series}.
    """
    cfg = cfg or BacktestConfig()
    spec = INSTRUMENTS[instrument]
    r, q = RISK_FREE_RATE, DIVIDEND_YIELD.get(instrument, 0.0)

    trend_vol = build_trend_vol_frame(spot_ohlc, vix)

    capital = cfg.initial_capital
    equity_curve = pd.Series(index=trend_vol.index, dtype=float)
    trade_log_rows: list[dict] = []
    open_positions: list[_OpenPosition] = []
    next_trade_id = 1

    for today in trend_vol.index:
        row = trend_vol.loc[today]
        spot_today = spot_ohlc.loc[today, "close"]
        iv_today = row["iv"]

        if len(open_positions) == 0:
            # idle capital accrues the risk-free rate -- see module docstring's
            # "Risk-free accrual on idle capital" section. Covers invalid-data
            # days, NO_TRADE days, and the entry day itself (which doesn't
            # mark-to-market until tomorrow) -- not days at least one position
            # is already open, which keep the unchanged logic below.
            capital *= (1 + r / TRADING_DAYS_PER_YEAR)

        if pd.isna(spot_today) or pd.isna(iv_today) or iv_today <= 0:
            equity_curve.loc[today] = capital
            continue

        # snapshot BEFORE processing today's exits -- a slot freed today only
        # becomes available for a new entry starting tomorrow, matching the
        # original single-position "no same-day re-entry" rule exactly
        positions_at_start = len(open_positions)

        total_unrealized = 0.0
        still_open: list[_OpenPosition] = []
        for pos in open_positions:
            T_remaining = max((pos.expiry_date - today.date()).days, 0) / 365.0
            dte_remaining = (pos.expiry_date - today.date()).days

            if cfg.use_iv_skew:
                marks = [
                    reprice_leg_with_skew(leg.option_type, leg.strike, spot_today, iv_today, T_remaining, r, q,
                                           cfg.skew_slope, cfg.smile_curvature)
                    for leg in pos.spread.legs
                ]
            else:
                marks = [
                    reprice_leg(leg.option_type, leg.strike, spot_today, iv_today, T_remaining, r, q)
                    for leg in pos.spread.legs
                ]
            unrealized_pnl = sum(
                _leg_pnl(leg.action, pos.entry_fills[i], marks[i], pos.spread.lot_size, pos.spread.n_lots)
                for i, leg in enumerate(pos.spread.legs)
            )

            net_credit_rupees = pos.spread.net_credit * pos.spread.lot_size * pos.spread.n_lots
            exit_reason = None
            if unrealized_pnl >= cfg.profit_target_pct * net_credit_rupees:
                exit_reason = "profit_target"
            elif unrealized_pnl <= -cfg.stop_loss_pct * net_credit_rupees:
                exit_reason = "stop_loss"
            elif dte_remaining <= cfg.min_dte_exit:
                exit_reason = "dte_exit"

            if exit_reason is not None:
                realized_pnl = 0.0
                for i, leg in enumerate(pos.spread.legs):
                    closing_action = "BUY" if leg.action == "SELL" else "SELL"
                    exit_fill = _apply_slippage(marks[i], closing_action, cfg.slippage_pct)
                    leg_pnl = _leg_pnl(leg.action, pos.entry_fills[i], exit_fill,
                                        pos.spread.lot_size, pos.spread.n_lots)
                    realized_pnl += leg_pnl
                    trade_log_rows.append({
                        "trade_id": pos.trade_id,
                        "entry_date": pos.entry_date,
                        "exit_date": today.date(),
                        "instrument": instrument,
                        "strategy_type": pos.spread.strategy_type,
                        "option_type": leg.option_type,
                        "strike": leg.strike,
                        "action": leg.action,
                        "entry_price": pos.entry_fills[i],
                        "exit_price": exit_fill,
                        "lot_size": pos.spread.lot_size,
                        "n_lots": pos.spread.n_lots,
                        "leg_pnl": leg_pnl,
                        "exit_reason": exit_reason,
                    })
                capital += realized_pnl
            else:
                total_unrealized += unrealized_pnl
                still_open.append(pos)

        open_positions = still_open
        equity_curve.loc[today] = capital + total_unrealized

        # consider at most one new entry today, only if a slot was already
        # free at the START of today (see snapshot above)
        if positions_at_start < cfg.max_concurrent_positions:
            strategy_type = cfg.strategy_override or recommend_strategy_type(
                row["trend"], row["iv_rank"], cfg.assumed_range_score, SIGNALS,
            )
            if strategy_type != NO_TRADE:
                expiry_date = today.date() + timedelta(days=cfg.tenor_days)
                T = cfg.tenor_days / 365.0
                if cfg.use_iv_skew:
                    chain = reconstruct_chain_with_skew(spot_today, iv_today, T, r, q, spec.strike_step,
                                                         cfg.skew_slope, cfg.smile_curvature)
                else:
                    chain = reconstruct_chain(spot_today, iv_today, T, r, q, spec.strike_step)
                forward = spot_today * np.exp((r - q) * T)

                spread = select_spread(chain, strategy_type, forward, T, r, instrument, expiry_date,
                                        spec.strike_step, lot_size=spec.lot_size, n_lots=cfg.n_lots,
                                        cfg=cfg.strike_selection_cfg)
                if spread is not None:
                    entry_fills = [_apply_slippage(leg.price, leg.action, cfg.slippage_pct) for leg in spread.legs]
                    open_positions.append(_OpenPosition(next_trade_id, today.date(), expiry_date, spread, entry_fills))
                    next_trade_id += 1

    trade_log = pd.DataFrame(trade_log_rows)
    return {"trade_log": trade_log, "equity_curve": equity_curve.ffill()}
