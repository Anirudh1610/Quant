"""
backtest.py
-----------
Event-driven futures backtesting engine for the NSE F&O strategy.

Execution model  (zero lookahead bias)
---------------------------------------
Signal computed at close of day T  →  executed at open of day T+1.

Futures P&L mechanics  (daily MTM settlement)
----------------------------------------------
Futures are mark-to-market daily.  All unrealised P&L is settled in cash
at end of each session.

Day T+1 (entry day):
    entry at open_{T+1}
    entry_day_mtm = (close_{T+1} − open_{T+1}) × lots × lot_size × direction
    cash += entry_day_mtm

Day T+k (held day, k > 1):
    daily_mtm = (close_{T+k} − close_{T+k−1}) × lots × lot_size × direction
    cash += daily_mtm

Exit day (at open of exit_day):
    exit_mtm = (open_{exit_day} − close_{prev_day}) × lots × lot_size × direction
    cash += exit_mtm
    cash −= exit_transaction_cost

Because P&L is settled daily, portfolio NAV = cash at all times.

Rollover
---------
Two business days before each monthly expiry (NSE last Thursday):
    close the near-month position  →  incur sell transaction cost
    open a far-month position      →  incur buy transaction cost
    The synthetic price is identical (same CoC formula), so no price gap.

Risk circuit breakers
----------------------
1. Daily loss limit  : if today's P&L < −daily_loss_limit × prev_nav  → close all
2. Max drawdown gate : if drawdown from peak > max_drawdown_limit     → stop trading

Capital accounting
------------------
    cash         = running available funds (increases with P&L, decreases with losses)
    margin_used  = Σ |position_notional| × margin_rate   (reserved from cash)
    free_cash    = cash − margin_used   (available for new positions)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .costs import compute_futures_cost, compute_rollover_cost
from .portfolio import (
    compute_position_size, compute_exposure,
    sector_cap_lots, beta_neutral_scale,
)
from .universe import LOT_SIZES, get_margin_rate, is_index_future


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FuturesBacktestConfig:
    """All tunable parameters for the futures backtest in one place."""

    # ── Capital ─────────────────────────────────────────────────────────────
    initial_capital: float = 5_000_000.0   # ₹50 lakh — min realistic futures capital

    # ── Universe limits ──────────────────────────────────────────────────────
    max_long:  int = 8    # max simultaneous long contracts
    max_short: int = 8    # max simultaneous short contracts

    # ── Signal parameters ────────────────────────────────────────────────────
    momentum_window:   int   = 120    # 6-month cross-sectional momentum (v2)
    ema_period:        int   = 20     # EMA period for pullback entry timing
    adx_period:        int   = 14     # ADX period for trend-strength filter
    adx_threshold:     float = 20.0   # minimum ADX to confirm trend (skip choppy)
    ema_tolerance:     float = 0.015  # max distance from EMA20 for pullback entry (1.5%)
    atr_period:        int   = 14     # ATR period for stops and sizing

    # ── Entry / ranking parameters ───────────────────────────────────────────
    rank_entry:  int = 5    # enter only when momentum rank ≤ rank_entry
    rank_exit:   int = 15   # exit when momentum rank > rank_exit (hysteresis buffer)

    # ── Exit rules (v2) ──────────────────────────────────────────────────────
    min_hold_days:       int   = 10    # minimum hold before rank/other exits fire
    trailing_stop_mult:  float = 2.5   # ATR trailing stop multiplier (ratchets with price)

    # ── Rebalancing ──────────────────────────────────────────────────────────
    rebalance_weekly: bool  = True     # only open new entries on first day of each week

    # ── Legacy v1 exit rules (used when ranks=None passed to engine.run()) ───
    rsi_period:        int   = 2      # ultra-short-term RSI
    rsi_long_entry:    float = 10.0   # enter long when RSI(2) < 10
    rsi_short_entry:   float = 90.0   # enter short when RSI(2) > 90
    bb_window:         int   = 20     # Bollinger Band period
    bb_std:            float = 2.0    # Bollinger Band width (σ)
    bb_buffer:         float = 0.005  # 0.5% buffer around band for entry
    ma_trend_window:   int   = 50     # SMA for trend gate
    hold_days:         int   = 5      # max days before time-based exit
    rsi_exit_long:     float = 70.0   # exit long when RSI(2) > 70
    rsi_exit_short:    float = 30.0   # exit short when RSI(2) < 30
    atr_stop_mult:     float = 2.0    # v1 fixed stop = entry ± atr_stop_mult × ATR

    # ── Sizing ───────────────────────────────────────────────────────────────
    risk_fraction:       float = 0.01   # max 1% NAV risk per trade (ATR stop)
    max_gross_multiple:  float = 3.0    # max gross notional / NAV
    max_position_pct:    float = 0.20   # max single-position notional / NAV
    min_lots:            int   = 1
    max_lots:            int   = 50

    # ── Risk circuit breakers ────────────────────────────────────────────────
    daily_loss_limit:   float = 0.03    # close all if day P&L < −3% of prev NAV
    max_drawdown_limit: float = 0.20    # stop trading if drawdown > 20% from peak

    # ── VIX regime ───────────────────────────────────────────────────────────
    vix_extreme_threshold: float = 35.0  # no new positions above this VIX

    # ── Momentum filter limits ────────────────────────────────────────────────
    top_n:    int = 5    # top-momentum long candidates (tighter = higher quality)
    bottom_n: int = 5    # bottom-momentum short candidates


# ──────────────────────────────────────────────────────────────────────────────
# INTERNAL TRADE RECORD
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _Trade:
    ticker:           str
    direction:        str        # "LONG" | "SHORT"
    entry_date:       pd.Timestamp
    exit_date:        pd.Timestamp
    entry_price:      float
    exit_price:       float
    num_lots:         int
    lot_size:         int
    notional:         float      # num_lots × lot_size × entry_price
    gross_pnl:        float
    entry_cost:       float
    exit_cost:        float
    rollover_costs:   float
    net_pnl:          float
    exit_reason:      str        # "TIME" | "ATR_STOP" | "RSI_EXIT" | "RANK_EXIT" | "TRAILING_STOP" | "ROLLOVER" | "EOB" | "CIRCUIT"


# ──────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ──────────────────────────────────────────────────────────────────────────────

class FuturesBacktestEngine:
    """
    Event-driven, daily-resolution futures backtesting engine.

    Usage
    -----
    >>> cfg    = FuturesBacktestConfig()
    >>> engine = FuturesBacktestEngine(cfg)
    >>> results = engine.run(dataset, long_entries, short_entries, rsi, atr)
    """

    def __init__(self, config: FuturesBacktestConfig):
        self.cfg = config
        self._trades: list[_Trade] = []

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────────────────

    def run(
        self,
        dataset:       dict,
        long_entries:  pd.DataFrame,
        short_entries: pd.DataFrame,
        rsi:           pd.DataFrame,
        atr:           pd.DataFrame,
        regime:        pd.DataFrame,
        ranks:         pd.DataFrame | None = None,
    ) -> dict:
        """
        Run the full backtest simulation.

        Parameters
        ----------
        dataset       : Output of data.load_futures_dataset().
        long_entries  : Boolean entry signals for longs (rows = dates, cols = tickers).
        short_entries : Boolean entry signals for shorts.
        rsi           : RSI(2) DataFrame (used for legacy v1 exits when ranks=None).
        atr           : ATR(14) DataFrame.
        regime        : Output of signals.detect_combined_regime().
        ranks         : Momentum rank DataFrame (from rank_momentum).  When provided,
                        rank-buffer exits replace RSI/time exits and trailing stops
                        replace the fixed ATR stop.  Pass None for legacy v1 behaviour.

        Returns
        -------
        dict with keys: equity_curve, daily_returns, trade_log, exposure_df,
                        margin_df, roll_dates_fired, nifty_close.
        """
        self._trades = []
        cfg = self.cfg

        futures_close  = dataset["futures"]["close"]
        futures_open   = dataset["futures"]["open"]
        futures_high   = dataset["futures"]["high"]
        futures_low    = dataset["futures"]["low"]
        roll_dates_set = set(dataset["roll_dates"])
        india_vix      = dataset["india_vix"]

        dates     = futures_close.index
        cash      = cfg.initial_capital
        peak_nav  = cfg.initial_capital
        trading_halted = False

        # active: {ticker → position_dict}
        # position_dict keys: direction(int), num_lots, lot_size, entry_date,
        #   entry_price, last_price, stop_price, days_held, entry_cost,
        #   rollover_costs, direction_str
        active: dict[str, dict] = {}

        equity_curve = pd.Series(np.nan,  index=dates)
        exposure_df  = pd.DataFrame(index=dates,
                                    columns=["gross_notional","net_notional",
                                             "long_notional","short_notional",
                                             "num_long","num_short"],
                                    dtype=float)
        margin_series = pd.Series(np.nan, index=dates)
        rolls_fired: list[pd.Timestamp] = []

        for i, date in enumerate(dates):
            if i == 0:
                equity_curve.iloc[0] = cash
                continue

            prev_date = dates[i - 1]

            # ── Step 1: MTM for currently open positions ─────────────────
            for ticker in list(active.keys()):
                pos = active[ticker]
                if ticker not in futures_close.columns:
                    continue
                curr_close = futures_close.loc[date, ticker]
                prev_price = pos["last_price"]
                if np.isnan(curr_close) or np.isnan(prev_price):
                    continue
                daily_mtm = (curr_close - prev_price) * pos["lot_size"] * pos["num_lots"] * pos["direction"]
                cash += daily_mtm
                pos["last_price"] = curr_close
                pos["days_held"] += 1

                # Update ATR trailing stop (ratchets in favour of the position)
                if ranks is not None and ticker in atr.columns and date in atr.index:
                    atr_val = atr.loc[date, ticker]
                    if not np.isnan(atr_val) and atr_val > 0:
                        if pos["direction"] == 1:   # LONG: stop moves up, never down
                            new_stop = curr_close - cfg.trailing_stop_mult * atr_val
                            pos["stop_price"] = max(pos["stop_price"], new_stop)
                        else:                        # SHORT: stop moves down, never up
                            new_stop = curr_close + cfg.trailing_stop_mult * atr_val
                            pos["stop_price"] = min(pos["stop_price"], new_stop)

            nav = cash
            daily_pnl_pct = nav / equity_curve.iloc[i - 1] - 1.0 if equity_curve.iloc[i - 1] > 0 else 0.0

            # ── Step 2: Daily loss limit / drawdown circuit ───────────────
            peak_nav = max(peak_nav, nav)
            drawdown = nav / peak_nav - 1.0
            if (daily_pnl_pct < -cfg.daily_loss_limit or
                    drawdown < -cfg.max_drawdown_limit):
                if active:
                    print(f"[CIRCUIT] {date.date()} — daily_pnl={daily_pnl_pct:.2%}  "
                          f"drawdown={drawdown:.2%}  → closing all positions")
                    for ticker in list(active.keys()):
                        if ticker in futures_open.columns:
                            ep = futures_open.loc[date, ticker]
                        else:
                            ep = active[ticker]["last_price"]
                        cash = self._close_position(active, ticker, ep, "CIRCUIT", date, cash)
                trading_halted = True

            # Re-enable trading when conditions normalise (next day)
            if trading_halted and drawdown > -cfg.max_drawdown_limit * 0.80:
                trading_halted = False

            # ── Step 3: Exit checks for open positions ────────────────────
            for ticker in list(active.keys()):
                pos  = active[ticker]
                if ticker not in futures_close.columns:
                    continue

                curr_close = futures_close.loc[date, ticker]
                curr_high  = futures_high.loc[date, ticker]  if ticker in futures_high.columns  else curr_close
                curr_low   = futures_low.loc[date, ticker]   if ticker in futures_low.columns   else curr_close

                exit_reason: Optional[str] = None
                exit_price  = curr_close   # default: close-of-day

                # ATR stop-loss (checked intraday via high/low)
                if pos["direction"] == 1:
                    if curr_low <= pos["stop_price"]:
                        exit_reason = "ATR_STOP"
                        exit_price  = pos["stop_price"]
                else:
                    if curr_high >= pos["stop_price"]:
                        exit_reason = "ATR_STOP"
                        exit_price  = pos["stop_price"]

                # RSI exit
                if exit_reason is None and ranks is None and ticker in rsi.columns:
                    cur_rsi = rsi.loc[date, ticker]
                    if pos["direction"] == 1  and cur_rsi > cfg.rsi_exit_long:
                        exit_reason = "RSI_EXIT"
                    elif pos["direction"] == -1 and cur_rsi < cfg.rsi_exit_short:
                        exit_reason = "RSI_EXIT"

                # Time exit (legacy v1 — only when ranks=None)
                if exit_reason is None and ranks is None and pos["days_held"] >= cfg.hold_days:
                    exit_reason = "TIME"

                # Rank-buffer exit (v2 — replaces RSI + time exits)
                # Only fires after min_hold_days to avoid premature rank-chasing.
                # Direction matters: long exits when rank deteriorates (rises above rank_exit);
                # short exits when rank IMPROVES (falls below total - rank_exit + 1).
                if exit_reason is None and ranks is not None and pos["days_held"] >= cfg.min_hold_days:
                    rank_date = prev_date if prev_date in ranks.index else date
                    if rank_date in ranks.index and ticker in ranks.columns:
                        curr_rank = ranks.loc[rank_date, ticker]
                        total_stocks = ranks.shape[1]
                        if pos["direction"] == 1:
                            # Long: exit if momentum has deteriorated past the buffer
                            if pd.isna(curr_rank) or curr_rank > cfg.rank_exit:
                                exit_reason = "RANK_EXIT"
                        else:
                            # Short: exit if momentum has improved back above the buffer
                            # entered when rank > total-bottom_n; hold while rank >= total-rank_exit+1
                            short_exit_threshold = total_stocks - cfg.rank_exit + 1
                            if pd.isna(curr_rank) or curr_rank < short_exit_threshold:
                                exit_reason = "RANK_EXIT"
                    else:
                        exit_reason = "RANK_EXIT"  # ticker no longer ranked

                if exit_reason:
                    # Correct last_price for stop exits (price already moved to stop)
                    if exit_reason == "ATR_STOP":
                        # Undo the close-based MTM above and re-do to stop price
                        correction = (exit_price - curr_close) * pos["lot_size"] * pos["num_lots"] * pos["direction"]
                        cash += correction
                        pos["last_price"] = exit_price
                    cash = self._close_position(active, ticker, exit_price, exit_reason, date, cash)

            # ── Step 4: Rollover positions ────────────────────────────────
            if date in roll_dates_set:
                for ticker in list(active.keys()):
                    pos = active[ticker]
                    if ticker not in futures_close.columns:
                        continue
                    roll_price   = futures_close.loc[date, ticker]
                    lot_size     = pos["lot_size"]
                    num_lots     = pos["num_lots"]
                    notional     = num_lots * lot_size * roll_price
                    roll_cost    = compute_rollover_cost(notional, is_index=is_index_future(ticker))
                    cash        -= roll_cost
                    pos["rollover_costs"] += roll_cost
                    # Update the stop to the new contract price
                    old_stop_pct = (pos["stop_price"] - pos["entry_price"]) / pos["entry_price"]
                    pos["entry_price"] = roll_price
                    pos["stop_price"]  = roll_price * (1.0 + old_stop_pct)
                rolls_fired.append(date)

            # ── Step 5: Open new positions ────────────────────────────────
            # Weekly rebalance gate: only open new entries on first day of each calendar week
            is_rebalance_day = (
                not cfg.rebalance_weekly
                or date.isocalendar().week != prev_date.isocalendar().week
            )
            if not trading_halted and is_rebalance_day and prev_date in long_entries.index:
                vix_today = india_vix.get(date, india_vix.iloc[-1]) if isinstance(india_vix, pd.Series) else india_vix.iloc[-1]
                if hasattr(vix_today, "item"):
                    vix_today = vix_today.item()
                vix_today = float(vix_today) if not np.isnan(float(vix_today)) else 15.0

                # Compute scalar from VIX
                from .signals import vix_position_scalar
                vix_scal_series = vix_position_scalar(pd.Series([vix_today]))[0]

                # Skip new positions when VIX is extreme
                if vix_today >= cfg.vix_extreme_threshold:
                    pass
                else:
                    # Determine allowed sides from regime
                    reg_row = regime.loc[prev_date] if prev_date in regime.index else None
                    allow_longs  = reg_row["allow_longs"]  if reg_row is not None else True
                    allow_shorts = reg_row["allow_shorts"] if reg_row is not None else True

                    # Current exposure & free cash
                    close_row_today = futures_close.loc[date] if date in futures_close.index else pd.Series(dtype=float)
                    exp = compute_exposure(active, close_row_today)
                    free_cash = cash - exp["margin_used"]

                    # Slots available
                    long_slots  = max(0, cfg.max_long  - exp["num_long"])
                    short_slots = max(0, cfg.max_short - exp["num_short"])

                    if allow_longs and long_slots > 0:
                        new_longs = [
                            t for t in long_entries.columns
                            if long_entries.loc[prev_date, t]
                            and t not in active
                            and t in futures_open.columns
                        ][:long_slots]

                        for ticker in new_longs:
                            entry_p = futures_open.loc[date, ticker]
                            if np.isnan(entry_p) or entry_p <= 0:
                                continue
                            lot_size = LOT_SIZES.get(ticker, 1)
                            atr_val  = atr.loc[prev_date, ticker] if prev_date in atr.index and ticker in atr.columns else entry_p * 0.02

                            lots, stop_p = compute_position_size(
                                ticker=ticker, direction=1, nav=nav,
                                futures_price=entry_p, atr=atr_val,
                                current_gross_notional=exp["gross_notional"],
                                available_margin=free_cash,
                                vix_scalar=vix_scal_series,
                                risk_fraction=cfg.risk_fraction,
                                atr_multiplier=cfg.trailing_stop_mult if ranks is not None else cfg.atr_stop_mult,
                                max_gross_multiple=cfg.max_gross_multiple,
                                max_position_pct=cfg.max_position_pct,
                                min_lots=cfg.min_lots, max_lots=cfg.max_lots,
                            )
                            if lots < cfg.min_lots:
                                continue

                            # Sector concentration cap + beta-neutral scaling (v2)
                            close_row_entry = futures_close.loc[date] if date in futures_close.index else pd.Series(dtype=float)
                            if ranks is not None:
                                lots = sector_cap_lots(
                                    ticker, lots, entry_p, lot_size, nav,
                                    active, close_row_entry, max_sector_pct=0.30,
                                )
                                exp_now = compute_exposure(active, close_row_entry)
                                b_scale = beta_neutral_scale(
                                    1, exp_now["long_notional"], exp_now["short_notional"]
                                )
                                lots = max(0, int(np.floor(lots * b_scale)))

                            if lots < cfg.min_lots:
                                continue

                            notional   = lots * lot_size * entry_p
                            entry_cost = compute_futures_cost(notional, "BUY", is_index=is_index_future(ticker))["total"]
                            # MTM from entry-open to close of entry day
                            entry_day_mtm = (futures_close.loc[date, ticker] - entry_p) * lot_size * lots
                            cash -= entry_cost
                            cash += entry_day_mtm

                            active[ticker] = dict(
                                direction=1, direction_str="LONG",
                                num_lots=lots, lot_size=lot_size,
                                entry_date=date, entry_price=entry_p,
                                last_price=futures_close.loc[date, ticker],
                                stop_price=stop_p,
                                days_held=0,
                                entry_cost=entry_cost,
                                rollover_costs=0.0,
                            )
                            free_cash -= lots * lot_size * entry_p * get_margin_rate(ticker)
                            exp["gross_notional"] += notional
                            exp["num_long"] += 1

                    if allow_shorts and short_slots > 0:
                        new_shorts = [
                            t for t in short_entries.columns
                            if short_entries.loc[prev_date, t]
                            and t not in active
                            and t in futures_open.columns
                        ][:short_slots]

                        for ticker in new_shorts:
                            entry_p = futures_open.loc[date, ticker]
                            if np.isnan(entry_p) or entry_p <= 0:
                                continue
                            lot_size = LOT_SIZES.get(ticker, 1)
                            atr_val  = atr.loc[prev_date, ticker] if prev_date in atr.index and ticker in atr.columns else entry_p * 0.02

                            lots, stop_p = compute_position_size(
                                ticker=ticker, direction=-1, nav=nav,
                                futures_price=entry_p, atr=atr_val,
                                current_gross_notional=exp["gross_notional"],
                                available_margin=free_cash,
                                vix_scalar=vix_scal_series,
                                risk_fraction=cfg.risk_fraction,
                                atr_multiplier=cfg.trailing_stop_mult if ranks is not None else cfg.atr_stop_mult,
                                max_gross_multiple=cfg.max_gross_multiple,
                                max_position_pct=cfg.max_position_pct,
                                min_lots=cfg.min_lots, max_lots=cfg.max_lots,
                            )
                            if lots < cfg.min_lots:
                                continue

                            # Sector concentration cap + beta-neutral scaling (v2)
                            close_row_entry_s = futures_close.loc[date] if date in futures_close.index else pd.Series(dtype=float)
                            if ranks is not None:
                                lots = sector_cap_lots(
                                    ticker, lots, entry_p, lot_size, nav,
                                    active, close_row_entry_s, max_sector_pct=0.30,
                                )
                                exp_now_s = compute_exposure(active, close_row_entry_s)
                                b_scale_s = beta_neutral_scale(
                                    -1, exp_now_s["long_notional"], exp_now_s["short_notional"]
                                )
                                lots = max(0, int(np.floor(lots * b_scale_s)))

                            if lots < cfg.min_lots:
                                continue

                            notional   = lots * lot_size * entry_p
                            entry_cost = compute_futures_cost(notional, "SELL", is_index=is_index_future(ticker))["total"]
                            entry_day_mtm = (futures_close.loc[date, ticker] - entry_p) * lot_size * lots * (-1)
                            cash -= entry_cost
                            cash += entry_day_mtm

                            active[ticker] = dict(
                                direction=-1, direction_str="SHORT",
                                num_lots=lots, lot_size=lot_size,
                                entry_date=date, entry_price=entry_p,
                                last_price=futures_close.loc[date, ticker],
                                stop_price=stop_p,
                                days_held=0,
                                entry_cost=entry_cost,
                                rollover_costs=0.0,
                            )
                            free_cash -= lots * lot_size * entry_p * get_margin_rate(ticker)
                            exp["gross_notional"] += notional
                            exp["num_short"] += 1

            # ── Step 6: Snapshot ──────────────────────────────────────────
            nav = cash
            equity_curve.iloc[i] = nav

            close_row = futures_close.loc[date] if date in futures_close.index else pd.Series(dtype=float)
            exp_snap  = compute_exposure(active, close_row)
            exposure_df.loc[date]  = [
                exp_snap["gross_notional"], exp_snap["net_notional"],
                exp_snap["long_notional"],  exp_snap["short_notional"],
                exp_snap["num_long"],       exp_snap["num_short"],
            ]
            margin_series.iloc[i] = exp_snap["margin_used"]

        # ── Force-close all remaining positions at last date ─────────────
        last_date = dates[-1]
        for ticker in list(active.keys()):
            ep = futures_close.loc[last_date, ticker] if ticker in futures_close.columns else active[ticker]["last_price"]
            cash = self._close_position(active, ticker, ep, "EOB", last_date, cash)
        equity_curve.iloc[-1] = cash

        daily_returns = equity_curve.pct_change().fillna(0.0)

        return {
            "equity_curve":    equity_curve,
            "daily_returns":   daily_returns,
            "trade_log":       self._build_trade_log(),
            "exposure_df":     exposure_df.astype(float),
            "margin_series":   margin_series,
            "rolls_fired":     pd.DatetimeIndex(rolls_fired),
            "nifty_close":     dataset["spot"]["close"].get("NIFTY"),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _close_position(
        self,
        active: dict,
        ticker: str,
        exit_price: float,
        reason: str,
        exit_date: pd.Timestamp,
        cash: float,
    ) -> float:
        """
        Close a position, apply exit transaction cost, record the trade.

        For futures, the MTM P&L is already in cash (settled daily).
        The exit step only deducts the transaction cost of the closing order.
        Gross P&L is the total price movement from entry to exit (for records).
        """
        pos      = active.pop(ticker)
        cfg      = self.cfg
        lot_size = pos["lot_size"]
        num_lots = pos["num_lots"]
        notional = num_lots * lot_size * exit_price
        is_idx   = is_index_future(ticker)

        # Direction of the CLOSING order is opposite to position direction
        close_dir = "SELL" if pos["direction"] == 1 else "BUY"
        exit_cost = compute_futures_cost(notional, close_dir, is_index=is_idx)["total"]
        cash -= exit_cost

        # Gross P&L = full price movement from entry to exit
        gross_pnl = (exit_price - pos["entry_price"]) * lot_size * num_lots * pos["direction"]
        net_pnl   = gross_pnl - pos["entry_cost"] - exit_cost - pos["rollover_costs"]

        self._trades.append(
            _Trade(
                ticker=ticker,
                direction=pos["direction_str"],
                entry_date=pos["entry_date"],
                exit_date=exit_date,
                entry_price=round(pos["entry_price"], 2),
                exit_price=round(exit_price, 2),
                num_lots=num_lots,
                lot_size=lot_size,
                notional=round(pos["entry_price"] * lot_size * num_lots, 2),
                gross_pnl=round(gross_pnl, 2),
                entry_cost=round(pos["entry_cost"], 2),
                exit_cost=round(exit_cost, 2),
                rollover_costs=round(pos["rollover_costs"], 2),
                net_pnl=round(net_pnl, 2),
                exit_reason=reason,
            )
        )
        return cash

    def _build_trade_log(self) -> pd.DataFrame:
        if not self._trades:
            return pd.DataFrame()
        rows = [vars(t) for t in self._trades]
        df = pd.DataFrame(rows)
        df["hold_days"] = (df["exit_date"] - df["entry_date"]).dt.days
        return df.sort_values("entry_date").reset_index(drop=True)
