"""
backtest.py
-----------
Event-driven options selling backtest engine.

Trade mechanics (zero lookahead)
---------------------------------
Signal computed at close of day T  →  position opened at open of day T+1.
In practice "entry_days_before=1" means we sell on the trading day immediately
before expiry — the most common weekly options selling approach.

P&L settlement
--------------
  NSE settles weekly and monthly index options at the closing value of the
  index on expiry day.  Seller's P&L per share:

      gross_pnl = call_premium + put_premium
                  − max(spot_expiry − call_strike, 0)   ← call assigned loss
                  − max(put_strike  − spot_expiry, 0)   ← put assigned loss

  All amounts are in index points; multiply by lot_size × n_lots for ₹ P&L.

Stop-loss rule (optional, cfg.stop_loss_multiple > 0)
------------------------------------------------------
  During the holding period the position is marked to market at each day's
  close using Black-Scholes with the current VIX.  If the mark-to-market loss
  exceeds (stop_loss_multiple × total premium received), the position is
  treated as closed on that day at the theoretical mid price.

Expiry calendars
----------------
  Nifty 50   : every Tuesday  (user-confirmed; verify on NSE if in doubt)
  Bank Nifty : weekly every Wednesday  +  monthly last Thursday of each month
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .options import (
    bs_price, trade_cost, select_strikes,
    NIFTY_LOT, BANKNIFTY_LOT,
    NIFTY_STRIKE_STEP, BANKNIFTY_STRIKE_STEP,
    RISK_FREE_RATE, NIFTY_DIV_YIELD, BANKNIFTY_DIV_YIELD,
)


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class OptionsBacktestConfig:
    """All tunable parameters for the options backtest."""

    initial_capital: float = 2_000_000.0   # ₹20 lakh

    # Instruments
    trade_nifty:     bool = True
    trade_banknifty: bool = True

    # Lot sizes — update when NSE revises
    nifty_lot:     int = NIFTY_LOT        # 25 as of Nov 2024
    banknifty_lot: int = BANKNIFTY_LOT    # 15 as of Nov 2024

    # Number of lots per trade
    n_lots_nifty:     int = 1
    n_lots_banknifty: int = 1

    # Entry: how many trading days before expiry to enter
    entry_days_before: int = 3           # 3 days = enough premium, reasonable gamma risk

    # OTM distance for STRANGLE legs (multiple of daily IV sigma)
    sigma_multiple: float = 1.0

    # Risk filters
    max_vix: float = 25.0    # skip if prev VIX > this
    max_adx: float = 30.0    # skip if ADX(14) > this

    # Stop-loss: 0 = disabled; 1.5 = exit when loss > 1.5× premium received
    stop_loss_multiple: float = 1.5

    # Take-profit: 0 = disabled; 0.5 = exit when 50% of premium is captured
    take_profit_pct: float = 0.5

    # Bank Nifty: include monthly expiry trades?
    trade_banknifty_monthly: bool = True


# ──────────────────────────────────────────────────────────────────────────────
# EXPIRY CALENDARS
# ──────────────────────────────────────────────────────────────────────────────

def _nifty_weekly_expiries(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Every Tuesday in the trading calendar."""
    return dates[dates.dayofweek == 1]


def _banknifty_weekly_expiries(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Every Wednesday in the trading calendar."""
    return dates[dates.dayofweek == 2]


def _banknifty_monthly_expiries(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last Thursday of each calendar month present in the trading calendar."""
    thursdays = dates[dates.dayofweek == 3]
    last_thu = (
        thursdays.to_series()
        .groupby([thursdays.year, thursdays.month])
        .last()
        .values
    )
    return pd.DatetimeIndex(last_thu)


def _entry_date(
    expiry: pd.Timestamp, n_days: int, dates: pd.DatetimeIndex
) -> Optional[pd.Timestamp]:
    """Trading date n_days before expiry, or None if out of range."""
    try:
        loc = dates.get_loc(expiry)
    except KeyError:
        return None
    if loc < n_days:
        return None
    return dates[loc - n_days]


# ──────────────────────────────────────────────────────────────────────────────
# ENGINE
# ──────────────────────────────────────────────────────────────────────────────

class OptionsBacktestEngine:

    def __init__(self, cfg: OptionsBacktestConfig):
        self.cfg = cfg

    # ── helpers ──────────────────────────────────────────────────────────────

    def _entry_premiums(
        self, spot: float, call_k: int, put_k: int,
        T: float, iv: float, q: float,
    ) -> tuple[float, float]:
        c = bs_price(spot, call_k, T, RISK_FREE_RATE, iv, q, "call")
        p = bs_price(spot, put_k,  T, RISK_FREE_RATE, iv, q, "put")
        return c, p

    def _expiry_pnl(
        self,
        spot_expiry: float,
        call_k: int, put_k: int,
        call_prem: float, put_prem: float,
        lot: int, n_lots: int,
    ) -> tuple[float, float, float]:
        """Returns (gross_pnl, costs, net_pnl) in ₹."""
        credit    = (call_prem + put_prem) * lot * n_lots
        call_loss = max(spot_expiry - call_k, 0.0) * lot * n_lots
        put_loss  = max(put_k - spot_expiry, 0.0) * lot * n_lots
        gross     = credit - call_loss - put_loss
        costs     = trade_cost(call_prem, lot, n_lots) + trade_cost(put_prem, lot, n_lots)
        return gross, costs, gross - costs

    def _build_schedule(self, dates: pd.DatetimeIndex) -> list[dict]:
        """Flat list of planned (instrument, expiry, entry) tuples."""
        cfg    = self.cfg
        plan   = []

        if cfg.trade_nifty:
            for exp in _nifty_weekly_expiries(dates):
                entry = _entry_date(exp, cfg.entry_days_before, dates)
                if entry is not None:
                    plan.append({
                        "instrument":  "NIFTY",
                        "expiry":      exp,
                        "entry":       entry,
                        "lot":         cfg.nifty_lot,
                        "n_lots":      cfg.n_lots_nifty,
                        "step":        NIFTY_STRIKE_STEP,
                        "q":           NIFTY_DIV_YIELD,
                        "expiry_type": "WEEKLY",
                    })

        if cfg.trade_banknifty:
            weekly  = set(_banknifty_weekly_expiries(dates))
            monthly = set(_banknifty_monthly_expiries(dates)) if cfg.trade_banknifty_monthly else set()
            for exp in sorted(weekly | monthly):
                entry = _entry_date(exp, cfg.entry_days_before, dates)
                if entry is not None:
                    plan.append({
                        "instrument":  "BANKNIFTY",
                        "expiry":      exp,
                        "entry":       entry,
                        "lot":         cfg.banknifty_lot,
                        "n_lots":      cfg.n_lots_banknifty,
                        "step":        BANKNIFTY_STRIKE_STEP,
                        "q":           BANKNIFTY_DIV_YIELD,
                        "expiry_type": "MONTHLY" if exp in monthly else "WEEKLY",
                    })

        return sorted(plan, key=lambda x: x["entry"])

    # ── main run ──────────────────────────────────────────────────────────────

    def run(
        self,
        nifty: pd.DataFrame,
        banknifty: pd.DataFrame,
        vix: pd.Series,
        trend_nifty: pd.Series,
        trend_banknifty: pd.Series,
    ) -> dict:
        """
        Simulate the options selling strategy over the full date range.

        Parameters
        ----------
        nifty, banknifty  : OHLCV DataFrames indexed by date.
        vix               : India VIX close series.
        trend_nifty       : Daily signal series for Nifty.
        trend_banknifty   : Daily signal series for Bank Nifty.

        Returns
        -------
        dict:
            "trade_log"    : pd.DataFrame  one row per closed trade
            "equity_curve" : pd.Series     portfolio NAV indexed by date
            "daily_returns": pd.Series     daily return fractions
        """
        cfg   = self.cfg
        dates = nifty.index

        schedule  = self._build_schedule(dates)
        pnl_events: list[tuple[pd.Timestamp, float]] = []
        trade_log: list[dict] = []

        for meta in schedule:
            entry  = meta["entry"]
            expiry = meta["expiry"]
            inst   = meta["instrument"]
            lot    = meta["lot"]
            n_lots = meta["n_lots"]
            step   = meta["step"]
            q      = meta["q"]

            src    = nifty if inst == "NIFTY" else banknifty
            signal = (trend_nifty if inst == "NIFTY" else trend_banknifty).get(entry)

            if signal is None or signal == "SKIP":
                continue
            if entry not in src.index or expiry not in src.index:
                continue

            spot_entry  = float(src.loc[entry, "open"])
            spot_expiry = float(src.loc[expiry, "close"])
            iv_entry    = float(vix.get(entry, 15.0)) / 100   # VIX% → decimal

            loc_entry  = dates.get_loc(entry)
            loc_expiry = dates.get_loc(expiry)
            T_entry    = max((loc_expiry - loc_entry) / 252, 1 / 252)
            days_left  = loc_expiry - loc_entry  # trading days remaining

            # Use period sigma (not daily) so 1σ-OTM is 1× the full holding-period move
            period_iv_pct = (iv_entry / np.sqrt(252)) * np.sqrt(days_left) * 100

            call_k, put_k, trade_type = select_strikes(
                spot_entry, signal, period_iv_pct, step, cfg.sigma_multiple
            )
            call_prem, put_prem = self._entry_premiums(
                spot_entry, call_k, put_k, T_entry, iv_entry, q
            )
            credit = (call_prem + put_prem) * lot * n_lots

            # ── optional stop-loss (mark-to-market each intermediate day) ────
            stopped   = False
            exit_date = expiry
            exit_spot = spot_expiry
            exit_call_prem = call_prem
            exit_put_prem  = put_prem
            exit_T         = 0.0
            exit_iv        = iv_entry

            use_sl = cfg.stop_loss_multiple > 0
            use_tp = cfg.take_profit_pct > 0

            if use_sl or use_tp:
                for mid_loc in range(loc_entry + 1, loc_expiry):
                    mid_date = dates[mid_loc]
                    if mid_date not in src.index:
                        continue
                    spot_mid = float(src.loc[mid_date, "close"])
                    iv_mid   = float(vix.get(mid_date, iv_entry * 100)) / 100
                    T_mid    = max((loc_expiry - mid_loc) / 252, 1 / 252)

                    c_mid = bs_price(spot_mid, call_k, T_mid, RISK_FREE_RATE, iv_mid, q, "call")
                    p_mid = bs_price(spot_mid, put_k,  T_mid, RISK_FREE_RATE, iv_mid, q, "put")
                    curr_val = (c_mid + p_mid) * lot * n_lots

                    # Take-profit: option value has decayed by take_profit_pct of initial credit
                    tp_hit = use_tp and curr_val <= credit * (1 - cfg.take_profit_pct)
                    # Stop-loss: current value exceeds entry by stop_loss_multiple × credit
                    sl_hit = use_sl and (curr_val - credit) > cfg.stop_loss_multiple * credit

                    if tp_hit or sl_hit:
                        stopped        = True
                        exit_date      = mid_date
                        exit_spot      = spot_mid
                        exit_call_prem = c_mid
                        exit_put_prem  = p_mid
                        exit_T         = T_mid
                        exit_iv        = iv_mid
                        exit_reason    = "TAKE_PROFIT" if tp_hit else "STOP_LOSS"
                        break

            # ── compute final P&L ─────────────────────────────────────────────
            if stopped:
                gross = ((call_prem + put_prem) - (exit_call_prem + exit_put_prem)) * lot * n_lots
                costs = (trade_cost(call_prem, lot, n_lots)
                         + trade_cost(put_prem, lot, n_lots)
                         + trade_cost(exit_call_prem, lot, n_lots)
                         + trade_cost(exit_put_prem, lot, n_lots))
                # exit_reason already set in the loop (TAKE_PROFIT or STOP_LOSS)
            else:
                gross, costs, _ = self._expiry_pnl(
                    spot_expiry, call_k, put_k, call_prem, put_prem, lot, n_lots
                )
                exit_reason = "EXPIRY"

            net_pnl = gross - costs
            pnl_events.append((exit_date, net_pnl))

            trade_log.append({
                "entry_date":   entry,
                "exit_date":    exit_date,
                "expiry_date":  expiry,
                "instrument":   inst,
                "expiry_type":  meta["expiry_type"],
                "trade_type":   trade_type,
                "trend_signal": signal,
                "call_strike":  call_k,
                "put_strike":   put_k,
                "call_premium": round(call_prem, 2),
                "put_premium":  round(put_prem, 2),
                "total_premium": round(call_prem + put_prem, 2),
                "total_credit": round(credit, 2),
                "spot_entry":   round(spot_entry, 2),
                "spot_exit":    round(exit_spot, 2),
                "vix_entry":    round(iv_entry * 100, 2),
                "gross_pnl":    round(gross, 2),
                "costs":        round(costs, 2),
                "net_pnl":      round(net_pnl, 2),
                "exit_reason":  exit_reason,
            })

        # ── build equity curve from P&L events ───────────────────────────────
        tl = pd.DataFrame(trade_log)

        if pnl_events:
            ev_dates = pd.DatetimeIndex([e[0] for e in pnl_events])
            ev_pnl   = pd.Series([e[1] for e in pnl_events], index=ev_dates)
            daily_pnl = ev_pnl.groupby(level=0).sum()
            equity = (cfg.initial_capital + daily_pnl.cumsum()).reindex(dates).ffill()
        else:
            equity = pd.Series(cfg.initial_capital, index=dates)

        equity = equity.fillna(cfg.initial_capital)
        equity.name = "equity"
        daily_returns = equity.pct_change().fillna(0.0)

        return {
            "trade_log":     tl,
            "equity_curve":  equity,
            "daily_returns": daily_returns,
        }
