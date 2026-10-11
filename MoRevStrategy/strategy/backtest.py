"""
backtest.py
-----------
Event-driven backtesting engine for the cross-sectional momentum
+ mean-reversion strategy on the NIFTY 50 universe.

Execution model (avoids lookahead bias)
----------------------------------------
Signal computed at close of day T  →  trade executed at open of day T+1.

Position lifecycle
------------------
1. ENTRY  : open of T+1  (next-day open after signal day)
2. MONITOR: check stop-loss intraday using high/low as proxy
3. RSI EXIT: close of each day — if RSI crosses 50 against position
4. TIME EXIT: forced close at close of T+1+hold_days

Cash accounting (margin model)
-------------------------------
LONG   open : cash  -= notional + entry_cost
LONG   close: cash  += shares × exit_price - exit_cost

SHORT  open : cash  -= notional + entry_cost   (notional posted as margin)
SHORT  close: cash  += notional + (entry_price - exit_price) × shares - exit_cost

Portfolio NAV = cash
              + Σ long  positions: shares × current_close
              + Σ short positions: notional + (entry_price - current_close) × shares

Indian market transaction costs  (Zerodha, CNC equity delivery)
--------------------------------
Brokerage       : ₹0          — Zerodha offers free equity delivery
STT             : 0.10% on buy AND sell  (delivery; both sides)
NSE exchange    : 0.00297% per side
SEBI charges    : ₹10/crore = 0.0001% per side
GST             : 18% on (exchange + SEBI fees)
Stamp duty      : 0.015% on buy side only
DP charge       : ₹15.93 per scrip per day on sell (₹13.5 + 18% GST)
Slippage        : 0.01% per side  (NIFTY 50 stocks are extremely liquid)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# COST CONSTANTS  (Zerodha CNC — equity delivery)
# ──────────────────────────────────────────────────────────────────────────────

_STT          = 0.0010     # 0.10% on turnover — both buy AND sell for CNC delivery
_EXCHANGE     = 0.0000297  # NSE exchange transaction charge: 0.00297%
_SEBI         = 0.000001   # SEBI: ₹10/crore = 0.0001%
_GST          = (_EXCHANGE + _SEBI) * 0.18   # 18% GST on exchange + SEBI fees
_STAMP_DUTY   = 0.00015    # 0.015% stamp duty on buy side only
_SLIPPAGE     = 0.0001     # 0.01% market impact (NIFTY 50 — highly liquid)

BUY_COST_PCT        = _STT + _EXCHANGE + _SEBI + _GST + _STAMP_DUTY + _SLIPPAGE
SELL_COST_PCT       = _STT + _EXCHANGE + _SEBI + _GST + _SLIPPAGE
DP_CHARGE_PER_SELL  = 15.93   # ₹15.93 flat per scrip per day (₹13.5 + 18% GST)


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class BacktestConfig:
    """All tunable strategy and cost parameters in one place."""

    # ── capital ────────────────────────────────────────────────
    initial_capital: float = 1_000_000.0  # ₹10 lakh starting NAV

    # ── portfolio limits ───────────────────────────────────────
    max_long: int = 7
    max_short: int = 7              # enable equal number of short trades as long

    # ── exit rules ─────────────────────────────────────────────
    hold_days: int = 5             # shorter hold period improves return per sweep
    stop_loss_pct: float = 0.05    # 5% stop loss each side (best tradeoff in sweep)
    rsi_exit_long: float = 70.0    # exit long when RSI(2) crosses above 70
    rsi_exit_short: float = 40.0   # exit short when RSI(2) crosses below 40

    # ── signal parameters ──────────────────────────────────────────────────────────────────────────────────────
    momentum_window: int = 25      # 25-day: best risk/return per sensitivity analysis
    rsi_period: int = 2            # period-2 RSI (default)
    rsi_long_entry: float = 20.0   # RSI(2)<20 captures broader pullbacks — best Sharpe per sweep
    rsi_short_entry: float = 85.0  # kept for completeness (shorts disabled above)

    # ── filters ────────────────────────────────────────────────
    vol_filter_window: int = 20
    vol_threshold_multiplier: float = 1.8  # 1.8× median vol (best tradeoff in sweep)
    regime_ma_window: int = 200            # 200-day MA — only blocks longs in deep bear markets

    # ── costs (can override for sensitivity analysis) ──────────
    buy_cost_pct: float = BUY_COST_PCT
    sell_cost_pct: float = SELL_COST_PCT
    dp_charge_per_sell: float = DP_CHARGE_PER_SELL   # flat ₹/scrip on each sell

    # ── adaptive risk management ───────────────────────────────
    momentum_skip_days: int = 0          # JT skip: exclude last N days from momentum
    min_momentum_pct: float = 0.0        # only enter if |momentum| > this (e.g. 0.03 = 3%)
    use_atr_stop: bool = False           # True = ATR-based stop; False = fixed %
    atr_multiplier: float = 2.0          # stop distance = atr_multiplier × ATR(14)
    trailing_stop_pct: float = 0.0       # trailing stop as % from best price (0 = off)


# ──────────────────────────────────────────────────────────────────────────────
# TRADE RECORD
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _Trade:
    """Internal trade record populated when a position is closed."""

    ticker: str
    direction: str               # "LONG" | "SHORT"
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: float
    notional: float
    gross_pnl: float
    entry_cost: float
    exit_cost: float
    net_pnl: float
    exit_reason: str             # "TIME" | "STOP_LOSS" | "TRAIL_STOP" | "RSI_EXIT" | "EOB"


# ──────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ──────────────────────────────────────────────────────────────────────────────


class BacktestEngine:
    """
    Event-driven, day-by-day backtesting engine.

    Usage
    -----
    >>> cfg = BacktestConfig()
    >>> engine = BacktestEngine(cfg)
    >>> results = engine.run(close, open_, high, low, long_entries, short_entries, rsi)

    The engine supports multiple runs (each ``run()`` resets internal state).
    """

    def __init__(self, config: BacktestConfig):
        self.cfg = config
        self._trades: list[_Trade] = []

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC: run
    # ─────────────────────────────────────────────────────────────────────────

    def run(
        self,
        close: pd.DataFrame,
        open_: pd.DataFrame,
        high: pd.DataFrame,
        low: pd.DataFrame,
        long_entries: pd.DataFrame,
        short_entries: pd.DataFrame,
        rsi: pd.DataFrame,
        nifty_close: Optional[pd.Series] = None,
        atr: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Run the full backtest simulation.

        Parameters
        ----------
        close, open_, high, low : pd.DataFrame
            OHLCV price matrices (rows = dates, cols = tickers).
            ``open_`` is used for entry (next-day execution).
            ``high`` / ``low`` are used for intraday stop-loss detection.
        long_entries  : pd.DataFrame  Boolean entry signals (longs).
        short_entries : pd.DataFrame  Boolean entry signals (shorts).
        rsi           : pd.DataFrame  RSI values for exit monitoring.
        nifty_close   : pd.Series     Optional — stored in results for plotting.

        Returns
        -------
        dict
            equity_curve  : pd.Series   Portfolio NAV each day.
            daily_returns : pd.Series   Daily % change in NAV.
            trade_log     : pd.DataFrame  One row per closed trade.
            positions     : pd.DataFrame  Daily long/short notional per ticker.
        """
        # Reset state for re-use
        self._trades = []

        cfg = self.cfg
        dates = close.index
        cash = cfg.initial_capital

        # active_positions: {ticker → position_dict}
        # position_dict keys:
        #   direction, entry_date, entry_price, shares, notional, entry_cost, days_held
        active: dict[str, dict] = {}

        equity_curve = pd.Series(np.nan, index=dates)
        positions_snap = pd.DataFrame(0.0, index=dates, columns=close.columns)

        for i, date in enumerate(dates):

            # ── Day 0: no prior bar; just record starting NAV ────────────
            if i == 0:
                equity_curve.iloc[0] = cash
                continue

            prev_date = dates[i - 1]

            # ── Step 1: Update open positions (MTM + exit checks) ────────
            for ticker in list(active.keys()):
                pos = active[ticker]
                pos["days_held"] += 1

                if ticker not in close.columns:
                    continue

                cp = close.loc[date, ticker]  # current close price

                # ── Update trailing stop (advances stop as price moves in our favour)
                if cfg.trailing_stop_pct > 0:
                    if pos["direction"] == "LONG":
                        pos["best_price"] = max(pos["best_price"], cp)
                        trail_level = pos["best_price"] * (1.0 - cfg.trailing_stop_pct)
                        pos["stop_price"] = max(pos["stop_price"], trail_level)
                    else:
                        pos["best_price"] = min(pos["best_price"], cp)
                        trail_level = pos["best_price"] * (1.0 + cfg.trailing_stop_pct)
                        pos["stop_price"] = min(pos["stop_price"], trail_level)

                # ── Stop-loss check via intraday proxy ───────────────────
                #    For longs  : check if intraday LOW breached stop
                #    For shorts : check if intraday HIGH breached stop
                stop_hit = False
                stop_exit_price = np.nan

                if pos["direction"] == "LONG":
                    intra_low = low.loc[date, ticker] if ticker in low.columns else cp
                    if intra_low <= pos["stop_price"]:
                        stop_hit = True
                        stop_exit_price = pos["stop_price"]
                else:  # SHORT
                    intra_high = high.loc[date, ticker] if ticker in high.columns else cp
                    if intra_high >= pos["stop_price"]:
                        stop_hit = True
                        stop_exit_price = pos["stop_price"]

                # ── RSI exit check ───────────────────────────────────────
                rsi_hit = False
                cur_rsi = rsi.loc[date, ticker] if ticker in rsi.columns else 50.0

                if pos["direction"] == "LONG" and cur_rsi > cfg.rsi_exit_long:
                    rsi_hit = True
                elif pos["direction"] == "SHORT" and cur_rsi < cfg.rsi_exit_short:
                    rsi_hit = True

                # ── Time-based exit ──────────────────────────────────────
                time_hit = pos["days_held"] >= cfg.hold_days

                # ── Close if any exit condition is met ───────────────────
                if stop_hit or rsi_hit or time_hit:
                    exit_reason = (
                        ("TRAIL_STOP" if cfg.trailing_stop_pct > 0 else "STOP_LOSS") if stop_hit
                        else "RSI_EXIT" if rsi_hit
                        else "TIME"
                    )
                    exit_price = stop_exit_price if stop_hit else cp
                    cash = self._close_position(active, ticker, exit_price, exit_reason, date, cash)

            # ── Step 2: Open new positions on signals from PREVIOUS close ─
            #    Signal on prev_date → execute at today's open
            if prev_date in long_entries.index:

                new_long_tickers = [
                    t for t in long_entries.columns
                    if long_entries.loc[prev_date, t]
                    and t not in active
                    and t in open_.columns
                ]
                new_short_tickers = [
                    t for t in short_entries.columns
                    if short_entries.loc[prev_date, t]
                    and t not in active
                    and t in open_.columns
                ]

                # Limit to available slots
                long_slots = cfg.max_long - sum(
                    1 for p in active.values() if p["direction"] == "LONG"
                )
                short_slots = cfg.max_short - sum(
                    1 for p in active.values() if p["direction"] == "SHORT"
                )

                new_long_tickers = sorted(new_long_tickers)[:max(0, long_slots)]
                new_short_tickers = sorted(new_short_tickers)[:max(0, short_slots)]

                # Equal-weight sizing:
                #   Long  side: 100% NAV ÷ max_positions (fully deploy capital, no short book competing)
                #   Short side: 50% NAV ÷ max_positions (posts full notional as margin)
                current_nav = self._compute_nav(cash, active, close, date)
                long_notional = (current_nav * 1.0) / cfg.max_long
                short_notional = (current_nav * 0.5) / cfg.max_short if cfg.max_short > 0 else 0.0

                for ticker in new_long_tickers:
                    ep = open_.loc[date, ticker]
                    if ep <= 0 or np.isnan(ep):
                        continue
                    cost = long_notional * cfg.buy_cost_pct
                    shares = long_notional / ep
                    cash -= long_notional + cost
                    if cfg.use_atr_stop and atr is not None and ticker in atr.columns:
                        atr_val = float(atr.loc[date, ticker]) if date in atr.index else np.nan
                        stop_px = (ep - cfg.atr_multiplier * atr_val
                                   if np.isfinite(atr_val) and atr_val > 0
                                   else ep * (1.0 - cfg.stop_loss_pct))
                    else:
                        stop_px = ep * (1.0 - cfg.stop_loss_pct)
                    active[ticker] = dict(
                        direction="LONG",
                        entry_date=date,
                        entry_price=ep,
                        shares=shares,
                        notional=long_notional,
                        entry_cost=cost,
                        days_held=0,
                        stop_price=stop_px,
                        best_price=ep,
                    )

                for ticker in new_short_tickers:
                    ep = open_.loc[date, ticker]
                    if ep <= 0 or np.isnan(ep):
                        continue
                    cost = short_notional * cfg.buy_cost_pct
                    shares = short_notional / ep
                    cash -= short_notional + cost
                    if cfg.use_atr_stop and atr is not None and ticker in atr.columns:
                        atr_val = float(atr.loc[date, ticker]) if date in atr.index else np.nan
                        stop_px = (ep + cfg.atr_multiplier * atr_val
                                   if np.isfinite(atr_val) and atr_val > 0
                                   else ep * (1.0 + cfg.stop_loss_pct))
                    else:
                        stop_px = ep * (1.0 + cfg.stop_loss_pct)
                    active[ticker] = dict(
                        direction="SHORT",
                        entry_date=date,
                        entry_price=ep,
                        shares=shares,
                        notional=short_notional,
                        entry_cost=cost,
                        days_held=0,
                        stop_price=stop_px,
                        best_price=ep,
                    )

            # ── Step 3: Snapshot and record NAV ─────────────────────────
            nav = self._compute_nav(cash, active, close, date)
            equity_curve.iloc[i] = nav

            for ticker, pos in active.items():
                if ticker in positions_snap.columns:
                    sign = 1.0 if pos["direction"] == "LONG" else -1.0
                    positions_snap.loc[date, ticker] = sign * pos["notional"]

        # ── Force-close all remaining positions at last date ─────────────
        last_date = dates[-1]
        for ticker in list(active.keys()):
            if ticker in close.columns:
                ep = close.loc[last_date, ticker]
                cash = self._close_position(active, ticker, ep, "EOB", last_date, cash)

        daily_returns = equity_curve.pct_change().fillna(0.0)

        return {
            "equity_curve": equity_curve,
            "daily_returns": daily_returns,
            "trade_log": self._build_trade_log(),
            "positions": positions_snap,
            "nifty_close": nifty_close,
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
        Close an open position, update cash, record the trade.

        Returns the updated cash balance.
        """
        pos = active.pop(ticker)
        cfg = self.cfg

        if pos["direction"] == "LONG":
            gross_pnl = (exit_price - pos["entry_price"]) * pos["shares"]
            exit_cost = pos["shares"] * exit_price * cfg.sell_cost_pct + cfg.dp_charge_per_sell
            # Return proceeds net of sell cost + DP charge
            cash += pos["shares"] * exit_price - exit_cost
        else:  # SHORT
            gross_pnl = (pos["entry_price"] - exit_price) * pos["shares"]
            exit_cost = pos["shares"] * exit_price * cfg.buy_cost_pct  # buy-to-cover
            # Return margin + realised P&L - buy-to-cover cost
            cash += pos["notional"] + gross_pnl - exit_cost

        net_pnl = gross_pnl - pos["entry_cost"] - exit_cost

        self._trades.append(
            _Trade(
                ticker=ticker,
                direction=pos["direction"],
                entry_date=pos["entry_date"],
                exit_date=exit_date,
                entry_price=round(pos["entry_price"], 4),
                exit_price=round(exit_price, 4),
                shares=round(pos["shares"], 6),
                notional=round(pos["notional"], 2),
                gross_pnl=round(gross_pnl, 2),
                entry_cost=round(pos["entry_cost"], 2),
                exit_cost=round(exit_cost, 2),
                net_pnl=round(net_pnl, 2),
                exit_reason=reason,
            )
        )
        return cash

    @staticmethod
    def _compute_nav(
        cash: float,
        active: dict,
        close: pd.DataFrame,
        date: pd.Timestamp,
    ) -> float:
        """
        Mark-to-market portfolio NAV.

        NAV = cash
            + Σ (LONG)  shares × current_close
            + Σ (SHORT) notional + (entry_price − current_close) × shares
        """
        nav = cash
        for ticker, pos in active.items():
            if ticker not in close.columns:
                nav += pos["notional"]  # can't mark → use notional as proxy
                continue
            cp = close.loc[date, ticker]
            if pos["direction"] == "LONG":
                nav += pos["shares"] * cp
            else:
                nav += pos["notional"] + (pos["entry_price"] - cp) * pos["shares"]
        return nav

    def _build_trade_log(self) -> pd.DataFrame:
        """Convert the internal ``_Trade`` list to a clean DataFrame."""
        if not self._trades:
            return pd.DataFrame(columns=[
                "ticker", "direction", "entry_date", "exit_date",
                "entry_price", "exit_price", "shares", "notional",
                "gross_pnl", "entry_cost", "exit_cost", "net_pnl",
                "pnl_pct", "holding_days", "exit_reason",
            ])

        rows = []
        for t in self._trades:
            hold = (t.exit_date - t.entry_date).days if t.exit_date else None
            pnl_pct = (t.net_pnl / t.notional * 100.0) if t.notional else 0.0
            rows.append({
                "ticker":        t.ticker,
                "direction":     t.direction,
                "entry_date":    t.entry_date,
                "exit_date":     t.exit_date,
                "entry_price":   t.entry_price,
                "exit_price":    t.exit_price,
                "shares":        t.shares,
                "notional":      t.notional,
                "gross_pnl":     t.gross_pnl,
                "entry_cost":    t.entry_cost,
                "exit_cost":     t.exit_cost,
                "net_pnl":       t.net_pnl,
                "pnl_pct":       round(pnl_pct, 3),
                "holding_days":  hold,
                "exit_reason":   t.exit_reason,
            })

        df = pd.DataFrame(rows)
        df["entry_date"] = pd.to_datetime(df["entry_date"])
        df["exit_date"]  = pd.to_datetime(df["exit_date"])
        return df.sort_values("entry_date").reset_index(drop=True)
