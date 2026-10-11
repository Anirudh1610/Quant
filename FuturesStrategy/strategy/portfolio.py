"""
portfolio.py
------------
Position sizing, margin management, and risk controls for the NSE Futures strategy.

Sizing methodology
------------------
Volatility-targeted sizing based on ATR:

    risk_per_trade   = capital × risk_fraction      (e.g. 1% of NAV per trade)
    atr_stop_points  = atr_multiplier × ATR          (price move to stop)
    raw_lots         = risk_per_trade / (atr_stop_points × lot_size)
    num_lots         = floor(raw_lots)               (must be a whole number)

Subject to hard constraints:
    1. Margin constraint: num_lots × lot_size × price × margin_rate ≤ available_margin
    2. Gross exposure cap: total notional ≤ max_gross_multiple × capital
    3. Concentration cap: single position notional ≤ max_position_pct × capital
    4. VIX scalar: scale num_lots down linearly with India VIX level
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .universe import get_margin_rate, is_index_future


# ──────────────────────────────────────────────────────────────────────────────
# ATR STOP LEVELS
# ──────────────────────────────────────────────────────────────────────────────

def compute_atr_stop(
    entry_price: float,
    atr_value: float,
    direction: int,
    atr_multiplier: float = 2.0,
) -> float:
    """
    Compute the ATR-based stop-loss price.

    stop = entry_price − atr_multiplier × ATR   (for LONG)
    stop = entry_price + atr_multiplier × ATR   (for SHORT)

    Parameters
    ----------
    entry_price     : Futures fill price.
    atr_value       : Current ATR for the contract.
    direction       : +1 (long) or -1 (short).
    atr_multiplier  : Distance to stop in ATR units. Default 2.0.
    """
    return entry_price - direction * atr_multiplier * atr_value


# ──────────────────────────────────────────────────────────────────────────────
# VOLATILITY-TARGETED LOT SIZING
# ──────────────────────────────────────────────────────────────────────────────

def volatility_targeted_lots(
    nav: float,
    futures_price: float,
    lot_size: int,
    atr: float,
    risk_fraction: float = 0.01,
    atr_multiplier: float = 2.0,
    min_lots: int = 1,
    max_lots: int = 50,
) -> int:
    """
    Calculate the number of lots such that an ATR-stop loss equals
    ``risk_fraction`` of current NAV.

    Derivation
    ----------
    risk_budget      = nav × risk_fraction
    stop_distance    = atr_multiplier × atr            (per share/index point)
    pnl_per_lot      = stop_distance × lot_size
    raw_lots         = risk_budget / pnl_per_lot
    num_lots         = clamp(floor(raw_lots), min_lots, max_lots)

    Parameters
    ----------
    nav            : Current portfolio NAV (₹).
    futures_price  : Current futures price.
    lot_size       : Number of shares / index units per lot.
    atr            : Current ATR value for the contract.
    risk_fraction  : Max loss per trade as a fraction of NAV. Default 1%.
    atr_multiplier : ATR multiplier for the stop. Default 2.0.
    min_lots, max_lots : Hard bounds on lot count.

    Returns
    -------
    int  Number of lots (≥ min_lots).
    """
    if atr <= 0 or lot_size <= 0 or futures_price <= 0:
        return min_lots

    risk_budget   = nav * risk_fraction
    pnl_per_lot   = atr_multiplier * atr * lot_size
    if pnl_per_lot <= 0:
        return min_lots

    raw_lots = risk_budget / pnl_per_lot
    return int(np.clip(np.floor(raw_lots), min_lots, max_lots))


# ──────────────────────────────────────────────────────────────────────────────
# MARGIN CONSTRAINT
# ──────────────────────────────────────────────────────────────────────────────

def margin_constrained_lots(
    num_lots: int,
    futures_price: float,
    lot_size: int,
    ticker: str,
    available_margin: float,
) -> int:
    """
    Reduce num_lots so that the margin required does not exceed available_margin.

    margin_per_lot = lot_size × futures_price × margin_rate
    max_lots       = floor(available_margin / margin_per_lot)
    """
    margin_rate = get_margin_rate(ticker)
    margin_per_lot = lot_size * futures_price * margin_rate
    if margin_per_lot <= 0:
        return 0
    max_by_margin = int(np.floor(available_margin / margin_per_lot))
    return max(0, min(num_lots, max_by_margin))


# ──────────────────────────────────────────────────────────────────────────────
# GROSS EXPOSURE SCALING
# ──────────────────────────────────────────────────────────────────────────────

def gross_exposure_cap_lots(
    num_lots: int,
    futures_price: float,
    lot_size: int,
    current_gross_notional: float,
    nav: float,
    max_gross_multiple: float = 3.0,
) -> int:
    """
    Reduce num_lots so that total gross notional stays within max_gross_multiple × NAV.

    max_additional_notional = max_gross_multiple × nav − current_gross_notional
    max_lots = floor(max_additional_notional / (lot_size × futures_price))
    """
    max_additional = max(0.0, max_gross_multiple * nav - current_gross_notional)
    notional_per_lot = lot_size * futures_price
    if notional_per_lot <= 0:
        return 0
    max_by_exposure = int(np.floor(max_additional / notional_per_lot))
    return max(0, min(num_lots, max_by_exposure))


# ──────────────────────────────────────────────────────────────────────────────
# CONCENTRATION LIMIT
# ──────────────────────────────────────────────────────────────────────────────

def concentration_limited_lots(
    num_lots: int,
    futures_price: float,
    lot_size: int,
    nav: float,
    max_position_pct: float = 0.20,
) -> int:
    """
    Reduce num_lots so that a single position's notional ≤ max_position_pct × NAV.
    Prevents any single contract from dominating the portfolio.
    """
    max_notional  = nav * max_position_pct
    notional_per_lot = lot_size * futures_price
    if notional_per_lot <= 0:
        return 0
    max_lots = int(np.floor(max_notional / notional_per_lot))
    return max(0, min(num_lots, max_lots))


# ──────────────────────────────────────────────────────────────────────────────
# FULL SIZING PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def compute_position_size(
    ticker: str,
    direction: int,
    nav: float,
    futures_price: float,
    atr: float,
    current_gross_notional: float,
    available_margin: float,
    vix_scalar: float = 1.0,
    risk_fraction: float = 0.01,
    atr_multiplier: float = 2.0,
    max_gross_multiple: float = 3.0,
    max_position_pct: float = 0.20,
    min_lots: int = 1,
    max_lots: int = 50,
) -> tuple[int, float]:
    """
    Full position sizing pipeline applying all constraints in order.

    Order of constraints (most binding wins)
    -----------------------------------------
    1. Volatility-targeted baseline (ATR stop = risk_fraction × NAV)
    2. Scale down by India VIX scalar
    3. Concentration limit (max_position_pct × NAV per position)
    4. Gross exposure cap (max_gross_multiple × NAV total)
    5. Available margin cap

    Parameters
    ----------
    ticker                 : Futures ticker (used to look up margin rate / lot size).
    direction              : +1 long, -1 short.
    nav                    : Current portfolio NAV (₹).
    futures_price          : Current futures close price.
    atr                    : Current ATR for the contract.
    current_gross_notional : Sum of |notional| of all currently open positions.
    available_margin       : Free cash available to post as new margin.
    vix_scalar             : Position scalar from india_vix [0, 1].
    risk_fraction          : Target risk per trade as fraction of NAV.
    atr_multiplier         : ATR stop distance multiplier.
    max_gross_multiple     : Maximum gross notional / NAV.
    max_position_pct       : Maximum single-position notional / NAV.
    min_lots, max_lots     : Hard lot bounds.

    Returns
    -------
    (num_lots, stop_price)
    """
    from .universe import LOT_SIZES
    lot_size = LOT_SIZES.get(ticker, 1)

    # 1. Baseline vol-targeted size
    lots = volatility_targeted_lots(
        nav, futures_price, lot_size, atr,
        risk_fraction, atr_multiplier, min_lots, max_lots,
    )

    # 2. VIX scaling
    lots = max(0, int(np.floor(lots * vix_scalar)))

    # 3. Concentration cap
    lots = concentration_limited_lots(lots, futures_price, lot_size, nav, max_position_pct)

    # 4. Gross exposure cap
    lots = gross_exposure_cap_lots(
        lots, futures_price, lot_size, current_gross_notional, nav, max_gross_multiple,
    )

    # 5. Margin constraint
    lots = margin_constrained_lots(lots, futures_price, lot_size, ticker, available_margin)

    stop_price = compute_atr_stop(futures_price, atr, direction, atr_multiplier)
    return lots, stop_price


# ──────────────────────────────────────────────────────────────────────────────
# PORTFOLIO EXPOSURE SNAPSHOT
# ──────────────────────────────────────────────────────────────────────────────

def compute_exposure(
    active_positions: dict,
    futures_close_row: pd.Series,
) -> dict[str, float]:
    """
    Compute gross / net notional exposure and margin used.

    Parameters
    ----------
    active_positions : dict ticker → position dict (as maintained by the engine).
    futures_close_row : today's futures close prices (pd.Series, index = tickers).

    Returns
    -------
    dict with keys: gross_notional, net_notional, long_notional, short_notional,
                    margin_used, num_long, num_short.
    """
    from .universe import LOT_SIZES
    long_notional  = 0.0
    short_notional = 0.0
    margin_used    = 0.0
    num_long       = 0
    num_short      = 0

    for ticker, pos in active_positions.items():
        price = futures_close_row.get(ticker, pos["last_price"])
        lot_size   = LOT_SIZES.get(ticker, 1)
        notional   = abs(pos["num_lots"]) * lot_size * price
        margin_req = notional * get_margin_rate(ticker)

        if pos["direction"] == 1:
            long_notional += notional
            num_long += 1
        else:
            short_notional += notional
            num_short += 1
        margin_used += margin_req

    return {
        "gross_notional":  long_notional + short_notional,
        "net_notional":    long_notional - short_notional,
        "long_notional":   long_notional,
        "short_notional":  short_notional,
        "margin_used":     margin_used,
        "num_long":        num_long,
        "num_short":       num_short,
    }


# ──────────────────────────────────────────────────────────────────────────────
# SECTOR CONCENTRATION CAP
# ──────────────────────────────────────────────────────────────────────────────

def sector_cap_lots(
    ticker: str,
    num_lots: int,
    futures_price: float,
    lot_size: int,
    nav: float,
    active_positions: dict,
    futures_close_row: pd.Series,
    max_sector_pct: float = 0.30,
) -> int:
    """
    Reduce ``num_lots`` so that total notional in the same sector stays within
    ``max_sector_pct`` × NAV.

    Prevents crowding in a single sector (e.g. all 5 longs in Financials).

    Parameters
    ----------
    ticker             : Ticker being added.
    num_lots           : Proposed lot count (post all other constraints).
    futures_price      : Current futures price for ``ticker``.
    lot_size           : Lot size for ``ticker``.
    nav                : Current portfolio NAV.
    active_positions   : Current active position dict from the engine.
    futures_close_row  : Today's futures close prices (pd.Series, index = tickers).
    max_sector_pct     : Maximum sector notional / NAV. Default 30%.
    """
    from .universe import SECTOR_MAP, LOT_SIZES
    sector = SECTOR_MAP.get(ticker, "Other")

    # Compute existing exposure in this sector
    sector_notional = 0.0
    for t, pos in active_positions.items():
        if SECTOR_MAP.get(t, "Other") == sector:
            price = float(futures_close_row.get(t, pos["last_price"]))
            t_lot = LOT_SIZES.get(t, 1)
            sector_notional += abs(pos["num_lots"]) * t_lot * price

    max_sector_notional = nav * max_sector_pct
    remaining = max(0.0, max_sector_notional - sector_notional)
    notional_per_lot = lot_size * futures_price
    if notional_per_lot <= 0:
        return 0
    max_by_sector = int(np.floor(remaining / notional_per_lot))
    return max(0, min(num_lots, max_by_sector))


# ──────────────────────────────────────────────────────────────────────────────
# BETA-NEUTRAL SCALING
# ──────────────────────────────────────────────────────────────────────────────

def beta_neutral_scale(
    direction: int,
    long_notional: float,
    short_notional: float,
    max_imbalance: float = 1.5,
) -> float:
    """
    Return a scalar [0, 1] that prevents gross long/short notional from
    becoming too one-sided (simplified beta-neutrality proxy).

    When the proposed direction would push the ratio beyond ``max_imbalance``,
    the scalar is reduced proportionally.  For a fully hedged L/S book, target
    is 1:1 notional; imbalances up to max_imbalance are allowed.

    Parameters
    ----------
    direction      : +1 for a new long, -1 for a new short.
    long_notional  : Current total long notional (before this trade).
    short_notional : Current total short notional (before this trade).
    max_imbalance  : Maximum allowed ratio of larger side / smaller side.
                     Default 1.5 (longs can be at most 1.5× shorts, or vice versa).

    Returns
    -------
    float scalar in [0, 1].  Multiply by proposed lots and take floor.
    """
    if long_notional <= 0 and short_notional <= 0:
        return 1.0

    if direction == 1:   # adding long → check long / short ratio
        larger  = long_notional
        smaller = short_notional if short_notional > 0 else long_notional / max_imbalance
    else:                # adding short → check short / long ratio
        larger  = short_notional
        smaller = long_notional if long_notional > 0 else short_notional / max_imbalance

    if smaller <= 0 or (larger / smaller) < max_imbalance:
        return 1.0
    # Scale so the resulting ratio = max_imbalance
    return min(1.0, max(0.0, smaller * max_imbalance / (larger + 1.0)))
