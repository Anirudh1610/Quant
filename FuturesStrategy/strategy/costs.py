"""
costs.py
--------
Zerodha NSE Futures & Options transaction cost model.

All costs are applied on TRADED NOTIONAL (num_lots × lot_size × price),
NOT on the margin deposited.  This is the correct basis per SEBI/NSE rules.

Zerodha Futures charges (as of 2024)
--------------------------------------
Brokerage       : ₹20 flat per executed order  (NOT percentage-based)
STT             : 0.0125% on sell side only    (futures, unlike 0.1% for CNC equity)
NSE exchange    : 0.00173% per side
SEBI charges    : ₹10 / crore  = 0.0001% per side
GST             : 18% on (brokerage + exchange + SEBI)
Stamp duty      : 0.002% on buy side only
Slippage        : 0.02% for index futures (NIFTY, BANKNIFTY)
                  0.03% for stock futures

Sources
-------
https://zerodha.com/charges/
NSE circular on STT for F&O contracts
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# PERCENTAGE-BASED COMPONENTS  (applied to traded notional)
# ──────────────────────────────────────────────────────────────────────────────

_STT_FUTURES_SELL  = 0.000125   # 0.0125%  sell side only for futures
_NSE_EXCHANGE      = 0.0000173  # 0.00173% per side
_SEBI              = 0.000001   # ₹10/crore ≈ 0.0001%
_STAMP_DUTY_BUY    = 0.00002    # 0.002%   buy side only
_GST_RATE          = 0.18       # 18% on brokerage + exchange + SEBI

# ── Slippage  ─────────────────────────────────────────────────────────────────
# NIFTY 50 index futures are among the most liquid contracts in Asia;
# stock futures have wider spreads
SLIPPAGE_INDEX = 0.0002    # 0.02% for NIFTY / BANKNIFTY
SLIPPAGE_STOCK = 0.0003    # 0.03% for single-stock futures

# ── Flat brokerage ────────────────────────────────────────────────────────────
BROKERAGE_PER_ORDER: float = 20.0   # ₹20 per order (Zerodha F&O flat fee)

# ──────────────────────────────────────────────────────────────────────────────
# ROLLOVER COST MULTIPLIER
# Rolling near-month → far-month = one round-trip (close + open)
# ──────────────────────────────────────────────────────────────────────────────

def compute_futures_cost(
    notional: float,
    direction: str,
    is_index: bool = False,
) -> dict[str, float]:
    """
    Compute all-in transaction cost for a single futures order.

    Parameters
    ----------
    notional  : Traded notional value (num_lots × lot_size × price).
    direction : "BUY" (entry long / exit short) or "SELL" (entry short / exit long).
    is_index  : True for NIFTY / BANKNIFTY; False for stock futures.

    Returns
    -------
    dict with keys brokerage, stt, exchange, sebi, gst, stamp, slippage, total.
    """
    brokerage = BROKERAGE_PER_ORDER

    stt      = notional * _STT_FUTURES_SELL if direction == "SELL" else 0.0
    exchange = notional * _NSE_EXCHANGE
    sebi     = notional * _SEBI
    gst      = (brokerage + exchange + sebi) * _GST_RATE
    stamp    = notional * _STAMP_DUTY_BUY  if direction == "BUY"  else 0.0
    slippage = notional * (SLIPPAGE_INDEX if is_index else SLIPPAGE_STOCK)

    total = brokerage + stt + exchange + sebi + gst + stamp + slippage
    return dict(
        brokerage=round(brokerage, 4),
        stt=round(stt, 4),
        exchange=round(exchange, 4),
        sebi=round(sebi, 4),
        gst=round(gst, 4),
        stamp=round(stamp, 4),
        slippage=round(slippage, 4),
        total=round(total, 4),
    )


def round_trip_cost_pct(is_index: bool = False) -> float:
    """
    Approximate all-in round-trip cost as a fraction of notional.

    Useful for quick break-even analysis without specifying exact notional.
    (Excludes flat brokerage — call compute_futures_cost for exact numbers.)
    """
    buy_pct  = _NSE_EXCHANGE + _SEBI + _STAMP_DUTY_BUY
    sell_pct = _STT_FUTURES_SELL + _NSE_EXCHANGE + _SEBI
    slip_pct = 2 * (SLIPPAGE_INDEX if is_index else SLIPPAGE_STOCK)
    gst_pct  = (buy_pct + sell_pct) * _GST_RATE
    return buy_pct + sell_pct + slip_pct + gst_pct


def compute_rollover_cost(
    notional: float,
    is_index: bool = False,
) -> float:
    """
    Approximate cost of rolling a futures position (close near-month + open far-month).

    The roll = one full round-trip cost (two brokerage orders + STT on sell + stamp on buy).
    """
    sell_cost = compute_futures_cost(notional, "SELL", is_index)["total"]
    buy_cost  = compute_futures_cost(notional, "BUY",  is_index)["total"]
    return sell_cost + buy_cost
