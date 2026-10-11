"""
strikes/margin.py
------------------
SPAN+exposure margin estimate for a candidate spread.

Primary path (get_live_margin): kite.basket_order_margins() — Kite
Connect's real margin calculator, returning the exchange's actual
SPAN+exposure requirement for a hypothetical multi-leg order without
placing it. Authoritative; nothing here should second-guess it.

Verified live against a real NIFTY bull put spread (24600 short PE / 24400
long PE, lot 75): the response has both an `initial` and a `final` total —
`initial.total` (₹203,989.75 in that test) prices each leg independently
with NO hedge benefit, while `final.total` (₹41,008) is the real
basket-level margin after the exchange recognizes the long leg caps risk.
Using `initial` would overstate margin ~5x for a defined-risk spread — this
module always reads `final`.

Fallback path (estimate_margin_fallback): used when no live Kite session is
available (e.g. Phase 4 backtesting over historical dates, where there's no
live margin calculator to query). Crude — NOT SPAN — a rough sanity bound,
not a trading input, until validated against a range of real
get_live_margin() figures.
"""

from __future__ import annotations

from datetime import date

from core.kite_client import KiteRateLimited
from data.instruments import get_option_chain_instruments
from strikes.spreads import SpreadDefinition


def get_live_margin(client: KiteRateLimited, instrument: str, expiry: date,
                     spread: SpreadDefinition) -> dict:
    """
    Queries kite.basket_order_margins() for the spread's legs. Mutates
    `spread` in place (sets margin_estimate / margin_source) and returns
    the raw Kite response for anyone who wants the full span/exposure/
    charges breakdown.
    """
    contracts = get_option_chain_instruments(client, instrument, expiry)
    symbol_map = {
        (row["strike"], row["instrument_type"]): row["tradingsymbol"]
        for _, row in contracts.iterrows()
    }

    orders = []
    for leg in spread.legs:
        tradingsymbol = symbol_map.get((leg.strike, leg.option_type))
        if tradingsymbol is None:
            raise ValueError(f"Could not resolve tradingsymbol for {leg.option_type} {leg.strike}")
        orders.append({
            "exchange": "NFO",
            "tradingsymbol": tradingsymbol,
            "transaction_type": leg.action,
            "variety": "regular",
            "product": "NRML",
            "order_type": "MARKET",
            "quantity": spread.lot_size * spread.n_lots,
        })

    result = client.call("basket_order_margins", orders)
    spread.margin_estimate = result["final"]["total"]
    spread.margin_source = "kite_basket_margin"
    return result


def estimate_margin_fallback(spread: SpreadDefinition, multiple: float = 1.0) -> float:
    """
    Crude fallback when no live margin calculator is available: NSE index
    option credit spreads are margined close to (often somewhat above) the
    defined max loss, since the long leg caps risk — unlike a naked short,
    which needs full SPAN+exposure on the uncapped side. `multiple` lets
    you pad this (default 1.0x max loss).

    NOT SPAN — validate against real get_live_margin() figures before
    trusting this for anything but a rough backtest sanity bound.
    """
    estimate = spread.max_loss * multiple
    spread.margin_estimate = estimate
    spread.margin_source = "fallback_max_loss_multiple"
    return estimate
