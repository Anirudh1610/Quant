"""
strikes/spreads.py
-------------------
Builds a full spread definition (legs, theoretical credit/debit, max
profit/loss, breakevens) from selected strikes and an option chain
snapshot. Each leg's theoretical fill price is the mid (bid/ask average),
falling back to LTP — the same convention data/option_chain.py already
uses for IV computation. Phase 4's backtester is where a configurable
slippage/bid-ask assumption on top of this theoretical fill belongs (per
the original spec) — not modeled here.

Sign convention: net_credit / max_profit / max_loss are all expressed as
positive-is-good-for-the-seller (credit received is positive, max_loss is
a positive number representing rupees at risk).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Leg:
    option_type: str   # "CE" | "PE"
    strike: float
    action: str          # "SELL" | "BUY"
    price: float          # theoretical fill price (mid, or LTP fallback)
    delta: float | None = None
    iv: float | None = None

    def to_dict(self) -> dict:
        return {"option_type": self.option_type, "strike": self.strike, "action": self.action,
                "price": self.price, "delta": self.delta, "iv": self.iv}


@dataclass
class SpreadDefinition:
    strategy_type: str
    instrument: str
    expiry: str
    legs: list[Leg]
    lot_size: int
    n_lots: int
    net_credit: float       # per share/point; positive = credit received
    max_profit: float        # rupees
    max_loss: float           # rupees
    breakevens: list[float]
    margin_estimate: float | None = None
    margin_source: str | None = None  # "kite_basket_margin" | "fallback_max_loss_multiple"

    def to_dict(self) -> dict:
        return {
            "strategy_type": self.strategy_type,
            "instrument": self.instrument,
            "expiry": self.expiry,
            "legs": [leg.to_dict() for leg in self.legs],
            "lot_size": self.lot_size,
            "n_lots": self.n_lots,
            "net_credit": self.net_credit,
            "max_profit": self.max_profit,
            "max_loss": self.max_loss,
            "breakevens": self.breakevens,
            "margin_estimate": self.margin_estimate,
            "margin_source": self.margin_source,
        }


def _leg_price(row: dict) -> float:
    bid, ask = row.get("bid"), row.get("ask")
    if bid and ask and bid > 0 and ask > 0:
        return (bid + ask) / 2
    return row["ltp"]


def _make_leg(row: dict, action: str) -> Leg:
    return Leg(option_type=row["option_type"], strike=row["strike"], action=action,
               price=_leg_price(row), delta=row.get("delta"), iv=row.get("iv"))


def build_bull_put_spread(short_put: dict, long_put: dict, instrument: str, expiry: str,
                           lot_size: int, n_lots: int = 1) -> SpreadDefinition:
    """short_put/long_put: row dicts from strikes/selection.py (strike, price fields, etc.)."""
    if long_put["strike"] >= short_put["strike"]:
        raise ValueError("Bull put spread requires long_put strike below short_put strike")

    short_leg = _make_leg(short_put, "SELL")
    long_leg = _make_leg(long_put, "BUY")
    net_credit = short_leg.price - long_leg.price
    width = short_put["strike"] - long_put["strike"]

    max_profit = net_credit * lot_size * n_lots
    max_loss = (width - net_credit) * lot_size * n_lots
    breakeven = short_put["strike"] - net_credit

    return SpreadDefinition("BULL_PUT_SPREAD", instrument, expiry, [short_leg, long_leg],
                             lot_size, n_lots, net_credit, max_profit, max_loss, [breakeven])


def build_bear_call_spread(short_call: dict, long_call: dict, instrument: str, expiry: str,
                            lot_size: int, n_lots: int = 1) -> SpreadDefinition:
    if long_call["strike"] <= short_call["strike"]:
        raise ValueError("Bear call spread requires long_call strike above short_call strike")

    short_leg = _make_leg(short_call, "SELL")
    long_leg = _make_leg(long_call, "BUY")
    net_credit = short_leg.price - long_leg.price
    width = long_call["strike"] - short_call["strike"]

    max_profit = net_credit * lot_size * n_lots
    max_loss = (width - net_credit) * lot_size * n_lots
    breakeven = short_call["strike"] + net_credit

    return SpreadDefinition("BEAR_CALL_SPREAD", instrument, expiry, [short_leg, long_leg],
                             lot_size, n_lots, net_credit, max_profit, max_loss, [breakeven])


def build_iron_condor(short_put: dict, long_put: dict, short_call: dict, long_call: dict,
                       instrument: str, expiry: str, lot_size: int, n_lots: int = 1) -> SpreadDefinition:
    put_side = build_bull_put_spread(short_put, long_put, instrument, expiry, lot_size, n_lots)
    call_side = build_bear_call_spread(short_call, long_call, instrument, expiry, lot_size, n_lots)

    net_credit = put_side.net_credit + call_side.net_credit
    put_width = short_put["strike"] - long_put["strike"]
    call_width = long_call["strike"] - short_call["strike"]

    # only one side can be breached at expiry, so the wider side sets the
    # worst case; the combined credit cushions whichever side gets tested
    max_profit = net_credit * lot_size * n_lots
    max_loss = (max(put_width, call_width) - net_credit) * lot_size * n_lots
    breakevens = sorted([short_put["strike"] - net_credit, short_call["strike"] + net_credit])

    legs = put_side.legs + call_side.legs
    return SpreadDefinition("IRON_CONDOR", instrument, expiry, legs, lot_size, n_lots,
                             net_credit, max_profit, max_loss, breakevens)
