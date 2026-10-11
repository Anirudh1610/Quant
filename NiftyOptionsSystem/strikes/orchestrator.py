"""
strikes/orchestrator.py
------------------------
Ties strike selection (strikes/selection.py) and spread construction
(strikes/spreads.py) together: given a live option chain snapshot, a
market-implied forward, and a target strategy type (from
signals/regime.py's RegimeSnapshot.recommended_strategy_type), returns a
fully-defined SpreadDefinition — or None if no valid spread could be built
(a leg failed selection, or the required strikes aren't listed).

Does not place orders. Does not require a live Kite client unless
`compute_margin=True` (margin needs a live basket_order_margins() call —
see strikes/margin.py).
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from config import STRIKE_SELECTION, StrikeSelectionConfig
from core.kite_client import KiteRateLimited
from signals.regime import BEAR_CALL_SPREAD, BULL_PUT_SPREAD, IRON_CONDOR
from strikes.margin import get_live_margin
from strikes.selection import compute_chain_greeks, select_long_strike, select_short_strike
from strikes.spreads import SpreadDefinition, build_bear_call_spread, build_bull_put_spread, build_iron_condor


def _build_put_leg_pair(greeked: pd.DataFrame, forward: float, T: float, strike_step: float,
                         cfg: StrikeSelectionConfig) -> tuple[dict, dict] | None:
    short_put = select_short_strike(greeked, "PE", forward, T, strike_step, cfg)
    if short_put is None:
        return None
    long_put = select_long_strike(greeked, "PE", short_put["strike"], strike_step, cfg)
    if long_put is None:
        return None
    return short_put, long_put


def _build_call_leg_pair(greeked: pd.DataFrame, forward: float, T: float, strike_step: float,
                          cfg: StrikeSelectionConfig) -> tuple[dict, dict] | None:
    short_call = select_short_strike(greeked, "CE", forward, T, strike_step, cfg)
    if short_call is None:
        return None
    long_call = select_long_strike(greeked, "CE", short_call["strike"], strike_step, cfg)
    if long_call is None:
        return None
    return short_call, long_call


def select_spread(chain: pd.DataFrame, strategy_type: str, forward: float, T: float, r: float,
                   instrument: str, expiry: date, strike_step: float, lot_size: int, n_lots: int = 1,
                   cfg: StrikeSelectionConfig = STRIKE_SELECTION,
                   client: KiteRateLimited | None = None, compute_margin: bool = False,
                   ) -> SpreadDefinition | None:
    """
    Returns a SpreadDefinition for `strategy_type`, or None if selection
    failed for any leg (e.g. nothing survived the probability-of-touch /
    OI-wall filters, or a required long-leg strike isn't listed).

    Pass a live `client` with `compute_margin=True` to also populate
    margin_estimate via Kite's real basket margin calculator
    (strikes/margin.py) — otherwise margin_estimate is left None; the
    caller can call strikes.margin.estimate_margin_fallback(spread) for the
    documented crude approximation instead (e.g. in Phase 4 backtesting,
    where there's no live client to query).
    """
    greeked = compute_chain_greeks(chain, forward, T, r)
    expiry_s = expiry.isoformat()

    spread: SpreadDefinition | None = None

    if strategy_type == BULL_PUT_SPREAD:
        pair = _build_put_leg_pair(greeked, forward, T, strike_step, cfg)
        if pair is not None:
            spread = build_bull_put_spread(*pair, instrument, expiry_s, lot_size, n_lots)

    elif strategy_type == BEAR_CALL_SPREAD:
        pair = _build_call_leg_pair(greeked, forward, T, strike_step, cfg)
        if pair is not None:
            spread = build_bear_call_spread(*pair, instrument, expiry_s, lot_size, n_lots)

    elif strategy_type == IRON_CONDOR:
        put_pair = _build_put_leg_pair(greeked, forward, T, strike_step, cfg)
        call_pair = _build_call_leg_pair(greeked, forward, T, strike_step, cfg)
        if put_pair is not None and call_pair is not None:
            short_put, long_put = put_pair
            short_call, long_call = call_pair
            spread = build_iron_condor(short_put, long_put, short_call, long_call,
                                        instrument, expiry_s, lot_size, n_lots)

    else:
        raise ValueError(f"select_spread does not handle strategy_type={strategy_type!r} "
                          f"(NO_TRADE should be handled by the caller before reaching here)")

    if spread is not None and compute_margin:
        if client is None:
            raise ValueError("compute_margin=True requires a live `client`")
        get_live_margin(client, instrument, expiry, spread)

    return spread
