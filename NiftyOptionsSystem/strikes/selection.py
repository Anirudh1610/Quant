"""
strikes/selection.py
---------------------
Strike selection for credit spreads: Greeks computation, delta targeting,
probability-of-touch filtering, and OI-wall avoidance, given a live option
chain snapshot (data/option_chain.py columns: strike, option_type, ltp,
bid, ask, oi, volume, iv, underlying_ltp).

Nothing here places orders — it only ranks/filters candidate strikes.
Phase 5's manual-confirmation gate is what turns a chosen strike into an
order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import STRIKE_SELECTION, StrikeSelectionConfig
from core.pricing import bs_greeks
from core.probability import probability_of_touch


def compute_chain_greeks(chain: pd.DataFrame, forward: float, T: float, r: float) -> pd.DataFrame:
    """
    Adds delta/gamma/theta/vega columns, computed Black-76 style (S=forward,
    q=r) — consistent with how IV itself was computed in
    data/option_chain.py (see core/pricing.py's module docstring). Rows
    with missing/None `iv` get NaN Greeks rather than being dropped, so
    callers can decide how to handle them.
    """
    out = chain.copy()

    def _row_greeks(row: pd.Series) -> pd.Series:
        if pd.isna(row["iv"]):
            return pd.Series({"delta": np.nan, "gamma": np.nan, "theta": np.nan, "vega": np.nan})
        bs_type = "call" if row["option_type"] == "CE" else "put"
        g = bs_greeks(forward, row["strike"], T, r, row["iv"], q=r, option_type=bs_type)
        return pd.Series(g)

    greeks = out.apply(_row_greeks, axis=1)
    return pd.concat([out, greeks], axis=1)


def find_oi_walls(chain: pd.DataFrame, option_type: str,
                   multiple: float = STRIKE_SELECTION.oi_wall_multiple,
                   local_window: int = STRIKE_SELECTION.oi_wall_local_window) -> set[float]:
    """
    Strikes (of a given option_type) whose OI exceeds `multiple` x the
    LOCAL median OI of the `local_window` nearest strikes on each side —
    flagged as "walls" per the Phase 3 spec: avoid selling strikes inside
    major OI concentration zones against your position.

    Deliberately uses a local reference, not the whole chain's median: OI
    decays sharply away from spot in a real chain, so comparing against the
    whole-chain median flags nearly the entire liquid near-ATM region as a
    "wall" — verified live: a 3x whole-chain-median threshold tripped 42 of
    105 NIFTY PE strikes, including every strike within 450 points of spot,
    making the filter block the only strikes worth selling. A local window
    instead picks out genuine round-strike OI concentration spikes (e.g.
    22500/23000/25000 stood out clearly at 5-20x their local neighborhood
    in that same live snapshot, while near-ATM strikes scored 0.3-3.1x)
    without blanket-blocking the tradeable region.
    """
    side = chain[chain["option_type"] == option_type].sort_values("strike").reset_index(drop=True)
    if side.empty or side["oi"].isna().all():
        return set()
    oi = side["oi"].fillna(0.0)
    span = 2 * local_window + 1
    local_median = oi.rolling(span, center=True, min_periods=local_window + 1).median()
    is_wall = (local_median > 0) & (oi > multiple * local_median)
    return set(side.loc[is_wall, "strike"].tolist())


def expand_wall_band(walls: set[float], strike_step: float,
                      band: int = STRIKE_SELECTION.oi_wall_band_strikes) -> set[float]:
    """Strikes within `band` strike-steps of any wall, inclusive of the wall itself."""
    expanded: set[float] = set()
    for w in walls:
        for i in range(-band, band + 1):
            expanded.add(w + i * strike_step)
    return expanded


def select_short_strike(chain_with_greeks: pd.DataFrame, option_type: str, forward: float, T: float,
                         strike_step: float, cfg: StrikeSelectionConfig = STRIKE_SELECTION) -> dict | None:
    """
    Picks the OTM strike of `option_type` ("CE"/"PE") whose |delta| is
    closest to `cfg.target_delta`, after filtering out:
      - strikes with probability-of-touch > cfg.max_probability_of_touch
      - strikes sitting inside/adjacent to an OI wall (cfg.oi_wall_multiple /
        cfg.oi_wall_band_strikes)

    Returns None if no strike survives filtering (caller should treat this
    as "can't build this spread right now", not silently skip a leg).
    """
    side = chain_with_greeks[chain_with_greeks["option_type"] == option_type].copy()
    side = side[side["strike"] > forward] if option_type == "CE" else side[side["strike"] < forward]
    side = side.dropna(subset=["delta", "iv"])
    if side.empty:
        return None

    side["prob_touch"] = side.apply(
        lambda r: probability_of_touch(forward, r["strike"], T, r["iv"]), axis=1
    )
    side = side[side["prob_touch"] <= cfg.max_probability_of_touch]
    if side.empty:
        return None

    walls = expand_wall_band(
        find_oi_walls(chain_with_greeks, option_type, cfg.oi_wall_multiple, cfg.oi_wall_local_window),
        strike_step, cfg.oi_wall_band_strikes,
    )
    side = side[~side["strike"].isin(walls)]
    if side.empty:
        return None

    side["delta_diff"] = (side["delta"].abs() - cfg.target_delta).abs()
    return side.sort_values("delta_diff").iloc[0].to_dict()


def select_long_strike(chain_with_greeks: pd.DataFrame, option_type: str, short_strike: float,
                        strike_step: float, cfg: StrikeSelectionConfig = STRIKE_SELECTION) -> dict | None:
    """
    Protective long leg: `cfg.spread_width_strikes` strike-steps further
    OTM than the short leg (further from the forward). No delta/probability
    filtering here — the long leg's job is capping risk at a defined
    width, not hitting a delta target.
    """
    direction = 1 if option_type == "CE" else -1
    target_strike = short_strike + direction * cfg.spread_width_strikes * strike_step

    side = chain_with_greeks[chain_with_greeks["option_type"] == option_type]
    matches = side[np.isclose(side["strike"], target_strike)]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()
