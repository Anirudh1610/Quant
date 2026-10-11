"""
signals/oi.py
-------------
Open-interest-derived context: max pain, PCR, OI concentration near spot,
and a composite "range-bound conviction" score.

ASSUMPTION FLAG: max pain and OI concentration are computed from Kite's
live cumulative OI per strike. Neither is a proven predictive signal on its
own — they're commonly used by discretionary options sellers as context
(which is how the user described their manual process), and are exposed
here as inputs to the regime object, not as a standalone trading signal.
The range_bound_conviction_score's weights are a documented starting
heuristic (see the function docstring) that should be validated/tuned once
the Phase 4 backtester exists — don't treat the default weights as final.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _pivot_oi(chain: pd.DataFrame) -> pd.DataFrame:
    """chain: columns strike, option_type, oi -> wide DataFrame indexed by strike with CE/PE OI columns."""
    pivot = chain.pivot_table(index="strike", columns="option_type", values="oi", aggfunc="sum").fillna(0.0)
    for col in ("CE", "PE"):
        if col not in pivot.columns:
            pivot[col] = 0.0
    return pivot[["CE", "PE"]].sort_index()


def max_pain(chain: pd.DataFrame) -> float:
    """
    Strike at which aggregate option-writer payout (across all listed
    strikes) is minimized. `chain` needs columns: strike, option_type
    ("CE"/"PE"), oi.
    """
    pivot = _pivot_oi(chain)
    strikes = pivot.index.to_numpy()
    ce_oi = pivot["CE"].to_numpy()
    pe_oi = pivot["PE"].to_numpy()

    # total_loss[i] = payout if the index expires at strikes[i]
    # call writers lose (S_c - K) for every K < S_c that's ITM; put writers
    # lose (K - S_c) for every K > S_c that's ITM.
    diffs = strikes[:, None] - strikes[None, :]          # diffs[c, k] = S_c - K
    call_loss = np.where(diffs > 0, diffs, 0.0) @ ce_oi   # sum over K of ITM call payout at each S_c
    put_loss = np.where(-diffs > 0, -diffs, 0.0) @ pe_oi
    total_loss = call_loss + put_loss

    return float(strikes[np.argmin(total_loss)])


def put_call_ratio(chain: pd.DataFrame) -> float:
    """Total Put OI / Total Call OI. >1 = more put OI (often read as bullish contrarian), <1 = more call OI."""
    pivot = _pivot_oi(chain)
    total_ce = pivot["CE"].sum()
    if total_ce == 0:
        return float("nan")
    return float(pivot["PE"].sum() / total_ce)


def oi_concentration_ratio(chain: pd.DataFrame, spot: float, strike_step: int,
                            n_strikes: int = 5) -> float:
    """
    Fraction of total OI (CE+PE combined) sitting within +/- n_strikes of
    spot. High concentration near spot is read as a pinning/range-bound cue;
    concentration far from spot suggests the market is positioned for a move.
    """
    pivot = _pivot_oi(chain)
    total_oi = (pivot["CE"] + pivot["PE"]).sum()
    if total_oi == 0:
        return float("nan")

    atm = round(spot / strike_step) * strike_step
    lo, hi = atm - n_strikes * strike_step, atm + n_strikes * strike_step
    near_mask = (pivot.index >= lo) & (pivot.index <= hi)
    near_oi = (pivot.loc[near_mask, "CE"] + pivot.loc[near_mask, "PE"]).sum()

    return float(near_oi / total_oi)


def range_bound_conviction_score(spot: float, max_pain_strike: float, pcr: float,
                                  concentration_ratio: float, strike_step: int) -> float:
    """
    Composite 0-100 "range-bound conviction" score from three OI-derived
    inputs. This is a heuristic starting point, documented so it can be
    replaced/reweighted after backtesting — not a validated model.

    Components (equal-weighted by default):
      1. Spot-to-max-pain proximity: closer = more range-bound conviction.
         Normalized by a 3% band (spot within 3% of max pain -> full score,
         linearly decaying to 0 at >=3% away).
      2. PCR balance: PCR near 1.0 (balanced positioning) = more range-bound
         conviction; PCR far from 1.0 (skewed) = less. Scored as
         1 - min(|log(PCR)|, 1) so it's symmetric in put-heavy/call-heavy
         skew.
      3. OI concentration near spot: higher concentration = more range-bound
         conviction (OI walls near spot suggest pinning).
    """
    pain_distance_pct = abs(spot - max_pain_strike) / spot
    pain_score = max(0.0, 1.0 - pain_distance_pct / 0.03)

    pcr_score = 1.0
    if pcr and not np.isnan(pcr) and pcr > 0:
        pcr_score = max(0.0, 1.0 - min(abs(np.log(pcr)), 1.0))

    concentration_score = concentration_ratio if not np.isnan(concentration_ratio) else 0.0

    composite = (pain_score + pcr_score + concentration_score) / 3.0
    return float(np.clip(composite * 100, 0, 100))
