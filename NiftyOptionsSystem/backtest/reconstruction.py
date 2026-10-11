"""
backtest/reconstruction.py
----------------------------
Reconstructs a synthetic option chain for a historical date, since Kite has
no historical option-chain endpoint (flagged in Phase 1's README — this is
the "reconstruct via Black-Scholes from historical spot + IV if raw chain
data isn't available" fallback the original spec anticipated). Built from
that date's cached spot close + India VIX via Black-76 (S=forward, q=r —
same convention as data/option_chain.py and strikes/selection.py).

Output has the SAME schema as data/option_chain.py:fetch_option_chain_snapshot
(strike, option_type, ltp, bid, ask, oi, volume, iv, underlying_ltp), so
Phase 3's strikes/selection.py and strikes/spreads.py run unchanged on
either a live or a reconstructed chain — that compatibility is the whole
point of matching the schema.

ASSUMPTION FLAGS (all stem from the same root cause: no historical option
chain data source)
------------------------------------------------------------------------
- **Flat IV across all strikes, by default.** `reconstruct_chain`/
  `reprice_leg` give every strike that day's VIX/100 as its IV (note: that
  IV *does* vary correctly day-to-day with real historical VIX — verified
  live, range 9.15-27.89 over 2023-2026, including the real 2024-06-04
  election-result vol spike; "flat" here means flat *across strikes on a
  given day*, not frozen over time). `reconstruct_chain_with_skew`/
  `reprice_leg_with_skew` below are an opt-in alternative that adds a
  documented synthetic skew on top of that same daily ATM IV — see
  `skewed_iv`'s docstring for why it's a heuristic, not a calibration.
  Delta-targeted strike selection under the flat-IV default will be
  systematically less accurate for OTM strikes than the live pipeline
  (which computes true per-strike IV from real quotes) or the skewed path.
- **No historical OI.** oi/volume columns are NaN. strikes.selection.
  find_oi_walls degrades to a no-op (empty wall set, since it explicitly
  handles chain["oi"].isna().all()) — OI-wall avoidance is NOT modeled in
  backtests, only in the live pipeline.
- **No historical forward/parity data.** The forward is the static
  S*e^{(r-q)T} carry approximation (config.RISK_FREE_RATE /
  DIVIDEND_YIELD) — the same static approximation the live pipeline moved
  away from (see core/pricing.py's module docstring) because it
  measurably diverges from the true market-implied forward. There is no
  live chain to imply a forward from on a historical date, so this is the
  best available fallback here, not a preference.
- **Synthetic expiry tenor, not the real historical NSE calendar.** NSE's
  actual weekly/monthly expiry days have changed over time (see
  data/instruments.py's docstring), and there is no dataset of historical
  expiry dates to align to. backtest/engine.py uses a fixed calendar-day
  tenor from each entry date as an idealized expiry, not a real historical
  Thursday/Wednesday.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.pricing import bs_price

DEFAULT_STRIKES_EACH_SIDE = 40


def reconstruct_chain(spot: float, iv: float, T: float, r: float, q: float, strike_step: float,
                       n_strikes_each_side: int = DEFAULT_STRIKES_EACH_SIDE) -> pd.DataFrame:
    """
    Synthetic chain centered on spot's nearest strike, +/- n_strikes_each_side.
    `iv` is a flat annualized decimal (e.g. that day's VIX/100) applied to
    every strike — see module docstring's first ASSUMPTION FLAG.
    """
    atm = round(spot / strike_step) * strike_step
    strikes = [atm + i * strike_step for i in range(-n_strikes_each_side, n_strikes_each_side + 1)]
    forward = spot * np.exp((r - q) * T)

    rows = []
    for k in strikes:
        for option_type, bs_type in (("CE", "call"), ("PE", "put")):
            # Black-76 (S=forward, q=r) -- consistent with core/pricing.py's
            # module-docstring convention used throughout the live pipeline
            price = bs_price(forward, k, T, r, iv, q=r, option_type=bs_type)
            rows.append({
                "strike": float(k),
                "option_type": option_type,
                "ltp": price,
                "bid": price,   # no synthetic bid/ask spread -- backtest/engine.py
                "ask": price,   # applies slippage separately at fill time
                "oi": np.nan,
                "volume": np.nan,
                "iv": iv,
                "underlying_ltp": spot,
            })
    return pd.DataFrame(rows)


def reprice_leg(option_type: str, strike: float, spot: float, iv: float, T: float, r: float, q: float) -> float:
    """
    Theoretical price of a single leg for daily mark-to-market, without
    regenerating a full chain (cheap — called once per open leg per day).
    Same Black-76 convention as reconstruct_chain.
    """
    forward = spot * np.exp((r - q) * T)
    bs_type = "call" if option_type == "CE" else "put"
    return bs_price(forward, strike, T, r, iv, q=r, option_type=bs_type)


# ── Optional cross-strike skew (opt-in — see BacktestConfig.use_iv_skew) ────
#
# There is no historical per-strike IV to calibrate a skew curve against
# (that's the entire reason reconstruction exists). DEFAULT_SKEW_SLOPE and
# DEFAULT_SMILE_CURVATURE below are illustrative, order-of-magnitude values
# reflecting the well-known qualitative shape of index option skew (OTM
# puts trade richer than ATM, OTM calls trade flat-to-cheaper than ATM) —
# they are NOT fitted to this project's own data. A live NIFTY snapshot
# pulled earlier in this project did show a similar qualitative pattern
# (far OTM/ITM strikes reading higher IV than near-ATM), but those specific
# readings were separately diagnosed as noise-dominated (thin trading on
# low-vega, near-zero-time-value contracts), so they were deliberately NOT
# used to fit these constants. Treat any result produced with
# use_iv_skew=True as "under one illustrative skew shape", not "with the
# real historical skew" — validate/replace these constants if the shape or
# magnitude matters to your conclusion.

DEFAULT_SKEW_SLOPE = 0.08       # linear term: positive -> puts richer than calls (standard equity-index direction)
DEFAULT_SMILE_CURVATURE = 0.02  # quadratic term: both tails pulled up relative to a pure straight-line skew
MIN_SKEWED_IV = 0.02            # floor, guards against a degenerate/negative IV at extreme moneyness


def skewed_iv(strike: float, forward: float, iv_atm: float, T: float,
              skew_slope: float = DEFAULT_SKEW_SLOPE, smile_curvature: float = DEFAULT_SMILE_CURVATURE,
              min_iv: float = MIN_SKEWED_IV) -> float:
    """
    IV at `strike`, given that day's ATM IV, as:

        x   = ln(strike/forward) / (iv_atm * sqrt(T))     # moneyness in ATM-implied std-dev units
        iv  = iv_atm * (1 - skew_slope*x + smile_curvature*x**2)

    Normalizing by iv_atm*sqrt(T) (rather than using raw log-moneyness)
    keeps the skew shape comparable across different DTEs and vol regimes
    — a strike 1 "sigma" OTM means roughly the same thing on a 5-day trade
    at 25% IV as on a 30-day trade at 12% IV. x<0 is OTM puts / ITM calls
    (richer under the default positive skew_slope); x>0 is OTM calls / ITM
    puts (flat-to-cheaper). See module docstring for why the coefficients
    are illustrative, not calibrated.
    """
    if T <= 0 or iv_atm <= 0:
        return iv_atm
    x = np.log(strike / forward) / (iv_atm * np.sqrt(T))
    iv = iv_atm * (1 - skew_slope * x + smile_curvature * x ** 2)
    return max(iv, min_iv)


def reconstruct_chain_with_skew(spot: float, iv_atm: float, T: float, r: float, q: float, strike_step: float,
                                 skew_slope: float = DEFAULT_SKEW_SLOPE,
                                 smile_curvature: float = DEFAULT_SMILE_CURVATURE,
                                 n_strikes_each_side: int = DEFAULT_STRIKES_EACH_SIDE) -> pd.DataFrame:
    """Same as reconstruct_chain, but each strike's IV comes from skewed_iv() instead of a flat iv_atm."""
    atm = round(spot / strike_step) * strike_step
    strikes = [atm + i * strike_step for i in range(-n_strikes_each_side, n_strikes_each_side + 1)]
    forward = spot * np.exp((r - q) * T)

    rows = []
    for k in strikes:
        iv_k = skewed_iv(k, forward, iv_atm, T, skew_slope, smile_curvature)
        for option_type, bs_type in (("CE", "call"), ("PE", "put")):
            price = bs_price(forward, k, T, r, iv_k, q=r, option_type=bs_type)
            rows.append({
                "strike": float(k),
                "option_type": option_type,
                "ltp": price,
                "bid": price,
                "ask": price,
                "oi": np.nan,
                "volume": np.nan,
                "iv": iv_k,
                "underlying_ltp": spot,
            })
    return pd.DataFrame(rows)


def reprice_leg_with_skew(option_type: str, strike: float, spot: float, iv_atm: float, T: float, r: float, q: float,
                           skew_slope: float = DEFAULT_SKEW_SLOPE,
                           smile_curvature: float = DEFAULT_SMILE_CURVATURE) -> float:
    """
    Same as reprice_leg, but applies skewed_iv() using *that day's* ATM IV
    and the leg's fixed strike — i.e. a held leg's skew offset relative to
    ATM is recomputed fresh each day (not frozen at entry), consistent with
    how the flat-IV path already re-derives IV daily rather than freezing
    it (verified in the Step 2 premise check).
    """
    forward = spot * np.exp((r - q) * T)
    iv_k = skewed_iv(strike, forward, iv_atm, T, skew_slope, smile_curvature)
    bs_type = "call" if option_type == "CE" else "put"
    return bs_price(forward, strike, T, r, iv_k, q=r, option_type=bs_type)
