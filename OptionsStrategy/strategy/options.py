"""
options.py
----------
Black-Scholes-Merton pricing, Greeks, strike selection, and transaction costs.

All prices are in index points.  ₹1 = 1 point of P&L per lot for both
Nifty 50 and Bank Nifty (i.e. lot_size points = 1 lot's rupee exposure).

Lot sizes (verify with NSE circulars — these are revised periodically):
    Nifty 50   : 25  (revised November 2024)
    Bank Nifty : 15  (revised November 2024)

Strike steps:
    Nifty 50   : ₹50 increments
    Bank Nifty : ₹100 increments
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

NIFTY_LOT           = 25
BANKNIFTY_LOT       = 15
NIFTY_STRIKE_STEP   = 50
BANKNIFTY_STRIKE_STEP = 100

RISK_FREE_RATE      = 0.068    # 10-year G-sec yield (~6.8%)
NIFTY_DIV_YIELD     = 0.015    # ~1.5% for Nifty 50
BANKNIFTY_DIV_YIELD = 0.010    # ~1.0% for Bank Nifty


# ──────────────────────────────────────────────────────────────────────────────
# PRICING
# ──────────────────────────────────────────────────────────────────────────────

def bs_price(
    S: float, K: float, T: float, r: float, sigma: float,
    q: float = 0.0, option_type: str = "call",
) -> float:
    """
    Black-Scholes-Merton option price with continuous dividend yield.

    Parameters
    ----------
    S     : Spot price (index level).
    K     : Strike price.
    T     : Time to expiry in years (use trading_days / 252).
    r     : Continuous risk-free rate (annual decimal, e.g. 0.068).
    sigma : Implied volatility (annual decimal, e.g. 0.15 for 15%).
    q     : Continuous dividend yield (annual decimal).

    Returns
    -------
    Option premium in index points (≥ 0).
    """
    if T <= 0 or sigma <= 0:
        intrinsic = S - K if option_type == "call" else K - S
        return max(intrinsic, 0.0)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = (S * np.exp(-q * T) * norm.cdf(d1)
                 - K * np.exp(-r * T) * norm.cdf(d2))
    else:
        price = (K * np.exp(-r * T) * norm.cdf(-d2)
                 - S * np.exp(-q * T) * norm.cdf(-d1))

    return max(price, 0.0)


def bs_greeks(
    S: float, K: float, T: float, r: float, sigma: float,
    q: float = 0.0, option_type: str = "call",
) -> dict:
    """
    Black-Scholes Greeks.

    Returns dict with:
        delta : Rate of change of price w.r.t. spot.
        gamma : Rate of change of delta w.r.t. spot.
        theta : Daily time decay (per calendar day, in index points).
        vega  : Sensitivity to 1% change in IV (in index points).
    """
    if T <= 0 or sigma <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    d1    = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2    = d1 - sigma * np.sqrt(T)
    phi1  = norm.pdf(d1)
    gamma = phi1 * np.exp(-q * T) / (S * sigma * np.sqrt(T))
    vega  = S * np.exp(-q * T) * phi1 * np.sqrt(T) / 100  # per 1% IV

    if option_type == "call":
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (
            -(S * np.exp(-q * T) * phi1 * sigma / (2 * np.sqrt(T)))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
            + q * S * np.exp(-q * T) * norm.cdf(d1)
        ) / 365
    else:
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        theta = (
            -(S * np.exp(-q * T) * phi1 * sigma / (2 * np.sqrt(T)))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - q * S * np.exp(-q * T) * norm.cdf(-d1)
        ) / 365

    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}


# ──────────────────────────────────────────────────────────────────────────────
# STRIKE SELECTION
# ──────────────────────────────────────────────────────────────────────────────

def _nearest_strike(price: float, step: int) -> int:
    return int(round(price / step) * step)


def select_strikes(
    spot: float,
    signal: str,
    daily_iv_pct: float,
    strike_step: int,
    sigma_multiple: float = 1.0,
) -> tuple[int, int, str]:
    """
    Choose call and put strikes for the short straddle / strangle.

    Strike skew logic
    -----------------
    NEUTRAL  → both legs at ATM  (straddle)
    BULL     → call shifted further OTM, put near ATM  (give room for upside)
    BEAR     → put shifted further OTM, call near ATM  (give room for downside)
    STRANGLE → both legs symmetrically OTM at sigma_multiple × daily_iv

    Parameters
    ----------
    spot           : Entry spot price.
    signal         : NEUTRAL | BULL | BEAR | STRANGLE
    daily_iv_pct   : Daily 1σ move in % (VIX / sqrt(252)).
    strike_step    : Strike increment (50 for Nifty, 100 for BankNifty).
    sigma_multiple : OTM distance multiplier for STRANGLE legs.

    Returns
    -------
    (call_strike, put_strike, trade_type)
    """
    atm  = _nearest_strike(spot, strike_step)
    move = spot * daily_iv_pct / 100  # 1σ in absolute points

    if signal == "NEUTRAL":
        return atm, atm, "STRADDLE"

    if signal == "BULL":
        call_k = _nearest_strike(atm + move * 0.8 * sigma_multiple, strike_step)
        put_k  = _nearest_strike(atm - move * 0.3 * sigma_multiple, strike_step)
        return call_k, put_k, "STRANGLE"

    if signal == "BEAR":
        call_k = _nearest_strike(atm + move * 0.3 * sigma_multiple, strike_step)
        put_k  = _nearest_strike(atm - move * 0.8 * sigma_multiple, strike_step)
        return call_k, put_k, "STRANGLE"

    # STRANGLE — symmetric OTM
    call_k = _nearest_strike(atm + move * sigma_multiple, strike_step)
    put_k  = _nearest_strike(atm - move * sigma_multiple, strike_step)
    return call_k, put_k, "STRANGLE"


# ──────────────────────────────────────────────────────────────────────────────
# TRANSACTION COSTS (India — approximate)
# ──────────────────────────────────────────────────────────────────────────────

def trade_cost(
    premium_per_share: float,
    lot_size: int,
    n_lots: int = 1,
    brokerage: float = 20.0,
    stt_rate: float = 0.000625,      # 0.0625% on sell-side premium
    exchange_rate: float = 0.0005,   # NSE + SEBI + IPFT combined ~0.05%
    gst_rate: float = 0.18,
) -> float:
    """
    Approximate transaction cost (₹) for one options leg (sell order).

    Components
    ----------
    Brokerage  : flat ₹20 per order (Zerodha/Upstox rate).
    STT        : 0.0625% on premium turnover (sell side).
    Exchange   : ~0.05% of premium turnover (NSE + SEBI + IPFT).
    GST        : 18% on brokerage + exchange charges.
    """
    turnover = premium_per_share * lot_size * n_lots
    stt      = turnover * stt_rate
    exc      = turnover * exchange_rate
    gst      = (brokerage + exc) * gst_rate
    return brokerage + stt + exc + gst
