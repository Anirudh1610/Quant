"""
core/pricing.py
----------------
Black-Scholes-Merton pricing, Greeks, and implied volatility inversion.

ASSUMPTION FLAG: Kite's quote()/option chain data does NOT include implied
volatility directly (unlike some retail platforms' option-chain widgets).
IV here is computed by inverting the BS price from the traded LTP (or mid
of bid/ask when available) — this is standard practice, but it means our IV
is only as good as (a) the BS/GBM assumption for index options and (b) the
last traded/quoted price being a fair reflection of current value (illiquid
far OTM strikes can have stale LTPs that produce noisy IV).

Pricing off a market-implied forward instead of spot
------------------------------------------------------
Pricing/inverting off spot `S` requires guessing a dividend yield `q` to
get the cost-of-carry right. A static guess (see config.DIVIDEND_YIELD)
measurably diverges from what the market is actually pricing — verified
empirically: an ~44-point forward mismatch on NIFTY produced a spurious
call/put IV skew (calls ~12-14%, puts ~8-10% at the *same* strikes, the
wrong direction for real index skew).

The fix used throughout this module: call `bs_price`/`bs_greeks`/
`implied_volatility` with `S=F` (a market-implied forward — see
`implied_forward()` below) and `q=r`. This is exactly the Black-76 model:
substituting S=F, q=r into the BS formula collapses `S*e^{-qT}` to
`F*e^{-rT}`, which is precisely Black-76's discounted-forward pricing — no
new formula needed, just different inputs. `data/option_chain.py` computes
`F` per snapshot from put-call parity on liquid near-ATM strikes rather
than trusting the static config.DIVIDEND_YIELD guess.

All prices are in index points. r and q are continuous annualized rates.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

_MIN_SIGMA = 1e-4
_MAX_SIGMA = 5.0


def bs_price(S: float, K: float, T: float, r: float, sigma: float,
             q: float = 0.0, option_type: str = "call") -> float:
    """Black-Scholes-Merton price. Returns intrinsic value if T<=0 or sigma<=0."""
    if T <= 0 or sigma <= 0:
        intrinsic = (S - K) if option_type == "call" else (K - S)
        return max(intrinsic, 0.0)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return max(S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2), 0.0)
    return max(K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1), 0.0)


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float,
              q: float = 0.0, option_type: str = "call") -> dict:
    """
    Returns delta, gamma, theta (per calendar day), vega (per 1% IV move).
    Delta sign convention: calls in [0, 1], puts in [-1, 0].
    """
    if T <= 0 or sigma <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    phi1 = norm.pdf(d1)
    gamma = phi1 * np.exp(-q * T) / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * phi1 * np.sqrt(T) / 100

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

    return {"delta": float(delta), "gamma": float(gamma), "theta": float(theta), "vega": float(vega)}


def implied_volatility(price: float, S: float, K: float, T: float, r: float,
                        q: float = 0.0, option_type: str = "call") -> float | None:
    """
    Invert BS to find sigma given a traded/quoted price, via Brent's method.
    Returns None if no solution exists in [_MIN_SIGMA, _MAX_SIGMA] (e.g. the
    price is below intrinsic value or the input is otherwise degenerate) —
    callers should treat None as "IV unavailable for this row", not zero.
    """
    if T <= 0 or price <= 0:
        return None

    # No-arbitrage floor is the *discounted* forward intrinsic (from put-call
    # parity: C >= S*e^{-qT} - K*e^{-rT}, since P >= 0), not raw S-K. Using
    # the undiscounted bound here would mis-reject/mis-accept borderline
    # deep ITM quotes, especially when q and r differ meaningfully (as they
    # do when called with q=r for Black-76 forward pricing — see this
    # module's docstring).
    disc_forward_value = S * np.exp(-q * T) - K * np.exp(-r * T)
    intrinsic = max(disc_forward_value if option_type == "call" else -disc_forward_value, 0.0)
    if price < intrinsic:
        return None  # arbitrage-violating quote (stale/illiquid) — don't fabricate an IV

    def objective(sigma: float) -> float:
        return bs_price(S, K, T, r, sigma, q, option_type) - price

    try:
        lo, hi = objective(_MIN_SIGMA), objective(_MAX_SIGMA)
        if lo * hi > 0:
            return None  # price out of the range spanned by [min,max] sigma
        return brentq(objective, _MIN_SIGMA, _MAX_SIGMA, xtol=1e-6)
    except (ValueError, RuntimeError):
        return None


def implied_forward(call_price: float, put_price: float, K: float, T: float, r: float) -> float:
    """
    Market-implied forward from a single strike's call/put (mid) prices via
    put-call parity: C - P = e^{-rT}(F - K)  =>  F = K + (C - P) * e^{rT}.

    A single strike's estimate is noisy (wide spreads, stale quotes); the
    standard approach — used by data/option_chain.py — is to compute this
    across several liquid near-ATM strikes and take the median.
    """
    return K + (call_price - put_price) * np.exp(r * T)
