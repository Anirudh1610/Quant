"""
core/probability.py
--------------------
Probability of a GBM path touching a barrier at any point before expiry
("probability of touch"), via the reflection principle. Used by
strikes/selection.py to filter candidate short strikes.

    P(max_{0<=t<=T} X_t >= h) = N((nu*T - h)/(sigma*sqrt(T)))
                                 + exp(2*nu*h/sigma^2) * N((-nu*T - h)/(sigma*sqrt(T)))

where X_t = ln(S_t/S_0) is a Brownian motion with drift nu = mu - 0.5*sigma^2
and diffusion sigma, and h = ln(barrier/S_0). The lower-barrier case follows
by the X -> -X symmetry. At h=0 (barrier==S) both branches correctly reduce
to 1.0 without needing to special-case it.

Drift convention: `mu` is exposed as a parameter, not hardcoded, but the
recommended usage — matching how the rest of this codebase already prices
options (core/pricing.py's S=forward, q=r substitution) — is to pass the
market-implied forward as `S` with `mu=0.0`. The forward is a martingale
under the T-forward measure, so zero drift is the risk-neutral-consistent
choice there, not an arbitrary simplification.

ASSUMPTION FLAG: this assumes continuous GBM with no jumps and constant
vol through expiry. Real index paths jump (results/events/circuit days),
so this will understate genuine tail/touch risk. Use it to rank/filter
candidate strikes against each other, not as a literal probability.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def probability_of_touch(S: float, barrier: float, T: float, sigma: float, mu: float = 0.0) -> float:
    """
    Probability that a GBM path starting at S, with annualized log-return
    drift `mu` and volatility `sigma`, touches `barrier` at any point in
    [0, T].
    """
    if T <= 0 or sigma <= 0:
        return 1.0 if barrier == S else 0.0

    nu = mu - 0.5 * sigma ** 2
    h = np.log(barrier / S)
    sqrtT = np.sqrt(T)
    reflection = np.exp(2 * nu * h / sigma ** 2)

    if barrier >= S:
        d1 = (nu * T - h) / (sigma * sqrtT)
        d2 = (-nu * T - h) / (sigma * sqrtT)
    else:
        d1 = (h - nu * T) / (sigma * sqrtT)
        d2 = (h + nu * T) / (sigma * sqrtT)

    prob = norm.cdf(d1) + reflection * norm.cdf(d2)
    return float(np.clip(prob, 0.0, 1.0))
