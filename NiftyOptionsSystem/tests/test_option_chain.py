from __future__ import annotations

import numpy as np
import pytest

from core.pricing import bs_price
from data.option_chain import _estimate_forward

T, R, Q = 6 / 365, 0.068, 0.015


def _contract(strike, option_type, mid):
    return {"strike": strike, "option_type": option_type, "mid": mid, "ltp": mid, "bid": mid, "ask": mid}


class TestEstimateForward:
    def test_recovers_known_forward_from_synthetic_chain(self):
        true_forward = 24339.5
        true_sigma = 0.13
        strikes = [24000, 24100, 24200, 24300, 24400, 24500, 24600]

        parsed = []
        for k in strikes:
            call = bs_price(true_forward, k, T, R, true_sigma, R, "call")
            put = bs_price(true_forward, k, T, R, true_sigma, R, "put")
            parsed.append(_contract(k, "CE", call))
            parsed.append(_contract(k, "PE", put))

        forward = _estimate_forward(parsed, underlying_ltp=24274.35, T=T, r=R, q=Q, n_strikes=8)
        assert forward == pytest.approx(true_forward, abs=1e-3)

    def test_robust_to_one_bad_strike(self):
        true_forward = 24339.5
        true_sigma = 0.13
        strikes = [24000, 24100, 24200, 24300, 24400, 24500, 24600]

        parsed = []
        for k in strikes:
            call = bs_price(true_forward, k, T, R, true_sigma, R, "call")
            put = bs_price(true_forward, k, T, R, true_sigma, R, "put")
            parsed.append(_contract(k, "CE", call))
            parsed.append(_contract(k, "PE", put))

        # corrupt one strike's put quote (e.g. a stale/wide-spread illiquid print)
        for c in parsed:
            if c["strike"] == 24600 and c["option_type"] == "PE":
                c["mid"] = c["mid"] * 3

        forward = _estimate_forward(parsed, underlying_ltp=24274.35, T=T, r=R, q=Q, n_strikes=8)
        # median across 7 strikes should shrug off one corrupted pair
        assert forward == pytest.approx(true_forward, abs=5.0)

    def test_falls_back_to_static_carry_when_illiquid(self):
        underlying_ltp = 24274.35
        parsed = [_contract(24300, "CE", None), _contract(24300, "PE", None)]  # no valid mids anywhere

        forward = _estimate_forward(parsed, underlying_ltp=underlying_ltp, T=T, r=R, q=Q, n_strikes=8)
        expected_fallback = underlying_ltp * np.exp((R - Q) * T)
        assert forward == pytest.approx(expected_fallback)

    def test_falls_back_with_only_one_liquid_pair(self):
        underlying_ltp = 24274.35
        true_forward = 24339.5
        true_sigma = 0.13
        call = bs_price(true_forward, 24300, T, R, true_sigma, R, "call")
        put = bs_price(true_forward, 24300, T, R, true_sigma, R, "put")
        parsed = [_contract(24300, "CE", call), _contract(24300, "PE", put)]

        # only 1 strike has both legs quoted -- below the n>=2 threshold for a median
        forward = _estimate_forward(parsed, underlying_ltp=underlying_ltp, T=T, r=R, q=Q, n_strikes=8)
        expected_fallback = underlying_ltp * np.exp((R - Q) * T)
        assert forward == pytest.approx(expected_fallback)

    def test_picks_nearest_the_money_strikes_when_more_available_than_n(self):
        true_forward = 24339.5
        true_sigma = 0.13
        near_strikes = [24200, 24300, 24400]
        far_strikes = [20000, 30000]  # would badly distort parity if included (huge extrinsic/discretization noise)

        parsed = []
        for k in near_strikes + far_strikes:
            call = bs_price(true_forward, k, T, R, true_sigma, R, "call")
            put = bs_price(true_forward, k, T, R, true_sigma, R, "put")
            parsed.append(_contract(k, "CE", call))
            parsed.append(_contract(k, "PE", put))

        forward = _estimate_forward(parsed, underlying_ltp=24274.35, T=T, r=R, q=Q, n_strikes=3)
        assert forward == pytest.approx(true_forward, abs=1e-3)
