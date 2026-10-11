from __future__ import annotations

import numpy as np
import pytest

from core.pricing import bs_greeks, bs_price, implied_forward, implied_volatility

S, K, T, R, SIGMA, Q = 22000.0, 22000.0, 30 / 365, 0.068, 0.15, 0.015


class TestBSPrice:
    def test_put_call_parity(self):
        call = bs_price(S, K, T, R, SIGMA, Q, "call")
        put = bs_price(S, K, T, R, SIGMA, Q, "put")
        lhs = call - put
        rhs = S * np.exp(-Q * T) - K * np.exp(-R * T)
        assert lhs == pytest.approx(rhs, abs=1e-6)

    def test_price_increases_with_vol(self):
        low = bs_price(S, K, T, R, 0.10, Q, "call")
        high = bs_price(S, K, T, R, 0.30, Q, "call")
        assert high > low

    def test_zero_time_returns_intrinsic(self):
        itm_call = bs_price(22500, 22000, 0, R, SIGMA, Q, "call")
        assert itm_call == pytest.approx(500.0)
        otm_call = bs_price(21500, 22000, 0, R, SIGMA, Q, "call")
        assert otm_call == pytest.approx(0.0)

    def test_deep_itm_call_approaches_intrinsic_forward(self):
        deep_itm = bs_price(30000, 10000, T, R, SIGMA, Q, "call")
        forward_intrinsic = 30000 * np.exp(-Q * T) - 10000 * np.exp(-R * T)
        assert deep_itm == pytest.approx(forward_intrinsic, rel=1e-3)

    def test_price_nonnegative_far_otm(self):
        far_otm_put = bs_price(30000, 10000, T, R, SIGMA, Q, "put")
        assert far_otm_put >= 0


class TestBSGreeks:
    def test_call_delta_bounds(self):
        g = bs_greeks(S, K, T, R, SIGMA, Q, "call")
        assert 0.0 <= g["delta"] <= 1.0

    def test_put_delta_bounds(self):
        g = bs_greeks(S, K, T, R, SIGMA, Q, "put")
        assert -1.0 <= g["delta"] <= 0.0

    def test_delta_matches_finite_difference(self):
        h = 1.0
        price_up = bs_price(S + h, K, T, R, SIGMA, Q, "call")
        price_down = bs_price(S - h, K, T, R, SIGMA, Q, "call")
        fd_delta = (price_up - price_down) / (2 * h)
        analytic_delta = bs_greeks(S, K, T, R, SIGMA, Q, "call")["delta"]
        assert analytic_delta == pytest.approx(fd_delta, abs=1e-3)

    def test_gamma_positive_and_matches_finite_difference(self):
        h = 1.0
        p_up = bs_price(S + h, K, T, R, SIGMA, Q, "call")
        p_mid = bs_price(S, K, T, R, SIGMA, Q, "call")
        p_down = bs_price(S - h, K, T, R, SIGMA, Q, "call")
        fd_gamma = (p_up - 2 * p_mid + p_down) / (h ** 2)
        analytic_gamma = bs_greeks(S, K, T, R, SIGMA, Q, "call")["gamma"]
        assert analytic_gamma > 0
        assert analytic_gamma == pytest.approx(fd_gamma, abs=1e-3)

    def test_vega_positive_and_same_for_call_and_put(self):
        call_vega = bs_greeks(S, K, T, R, SIGMA, Q, "call")["vega"]
        put_vega = bs_greeks(S, K, T, R, SIGMA, Q, "put")["vega"]
        assert call_vega > 0
        assert call_vega == pytest.approx(put_vega, rel=1e-9)

    def test_theta_negative_for_atm_long_option(self):
        # long-option theta is (near-universally, for non-dividend-heavy ATM options) negative --
        # time decay erodes value. We're pricing the long side here (bs_price convention),
        # a short seller's P&L theta is the negation of this.
        call_theta = bs_greeks(S, K, T, R, SIGMA, Q, "call")["theta"]
        assert call_theta < 0

    def test_zero_time_or_vol_returns_zeroed_greeks(self):
        g = bs_greeks(S, K, 0, R, SIGMA, Q, "call")
        assert g == {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}


class TestImpliedVolatility:
    @pytest.mark.parametrize("true_sigma", [0.08, 0.15, 0.25, 0.45])
    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_round_trip_recovers_input_sigma(self, true_sigma, option_type):
        price = bs_price(S, K, T, R, true_sigma, Q, option_type)
        recovered = implied_volatility(price, S, K, T, R, Q, option_type)
        assert recovered == pytest.approx(true_sigma, abs=1e-4)

    def test_price_below_intrinsic_returns_none(self):
        # A "call" price cheaper than intrinsic value is an arbitrage-violating
        # (stale/illiquid) quote -- must not fabricate an IV for it.
        deep_itm_intrinsic = 30000 - 10000
        bad_price = deep_itm_intrinsic - 100
        result = implied_volatility(bad_price, 30000, 10000, T, R, Q, "call")
        assert result is None

    def test_zero_price_returns_none(self):
        assert implied_volatility(0.0, S, K, T, R, Q, "call") is None

    def test_zero_time_returns_none(self):
        assert implied_volatility(100.0, S, K, 0, R, Q, "call") is None

    def test_otm_strike_round_trip(self):
        otm_k = 23000.0
        price = bs_price(S, otm_k, T, R, 0.18, Q, "call")
        recovered = implied_volatility(price, S, otm_k, T, R, Q, "call")
        assert recovered == pytest.approx(0.18, abs=1e-4)


class TestBlack76ForwardPricing:
    """
    Black-76 (pricing off a forward instead of spot) is implemented as
    bs_price/bs_greeks/implied_volatility called with S=F, q=r -- see
    core/pricing.py's module docstring for the derivation. These tests
    verify that equivalence holds, and that implied_forward() correctly
    recovers a known forward from parity.
    """

    F, true_sigma = 24339.5, 0.13

    def test_call_and_put_price_off_forward_satisfy_parity(self):
        call = bs_price(self.F, K, T, R, self.true_sigma, R, "call")
        put = bs_price(self.F, K, T, R, self.true_sigma, R, "put")
        # Black-76 parity: C - P = e^{-rT}(F - K)
        assert call - put == pytest.approx(np.exp(-R * T) * (self.F - K), abs=1e-6)

    def test_implied_forward_recovers_known_forward(self):
        call = bs_price(self.F, K, T, R, self.true_sigma, R, "call")
        put = bs_price(self.F, K, T, R, self.true_sigma, R, "put")
        recovered = implied_forward(call, put, K, T, R)
        assert recovered == pytest.approx(self.F, abs=1e-4)

    @pytest.mark.parametrize("strike", [23500.0, 24000.0, 24339.5, 24800.0, 25500.0])
    def test_implied_forward_consistent_across_strikes(self, strike):
        # Parity-implied forward should recover F regardless of which
        # strike's quotes you use -- this is what made the live sanity
        # check trustworthy (8 strikes agreed to within a few cents).
        call = bs_price(self.F, strike, T, R, self.true_sigma, R, "call")
        put = bs_price(self.F, strike, T, R, self.true_sigma, R, "put")
        recovered = implied_forward(call, put, strike, T, R)
        assert recovered == pytest.approx(self.F, abs=1e-4)

    def test_iv_recovered_off_forward_matches_true_sigma(self):
        price = bs_price(self.F, K, T, R, self.true_sigma, R, "call")
        recovered = implied_volatility(price, self.F, K, T, R, R, "call")
        assert recovered == pytest.approx(self.true_sigma, abs=1e-4)

    def test_wrong_static_q_produces_spurious_call_put_skew(self):
        # Reproduces the bug found during the live sanity check on real
        # NIFTY data: the market's true forward was ~44pts above what our
        # static-q assumption implied. Pricing off spot with that too-low
        # implied forward makes calls invert to a higher IV and puts to a
        # lower IV than the single true sigma both were actually priced
        # at -- confirming the mechanism, not just the symptom.
        true_sigma = 0.13
        S_spot = 24274.35
        wrong_q = 0.015  # config.DIVIDEND_YIELD["NIFTY"]
        forward_assumed = S_spot * np.exp((R - wrong_q) * T)  # what the static-q path believes the forward is
        forward_true = forward_assumed + 44.0                  # what the market actually priced in (empirical gap)

        call_price = bs_price(forward_true, K, T, R, true_sigma, R, "call")
        put_price = bs_price(forward_true, K, T, R, true_sigma, R, "put")

        call_iv_wrong = implied_volatility(call_price, S_spot, K, T, R, wrong_q, "call")
        put_iv_wrong = implied_volatility(put_price, S_spot, K, T, R, wrong_q, "put")
        assert call_iv_wrong > true_sigma
        assert put_iv_wrong < true_sigma

        # pricing off the correctly-implied forward recovers the true sigma for both legs
        call_iv_fixed = implied_volatility(call_price, forward_true, K, T, R, R, "call")
        put_iv_fixed = implied_volatility(put_price, forward_true, K, T, R, R, "put")
        assert call_iv_fixed == pytest.approx(true_sigma, abs=1e-4)
        assert put_iv_fixed == pytest.approx(true_sigma, abs=1e-4)
