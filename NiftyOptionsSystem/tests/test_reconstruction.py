from __future__ import annotations

import numpy as np
import pytest

from backtest.reconstruction import (
    reconstruct_chain,
    reconstruct_chain_with_skew,
    reprice_leg,
    reprice_leg_with_skew,
    skewed_iv,
)
from core.pricing import bs_price

SPOT, IV, T, R, Q, STEP = 24300.0, 0.13, 20 / 365, 0.068, 0.015, 50.0


class TestReconstructChain:
    def test_schema_matches_live_option_chain_columns(self):
        chain = reconstruct_chain(SPOT, IV, T, R, Q, STEP, n_strikes_each_side=5)
        expected_cols = {"strike", "option_type", "ltp", "bid", "ask", "oi", "volume", "iv", "underlying_ltp"}
        assert expected_cols <= set(chain.columns)

    def test_strike_grid_centered_on_nearest_strike_to_spot(self):
        chain = reconstruct_chain(SPOT, IV, T, R, Q, STEP, n_strikes_each_side=5)
        atm = round(SPOT / STEP) * STEP
        assert atm in chain["strike"].values
        assert chain["strike"].min() == atm - 5 * STEP
        assert chain["strike"].max() == atm + 5 * STEP

    def test_both_option_types_present_at_every_strike(self):
        chain = reconstruct_chain(SPOT, IV, T, R, Q, STEP, n_strikes_each_side=5)
        counts = chain.groupby("strike")["option_type"].nunique()
        assert (counts == 2).all()

    def test_no_synthetic_bid_ask_spread(self):
        chain = reconstruct_chain(SPOT, IV, T, R, Q, STEP, n_strikes_each_side=5)
        assert (chain["bid"] == chain["ltp"]).all()
        assert (chain["ask"] == chain["ltp"]).all()

    def test_oi_and_volume_are_nan(self):
        chain = reconstruct_chain(SPOT, IV, T, R, Q, STEP, n_strikes_each_side=5)
        assert chain["oi"].isna().all()
        assert chain["volume"].isna().all()

    def test_prices_satisfy_put_call_parity(self):
        chain = reconstruct_chain(SPOT, IV, T, R, Q, STEP, n_strikes_each_side=5)
        forward = SPOT * np.exp((R - Q) * T)
        pivot = chain.pivot_table(index="strike", columns="option_type", values="ltp")
        for k in pivot.index:
            lhs = pivot.loc[k, "CE"] - pivot.loc[k, "PE"]
            rhs = np.exp(-R * T) * (forward - k)
            assert lhs == pytest.approx(rhs, abs=1e-6)

    def test_matches_direct_bs_price_call(self):
        chain = reconstruct_chain(SPOT, IV, T, R, Q, STEP, n_strikes_each_side=5)
        forward = SPOT * np.exp((R - Q) * T)
        atm = round(SPOT / STEP) * STEP
        expected = bs_price(forward, atm, T, R, IV, q=R, option_type="call")
        actual = chain[(chain.strike == atm) & (chain.option_type == "CE")]["ltp"].iloc[0]
        assert actual == pytest.approx(expected)


class TestRepriceLeg:
    def test_matches_reconstruct_chain_for_same_inputs(self):
        chain = reconstruct_chain(SPOT, IV, T, R, Q, STEP, n_strikes_each_side=5)
        atm = round(SPOT / STEP) * STEP
        chain_price = chain[(chain.strike == atm) & (chain.option_type == "PE")]["ltp"].iloc[0]
        leg_price = reprice_leg("PE", atm, SPOT, IV, T, R, Q)
        assert leg_price == pytest.approx(chain_price)

    def test_decreasing_time_decreases_extrinsic_value(self):
        atm = round(SPOT / STEP) * STEP
        far_t = reprice_leg("CE", atm, SPOT, IV, 20 / 365, R, Q)
        near_t = reprice_leg("CE", atm, SPOT, IV, 1 / 365, R, Q)
        assert near_t < far_t

    def test_zero_time_returns_intrinsic(self):
        price = reprice_leg("CE", 24000.0, 24300.0, IV, 0.0, R, Q)
        forward_at_zero_t = 24300.0 * np.exp((R - Q) * 0.0)  # == spot
        assert price == pytest.approx(max(forward_at_zero_t - 24000.0, 0.0))


FORWARD = SPOT * np.exp((R - Q) * T)


class TestSkewedIV:
    def test_atm_strike_equals_flat_iv(self):
        assert skewed_iv(FORWARD, FORWARD, IV, T) == pytest.approx(IV)

    def test_matches_hand_computation_put_side(self):
        # x = ln(K/F) / (iv*sqrt(T)); choose K so x is exactly -2
        x = -2.0
        k = FORWARD * np.exp(x * IV * np.sqrt(T))
        expected = IV * (1 - 0.08 * x + 0.02 * x ** 2)  # default skew_slope=0.08, smile_curvature=0.02
        assert skewed_iv(k, FORWARD, IV, T) == pytest.approx(expected)

    def test_matches_hand_computation_call_side(self):
        x = 2.0
        k = FORWARD * np.exp(x * IV * np.sqrt(T))
        expected = IV * (1 - 0.08 * x + 0.02 * x ** 2)
        assert skewed_iv(k, FORWARD, IV, T) == pytest.approx(expected)

    def test_otm_put_richer_than_atm_under_default_positive_skew(self):
        otm_put_strike = FORWARD * 0.95
        assert skewed_iv(otm_put_strike, FORWARD, IV, T) > IV

    def test_otm_call_cheaper_than_atm_under_default_positive_skew(self):
        otm_call_strike = FORWARD * 1.05
        assert skewed_iv(otm_call_strike, FORWARD, IV, T) < IV

    def test_floor_clamps_extreme_negative_multiplier(self):
        # deliberately extreme skew_slope to drive the raw multiplier negative
        # on the call side, to test the clamp itself (not the sane defaults)
        otm_call_strike = FORWARD * 1.20
        result = skewed_iv(otm_call_strike, FORWARD, IV, T, skew_slope=1.0, smile_curvature=0.0, min_iv=0.02)
        assert result == pytest.approx(0.02)

    def test_zero_time_or_zero_iv_returns_iv_atm_unchanged(self):
        assert skewed_iv(FORWARD * 1.1, FORWARD, IV, 0.0) == IV
        assert skewed_iv(FORWARD * 1.1, FORWARD, 0.0, T) == 0.0


class TestReconstructChainWithSkew:
    def test_schema_matches_flat_version(self):
        skewed = reconstruct_chain_with_skew(SPOT, IV, T, R, Q, STEP, n_strikes_each_side=5)
        flat = reconstruct_chain(SPOT, IV, T, R, Q, STEP, n_strikes_each_side=5)
        assert set(skewed.columns) == set(flat.columns)
        assert len(skewed) == len(flat)

    def test_atm_strike_iv_close_to_flat_iv(self):
        chain = reconstruct_chain_with_skew(SPOT, IV, T, R, Q, STEP, n_strikes_each_side=5)
        atm = round(SPOT / STEP) * STEP
        atm_iv = chain[(chain.strike == atm) & (chain.option_type == "CE")]["iv"].iloc[0]
        assert atm_iv == pytest.approx(IV, abs=0.01)  # forward != spot exactly, so not a bit-exact match

    def test_iv_is_not_flat_across_strikes(self):
        chain = reconstruct_chain_with_skew(SPOT, IV, T, R, Q, STEP, n_strikes_each_side=10)
        assert chain["iv"].nunique() > 1

    def test_far_otm_put_iv_exceeds_far_otm_call_iv(self):
        chain = reconstruct_chain_with_skew(SPOT, IV, T, R, Q, STEP, n_strikes_each_side=15)
        atm = round(SPOT / STEP) * STEP
        far_put = chain[(chain.strike == atm - 10 * STEP) & (chain.option_type == "PE")]["iv"].iloc[0]
        far_call = chain[(chain.strike == atm + 10 * STEP) & (chain.option_type == "CE")]["iv"].iloc[0]
        assert far_put > far_call


class TestRepriceLegWithSkew:
    def test_matches_reconstruct_chain_with_skew_for_same_inputs(self):
        chain = reconstruct_chain_with_skew(SPOT, IV, T, R, Q, STEP, n_strikes_each_side=5)
        atm = round(SPOT / STEP) * STEP
        target_strike = atm - 3 * STEP
        chain_price = chain[(chain.strike == target_strike) & (chain.option_type == "PE")]["ltp"].iloc[0]
        leg_price = reprice_leg_with_skew("PE", target_strike, SPOT, IV, T, R, Q)
        assert leg_price == pytest.approx(chain_price)

    def test_differs_from_flat_reprice_for_otm_strike(self):
        atm = round(SPOT / STEP) * STEP
        otm_put_strike = atm - 5 * STEP
        flat_price = reprice_leg("PE", otm_put_strike, SPOT, IV, T, R, Q)
        skewed_price = reprice_leg_with_skew("PE", otm_put_strike, SPOT, IV, T, R, Q)
        assert skewed_price != pytest.approx(flat_price)
