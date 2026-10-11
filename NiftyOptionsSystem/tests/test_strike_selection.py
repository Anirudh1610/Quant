from __future__ import annotations

import pandas as pd
import pytest

from config import StrikeSelectionConfig
from core.pricing import bs_price
from strikes.selection import (
    compute_chain_greeks,
    expand_wall_band,
    find_oi_walls,
    select_long_strike,
    select_short_strike,
)

T, R = 6 / 365, 0.068
FORWARD = 24339.5
TRUE_SIGMA = 0.13
STRIKE_STEP = 50


def _synthetic_chain(strikes: list[float], oi_by_strike: dict[float, float] | None = None) -> pd.DataFrame:
    """Build a chain priced consistently off FORWARD/TRUE_SIGMA via bs_price, so
    Greeks/prob-of-touch computed on it are internally consistent and testable."""
    oi_by_strike = oi_by_strike or {}
    rows = []
    for k in strikes:
        for opt_type, bs_type in (("CE", "call"), ("PE", "put")):
            price = bs_price(FORWARD, k, T, R, TRUE_SIGMA, R, bs_type)
            rows.append({
                "strike": k, "option_type": opt_type,
                "ltp": price, "bid": price - 0.25, "ask": price + 0.25,
                "oi": oi_by_strike.get(k, 1000.0), "volume": 5000.0,
                "iv": TRUE_SIGMA, "underlying_ltp": FORWARD,
            })
    return pd.DataFrame(rows)


class TestComputeChainGreeks:
    def test_adds_expected_columns(self):
        chain = _synthetic_chain([24200, 24300, 24400])
        out = compute_chain_greeks(chain, FORWARD, T, R)
        assert {"delta", "gamma", "theta", "vega"} <= set(out.columns)

    def test_call_delta_positive_put_delta_negative(self):
        chain = _synthetic_chain([24200, 24300, 24400])
        out = compute_chain_greeks(chain, FORWARD, T, R)
        assert (out.loc[out.option_type == "CE", "delta"] > 0).all()
        assert (out.loc[out.option_type == "PE", "delta"] < 0).all()

    def test_missing_iv_gives_nan_greeks_not_a_crash(self):
        chain = _synthetic_chain([24300])
        chain.loc[chain.index[0], "iv"] = None
        out = compute_chain_greeks(chain, FORWARD, T, R)
        assert pd.isna(out.loc[out.index[0], "delta"])


class TestFindOIWalls:
    # 21 strikes (23800..24800 step 50) -- enough for the default local_window=5
    # (11-strike span) to have full neighborhoods away from the chain edges.
    _WALL_TEST_STRIKES = list(range(23800, 24850, 50))

    def test_detects_local_spike_not_whole_chain_median(self):
        oi = {k: 1000.0 for k in self._WALL_TEST_STRIKES}
        oi[24400] = 20_000.0  # 20x its local neighborhood -> a wall
        chain = _synthetic_chain(self._WALL_TEST_STRIKES, oi_by_strike=oi)
        walls = find_oi_walls(chain, "CE", multiple=3.0, local_window=5)
        assert walls == {24400.0}

    def test_neighbors_of_a_spike_are_not_flagged(self):
        # regression: a naive whole-chain-median comparison would flag the
        # entire liquid region once one strike spikes hard enough to drag
        # the chain median up -- local comparison should not do that.
        oi = {k: 1000.0 for k in self._WALL_TEST_STRIKES}
        oi[24400] = 20_000.0
        chain = _synthetic_chain(self._WALL_TEST_STRIKES, oi_by_strike=oi)
        walls = find_oi_walls(chain, "CE", multiple=3.0, local_window=5)
        assert 24350.0 not in walls
        assert 24450.0 not in walls

    def test_no_walls_when_oi_uniform(self):
        oi = {k: 1000.0 for k in self._WALL_TEST_STRIKES}
        chain = _synthetic_chain(self._WALL_TEST_STRIKES, oi_by_strike=oi)
        walls = find_oi_walls(chain, "CE", multiple=3.0, local_window=5)
        assert walls == set()

    def test_realistic_decaying_oi_away_from_spot_does_not_blanket_flag(self):
        # OI naturally decays away from spot in a real chain (verified live);
        # that alone must not trip the wall filter across the board.
        oi = {k: max(1.0, 5000.0 - abs(k - 24300) * 8) for k in self._WALL_TEST_STRIKES}
        chain = _synthetic_chain(self._WALL_TEST_STRIKES, oi_by_strike=oi)
        walls = find_oi_walls(chain, "CE", multiple=3.0, local_window=5)
        assert len(walls) == 0

    def test_empty_chain_returns_empty_set(self):
        chain = pd.DataFrame(columns=["strike", "option_type", "oi"])
        assert find_oi_walls(chain, "CE") == set()


class TestExpandWallBand:
    def test_expands_by_band(self):
        walls = {24400.0}
        expanded = expand_wall_band(walls, strike_step=50, band=1)
        assert expanded == {24350.0, 24400.0, 24450.0}

    def test_band_zero_is_just_the_wall(self):
        walls = {24400.0}
        assert expand_wall_band(walls, strike_step=50, band=0) == {24400.0}


class TestSelectShortStrike:
    def test_picks_strike_closest_to_target_delta(self):
        strikes = list(range(23800, 24900, 50))
        chain = _synthetic_chain(strikes)
        greeked = compute_chain_greeks(chain, FORWARD, T, R)
        cfg = StrikeSelectionConfig(target_delta=0.175, max_probability_of_touch=1.0,
                                     oi_wall_multiple=999.0, oi_wall_band_strikes=0)

        chosen = select_short_strike(greeked, "PE", FORWARD, T, STRIKE_STEP, cfg)
        assert chosen is not None

        # independently recompute the expected best strike (closest |delta| to target)
        puts = greeked[(greeked.option_type == "PE") & (greeked.strike < FORWARD)].copy()
        puts["diff"] = (puts["delta"].abs() - cfg.target_delta).abs()
        expected_strike = puts.sort_values("diff").iloc[0]["strike"]
        assert chosen["strike"] == pytest.approx(expected_strike)

        # sanity: delta is reasonably close to target (within the discreteness of a 50pt grid)
        assert abs(abs(chosen["delta"]) - 0.175) < 0.05

    def test_only_considers_otm_strikes(self):
        strikes = list(range(23800, 24900, 50))
        chain = _synthetic_chain(strikes)
        greeked = compute_chain_greeks(chain, FORWARD, T, R)
        cfg = StrikeSelectionConfig(max_probability_of_touch=1.0, oi_wall_multiple=999.0, oi_wall_band_strikes=0)

        put_choice = select_short_strike(greeked, "PE", FORWARD, T, STRIKE_STEP, cfg)
        call_choice = select_short_strike(greeked, "CE", FORWARD, T, STRIKE_STEP, cfg)
        assert put_choice["strike"] < FORWARD
        assert call_choice["strike"] > FORWARD

    def test_probability_of_touch_filter_excludes_near_strikes(self):
        strikes = list(range(23800, 24900, 50))
        chain = _synthetic_chain(strikes)
        greeked = compute_chain_greeks(chain, FORWARD, T, R)

        # an extremely tight probability-of-touch threshold should reject
        # everything close to the money, forcing selection further out
        # (or None if nothing survives)
        strict_cfg = StrikeSelectionConfig(target_delta=0.175, max_probability_of_touch=0.01,
                                            oi_wall_multiple=999.0, oi_wall_band_strikes=0)
        loose_cfg = StrikeSelectionConfig(target_delta=0.175, max_probability_of_touch=1.0,
                                           oi_wall_multiple=999.0, oi_wall_band_strikes=0)

        strict_choice = select_short_strike(greeked, "PE", FORWARD, T, STRIKE_STEP, strict_cfg)
        loose_choice = select_short_strike(greeked, "PE", FORWARD, T, STRIKE_STEP, loose_cfg)
        if strict_choice is not None:
            assert strict_choice["strike"] <= loose_choice["strike"]  # forced further OTM (or equal/None)

    def test_oi_wall_avoidance_skips_wall_strike(self):
        strikes = list(range(23800, 24900, 50))
        oi = {k: 1000.0 for k in strikes}
        # put a wall exactly at whatever strike would otherwise be chosen
        loose_cfg = StrikeSelectionConfig(target_delta=0.175, max_probability_of_touch=1.0,
                                           oi_wall_multiple=999.0, oi_wall_band_strikes=0)
        chain = _synthetic_chain(strikes, oi_by_strike=oi)
        greeked = compute_chain_greeks(chain, FORWARD, T, R)
        baseline = select_short_strike(greeked, "PE", FORWARD, T, STRIKE_STEP, loose_cfg)
        target_strike = baseline["strike"]

        oi_with_wall = dict(oi)
        oi_with_wall[target_strike] = 10_000.0  # 10x median -> wall
        chain_wall = _synthetic_chain(strikes, oi_by_strike=oi_with_wall)
        greeked_wall = compute_chain_greeks(chain_wall, FORWARD, T, R)
        wall_cfg = StrikeSelectionConfig(target_delta=0.175, max_probability_of_touch=1.0,
                                          oi_wall_multiple=3.0, oi_wall_band_strikes=0)
        with_wall_choice = select_short_strike(greeked_wall, "PE", FORWARD, T, STRIKE_STEP, wall_cfg)

        assert with_wall_choice is not None
        assert with_wall_choice["strike"] != target_strike

    def test_returns_none_when_nothing_survives_filtering(self):
        strikes = [24300, 24350]  # deep in the money-adjacent, essentially guaranteed high touch prob
        chain = _synthetic_chain(strikes)
        greeked = compute_chain_greeks(chain, FORWARD, T, R)
        impossible_cfg = StrikeSelectionConfig(max_probability_of_touch=0.0)
        assert select_short_strike(greeked, "PE", FORWARD, T, STRIKE_STEP, impossible_cfg) is None


class TestSelectLongStrike:
    def test_picks_strike_width_steps_further_otm(self):
        strikes = list(range(23800, 24900, 50))
        chain = _synthetic_chain(strikes)
        greeked = compute_chain_greeks(chain, FORWARD, T, R)
        cfg = StrikeSelectionConfig(spread_width_strikes=4)

        long_put = select_long_strike(greeked, "PE", short_strike=24300, strike_step=50, cfg=cfg)
        assert long_put["strike"] == pytest.approx(24100)  # 4 * 50 below 24300

        long_call = select_long_strike(greeked, "CE", short_strike=24400, strike_step=50, cfg=cfg)
        assert long_call["strike"] == pytest.approx(24600)  # 4 * 50 above 24400

    def test_returns_none_if_target_strike_not_listed(self):
        strikes = [24300, 24350]  # too narrow a chain to have a strike 200pts away
        chain = _synthetic_chain(strikes)
        greeked = compute_chain_greeks(chain, FORWARD, T, R)
        cfg = StrikeSelectionConfig(spread_width_strikes=4)
        assert select_long_strike(greeked, "PE", short_strike=24300, strike_step=50, cfg=cfg) is None
