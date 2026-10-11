from __future__ import annotations

import pandas as pd
import pytest

from signals.oi import max_pain, oi_concentration_ratio, put_call_ratio, range_bound_conviction_score


def _chain(rows: list[tuple[float, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["strike", "option_type", "oi"])


class TestMaxPain:
    def test_all_oi_at_one_strike_is_max_pain(self):
        chain = _chain([
            (95, "CE", 0), (95, "PE", 0),
            (100, "CE", 100), (100, "PE", 100),
            (105, "CE", 0), (105, "PE", 0),
        ])
        assert max_pain(chain) == pytest.approx(100)

    def test_call_only_oi_max_pain_at_or_below_strike(self):
        # only call OI at strike 100 -> writer loss is zero for any expiry
        # <= 100, so the lowest listed strike (100) minimizes loss.
        chain = _chain([(100, "CE", 200), (100, "PE", 0), (110, "CE", 0), (110, "PE", 0)])
        assert max_pain(chain) == pytest.approx(100)

    def test_asymmetric_oi_shifts_max_pain_toward_heavier_side(self):
        # heavy put OI far above spot should NOT pull max pain toward it as much
        # as heavy call OI at the same strike would pull it down; verify by
        # comparing two mirrored scenarios against direct computation.
        chain_calls_heavy = _chain([
            (100, "CE", 500), (100, "PE", 10),
            (110, "CE", 10), (110, "PE", 10),
        ])
        chain_puts_heavy = _chain([
            (100, "CE", 10), (100, "PE", 10),
            (110, "CE", 10), (110, "PE", 500),
        ])
        assert max_pain(chain_calls_heavy) == pytest.approx(100)
        assert max_pain(chain_puts_heavy) == pytest.approx(110)


class TestPutCallRatio:
    def test_simple_ratio(self):
        chain = _chain([(100, "CE", 300), (100, "PE", 600)])
        assert put_call_ratio(chain) == pytest.approx(2.0)

    def test_zero_call_oi_returns_nan(self):
        chain = _chain([(100, "CE", 0), (100, "PE", 100)])
        assert pd.isna(put_call_ratio(chain))


class TestOIConcentrationRatio:
    def test_matches_hand_computation(self):
        chain = _chain([
            (21800, "CE", 50), (21800, "PE", 0),
            (21900, "CE", 100), (21900, "PE", 50),
            (22000, "CE", 200), (22000, "PE", 200),
            (22100, "CE", 150), (22100, "PE", 100),
            (22200, "CE", 0), (22200, "PE", 80),
        ])
        ratio = oi_concentration_ratio(chain, spot=22000, strike_step=50, n_strikes=2)
        expected = 800 / 930
        assert ratio == pytest.approx(expected)

    def test_zero_total_oi_returns_nan(self):
        chain = _chain([(100, "CE", 0), (100, "PE", 0)])
        assert pd.isna(oi_concentration_ratio(chain, spot=100, strike_step=50, n_strikes=2))

    def test_all_oi_near_spot_gives_ratio_one(self):
        chain = _chain([(22000, "CE", 100), (22000, "PE", 100)])
        ratio = oi_concentration_ratio(chain, spot=22000, strike_step=50, n_strikes=2)
        assert ratio == pytest.approx(1.0)


class TestRangeBoundConvictionScore:
    def test_matches_hand_computation_high_conviction(self):
        score = range_bound_conviction_score(
            spot=22000, max_pain_strike=22000, pcr=1.0,
            concentration_ratio=0.8, strike_step=50,
        )
        assert score == pytest.approx(100 * (1.0 + 1.0 + 0.8) / 3.0)

    def test_matches_hand_computation_far_from_max_pain(self):
        score = range_bound_conviction_score(
            spot=22000, max_pain_strike=22000 * 1.05, pcr=1.0,
            concentration_ratio=0.8, strike_step=50,
        )
        assert score == pytest.approx(100 * (0.0 + 1.0 + 0.8) / 3.0)

    def test_bounded_0_100(self):
        for mp_mult in (0.8, 0.95, 1.0, 1.05, 1.2):
            for pcr in (0.1, 1.0, 5.0):
                score = range_bound_conviction_score(
                    spot=22000, max_pain_strike=22000 * mp_mult, pcr=pcr,
                    concentration_ratio=0.5, strike_step=50,
                )
                assert 0 <= score <= 100
