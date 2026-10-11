from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signals.volatility import (
    iv_percentile,
    iv_rank,
    iv_rv_spread,
    realized_vol_close_to_close,
    realized_vol_parkinson,
    realized_vol_yang_zhang,
)


class TestRealizedVolCloseToClose:
    def test_flat_price_gives_zero_vol(self):
        close = pd.Series([100.0] * 30)
        rv = realized_vol_close_to_close(close, window=10)
        assert (rv.dropna() == 0).all()

    def test_matches_manual_calculation(self):
        # Two alternating daily returns of +1% / -1% (approx, via log returns)
        # over a short deterministic series -- hand-computable rolling std.
        closes = [100.0]
        for i in range(9):
            closes.append(closes[-1] * (1.01 if i % 2 == 0 else 0.99))
        close = pd.Series(closes)
        log_ret = np.log(close / close.shift(1))
        expected = log_ret.rolling(4).std() * np.sqrt(252)
        rv = realized_vol_close_to_close(close, window=4)
        pd.testing.assert_series_equal(rv, expected, check_names=False)


class TestRealizedVolParkinson:
    def test_zero_range_gives_zero_vol(self):
        high = pd.Series([100.0] * 30)
        low = pd.Series([100.0] * 30)
        rv = realized_vol_parkinson(high, low, window=10)
        assert (rv.dropna() == 0).all()

    def test_matches_analytic_constant_range(self):
        # constant high/low ratio every day -> rolling mean of log(h/l)^2 is
        # just that constant, so the estimator's value is exactly solvable.
        n = 20
        ratio = 1.02  # high = 1.02 * low every day
        low = pd.Series(np.full(n, 100.0))
        high = low * ratio
        window = 10
        rv = realized_vol_parkinson(high, low, window=window)

        c = np.log(ratio)
        expected_var = (1.0 / (4 * np.log(2))) * c ** 2
        expected_rv = np.sqrt(expected_var * 252)
        assert rv.dropna().to_numpy() == pytest.approx(expected_rv)


class TestRealizedVolYangZhang:
    def test_flat_price_gives_zero_vol(self):
        n = 30
        flat = pd.Series([100.0] * n)
        rv = realized_vol_yang_zhang(flat, flat, flat, flat, window=10)
        assert (rv.dropna() == 0).all()

    def test_nonzero_for_moving_prices(self):
        n = 40
        rng = np.random.default_rng(7)
        close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))))
        open_ = close.shift(1).fillna(100.0) * (1 + rng.normal(0, 0.002, n))
        high = pd.concat([open_, close], axis=1).max(axis=1) * 1.003
        low = pd.concat([open_, close], axis=1).min(axis=1) * 0.997
        rv = realized_vol_yang_zhang(open_, high, low, close, window=10)
        assert (rv.dropna() > 0).all()


class TestIVRank:
    @pytest.fixture
    def iv_series(self) -> pd.Series:
        return pd.Series([10, 20, 15, 25, 12, 30, 18, 22, 16, 28], dtype=float)

    def test_zero_span_window_is_nan(self, iv_series):
        rank = iv_rank(iv_series, window=5)
        assert np.isnan(rank.iloc[0])  # single-value window: min==max==span 0

    @pytest.mark.parametrize("idx,expected", [
        (1, 100.0),
        (2, 50.0),
        (3, 100.0),
        (4, 100.0 * (12 - 10) / 15),
        (5, 100.0),
        (6, 100.0 * (18 - 12) / 18),
        (7, 100.0 * (22 - 12) / 18),
        (8, 100.0 * (16 - 12) / 18),
        (9, 100.0 * (28 - 16) / 14),
    ])
    def test_matches_hand_computed_values(self, iv_series, idx, expected):
        rank = iv_rank(iv_series, window=5)
        assert rank.iloc[idx] == pytest.approx(expected)

    def test_bounded_0_100(self, iv_series):
        rank = iv_rank(iv_series, window=5)
        valid = rank.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()


class TestIVPercentile:
    @pytest.fixture
    def iv_series(self) -> pd.Series:
        return pd.Series([10, 20, 15, 25, 12, 30, 18, 22, 16, 28], dtype=float)

    def test_single_value_window_is_nan(self, iv_series):
        pct = iv_percentile(iv_series, window=5)
        assert np.isnan(pct.iloc[0])

    @pytest.mark.parametrize("idx,expected", [
        (1, 100.0),
        (2, 50.0),
        (3, 100.0),
        (4, 25.0),
        (5, 100.0),
        (6, 50.0),
        (7, 50.0),
        (8, 25.0),
        (9, 75.0),
    ])
    def test_matches_hand_computed_values(self, iv_series, idx, expected):
        pct = iv_percentile(iv_series, window=5)
        assert pct.iloc[idx] == pytest.approx(expected)


class TestIVRVSpread:
    def test_simple_subtraction(self):
        iv = pd.Series([0.20, 0.25, 0.30])
        rv = pd.Series([0.15, 0.15, 0.35])
        spread = iv_rv_spread(iv, rv)
        pd.testing.assert_series_equal(spread, pd.Series([0.05, 0.10, -0.05]), check_names=False)
