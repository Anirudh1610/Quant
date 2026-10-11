from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signals.trend import BEARISH, BULLISH, RANGE_BOUND, adx, atr, classify_trend_regime, supertrend


class TestATR:
    def test_nonnegative(self, trending_up_ohlc):
        a = atr(trending_up_ohlc["high"], trending_up_ohlc["low"], trending_up_ohlc["close"])
        assert (a.dropna() >= 0).all()

    def test_zero_range_gives_zero_atr(self):
        n = 30
        flat = pd.Series([100.0] * n)
        a = atr(flat, flat, flat, period=10)
        assert (a.dropna() == 0).all()


class TestADX:
    def test_bounded_0_100(self, trending_up_ohlc):
        a = adx(trending_up_ohlc["high"], trending_up_ohlc["low"], trending_up_ohlc["close"])
        valid = a.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_strong_uptrend_has_high_adx_eventually(self, trending_up_ohlc):
        a = adx(trending_up_ohlc["high"], trending_up_ohlc["low"], trending_up_ohlc["close"], period=14)
        # a persistent strong drift should push ADX well above the "trending" threshold
        # by the end of a 300-day series, once the indicator has warmed up
        assert a.iloc[-1] > 25


class TestSupertrend:
    def test_direction_is_plus_or_minus_one(self, choppy_ohlc):
        st = supertrend(choppy_ohlc["high"], choppy_ohlc["low"], choppy_ohlc["close"])
        valid = st["direction"][st["direction"] != 0]
        assert set(valid.unique()) <= {1, -1}

    def test_line_below_close_when_uptrend(self, trending_up_ohlc):
        st = supertrend(trending_up_ohlc["high"], trending_up_ohlc["low"], trending_up_ohlc["close"])
        close = trending_up_ohlc["close"]
        up = st["direction"] == 1
        valid = up & st["supertrend"].notna()
        # in an uptrend the trailing stop sits at/below price (support), by construction
        assert (st.loc[valid, "supertrend"] <= close.loc[valid] + 1e-6).all()

    def test_line_above_close_when_downtrend(self, trending_down_ohlc):
        st = supertrend(trending_down_ohlc["high"], trending_down_ohlc["low"], trending_down_ohlc["close"])
        close = trending_down_ohlc["close"]
        down = st["direction"] == -1
        valid = down & st["supertrend"].notna()
        assert (st.loc[valid, "supertrend"] >= close.loc[valid] - 1e-6).all()


class TestClassifyTrendRegime:
    def test_strong_uptrend_classified_bullish_at_end(self, trending_up_ohlc):
        result = classify_trend_regime(trending_up_ohlc["high"], trending_up_ohlc["low"], trending_up_ohlc["close"])
        assert result["trend"].iloc[-1] == BULLISH

    def test_strong_downtrend_classified_bearish_at_end(self, trending_down_ohlc):
        result = classify_trend_regime(trending_down_ohlc["high"], trending_down_ohlc["low"], trending_down_ohlc["close"])
        assert result["trend"].iloc[-1] == BEARISH

    def test_low_adx_forces_range_bound_regardless_of_supertrend(self):
        # construct a case where ADX is (by threshold) forced low and confirm
        # RANGE_BOUND wins even if raw Supertrend direction is bullish
        n = 60
        close = pd.Series(100 + np.sin(np.linspace(0, 1, n)) * 0.01)  # near-flat, negligible trend
        high = close + 0.05
        low = close - 0.05
        result = classify_trend_regime(high, low, close, adx_range_bound_threshold=99999)
        assert (result["trend"] == RANGE_BOUND).all()

    def test_output_columns(self, choppy_ohlc):
        result = classify_trend_regime(choppy_ohlc["high"], choppy_ohlc["low"], choppy_ohlc["close"])
        assert set(result.columns) == {"trend", "adx", "st_direction"}
        assert set(result["trend"].dropna().unique()) <= {BULLISH, BEARISH, RANGE_BOUND}
