"""
signals/trend.py
-----------------
Trend/regime classification from Supertrend (direction) + ADX (strength).

Supertrend gives a directional flip-flop (bullish/bearish) but no sense of
whether that direction is a strong trend or a weak chop that happens to be
tilted one way. ADX filters that: below `adx_range_bound_threshold`, we
call the regime RANGE_BOUND regardless of what Supertrend says, since a
low-ADX "trend" is directionless enough that selling premium against it is
reasonable (this is what feeds the OI-based range_score and, later, the
regime-to-strategy mapping in signals/regime.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import SIGNALS

RANGE_BOUND = "RANGE_BOUND"
BULLISH = "BULLISH"
BEARISH = "BEARISH"


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's ADX(period)."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    smoothed_tr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / smoothed_tr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / smoothed_tr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Standard Supertrend. Returns a DataFrame with columns:
      supertrend : the trailing stop line
      direction  : +1 (uptrend / price above line) or -1 (downtrend)

    Path-dependent (each bar's band depends on the prior bar's band), so
    this is computed iteratively rather than vectorized.
    """
    n = len(close)
    atr_ = atr(high, low, close, period)
    hl2 = (high + low) / 2
    basic_upper = hl2 + multiplier * atr_
    basic_lower = hl2 - multiplier * atr_

    final_upper = pd.Series(np.nan, index=close.index)
    final_lower = pd.Series(np.nan, index=close.index)
    st = pd.Series(np.nan, index=close.index)
    direction = pd.Series(0, index=close.index, dtype=int)

    for i in range(n):
        if i == 0 or np.isnan(atr_.iloc[i]):
            continue

        if np.isnan(final_upper.iloc[i - 1]):
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = (
                basic_upper.iloc[i]
                if (basic_upper.iloc[i] < final_upper.iloc[i - 1] or close.iloc[i - 1] > final_upper.iloc[i - 1])
                else final_upper.iloc[i - 1]
            )

        if np.isnan(final_lower.iloc[i - 1]):
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = (
                basic_lower.iloc[i]
                if (basic_lower.iloc[i] > final_lower.iloc[i - 1] or close.iloc[i - 1] < final_lower.iloc[i - 1])
                else final_lower.iloc[i - 1]
            )

        prev_st = st.iloc[i - 1]
        if np.isnan(prev_st):
            st.iloc[i] = final_lower.iloc[i]
            direction.iloc[i] = 1
        elif prev_st == final_upper.iloc[i - 1]:
            if close.iloc[i] <= final_upper.iloc[i]:
                st.iloc[i] = final_upper.iloc[i]
                direction.iloc[i] = -1
            else:
                st.iloc[i] = final_lower.iloc[i]
                direction.iloc[i] = 1
        else:  # prev_st == final_lower.iloc[i - 1]
            if close.iloc[i] >= final_lower.iloc[i]:
                st.iloc[i] = final_lower.iloc[i]
                direction.iloc[i] = 1
            else:
                st.iloc[i] = final_upper.iloc[i]
                direction.iloc[i] = -1

    return pd.DataFrame({"supertrend": st, "direction": direction})


def classify_trend_regime(high: pd.Series, low: pd.Series, close: pd.Series,
                           supertrend_period: int = SIGNALS.supertrend_period,
                           supertrend_multiplier: float = SIGNALS.supertrend_multiplier,
                           adx_period: int = SIGNALS.adx_period,
                           adx_range_bound_threshold: float = SIGNALS.adx_range_bound_threshold) -> pd.DataFrame:
    """
    Returns a DataFrame indexed like `close` with columns:
      trend          : "BULLISH" | "BEARISH" | "RANGE_BOUND"
      adx            : ADX(adx_period) value
      st_direction   : raw Supertrend direction (+1/-1), pre-ADX-filter
    """
    st = supertrend(high, low, close, supertrend_period, supertrend_multiplier)
    adx_ = adx(high, low, close, adx_period)

    trend = pd.Series(RANGE_BOUND, index=close.index)
    trend[(adx_ >= adx_range_bound_threshold) & (st["direction"] == 1)] = BULLISH
    trend[(adx_ >= adx_range_bound_threshold) & (st["direction"] == -1)] = BEARISH

    return pd.DataFrame({
        "trend": trend,
        "adx": adx_,
        "st_direction": st["direction"],
    })
