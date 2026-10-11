"""
signals/volatility.py
----------------------
IV Rank / IV Percentile, realized volatility estimators, and the IV-RV
spread. All realized-vol functions return annualized decimal volatility
(e.g. 0.15 = 15%/yr) using 252 trading days/year.

IV Rank and IV Percentile both describe "where is current IV relative to
its own trailing history", but answer different questions:
  - IV Rank:       (current - min) / (max - min) over the window — sensitive
                    to single outlier days (a one-off VIX spike stretches the
                    range and permanently compresses future rank readings).
  - IV Percentile:  fraction of trailing days strictly below the current
                    value — robust to outliers, more of a "how often was IV
                    this low" read.
Both are exposed since practitioners commonly want to see both; there's no
consensus on which is "more correct" for regime classification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def realized_vol_close_to_close(close: pd.Series, window: int = 20) -> pd.Series:
    """Rolling annualized close-to-close realized vol from log returns."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def realized_vol_parkinson(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """
    Parkinson (1980) high-low range estimator. More efficient than
    close-to-close when there's no jumps/drift, but ignores overnight gaps
    (understates vol on gappy names like index options underlyings).
    """
    log_hl2 = np.log(high / low) ** 2
    factor = 1.0 / (4.0 * np.log(2.0))
    variance = factor * log_hl2.rolling(window).mean()
    return np.sqrt(variance * TRADING_DAYS_PER_YEAR)


def realized_vol_yang_zhang(open_: pd.Series, high: pd.Series, low: pd.Series,
                             close: pd.Series, window: int = 20) -> pd.Series:
    """
    Yang-Zhang (2000) estimator — combines overnight (close-to-open),
    open-to-close, and Rogers-Satchell components. Handles both drift and
    overnight jumps, generally the most robust of the three for index
    options where overnight gaps matter.
    """
    log_o_prevc = np.log(open_ / close.shift(1))
    log_c_o = np.log(close / open_)
    log_h_o = np.log(high / open_)
    log_l_o = np.log(low / open_)

    overnight_var = log_o_prevc.rolling(window).var()
    open_close_var = log_c_o.rolling(window).var()
    rs = log_h_o * (log_h_o - log_c_o) + log_l_o * (log_l_o - log_c_o)
    rs_var = rs.rolling(window).mean()

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    variance = overnight_var + k * open_close_var + (1 - k) * rs_var
    return np.sqrt(variance * TRADING_DAYS_PER_YEAR)


def iv_rank(iv: pd.Series, window: int = 252) -> pd.Series:
    """(current - trailing_min) / (trailing_max - trailing_min) * 100, in [0, 100]."""
    roll_min = iv.rolling(window, min_periods=window // 4).min()
    roll_max = iv.rolling(window, min_periods=window // 4).max()
    span = (roll_max - roll_min).replace(0, np.nan)
    return ((iv - roll_min) / span * 100).clip(0, 100)


def iv_percentile(iv: pd.Series, window: int = 252) -> pd.Series:
    """% of trailing-window days with IV strictly below today's IV, in [0, 100]."""
    def _pct_rank(vals: np.ndarray) -> float:
        current = vals[-1]
        return float((vals[:-1] < current).mean() * 100) if len(vals) > 1 else np.nan

    return iv.rolling(window, min_periods=window // 4).apply(_pct_rank, raw=True)


def iv_rv_spread(iv: pd.Series, rv: pd.Series) -> pd.Series:
    """IV minus realized vol, same (decimal, annualized) units. Positive = options rich vs realized."""
    return iv - rv
