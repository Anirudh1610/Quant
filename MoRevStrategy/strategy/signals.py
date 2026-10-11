"""
signals.py
----------
Momentum ranking and mean-reversion entry signals for the
cross-sectional momentum + mean-reversion strategy.

Signal pipeline
---------------
1. Cross-sectional 20-day momentum  → rank stocks daily
2. RSI(2)                           → time entries within ranked baskets
3. Volatility filter                → exclude excessively volatile names
4. NIFTY 50 regime filter           → only take trades aligned with index trend

All computations use *past* data only (no lookahead bias).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# MOMENTUM
# ──────────────────────────────────────────────────────────────────────────────


def calculate_momentum(close: pd.DataFrame, window: int = 20, skip_days: int = 0) -> pd.DataFrame:
    """
    Cross-sectional price momentum.

    Formula
    -------
    momentum[t] = close[t - skip] / close[t - skip - window] - 1

    Using ``pct_change(window)`` ensures that on day ``t`` we only use data
    available at the *close* of day ``t``; no forward-looking data.

    Parameters
    ----------
    close : pd.DataFrame
        Daily adjusted close prices (rows = dates, columns = tickers).
    window : int
        Look-back window in trading days. Default 20 ≈ 1 calendar month.
    skip_days : int
        Recent bars to exclude before the look-back (Jegadeesh–Titman
        convention). ``skip_days=5`` sidesteps the short-term reversal
        effect that inflates naive momentum signals. Default 0 (off).

    Returns
    -------
    pd.DataFrame
        Momentum scores, same shape as ``close``. NaN for the first
        ``window + skip_days`` rows.
    """
    if skip_days > 0:
        return close.shift(skip_days).pct_change(window)
    return close.pct_change(window)


def rank_momentum(momentum: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectionally rank stocks by momentum each day.

    Rank 1 = highest momentum (strongest); rank N = lowest (weakest).
    ``method='first'`` gives unique integer ranks (no ties).

    Parameters
    ----------
    momentum : pd.DataFrame
        Output of :func:`calculate_momentum`.

    Returns
    -------
    pd.DataFrame
        Integer ranks per row (ascending = False → rank 1 is best).
    """
    return momentum.rank(axis=1, ascending=False, method="first")


def get_long_candidates(ranks: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Boolean mask — True where stock rank <= ``top_n`` (strongest momentum).

    Parameters
    ----------
    ranks : pd.DataFrame
        Output of :func:`rank_momentum`.
    top_n : int
        Number of top-momentum stocks to flag as long candidates.

    Returns
    -------
    pd.DataFrame
        Boolean mask, True = long candidate.
    """
    return ranks <= top_n


def get_short_candidates(
    ranks: pd.DataFrame, total_stocks: int, bottom_n: int = 10
) -> pd.DataFrame:
    """
    Boolean mask — True where rank > (total_stocks - bottom_n) (weakest momentum).

    Parameters
    ----------
    ranks : pd.DataFrame
        Output of :func:`rank_momentum`.
    total_stocks : int
        Total number of stocks in the universe (used to infer bottom).
    bottom_n : int
        Number of bottom-momentum stocks to flag as short candidates.

    Returns
    -------
    pd.DataFrame
        Boolean mask, True = short candidate.
    """
    return ranks > (total_stocks - bottom_n)


# ──────────────────────────────────────────────────────────────────────────────
# RSI
# ──────────────────────────────────────────────────────────────────────────────


def calculate_rsi(close: pd.DataFrame, period: int = 2) -> pd.DataFrame:
    """
    Wilder's RSI for each ticker using exponential smoothing.

    Formula
    -------
    delta  = close.diff()
    RS     = ewm_avg_gain / ewm_avg_loss   (alpha = 1/period, Wilder)
    RSI    = 100 - 100 / (1 + RS)

    A period-2 RSI oscillates rapidly, making it ideal for identifying
    very short-term oversold / overbought conditions within a trend.

    Parameters
    ----------
    close : pd.DataFrame
        Daily adjusted close prices.
    period : int
        RSI window. Default 2 for ultra-short-term mean reversion timing.

    Returns
    -------
    pd.DataFrame
        RSI values in [0, 100]. Undefined rows filled with 50 (neutral).
    """
    delta = close.diff()

    # Separate gains and losses; clip ensures no negative values on either side
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder smoothing: exponential moving average with alpha = 1/period
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    # Avoid division by zero (pure up-move → RSI = 100)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Fill undefined with 50 (neutral) to avoid unintended entries at startup
    return rsi.fillna(50.0)


# ──────────────────────────────────────────────────────────────────────────────
# COMBINED ENTRY SIGNALS
# ──────────────────────────────────────────────────────────────────────────────


def get_entry_signals(
    long_candidates: pd.DataFrame,
    short_candidates: pd.DataFrame,
    rsi: pd.DataFrame,
    rsi_long_thresh: float = 10.0,
    rsi_short_thresh: float = 90.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine momentum rank filter with RSI mean-reversion timing.

    Entry logic
    -----------
    LONG  : stock is in top-momentum basket  AND  RSI(2) < rsi_long_thresh
    SHORT : stock is in bottom-momentum basket AND  RSI(2) > rsi_short_thresh

    This ensures we only buy *strong* stocks when they are oversold, and
    only short *weak* stocks when they are overbought — a classic
    momentum-with-pullback approach.

    Parameters
    ----------
    long_candidates : pd.DataFrame
        Boolean mask from :func:`get_long_candidates`.
    short_candidates : pd.DataFrame
        Boolean mask from :func:`get_short_candidates`.
    rsi : pd.DataFrame
        RSI values from :func:`calculate_rsi`.
    rsi_long_thresh : float
        RSI threshold below which longs are triggered. Default 10.
    rsi_short_thresh : float
        RSI threshold above which shorts are triggered. Default 90.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(long_entries, short_entries)`` — Boolean DataFrames.
        A signal on day T means: *execute at the open of day T+1*.
    """
    long_entries = long_candidates & (rsi < rsi_long_thresh)
    short_entries = short_candidates & (rsi > rsi_short_thresh)
    return long_entries, short_entries


# ──────────────────────────────────────────────────────────────────────────────
# VOLATILITY FILTER
# ──────────────────────────────────────────────────────────────────────────────


def calculate_volatility_filter(
    close: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """
    Rolling annualised volatility for each stock.

    Volatility = std(daily_returns, window) × √252

    Used by the portfolio layer to exclude stocks whose volatility exceeds
    a multiple of the daily cross-sectional median (default: 2×).

    Parameters
    ----------
    close : pd.DataFrame
        Daily adjusted close prices.
    window : int
        Rolling window for the standard deviation estimate.

    Returns
    -------
    pd.DataFrame
        Annualised volatility (decimal), same shape as ``close``.
    """
    daily_returns = close.pct_change()
    return daily_returns.rolling(window).std() * np.sqrt(252)


# ──────────────────────────────────────────────────────────────────────────────
# NIFTY REGIME FILTER
# ──────────────────────────────────────────────────────────────────────────────


def nifty_regime_filter(
    nifty_close: pd.Series, ma_window: int = 50
) -> pd.Series:
    """
    Classify market regime using NIFTY 50 vs its N-day simple moving average.

    Rules
    -----
    BULL  (+1) : NIFTY close > MA  → allow long entries, restrict shorts
    BEAR  (-1) : NIFTY close ≤ MA  → allow short entries, restrict longs

    A 50-day MA is a robust, widely-watched trend indicator that filters
    out whipsaw periods and aligns trades with the macro market direction.

    Parameters
    ----------
    nifty_close : pd.Series
        Daily NIFTY 50 index close prices.
    ma_window : int
        Moving average window in trading days. Default 50.

    Returns
    -------
    pd.Series
        Integer Series: +1 (bull) or -1 (bear), indexed by date.
        NaN during the warm-up period is filled with -1 (conservative).
    """
    ma = nifty_close.rolling(ma_window).mean()
    regime = pd.Series(
        np.where(nifty_close > ma, 1, -1),
        index=nifty_close.index,
        name="regime",
    )
    # During warm-up (first 50 bars), MA is NaN → regime = -1 (no longs)
    regime = regime.fillna(-1)
    return regime


# ──────────────────────────────────────────────────────────────────────────────
# ATR
# ──────────────────────────────────────────────────────────────────────────────


def calculate_atr(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """
    Average True Range (ATR) using Wilder's exponential smoothing.

    True Range[t] = max(H[t]-L[t], |H[t]-C[t-1]|, |L[t]-C[t-1]|)
    ATR[t]        = Wilder EWM of True Range  (alpha = 1/period)

    ATR-based stops scale the stop distance to each stock's current
    volatility regime, making entries more adaptive than a fixed percentage.

    Parameters
    ----------
    high, low, close : pd.DataFrame
        OHLC price matrices with matching index and columns.
    period : int
        Smoothing window. Default 14 (Wilder's original setting).

    Returns
    -------
    pd.DataFrame
        ATR in price units, same shape as ``close``.
        NaN for the first ``period`` rows.
    """
    prev_close = close.shift(1)
    tr = pd.DataFrame(
        np.maximum(
            np.maximum(
                (high - low).values,
                np.abs(high.values - prev_close.values),
            ),
            np.abs(low.values - prev_close.values),
        ),
        index=close.index,
        columns=close.columns,
    )
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
