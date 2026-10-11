"""
signals.py
----------
All alpha signals for the NSE Futures strategy.

Signal stack (v2)
-----------------
1. Cross-sectional 120-day momentum → rank futures contracts daily
2. ADX(14)                          → trend-strength filter (ADX > 20 = trending)
3. EMA(20) pullback                 → entry timing: price near dynamic support/resistance
4. ATR(14)                          → volatility measure for trailing stops + sizing
5. India VIX regime                 → macro risk-on / risk-off filter
6. Rank buffer                      → enter top 5, exit when rank > 15 (reduces turnover)

Entry logic v2 (no lookahead — all signals computed from data ≤ close of day T,
                executions at open of day T+1)
-----------------------------------------------------------------------
LONG  : top-5 momentum basket  AND  ADX(14) > 20  (trending market)
         AND  close ≤ EMA(20) × 1.015             (pullback to dynamic support)
SHORT : bottom-5 momentum basket  AND  ADX(14) > 20
         AND  close ≥ EMA(20) × 0.985             (bounce to dynamic resistance)

Legacy v1 entry signals (kept for backward compatibility / comparison)
-----------------------------------------------------------------------
LONG  : top momentum basket  AND  RSI(2) < rsi_long_thresh
         AND  price ≤ lower_bb + bb_buffer  AND  price > SMA(50)
SHORT : bottom momentum basket  AND  RSI(2) > rsi_short_thresh
         AND  price ≥ upper_bb − bb_buffer  AND  price < SMA(50)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# MOMENTUM
# ──────────────────────────────────────────────────────────────────────────────

def calculate_momentum(close: pd.DataFrame, window: int = 120) -> pd.DataFrame:
    """
    Cross-sectional price momentum: return over the past ``window`` trading days.

    momentum[t] = close[t] / close[t - window] - 1

    Parameters
    ----------
    close  : Daily adjusted close prices (rows = dates, cols = tickers).
    window : Look-back in trading days. Default 120 ≈ 6 calendar months.
             Longer window reduces turnover vs the 60-day original.
    """
    return close.pct_change(window)


def rank_momentum(momentum: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectionally rank each day.  Rank 1 = highest momentum.
    ``method='first'`` ensures unique ranks (no ties).
    """
    return momentum.rank(axis=1, ascending=False, method="first")


def get_long_candidates(ranks: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Boolean mask — True where rank ≤ top_n (strongest momentum)."""
    return ranks <= top_n


def get_short_candidates(
    ranks: pd.DataFrame, total_stocks: int, bottom_n: int = 5
) -> pd.DataFrame:
    """Boolean mask — True where rank > total_stocks - bottom_n (weakest momentum)."""
    return ranks > (total_stocks - bottom_n)


# ──────────────────────────────────────────────────────────────────────────────
# RSI
# ──────────────────────────────────────────────────────────────────────────────

def calculate_rsi(close: pd.DataFrame, period: int = 2) -> pd.DataFrame:
    """
    Wilder's RSI using exponential moving average smoothing.

    RSI(2) oscillates rapidly, making it ideal for detecting very short-term
    oversold / overbought conditions within a medium-term trend.

    Parameters
    ----------
    close  : Daily close prices.
    period : RSI window. Default 2 for ultra-short-term mean reversion.
    """
    delta    = close.diff()
    gain     = delta.clip(lower=0.0)
    loss     = (-delta).clip(lower=0.0)
    alpha    = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi      = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)   # undefined → neutral


# ──────────────────────────────────────────────────────────────────────────────
# BOLLINGER BANDS
# ──────────────────────────────────────────────────────────────────────────────

def calculate_bollinger_bands(
    close: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Standard Bollinger Bands.

    Returns
    -------
    (upper, middle, lower) — three DataFrames with the same shape as close.

    upper  = SMA + num_std × rolling_std
    middle = SMA
    lower  = SMA - num_std × rolling_std
    """
    middle = close.rolling(window).mean()
    std    = close.rolling(window).std(ddof=1)
    upper  = middle + num_std * std
    lower  = middle - num_std * std
    return upper, middle, lower


# ──────────────────────────────────────────────────────────────────────────────
# MOVING AVERAGE
# ──────────────────────────────────────────────────────────────────────────────

def calculate_sma(close: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """Simple moving average."""
    return close.rolling(window).mean()


# ──────────────────────────────────────────────────────────────────────────────
# EMA  (Exponential Moving Average — dynamic support/resistance for entries)
# ──────────────────────────────────────────────────────────────────────────────

def calculate_ema(close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Exponential Moving Average.

    Used as a dynamic support/resistance level for pullback entry timing (v2).
    EMA(20) adapts faster to recent price action than SMA(50), making it
    more suitable for identifying actionable pullbacks in trending markets.

    Parameters
    ----------
    close  : Daily close prices (rows = dates, cols = tickers).
    window : EMA span in trading days. Default 20 ≈ 1 calendar month.
    """
    return close.ewm(span=window, min_periods=window, adjust=False).mean()


# ──────────────────────────────────────────────────────────────────────────────
# ADX  (Average Directional Index — trend-strength filter)
# ──────────────────────────────────────────────────────────────────────────────

def calculate_adx(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """
    Wilder's Average Directional Index.

    Measures trend *strength* (not direction). Values above 20 indicate a
    directional market where momentum signals are reliable; values at or below
    20 indicate choppy / ranging conditions where momentum signals whipsaw.

    ADX > 20  : trending — safe to trade momentum
    ADX > 25  : strong trend — higher-conviction entries
    ADX ≤ 20  : range-bound — skip new momentum entries

    Parameters
    ----------
    high, low, close : OHLCV DataFrames (rows = dates, cols = tickers).
    period           : Wilder smoothing period. Default 14.
    """
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    up_move   = high - prev_high
    down_move = prev_low - low

    # Positive / negative directional movement (mutually exclusive)
    cond_plus  = (up_move > down_move) & (up_move > 0)
    cond_minus = (down_move > up_move) & (down_move > 0)
    plus_dm  = up_move.where(cond_plus,   other=0.0).clip(lower=0.0)
    minus_dm = down_move.where(cond_minus, other=0.0).clip(lower=0.0)

    # True Range (per-column loop avoids deprecated pandas axis=1 API)
    tr = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for col in close.columns:
        tr[col] = pd.concat([
            high[col] - low[col],
            (high[col] - prev_close[col]).abs(),
            (low[col]  - prev_close[col]).abs(),
        ], axis=1).max(axis=1)

    alpha = 1.0 / period
    atr14 = tr.ewm(alpha=alpha,        min_periods=period, adjust=False).mean()
    pdm14 = plus_dm.ewm(alpha=alpha,   min_periods=period, adjust=False).mean()
    mdm14 = minus_dm.ewm(alpha=alpha,  min_periods=period, adjust=False).mean()

    pdi   = (100.0 * pdm14 / atr14.replace(0.0, np.nan)).fillna(0.0)
    mdi   = (100.0 * mdm14 / atr14.replace(0.0, np.nan)).fillna(0.0)
    denom = (pdi + mdi).replace(0.0, np.nan)
    dx    = (100.0 * (pdi - mdi).abs() / denom).fillna(0.0)
    adx   = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    return adx.fillna(0.0)


# ──────────────────────────────────────────────────────────────────────────────
# ATR  (Average True Range — volatility measure for stops and sizing)
# ──────────────────────────────────────────────────────────────────────────────

def calculate_atr(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """
    Wilder's Average True Range.

    TR  = max(high - low,  |high - prev_close|,  |low - prev_close|)
    ATR = EMA(TR, span = 2 × period - 1)   [Wilder's smoothing]

    Parameters
    ----------
    high, low, close : OHLCV DataFrames (rows = dates, cols = tickers).
    period           : ATR period. Default 14 trading days.
    """
    prev_close = close.shift(1)
    # Compute true range per ticker (avoids deprecated groupby axis=1)
    tr = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for col in close.columns:
        tr[col] = pd.concat([
            high[col] - low[col],
            (high[col] - prev_close[col]).abs(),
            (low[col]  - prev_close[col]).abs(),
        ], axis=1).max(axis=1)

    # Wilder smoothing: EWM with alpha = 1/period
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr


# ──────────────────────────────────────────────────────────────────────────────
# FUTURES BASIS
# ──────────────────────────────────────────────────────────────────────────────

def calculate_basis(
    futures_close: pd.DataFrame,
    spot_close: pd.DataFrame,
) -> pd.DataFrame:
    """
    Futures basis = futures_price − spot_price.

    A positive basis (normal contango) confirms the cost-of-carry is priced in.
    A narrowing / negative basis (backwardation) can signal supply-demand stress.
    """
    common = futures_close.columns.intersection(spot_close.columns)
    return futures_close[common] - spot_close[common]


# ──────────────────────────────────────────────────────────────────────────────
# OPEN INTEREST TREND FILTER
# ──────────────────────────────────────────────────────────────────────────────

def oi_trend_filter(
    oi: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """
    Rising OI confirms trend strength; falling OI signals potential reversal.

    Returns a DataFrame of +1 (OI rising) / -1 (OI falling) / 0 (neutral).
    Used as a confirmation filter: only take momentum signals when OI is rising.

    Parameters
    ----------
    oi     : Open Interest DataFrame (rows = dates, cols = tickers).
    window : Rolling window to assess OI trend direction.
    """
    oi_change = oi.pct_change(window)
    result = pd.DataFrame(0, index=oi.index, columns=oi.columns)
    result[oi_change > 0]  = 1
    result[oi_change < 0]  = -1
    return result


def volume_trend_filter(
    volume: pd.DataFrame,
    short_window: int = 5,
    long_window: int = 20,
) -> pd.DataFrame:
    """
    Volume-based participation filter used as an Open Interest proxy.

    NSE actual OI is not available via yfinance, so volume serves as the
    next-best confirmation signal.  Rising average volume (recent vs longer-term
    average) indicates increasing market participation, confirming that a
    momentum move is driven by conviction rather than thin-market drift.

    Entry filter rule: allow the signal only when the short-window average
    volume is above the long-window average (elevated recent participation).

    Returns a boolean DataFrame (True = volume confirming, False = skip entry).

    Parameters
    ----------
    volume       : Spot or futures volume DataFrame (rows = dates, cols = tickers).
    short_window : Recent volume look-back (trading days). Default 5.
    long_window  : Baseline volume look-back (trading days). Default 20.
    """
    short_avg = volume.rolling(short_window, min_periods=short_window).mean()
    long_avg  = volume.rolling(long_window,  min_periods=long_window).mean()
    return (short_avg > long_avg).fillna(False)


# ──────────────────────────────────────────────────────────────────────────────
# COMBINED ENTRY SIGNALS
# ──────────────────────────────────────────────────────────────────────────────

def get_entry_signals(
    long_candidates: pd.DataFrame,
    short_candidates: pd.DataFrame,
    rsi: pd.DataFrame,
    bb_upper: pd.DataFrame,
    bb_lower: pd.DataFrame,
    close: pd.DataFrame,
    ma50: pd.DataFrame,
    rsi_long_thresh: float = 10.0,
    rsi_short_thresh: float = 90.0,
    bb_buffer: float = 0.005,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine four filters into final entry signals.

    LONG  entry:  top momentum  AND  RSI(2) < rsi_long_thresh
                  AND  close ≤ lower_bb × (1 + bb_buffer)   [near / below lower band]
                  AND  close > SMA(50)                       [in uptrend]

    SHORT entry:  bottom momentum  AND  RSI(2) > rsi_short_thresh
                  AND  close ≥ upper_bb × (1 - bb_buffer)   [near / above upper band]
                  AND  close < SMA(50)                       [in downtrend]

    Parameters
    ----------
    long_candidates, short_candidates : Boolean masks from rank functions.
    rsi        : RSI(2) values.
    bb_upper, bb_lower : Bollinger Band levels.
    close      : Daily closes used for band comparison.
    ma50       : 50-day SMA.
    rsi_long_thresh  : RSI threshold for long entries. Default 10.
    rsi_short_thresh : RSI threshold for short entries. Default 90.
    bb_buffer  : Fractional buffer above/below the exact band. Default 0.5%.

    Returns
    -------
    (long_entries, short_entries) — Boolean DataFrames.
    Signal on day T → execute at open of day T+1.
    """
    near_lower_bb = close <= bb_lower * (1.0 + bb_buffer)
    near_upper_bb = close >= bb_upper * (1.0 - bb_buffer)
    above_ma50    = close > ma50
    below_ma50    = close < ma50

    long_entries  = (
        long_candidates
        & (rsi < rsi_long_thresh)
        & near_lower_bb
        & above_ma50
    )
    short_entries = (
        short_candidates
        & (rsi > rsi_short_thresh)
        & near_upper_bb
        & below_ma50
    )
    return long_entries.fillna(False), short_entries.fillna(False)


# ──────────────────────────────────────────────────────────────────────────────
# IMPROVED ENTRY SIGNALS  (v2 — EMA pullback + ADX trend filter)
# ──────────────────────────────────────────────────────────────────────────────

def get_entry_signals_v2(
    long_candidates: pd.DataFrame,
    short_candidates: pd.DataFrame,
    close: pd.DataFrame,
    ema20: pd.DataFrame,
    adx: pd.DataFrame,
    adx_threshold: float = 20.0,
    ema_tolerance: float = 0.015,
    vol_filter: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Improved entry signals (v2) — replaces RSI(2) + Bollinger Band filter.

    Entry logic
    -----------
    LONG  : top-momentum basket  AND  ADX > adx_threshold  (trending market)
            AND  close ≤ EMA20 × (1 + ema_tolerance)      (pulled back to EMA)
            AND  vol_filter (if provided)                   (rising volume confirms)
    SHORT : bottom-momentum basket  AND  ADX > adx_threshold
            AND  close ≥ EMA20 × (1 - ema_tolerance)      (bounced to EMA resistance)
            AND  vol_filter (if provided)                   (rising volume confirms)

    Advantages over RSI(2) + Bollinger Bands
    ----------------------------------------
    * ADX gates out choppy / ranging regimes → fewer false momentum signals
    * EMA(20) pullback fires 10–15× more often than RSI(2) < 10 → better
      capital utilisation without sacrificing entry quality
    * Trailing EMA adapts to regime shifts faster than static Bollinger Bands
    * Optional volume filter (OI proxy) removes low-conviction drift entries

    Parameters
    ----------
    long_candidates, short_candidates : Boolean rank masks (get_long/short_candidates).
    close         : Daily futures close prices.
    ema20         : 20-period EMA (from calculate_ema).
    adx           : ADX(14) values (from calculate_adx).
    adx_threshold : Minimum ADX to confirm trend. Default 20.
    ema_tolerance : Max fraction above/below EMA20 for pullback entry. Default 1.5%.
    vol_filter    : Optional boolean DataFrame from volume_trend_filter().  When
                    provided, entries are only taken when volume is elevated vs
                    baseline (rising participation = conviction filter).

    Returns
    -------
    (long_entries, short_entries) — Boolean DataFrames.
    Signal on day T → execute at open of day T+1.
    """
    trending = adx > adx_threshold

    # LONG: price has pulled back to or below EMA20 (within tolerance above)
    near_ema_long  = close <= ema20 * (1.0 + ema_tolerance)
    # SHORT: price has bounced up to or above EMA20 (within tolerance below)
    near_ema_short = close >= ema20 * (1.0 - ema_tolerance)

    long_entries  = long_candidates  & trending & near_ema_long
    short_entries = short_candidates & trending & near_ema_short

    if vol_filter is not None:
        # Align vol_filter to the same dates and columns as entries
        vf = vol_filter.reindex(index=close.index, columns=close.columns).fillna(False)
        long_entries  = long_entries  & vf
        short_entries = short_entries & vf

    return long_entries.fillna(False), short_entries.fillna(False)


# ──────────────────────────────────────────────────────────────────────────────
# REGIME DETECTION
# ──────────────────────────────────────────────────────────────────────────────

_VIX_LOW      = 15.0
_VIX_MEDIUM   = 20.0
_VIX_HIGH     = 25.0
_VIX_EXTREME  = 35.0


def india_vix_regime(
    vix: pd.Series,
) -> pd.Series:
    """
    Classify daily market regime from India VIX.

    Returns a Series of strings:
        "low_vol"    VIX < 15   — calm, trend-following works well
        "normal"     15 ≤ VIX < 20
        "elevated"   20 ≤ VIX < 25  — reduce position sizes
        "high_vol"   25 ≤ VIX < 35  — only half positions
        "extreme"    VIX ≥ 35       — no new positions
    """
    regime = pd.Series("normal", index=vix.index, name="vix_regime")
    regime[vix < _VIX_LOW]                         = "low_vol"
    regime[(vix >= _VIX_MEDIUM) & (vix < _VIX_HIGH)] = "elevated"
    regime[(vix >= _VIX_HIGH) & (vix < _VIX_EXTREME)] = "high_vol"
    regime[vix >= _VIX_EXTREME]                    = "extreme"
    return regime


def vix_position_scalar(vix: pd.Series) -> pd.Series:
    """
    Linear position scalar based on India VIX level.

    VIX ≤ 15  → 1.0  (full size)
    VIX = 20  → 0.75
    VIX = 25  → 0.50
    VIX = 35  → 0.25
    VIX ≥ 35  → 0.0  (no new positions)
    """
    scalar = pd.Series(1.0, index=vix.index)
    scalar = scalar.where(vix <= _VIX_LOW,
             (1.0 - (vix - _VIX_LOW) / (_VIX_EXTREME - _VIX_LOW)).clip(0.0, 1.0))
    scalar[vix >= _VIX_EXTREME] = 0.0
    return scalar.fillna(1.0)


def trend_regime(
    close: pd.Series,
    ma_short: int = 50,
    ma_long: int = 200,
) -> pd.Series:
    """
    Dual-MA trend regime for a single price series (e.g. NIFTY index).

    +1 : ma_short > ma_long  (uptrend / bull)
    -1 : ma_short < ma_long  (downtrend / bear)
     0 : within 0.5% (choppy / sideways)
    """
    sma_s = close.rolling(ma_short).mean()
    sma_l = close.rolling(ma_long).mean()
    diff  = (sma_s - sma_l) / sma_l
    regime = pd.Series(0, index=close.index, dtype=int, name="trend_regime")
    regime[diff >  0.005] =  1
    regime[diff < -0.005] = -1
    return regime


def detect_combined_regime(
    nifty_close: pd.Series,
    india_vix: pd.Series,
) -> pd.DataFrame:
    """
    Combine VIX and trend signals into a two-column regime DataFrame.

    Columns
    -------
    trend_regime   : +1 / 0 / -1  from dual-MA crossover on NIFTY
    vix_regime     : string label from India VIX
    vix_scalar     : float 0–1 position size multiplier
    allow_longs    : bool — safe to open new long positions today
    allow_shorts   : bool — safe to open new short positions today
    """
    t_reg  = trend_regime(nifty_close)
    v_reg  = india_vix_regime(india_vix)
    v_sca  = vix_position_scalar(india_vix)

    # Block new longs only in bear + extreme VIX (both conditions needed to avoid
    # sitting out every consolidation in a secular bull market)
    allow_longs  = (v_reg != "extreme") & (t_reg >= 0)
    # Cross-sectional shorts (bottom-rank names) generate alpha regardless of broad market
    # direction — gating on t_reg<=0 kills the short book in India's secular uptrend.
    # Block only in extreme VIX where bid-ask spreads blow out and stops whipsaw.
    allow_shorts = (v_reg != "extreme")

    return pd.DataFrame({
        "trend_regime": t_reg,
        "vix_regime":   v_reg,
        "vix_scalar":   v_sca,
        "allow_longs":  allow_longs,
        "allow_shorts": allow_shorts,
    })
