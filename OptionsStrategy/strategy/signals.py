"""
signals.py
----------
Daily market trend classification for the options selling strategy.

Signal pipeline (all inputs from previous day — zero lookahead)
---------------------------------------------------------------
1. Gift Nifty proxy  — today's open gap vs prev close  (proxy for pre-market bias)
2. Previous day return — close-to-close % change
3. Previous day candle body — (close − open) / open
4. RSI(14)           — overbought / oversold state
5. EMA(20) deviation — is price above or below its dynamic support?
6. ADX(14)           — trend strength (SKIP if market is trending hard)
7. India VIX         — absolute volatility level (SKIP if extreme fear/greed)

Output labels
-------------
  NEUTRAL   → sell ATM straddle  (low absolute score, low ADX)
  BULL      → sell strangle skewed upward  (positive score)
  BEAR      → sell strangle skewed downward (negative score)
  STRANGLE  → sell symmetric OTM strangle  (moderate score, moderate ADX)
  SKIP      → do not trade  (ADX too high OR VIX too high)

Live trading note
-----------------
Replace the open-gap proxy with the actual Gift Nifty reading (available on
NSE GIFT City terminal before 09:00 IST) and the actual Nifty Futures premium
from your broker's live feed for sharper pre-open signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS
# ──────────────────────────────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_l = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _ema(close: pd.Series, window: int) -> pd.Series:
    return close.ewm(span=window, min_periods=window).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int = 14) -> pd.Series:
    tr   = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    hdiff = high.diff()
    ldiff = -low.diff()
    pdm   = hdiff.where((hdiff > ldiff) & (hdiff > 0), 0.0)
    ndm   = ldiff.where((ldiff > hdiff) & (ldiff > 0), 0.0)

    atr   = tr.ewm(alpha=1 / period, min_periods=period).mean()
    pdi   = 100 * pdm.ewm(alpha=1 / period, min_periods=period).mean() / atr.replace(0, np.nan)
    ndi   = 100 * ndm.ewm(alpha=1 / period, min_periods=period).mean() / atr.replace(0, np.nan)
    dx    = (100 * (pdi - ndi).abs() / (pdi + ndi + 1e-9))
    return dx.ewm(alpha=1 / period, min_periods=period).mean()


# ──────────────────────────────────────────────────────────────────────────────
# TREND SCORE
# ──────────────────────────────────────────────────────────────────────────────

def _build_score(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    gap_strong: float,
    gap_mild: float,
) -> pd.Series:
    """Scalar trend score for each day. Positive = bullish bias, negative = bearish."""
    prev_close = close.shift(1)
    prev_open  = open_.shift(1)

    gap         = open_ / prev_close - 1
    prev_return = prev_close / close.shift(2) - 1
    prev_body   = (prev_close - prev_open) / prev_open.replace(0, np.nan)

    rsi     = _rsi(close).shift(1)
    ema20   = _ema(close, 20).shift(1)
    ema_dev = (close.shift(1) - ema20) / ema20.replace(0, np.nan)

    score = pd.Series(0.0, index=close.index)

    # 1. Gift Nifty proxy (open gap)
    score += np.where(gap >  gap_strong,  2.0,
             np.where(gap >  gap_mild,    1.0,
             np.where(gap < -gap_strong, -2.0,
             np.where(gap < -gap_mild,   -1.0, 0.0))))

    # 2. Previous day return
    score += np.where(prev_return >  0.010,  1.0,
             np.where(prev_return >  0.005,  0.5,
             np.where(prev_return < -0.010, -1.0,
             np.where(prev_return < -0.005, -0.5, 0.0))))

    # 3. Candle body (bullish or bearish candle yesterday)
    score += np.where(prev_body >  0.005,  0.5,
             np.where(prev_body < -0.005, -0.5, 0.0))

    # 4. RSI momentum
    score += np.where(rsi > 65,  0.5,
             np.where(rsi < 35, -0.5, 0.0))

    # 5. EMA deviation
    score += np.where(ema_dev >  0.005,  0.5,
             np.where(ema_dev < -0.005, -0.5, 0.0))

    return score


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC CLASSIFIER
# ──────────────────────────────────────────────────────────────────────────────

def classify_trend(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    vix: pd.Series,
    gap_strong: float   = 0.0075,
    gap_mild: float     = 0.0030,
    min_vix: float      = 14.0,
    max_vix: float      = 20.0,
    max_adx: float      = 30.0,
    max_gap: float      = 0.015,
    neutral_band: float = 1.0,
    mild_band: float    = 2.0,
) -> pd.Series:
    """
    Classify each trading day for options trade decision.

    All signals use data from the *previous* close — no lookahead.

    Parameters
    ----------
    open_, high, low, close : OHLC series for the index.
    vix          : India VIX close series.
    min_vix      : Skip if VIX < this — premiums too small to cover costs.
    max_vix      : Skip if VIX > this — too dangerous to sell.
    max_adx      : Skip if ADX(14) > this — market is trending hard.
    max_gap      : Skip if opening gap > this fraction — overnight crisis move.
    neutral_band : Score within ±this → NEUTRAL (sell ATM straddle).
    mild_band    : Score between ±neutral_band and ±mild_band → BULL / BEAR.

    Returns
    -------
    pd.Series  Values: NEUTRAL | BULL | BEAR | STRANGLE | SKIP
    """
    score = _build_score(open_, high, low, close, gap_strong, gap_mild)
    adx   = _adx(high, low, close).shift(1)
    gap   = (open_ / close.shift(1) - 1).abs()  # today's open gap magnitude

    skip = (
        (vix.shift(1) < min_vix)   # premium too thin
        | (vix.shift(1) > max_vix) # too risky
        | (adx > max_adx)          # trending market
        | (gap > max_gap)          # crash / gap-up day — don't sell into uncertainty
    )

    sig = pd.Series("NEUTRAL", index=close.index)

    # Directional signals — keep skew even when ADX is elevated
    sig[score >  mild_band]  = "BULL"
    sig[score < -mild_band]  = "BEAR"
    sig[(score >  neutral_band) & (score <= mild_band)] = "BULL"
    sig[(score < -neutral_band) & (score >= -mild_band)] = "BEAR"

    # Widen NEUTRAL-only to STRANGLE when ADX is moderate
    sig[(sig == "NEUTRAL") & (adx > 20) & (adx <= max_adx)] = "STRANGLE"

    sig[skip] = "SKIP"

    return sig
