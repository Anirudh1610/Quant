"""
signals/regime.py
------------------
Combines trend, volatility, and OI signals into a single regime object per
day/timestamp: {trend, iv_rank, range_score, recommended_strategy_type, ...}.

Two entry points, because the underlying data has different history depths:

  build_trend_vol_frame(...)
      Fully backtestable — only needs historical spot OHLC + VIX (Phase 1
      data), both cached daily. Produces trend, ADX, realized vol (3
      estimators), IV rank/percentile (using India VIX as the instrument-
      level IV proxy — see ASSUMPTION FLAG below), and IV-RV spread for
      every historical date.

  build_regime_snapshot(...)
      Adds the OI-derived signals (max pain, PCR, concentration,
      range_score), which require a *live or cached option chain snapshot*.
      We don't have historical option chain data (flagged in Phase 1/4
      scope), so this can only be computed for dates where a chain snapshot
      was actually captured — in practice, "now" plus whatever chain
      history accumulates going forward. This is the function that
      produces the final regime object described in the Phase 2 spec.

ASSUMPTION FLAG: IV rank/percentile/spread use India VIX as the IV proxy
(a 30-day-forward, Nifty-only implied vol index) for both NIFTY and
BANKNIFTY, since Kite doesn't expose a comparable instrument-specific IV
history. This is standard practice for Nifty but is an approximation for
Bank Nifty (which can have a meaningfully different vol regime, e.g. around
sector-specific events). Per-strike IV computed in data/option_chain.py
could replace this for BANKNIFTY once enough chain-snapshot history has
accumulated locally to build a trailing window.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime

import pandas as pd

from config import INSTRUMENTS, SIGNALS
from signals.oi import max_pain, oi_concentration_ratio, put_call_ratio, range_bound_conviction_score
from signals.trend import BEARISH, BULLISH, RANGE_BOUND, classify_trend_regime
from signals.volatility import (
    iv_percentile,
    iv_rank,
    iv_rv_spread,
    realized_vol_close_to_close,
    realized_vol_parkinson,
    realized_vol_yang_zhang,
)

NO_TRADE = "NO_TRADE"
BULL_PUT_SPREAD = "BULL_PUT_SPREAD"
BEAR_CALL_SPREAD = "BEAR_CALL_SPREAD"
IRON_CONDOR = "IRON_CONDOR"


def build_trend_vol_frame(spot_ohlc: pd.DataFrame, vix: pd.Series,
                           signal_cfg=SIGNALS) -> pd.DataFrame:
    """
    spot_ohlc: DataFrame indexed by date with open/high/low/close (Phase 1 output).
    vix: Series indexed by date, India VIX close (Phase 1 output).

    Returns a DataFrame indexed by date with columns:
      trend, adx, st_direction,
      rv_close_20d, rv_parkinson_20d, rv_yang_zhang_20d,
      iv (=vix/100, decimal annualized), iv_rank, iv_percentile, iv_rv_spread
    """
    trend_df = classify_trend_regime(
        spot_ohlc["high"], spot_ohlc["low"], spot_ohlc["close"],
        supertrend_period=signal_cfg.supertrend_period,
        supertrend_multiplier=signal_cfg.supertrend_multiplier,
        adx_period=signal_cfg.adx_period,
        adx_range_bound_threshold=signal_cfg.adx_range_bound_threshold,
    )

    iv = vix.reindex(spot_ohlc.index) / 100.0
    rv_close = realized_vol_close_to_close(spot_ohlc["close"], signal_cfg.realized_vol_window)
    rv_parkinson = realized_vol_parkinson(spot_ohlc["high"], spot_ohlc["low"], signal_cfg.realized_vol_window)
    rv_yz = realized_vol_yang_zhang(
        spot_ohlc["open"], spot_ohlc["high"], spot_ohlc["low"], spot_ohlc["close"],
        signal_cfg.realized_vol_window,
    )

    out = trend_df.copy()
    out["rv_close_20d"] = rv_close
    out["rv_parkinson_20d"] = rv_parkinson
    out["rv_yang_zhang_20d"] = rv_yz
    out["iv"] = iv
    out["iv_rank"] = iv_rank(iv, signal_cfg.iv_lookback_days)
    out["iv_percentile"] = iv_percentile(iv, signal_cfg.iv_lookback_days)
    out["iv_rv_spread"] = iv_rv_spread(iv, rv_close)
    return out


@dataclass
class RegimeSnapshot:
    timestamp: datetime
    instrument: str
    spot: float
    trend: str
    adx: float
    iv: float
    iv_rank: float
    iv_percentile: float
    realized_vol_20d: float
    iv_rv_spread: float
    pcr: float
    max_pain_strike: float
    oi_concentration_ratio: float
    range_score: float
    recommended_strategy_type: str

    def to_dict(self) -> dict:
        return asdict(self)


def recommend_strategy_type(trend: str, iv_rank_value: float, range_score: float,
                             signal_cfg=SIGNALS) -> str:
    """
    Heuristic decision table mapping regime -> strategy type. This is a
    documented starting point for Phase 3 strike selection, not a validated
    model — expect to revisit thresholds once Phase 4 backtests exist.

      IV rank below `iv_rank_no_trade_below`  -> NO_TRADE (premium too thin
                                                  to justify selling, regardless
                                                  of trend/range read)
      RANGE_BOUND + range_score >= 60         -> IRON_CONDOR (defined-risk
                                                  neutral premium sale)
      BULLISH                                 -> BULL_PUT_SPREAD
      BEARISH                                 -> BEAR_CALL_SPREAD
      RANGE_BOUND + range_score < 60           -> NO_TRADE (ambiguous: low
                                                  ADX but OI signals don't
                                                  back up a range-bound read)
    """
    if pd.isna(iv_rank_value) or iv_rank_value < signal_cfg.iv_rank_no_trade_below:
        return NO_TRADE
    if trend == RANGE_BOUND:
        return IRON_CONDOR if range_score >= 60 else NO_TRADE
    if trend == BULLISH:
        return BULL_PUT_SPREAD
    if trend == BEARISH:
        return BEAR_CALL_SPREAD
    return NO_TRADE


def build_regime_snapshot(instrument: str, trend_vol_row: pd.Series, spot: float,
                           chain: pd.DataFrame, timestamp: datetime | None = None,
                           signal_cfg=SIGNALS) -> RegimeSnapshot:
    """
    trend_vol_row: a row from build_trend_vol_frame(...) (typically .iloc[-1]).
    chain: live option chain snapshot DataFrame for `instrument`
           (columns: strike, option_type, oi, ... — see data/option_chain.py).
    """
    spec = INSTRUMENTS[instrument]
    timestamp = timestamp or datetime.now()

    pcr = put_call_ratio(chain)
    mp = max_pain(chain)
    concentration = oi_concentration_ratio(chain, spot, spec.strike_step, signal_cfg.oi_concentration_strikes)
    range_score = range_bound_conviction_score(spot, mp, pcr, concentration, spec.strike_step)

    recommended = recommend_strategy_type(trend_vol_row["trend"], trend_vol_row["iv_rank"], range_score, signal_cfg)

    return RegimeSnapshot(
        timestamp=timestamp,
        instrument=instrument,
        spot=spot,
        trend=trend_vol_row["trend"],
        adx=trend_vol_row["adx"],
        iv=trend_vol_row["iv"],
        iv_rank=trend_vol_row["iv_rank"],
        iv_percentile=trend_vol_row["iv_percentile"],
        realized_vol_20d=trend_vol_row["rv_close_20d"],
        iv_rv_spread=trend_vol_row["iv_rv_spread"],
        pcr=pcr,
        max_pain_strike=mp,
        oi_concentration_ratio=concentration,
        range_score=range_score,
        recommended_strategy_type=recommended,
    )
