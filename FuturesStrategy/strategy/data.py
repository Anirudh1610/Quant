"""
data.py
-------
Data acquisition and futures price construction for the NSE F&O strategy.

Since tick-level NSE futures data is not freely available, this module
constructs SYNTHETIC near-month futures prices from spot OHLCV data using
the standard cost-of-carry (CoC) model:

    F(t, T) = S(t) × (1 + r × dte / 365)

where
    S(t)  = spot price on date t
    r     = risk-free rate (RBI repo rate proxy, default 6.5%)
    dte   = calendar days remaining to near-month expiry

This is a sound approximation because:
  1. NSE stock/index futures are cash-settled.
  2. Dividends on NIFTY 50 stocks are modest (dividend yield ~1-1.5%).
  3. Actual futures trade within a few basis points of fair value.

NSE expiry schedule
-------------------
Monthly contracts expire on the LAST THURSDAY of each calendar month.
If that Thursday is a BSE/NSE holiday, expiry shifts to the preceding Wednesday.

Roll convention used here
--------------------------
Roll to the next month contract 2 BUSINESS DAYS before the expiry Thursday.
This avoids expiry-week gamma risk and matches common practice.
"""

from __future__ import annotations

import calendar
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from .universe import DEFAULT_UNIVERSE, INDEX_YF_MAP, INDIA_VIX_TICKER, to_yf_ticker


# ──────────────────────────────────────────────────────────────────────────────
# NSE EXPIRY & ROLL DATE UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def get_last_thursday(year: int, month: int) -> pd.Timestamp:
    """
    Return the date of the last Thursday in (year, month).

    Parameters
    ----------
    year, month : Calendar year and month (1–12).

    Returns
    -------
    pd.Timestamp of the last Thursday (ignores market holidays for simplicity).
    """
    cal = calendar.monthcalendar(year, month)
    # In monthcalendar, Thursday is at index 3 (Mon=0 … Sun=6)
    thursdays = [week[3] for week in cal if week[3] != 0]
    return pd.Timestamp(year, month, thursdays[-1])


def build_expiry_calendar(
    start: str,
    end: str,
) -> pd.Series:
    """
    Build a daily Series mapping each trading date → its near-month expiry date.

    The near-month contract is the one expiring soonest.
    On (and after) the roll date (2 business days before expiry) the near-month
    switches to the following month.

    Parameters
    ----------
    start, end : Date strings "YYYY-MM-DD".

    Returns
    -------
    pd.Series[pd.Timestamp], indexed by business days in [start, end].
    """
    dates = pd.bdate_range(start=start, end=end)

    # Enumerate all monthly expiries covering the date range
    s = pd.Timestamp(start)
    e = pd.Timestamp(end) + pd.offsets.MonthEnd(1)
    expiries: list[pd.Timestamp] = []
    y, m = s.year, s.month
    while True:
        exp = get_last_thursday(y, m)
        expiries.append(exp)
        m += 1
        if m > 12:
            m, y = 1, y + 1
        if pd.Timestamp(y, m, 1) > e:
            break

    # For each date, find the active near-month expiry
    # Roll date = expiry − 2 business days
    result: list[pd.Timestamp] = []
    for date in dates:
        active_expiry: pd.Timestamp | None = None
        for exp in expiries:
            roll_date = exp - pd.tseries.offsets.BusinessDay(2)
            if date < roll_date:
                active_expiry = exp
                break
        if active_expiry is None:
            active_expiry = expiries[-1]
        result.append(active_expiry)

    return pd.Series(result, index=dates, name="expiry")


def get_roll_dates(expiry_calendar: pd.Series) -> pd.DatetimeIndex:
    """
    Return all dates on which a futures rollover is triggered.

    A roll date is the first date where the active expiry jumps forward
    (i.e., expiry_calendar[t] != expiry_calendar[t-1]).
    """
    shifted = expiry_calendar.shift(1)
    mask = expiry_calendar != shifted
    # Drop the very first row (NaN from shift)
    return expiry_calendar.index[mask & ~shifted.isna()]


# ──────────────────────────────────────────────────────────────────────────────
# SYNTHETIC FUTURES PRICE CONSTRUCTION
# ──────────────────────────────────────────────────────────────────────────────

def construct_futures_prices(
    spot_df: pd.DataFrame,
    expiry_calendar: pd.Series,
    risk_free_rate: float = 0.065,
) -> pd.DataFrame:
    """
    Construct daily near-month synthetic futures prices from spot OHLCV.

    Formula: F(t, T) = S(t) × (1 + r × dte / 365)

    Parameters
    ----------
    spot_df         : DataFrame of spot close prices (rows = dates, cols = tickers).
    expiry_calendar : Series mapping each date → near-month expiry (pd.Timestamp).
    risk_free_rate  : Annualised risk-free rate (RBI repo rate proxy).

    Returns
    -------
    pd.DataFrame with the same shape as spot_df.
    """
    # Calendar days to expiry for each date
    dates = spot_df.index.intersection(expiry_calendar.index)
    expiry_aligned = expiry_calendar.reindex(dates, method="ffill")
    dte = (expiry_aligned.values - dates).days.astype(float)
    dte = np.maximum(dte, 0.0)   # no negative dte at/after expiry

    carry_factor = pd.Series(1.0 + risk_free_rate * dte / 365.0, index=dates)
    futures = spot_df.reindex(dates).multiply(carry_factor, axis=0)
    return futures.round(2)


def _apply_carry_to_ohlcv(
    ohlcv: dict[str, pd.DataFrame],
    expiry_calendar: pd.Series,
    risk_free_rate: float,
) -> dict[str, pd.DataFrame]:
    """Apply the cost-of-carry multiplier to Open, High, Low, Close."""
    out: dict[str, pd.DataFrame] = {}
    dates = ohlcv["close"].index.intersection(expiry_calendar.index)
    expiry_aligned = expiry_calendar.reindex(dates, method="ffill")
    dte = (expiry_aligned.values - dates).days.astype(float)
    dte = np.maximum(dte, 0.0)
    carry = pd.Series(1.0 + risk_free_rate * dte / 365.0, index=dates)

    for field in ("open", "high", "low", "close"):
        df = ohlcv[field].reindex(dates)
        out[field] = df.multiply(carry, axis=0).round(2)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# DOWNLOAD LAYER
# ──────────────────────────────────────────────────────────────────────────────

def download_spot_data(
    tickers: list[str],
    start: str,
    end: str,
    data_dir: Path,
) -> dict[str, pd.DataFrame]:
    """
    Download daily OHLCV spot data from Yahoo Finance and cache as pickle.

    Parameters
    ----------
    tickers  : List of futures tickers (e.g. ["NIFTY", "RELIANCE"]).
    start, end : Date strings "YYYY-MM-DD".
    data_dir : Directory to write/read the pickle cache.

    Returns
    -------
    dict with keys "open", "high", "low", "close", "volume".
    Each value is a DataFrame (rows = dates, cols = futures tickers).
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = data_dir / f"spot_{start}_{end}.pkl"

    if cache.exists():
        print(f"[DATA] Loading spot cache → {cache}")
        return pd.read_pickle(cache)

    yf_tickers = [to_yf_ticker(t) for t in tickers]
    print(f"[DATA] Downloading spot data for {len(yf_tickers)} symbols …")
    raw = yf.download(
        yf_tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=True,
    )

    # Build reverse map: yf_ticker → futures_ticker
    yf_to_fut: dict[str, str] = {to_yf_ticker(t): t for t in tickers}

    def _extract(field: str) -> pd.DataFrame:
        if isinstance(raw.columns, pd.MultiIndex):
            if field not in raw.columns.get_level_values(0):
                return pd.DataFrame()
            df = raw[field].copy()
            df.columns.name = None
        else:
            df = raw[[field]].copy() if field in raw.columns else pd.DataFrame()
        # Rename yfinance symbols → futures tickers
        df = df.rename(columns=yf_to_fut)
        return df.dropna(how="all", axis=1)

    result = {
        "close":  _extract("Close"),
        "open":   _extract("Open"),
        "high":   _extract("High"),
        "low":    _extract("Low"),
        "volume": _extract("Volume"),
    }
    pd.to_pickle(result, cache)
    print(f"[DATA] Spot data cached → {cache}")
    return result


def download_india_vix(
    start: str,
    end: str,
    data_dir: Path,
) -> pd.Series:
    """
    Download India VIX (NSE fear gauge) from Yahoo Finance.

    India VIX > 20  → elevated fear; reduce position sizes.
    India VIX > 30  → high fear; halve positions.
    India VIX > 40  → extreme; no new positions.

    Returns
    -------
    pd.Series of daily India VIX close values.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = data_dir / f"india_vix_{start}_{end}.pkl"

    if cache.exists():
        return pd.read_pickle(cache)

    print("[DATA] Downloading India VIX …")
    raw = yf.download(INDIA_VIX_TICKER, start=start, end=end,
                      auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        series = raw["Close"].squeeze()
    else:
        series = raw["Close"].squeeze()
    series.name = "INDIAVIX"
    series = series.ffill()
    pd.to_pickle(series, cache)
    return series


def load_futures_dataset(
    tickers: list[str],
    start: str,
    end: str,
    data_dir: Path,
    risk_free_rate: float = 0.065,
) -> dict:
    """
    Full data pipeline: download spot OHLCV + India VIX, build synthetic futures,
    compute expiry calendar, and return a single dataset dict.

    Parameters
    ----------
    tickers       : Futures tickers (NSE codes, e.g. ["NIFTY", "RELIANCE"]).
    start, end    : Backtest date range "YYYY-MM-DD".
    data_dir      : Local cache directory.
    risk_free_rate: Cost-of-carry rate for synthetic futures construction.

    Returns
    -------
    dict with keys:
        "spot"            → dict("close", "open", "high", "low", "volume")
        "futures"         → dict("close", "open", "high", "low")  [synthetic]
        "expiry_calendar" → pd.Series  (date → near-month expiry)
        "roll_dates"      → pd.DatetimeIndex
        "india_vix"       → pd.Series
        "tickers"         → list[str]  (tickers with sufficient history)
    """
    data_dir = Path(data_dir)   # accept str or Path from callers
    spot   = download_spot_data(tickers, start, end, data_dir)
    vix    = download_india_vix(start, end, data_dir)
    expiry = build_expiry_calendar(start, end)

    # Drop tickers with insufficient spot data (< 80% of dates populated)
    close  = spot["close"]
    min_obs = int(0.8 * len(close))
    valid   = close.columns[close.notna().sum() >= min_obs].tolist()
    if len(valid) < len(tickers):
        dropped = set(tickers) - set(valid)
        print(f"[DATA] Dropped {len(dropped)} tickers due to missing data: {dropped}")

    # Trim all OHLCV to valid tickers
    spot = {k: v[valid] for k, v in spot.items()}

    # Forward-fill small gaps (e.g. holidays, circuit breakers)
    spot = {k: v.ffill(limit=3) for k, v in spot.items()}

    # Construct synthetic futures OHLCV
    futures = _apply_carry_to_ohlcv(spot, expiry, risk_free_rate)

    roll_dates = get_roll_dates(expiry)

    return {
        "spot":            spot,
        "futures":         futures,
        "expiry_calendar": expiry,
        "roll_dates":      roll_dates,
        "india_vix":       vix,
        "tickers":         valid,
    }
