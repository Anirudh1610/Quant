"""
data.py
-------
Download and cache Nifty 50, Bank Nifty, and India VIX data from Yahoo Finance.

Notes on data limitations
--------------------------
Gift Nifty (SGX Nifty):
    Historical Gift Nifty data is not available on Yahoo Finance.  In the
    backtest the opening gap (today's open / prev close − 1) is used as a
    proxy — this is the *realised* version of what Gift Nifty predicted.
    In live trading substitute this with the actual Gift Nifty level from
    NSE's GIFT City feed before 09:00 IST.

Nifty / BankNifty Futures:
    NSE futures are not on Yahoo Finance.  The strategy approximates the
    basis using the theoretical cost-of-carry formula (F = S·e^((r−q)·T)).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

NIFTY_TICKER     = "^NSEI"
BANKNIFTY_TICKER = "^NSEBANK"
VIX_TICKER       = "^INDIAVIX"


def _download_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [c.lower() for c in raw.columns]
    raw.index   = pd.to_datetime(raw.index)
    return raw[["open", "high", "low", "close", "volume"]].dropna(how="all")


def load_options_dataset(
    start: str,
    end: str,
    data_dir: str | Path = "data/cache",
    refresh: bool = False,
) -> dict:
    """
    Download (or load cached) Nifty 50, Bank Nifty, and India VIX data.

    Returns
    -------
    dict:
        "nifty"     : pd.DataFrame  OHLCV for Nifty 50 spot
        "banknifty" : pd.DataFrame  OHLCV for Bank Nifty spot
        "vix"       : pd.Series     India VIX daily close
        "dates"     : pd.DatetimeIndex  common trading days
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = data_dir / f"options_{start}_{end}.pkl"

    if cache.exists() and not refresh:
        print(f"[DATA] Loading cache → {cache}")
        return pd.read_pickle(cache)

    print("[DATA] Downloading Nifty 50, Bank Nifty, India VIX …")
    nifty     = _download_ohlcv(NIFTY_TICKER,     start, end)
    banknifty = _download_ohlcv(BANKNIFTY_TICKER, start, end)
    vix_df    = _download_ohlcv(VIX_TICKER,       start, end)
    vix       = vix_df["close"].rename("vix")

    common    = nifty.index.intersection(banknifty.index)
    nifty     = nifty.reindex(common).ffill()
    banknifty = banknifty.reindex(common).ffill()
    vix       = vix.reindex(common).ffill().fillna(15.0)

    dataset = {
        "nifty":     nifty,
        "banknifty": banknifty,
        "vix":       vix,
        "dates":     common,
    }
    pd.to_pickle(dataset, cache)
    print(f"[DATA] Saved → {cache}  ({len(common)} trading days)")
    return dataset
