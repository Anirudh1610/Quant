"""
data/vix.py
-----------
Live and historical India VIX, cached alongside (but in a separate table
from) spot OHLC — see data/cache.py schema notes. `get_spot_and_vix` joins
the two on date for regime-layer convenience.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from config import VIX_SPOT_SYMBOL
from core.kite_client import KiteRateLimited
from data.cache import read_spot_ohlc, read_vix, upsert_vix
from data.instruments import resolve_vix_symbol


def get_live_vix(client: KiteRateLimited) -> dict:
    _, symbol = resolve_vix_symbol(client)
    quote = client.call("quote", [symbol])[symbol]
    return {
        "timestamp": datetime.now(),
        "last_price": quote["last_price"],
        "ohlc": quote.get("ohlc", {}),
    }


def get_historical_vix(client: KiteRateLimited, start: date, end: date,
                        interval: str = "day", refresh: bool = False) -> pd.DataFrame:
    start_s, end_s = start.isoformat(), end.isoformat()

    if not refresh:
        cached = read_vix(start_s, end_s)
        cached_dates = set(cached.index.date) if not cached.empty else set()
        all_dates = {d.date() for d in pd.bdate_range(start, end)}
        missing = sorted(all_dates - cached_dates)
        if not missing:
            return cached
        fetch_start, fetch_end = min(missing), max(missing)
    else:
        fetch_start, fetch_end = start, end

    token, _ = resolve_vix_symbol(client)
    candles = client.call(
        "historical_data", token,
        from_date=fetch_start.isoformat(), to_date=fetch_end.isoformat(),
        interval=interval,
    )
    df = pd.DataFrame(candles)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.set_index("date")[["open", "high", "low", "close"]]
        upsert_vix(df, source="kite")

    return read_vix(start_s, end_s)


def get_spot_and_vix(instrument: str, start: date, end: date) -> pd.DataFrame:
    """Read-only join of cached spot OHLC + VIX close on date. Does not hit the API."""
    start_s, end_s = start.isoformat(), end.isoformat()
    spot = read_spot_ohlc(instrument, start_s, end_s)
    vix = read_vix(start_s, end_s)[["close"]].rename(columns={"close": "vix"})
    return spot.join(vix, how="left")
