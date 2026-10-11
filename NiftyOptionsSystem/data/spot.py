"""
data/spot.py
------------
Live and historical spot price data for NIFTY/BANKNIFTY, with SQLite
caching of historical data so re-running a backtest doesn't redownload
unchanged history.

ASSUMPTION FLAG: kite.historical_data requires the paid Historical Data API
add-on (confirmed active per user). Interval is daily ("day") for the regime
layer's needs; intraday intervals are supported by the same function via
the `interval` param if Phase 3/4 need finer granularity later.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from config import INSTRUMENTS
from core.kite_client import KiteRateLimited
from data.cache import read_spot_ohlc, upsert_spot_ohlc
from data.instruments import resolve_index_symbol


def get_live_spot(client: KiteRateLimited, instrument: str) -> dict:
    """Live LTP + OHLC snapshot for NIFTY or BANKNIFTY via kite.quote()."""
    spec = INSTRUMENTS[instrument]
    _, symbol = resolve_index_symbol(client, spec.spot_symbol)
    quote = client.call("quote", [symbol])[symbol]
    return {
        "instrument": instrument,
        "timestamp": datetime.now(),
        "last_price": quote["last_price"],
        "ohlc": quote.get("ohlc", {}),
    }


def get_historical_spot(client: KiteRateLimited, instrument: str,
                         start: date, end: date,
                         interval: str = "day", refresh: bool = False) -> pd.DataFrame:
    """
    Historical daily OHLCV for NIFTY/BANKNIFTY, backed by the local cache.
    Only fetches from Kite the date sub-ranges missing from the cache
    (unless refresh=True, which re-fetches everything in [start, end]).
    """
    start_s, end_s = start.isoformat(), end.isoformat()

    if not refresh:
        cached = read_spot_ohlc(instrument, start_s, end_s)
        cached_dates = set(cached.index.date) if not cached.empty else set()
        all_dates = {d.date() for d in pd.bdate_range(start, end)}
        missing = sorted(all_dates - cached_dates)
        if not missing:
            return cached
        fetch_start, fetch_end = min(missing), max(missing)
    else:
        fetch_start, fetch_end = start, end

    spec = INSTRUMENTS[instrument]
    token, _ = resolve_index_symbol(client, spec.spot_symbol)

    candles = client.call(
        "historical_data", token,
        from_date=fetch_start.isoformat(), to_date=fetch_end.isoformat(),
        interval=interval,
    )
    df = pd.DataFrame(candles)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.set_index("date")[["open", "high", "low", "close", "volume"]]
        upsert_spot_ohlc(df, instrument, source="kite")

    return read_spot_ohlc(instrument, start_s, end_s)
