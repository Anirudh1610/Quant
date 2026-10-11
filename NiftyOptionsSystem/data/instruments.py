"""
data/instruments.py
--------------------
Resolves Kite instrument tokens and option contract metadata dynamically
from the NFO/NSE instrument dumps, rather than hardcoding tokens or expiry
day-of-week — NSE has changed both lot sizes and the weekly expiry calendar
before (e.g. Bank Nifty weekly expiry was discontinued in Nov 2024), and
hardcoding would silently rot.

The instrument dump is large (tens of thousands of rows for NFO) and only
changes once a day, so it's cached to a local parquet file and refreshed at
most once per day.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from config import CACHE_DIR, INSTRUMENTS, VIX_SPOT_SYMBOL
from core.kite_client import KiteRateLimited
from data.cache import get_instrument_dump_fetched_at, set_instrument_dump_fetched_at

_NFO_PARQUET = CACHE_DIR / "instruments_nfo.parquet"
_NSE_PARQUET = CACHE_DIR / "instruments_nse.parquet"


def _load_dump(client: KiteRateLimited, exchange: str, parquet_path,
                force_refresh: bool = False) -> pd.DataFrame:
    fetched_at = get_instrument_dump_fetched_at(exchange)
    today = date.today().isoformat()
    if not force_refresh and fetched_at == today and parquet_path.exists():
        return pd.read_parquet(parquet_path)

    raw = client.call("instruments", exchange)
    df = pd.DataFrame(raw)
    if "expiry" in df.columns:
        df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
    df.to_parquet(parquet_path, index=False)
    set_instrument_dump_fetched_at(exchange, today)
    return df


def load_nfo_dump(client: KiteRateLimited, force_refresh: bool = False) -> pd.DataFrame:
    return _load_dump(client, "NFO", _NFO_PARQUET, force_refresh)


def load_nse_dump(client: KiteRateLimited, force_refresh: bool = False) -> pd.DataFrame:
    return _load_dump(client, "NSE", _NSE_PARQUET, force_refresh)


def get_available_expiries(client: KiteRateLimited, instrument: str) -> list[date]:
    """All future option expiries currently listed for NIFTY/BANKNIFTY, sorted ascending."""
    spec = INSTRUMENTS[instrument]
    nfo = load_nfo_dump(client)
    mask = (nfo["name"] == spec.name) & (nfo["instrument_type"].isin(["CE", "PE"]))
    expiries = nfo.loc[mask, "expiry"].dropna().dt.date.unique()
    today = date.today()
    return sorted(e for e in expiries if e >= today)


def get_nearest_expiry(client: KiteRateLimited, instrument: str,
                        weekly: bool = True) -> date:
    """
    Nearest listed expiry. `weekly` is a hint only — since NSE's expiry
    calendar changes over time (see module docstring), this just returns the
    soonest expiry in the dump rather than assuming a fixed weekday.
    """
    expiries = get_available_expiries(client, instrument)
    if not expiries:
        raise ValueError(f"No listed option expiries found for {instrument}")
    return expiries[0]


def get_option_chain_instruments(client: KiteRateLimited, instrument: str,
                                  expiry: date) -> pd.DataFrame:
    """
    All CE/PE contracts for `instrument` at `expiry`. Columns include
    instrument_token, tradingsymbol, strike, instrument_type, lot_size,
    tick_size.
    """
    spec = INSTRUMENTS[instrument]
    nfo = load_nfo_dump(client)
    mask = (
        (nfo["name"] == spec.name)
        & (nfo["instrument_type"].isin(["CE", "PE"]))
        & (nfo["expiry"].dt.date == expiry)
    )
    chain = nfo.loc[mask].copy()
    if chain.empty:
        raise ValueError(f"No option contracts found for {instrument} expiry {expiry}")
    return chain.sort_values(["strike", "instrument_type"]).reset_index(drop=True)


def get_current_lot_size(client: KiteRateLimited, instrument: str) -> int:
    """Lot size as currently listed on NFO — prefer this over config.py's placeholder."""
    spec = INSTRUMENTS[instrument]
    nfo = load_nfo_dump(client)
    mask = (nfo["name"] == spec.name) & (nfo["instrument_type"].isin(["CE", "PE"]))
    lots = nfo.loc[mask, "lot_size"].dropna().unique()
    if len(lots) == 0:
        raise ValueError(f"Could not resolve current lot size for {instrument}")
    return int(lots[0])


def resolve_index_symbol(client: KiteRateLimited, tradingsymbol: str) -> tuple[int, str]:
    """
    Resolve an NSE index (e.g. "NIFTY 50", "NIFTY BANK", "INDIA VIX") to its
    instrument_token and canonical `exchange:tradingsymbol` string.
    """
    nse = load_nse_dump(client)
    mask = (nse["segment"] == "INDICES") & (nse["tradingsymbol"] == tradingsymbol)
    matches = nse.loc[mask]
    if matches.empty:
        raise ValueError(f"Index symbol '{tradingsymbol}' not found in NSE instrument dump")
    token = int(matches.iloc[0]["instrument_token"])
    return token, f"NSE:{tradingsymbol}"


def resolve_vix_symbol(client: KiteRateLimited) -> tuple[int, str]:
    return resolve_index_symbol(client, VIX_SPOT_SYMBOL)
