"""
data/option_chain.py
---------------------
Live option chain snapshot for a given instrument + expiry: strikes, LTP,
OI, volume, bid/ask, and IV (computed — see core/pricing.py docstring on
why Kite doesn't give us IV directly).

Kite has no single "option chain" endpoint. This module:
  1. resolves all CE/PE contracts for (instrument, expiry) from the NFO
     instrument dump (data/instruments.py)
  2. batches them through kite.quote() (chunked — see KiteRateLimited)
  3. estimates a market-implied forward from put-call parity on liquid
     near-ATM strikes (core/pricing.py:implied_forward) rather than
     assuming a static dividend yield — see core/pricing.py's module
     docstring for why: a static-q assumption measurably diverges from
     what the market actually prices, producing a spurious call/put IV
     skew (verified empirically: ~44pt forward mismatch on NIFTY -> calls
     read ~12-14% IV, puts ~8-10% at the same strikes, backwards from real
     index skew)
  4. inverts BS for IV against that forward (Black-76: S=F, q=r)

Snapshots are cached (append-only, timestamped) in SQLite. To avoid
hammering the API, a fresh Kite call is only made if the most recent cached
snapshot is older than `min_interval_seconds` — pass force_refresh=True to
bypass that.
"""

from __future__ import annotations

import logging
from datetime import date, datetime

import numpy as np
import pandas as pd

from config import DIVIDEND_YIELD, RISK_FREE_RATE
from core.kite_client import KiteRateLimited
from core.pricing import implied_forward, implied_volatility
from data.cache import insert_option_chain_snapshot, latest_snapshot_age_seconds, read_latest_option_chain
from data.instruments import get_option_chain_instruments
from data.spot import get_live_spot

logger = logging.getLogger("nifty_options.option_chain")

DEFAULT_MIN_INTERVAL_SECONDS = 60.0
FORWARD_ESTIMATE_STRIKES = 8  # nearest-the-money liquid strikes used to imply the forward


def time_to_expiry_years(expiry: date) -> float:
    now = datetime.now()
    expiry_close = datetime.combine(expiry, datetime.min.time()).replace(hour=15, minute=30)
    seconds_left = max((expiry_close - now).total_seconds(), 60.0)  # floor: avoid T=0 mid-session on expiry day
    return seconds_left / (365.0 * 24 * 3600)


def _parse_quotes(contracts: pd.DataFrame, quotes: dict) -> list[dict]:
    """Extract strike/type/ltp/bid/ask/mid/oi/volume per contract from raw kite.quote() output."""
    parsed = []
    for _, contract in contracts.iterrows():
        symbol = f"NFO:{contract['tradingsymbol']}"
        q_data = quotes.get(symbol)
        if q_data is None:
            continue

        depth = q_data.get("depth", {})
        best_bid = depth.get("buy", [{}])[0].get("price") if depth.get("buy") else None
        best_ask = depth.get("sell", [{}])[0].get("price") if depth.get("sell") else None
        mid = (best_bid + best_ask) / 2 if best_bid and best_ask and best_bid > 0 and best_ask > 0 else None

        parsed.append({
            "strike": float(contract["strike"]),
            "option_type": contract["instrument_type"],  # "CE" | "PE"
            "ltp": q_data.get("last_price"),
            "bid": best_bid,
            "ask": best_ask,
            "mid": mid,
            "oi": q_data.get("oi"),
            "volume": q_data.get("volume"),
        })
    return parsed


def _estimate_forward(parsed: list[dict], underlying_ltp: float, T: float, r: float, q: float,
                       n_strikes: int = FORWARD_ESTIMATE_STRIKES) -> float:
    """
    Market-implied forward via put-call parity, median'd across the
    `n_strikes` nearest-the-money contracts with valid two-sided quotes on
    both legs (robust to any single strike's wide/stale spread).

    Falls back to the static S*e^{(r-q)T} approximation — logging a warning
    — if fewer than 2 such liquid pairs exist (e.g. an illiquid session).
    """
    by_strike: dict[float, dict[str, float]] = {}
    for p in parsed:
        if p["mid"] is None:
            continue
        by_strike.setdefault(p["strike"], {})[p["option_type"]] = p["mid"]

    candidates = sorted(
        (k for k, v in by_strike.items() if "CE" in v and "PE" in v),
        key=lambda k: abs(k - underlying_ltp),
    )[:n_strikes]

    if len(candidates) < 2:
        logger.warning(
            "Only %d liquid two-sided strikes available to imply a forward "
            "(need >=2) — falling back to static r/q approximation",
            len(candidates),
        )
        return underlying_ltp * np.exp((r - q) * T)

    forwards = [implied_forward(by_strike[k]["CE"], by_strike[k]["PE"], k, T, r) for k in candidates]
    return float(np.median(forwards))


def estimate_forward_from_chain(chain: pd.DataFrame, underlying_ltp: float, T: float, r: float, q: float,
                                 n_strikes: int = FORWARD_ESTIMATE_STRIKES) -> float:
    """
    Public entry point to re-derive the same market-implied forward
    fetch_option_chain_snapshot() priced IV off, from an already-fetched
    chain DataFrame (columns: strike, option_type, bid, ask) — e.g. for a
    caller (strikes/orchestrator.py via scripts/select_spread.py) that
    needs the forward for Greeks/probability-of-touch but only has the
    cached/returned DataFrame, not the raw quote() response.
    """
    parsed = [
        {
            "strike": row["strike"],
            "option_type": row["option_type"],
            "mid": (row["bid"] + row["ask"]) / 2
            if row["bid"] and row["ask"] and row["bid"] > 0 and row["ask"] > 0 else None,
        }
        for _, row in chain.iterrows()
    ]
    return _estimate_forward(parsed, underlying_ltp, T, r, q, n_strikes)


def fetch_option_chain_snapshot(client: KiteRateLimited, instrument: str, expiry: date,
                                 min_interval_seconds: float = DEFAULT_MIN_INTERVAL_SECONDS,
                                 force_refresh: bool = False) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per (strike, option_type):
    strike, option_type, ltp, bid, ask, oi, oi_day_change, volume, iv,
    underlying_ltp, fetch_timestamp.

    Serves from cache if the last snapshot is fresher than
    `min_interval_seconds`; otherwise pulls a new one from Kite and caches it.
    """
    expiry_s = expiry.isoformat()
    age = latest_snapshot_age_seconds(instrument, expiry_s)
    if not force_refresh and age is not None and age < min_interval_seconds:
        return read_latest_option_chain(instrument, expiry_s)

    contracts = get_option_chain_instruments(client, instrument, expiry)
    symbols = [f"NFO:{ts}" for ts in contracts["tradingsymbol"]]
    quotes = client.call_in_chunks("quote", symbols)

    spot = get_live_spot(client, instrument)
    underlying_ltp = spot["last_price"]
    T = time_to_expiry_years(expiry)
    r = RISK_FREE_RATE
    q = DIVIDEND_YIELD.get(instrument, 0.0)
    fetch_ts = datetime.now().isoformat(timespec="seconds")

    parsed = _parse_quotes(contracts, quotes)
    forward = _estimate_forward(parsed, underlying_ltp, T, r, q)

    rows = []
    for p in parsed:
        price_for_iv = p["mid"] if p["mid"] is not None else p["ltp"]

        iv = None
        if price_for_iv and price_for_iv > 0:
            bs_type = "call" if p["option_type"] == "CE" else "put"
            # Black-76 (S=forward, q=r) — see core/pricing.py module docstring
            iv = implied_volatility(price_for_iv, forward, p["strike"], T, r, r, bs_type)

        rows.append({
            "instrument": instrument,
            "expiry": expiry_s,
            "fetch_timestamp": fetch_ts,
            "strike": p["strike"],
            "option_type": p["option_type"],
            "ltp": p["ltp"],
            "bid": p["bid"],
            "ask": p["ask"],
            "oi": p["oi"],
            # ASSUMPTION FLAG: kite.quote() has no "OI change vs prior close" field.
            # True OI change requires diffing against yesterday's closing OI
            # (via historical_data(..., oi=True)) — not computed here. Left
            # NULL rather than approximated from oi_day_high/low, which are
            # today's intraday OI extremes, not the prior-day baseline.
            "oi_day_change": None,
            "volume": p["volume"],
            "iv": iv,
            "underlying_ltp": underlying_ltp,
        })

    df = pd.DataFrame(rows)
    insert_option_chain_snapshot(df)
    return df.sort_values(["strike", "option_type"]).reset_index(drop=True)
