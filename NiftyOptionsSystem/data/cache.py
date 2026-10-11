"""
data/cache.py
-------------
SQLite-backed local cache for spot OHLC, India VIX, and option chain
snapshots. Goals: avoid redundant Kite API calls, and give the signal layer
a stable, queryable local source of truth.

Schema
------
spot_ohlc(instrument, date, open, high, low, close, volume, source)
    PRIMARY KEY (instrument, date)
    One row per trading day per instrument ("NIFTY", "BANKNIFTY").

india_vix(date, open, high, low, close)
    PRIMARY KEY (date)
    Stored separately from spot_ohlc (it's an index, not a tradable
    instrument tied to NIFTY/BANKNIFTY) but joinable on `date`.

option_chain_snapshots(
    id, instrument, expiry, fetch_timestamp, strike, option_type,
    ltp, bid, ask, oi, oi_day_change, volume, iv, underlying_ltp
)
    Append-only — every fetch inserts a new batch of rows stamped with
    fetch_timestamp, so intraday history of the chain is preserved rather
    than overwritten. `id` is an autoincrement surrogate key.

    UNIQUE (instrument, expiry, strike, option_type, fetch_timestamp)
    prevents accidental duplicate inserts within the same fetch.

instrument_dump_meta(exchange, fetched_at)
    Tracks when the NFO instrument master was last refreshed, so
    data/instruments.py knows whether to re-download it.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from config import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS spot_ohlc (
    instrument TEXT NOT NULL,
    date       TEXT NOT NULL,   -- ISO date, YYYY-MM-DD
    open       REAL NOT NULL,
    high       REAL NOT NULL,
    low        REAL NOT NULL,
    close      REAL NOT NULL,
    volume     REAL,
    source     TEXT NOT NULL,   -- e.g. "kite"
    PRIMARY KEY (instrument, date)
);

CREATE TABLE IF NOT EXISTS india_vix (
    date  TEXT NOT NULL PRIMARY KEY,
    open  REAL NOT NULL,
    high  REAL NOT NULL,
    low   REAL NOT NULL,
    close REAL NOT NULL,
    source TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS option_chain_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument      TEXT NOT NULL,
    expiry          TEXT NOT NULL,   -- ISO date
    fetch_timestamp TEXT NOT NULL,   -- ISO datetime, when this snapshot was pulled
    strike          REAL NOT NULL,
    option_type     TEXT NOT NULL,   -- "CE" | "PE"
    ltp             REAL,
    bid             REAL,
    ask             REAL,
    oi              REAL,
    oi_day_change   REAL,
    volume          REAL,
    iv              REAL,            -- computed via BS inversion — see data/option_chain.py
    underlying_ltp  REAL,
    UNIQUE (instrument, expiry, strike, option_type, fetch_timestamp)
);

CREATE INDEX IF NOT EXISTS idx_chain_lookup
    ON option_chain_snapshots (instrument, expiry, fetch_timestamp);

CREATE TABLE IF NOT EXISTS instrument_dump_meta (
    exchange   TEXT NOT NULL PRIMARY KEY,
    fetched_at TEXT NOT NULL
);
"""


@contextmanager
def get_conn(db_path: Path = DB_PATH):
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path = DB_PATH) -> None:
    with get_conn(db_path) as conn:
        conn.executescript(_SCHEMA)


# ── spot / vix ──────────────────────────────────────────────────────────────

def upsert_spot_ohlc(df: pd.DataFrame, instrument: str, source: str = "kite",
                      db_path: Path = DB_PATH) -> int:
    """df must have a DatetimeIndex and columns open/high/low/close[/volume]."""
    if df.empty:
        return 0
    rows = [
        (instrument, idx.strftime("%Y-%m-%d"), r["open"], r["high"], r["low"],
         r["close"], r.get("volume"), source)
        for idx, r in df.iterrows()
    ]
    with get_conn(db_path) as conn:
        conn.executemany(
            """INSERT INTO spot_ohlc (instrument, date, open, high, low, close, volume, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(instrument, date) DO UPDATE SET
                 open=excluded.open, high=excluded.high, low=excluded.low,
                 close=excluded.close, volume=excluded.volume, source=excluded.source""",
            rows,
        )
    return len(rows)


def read_spot_ohlc(instrument: str, start: str, end: str,
                    db_path: Path = DB_PATH) -> pd.DataFrame:
    with get_conn(db_path) as conn:
        df = pd.read_sql_query(
            """SELECT date, open, high, low, close, volume FROM spot_ohlc
               WHERE instrument = ? AND date BETWEEN ? AND ?
               ORDER BY date""",
            conn, params=(instrument, start, end),
        )
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def upsert_vix(df: pd.DataFrame, source: str = "kite", db_path: Path = DB_PATH) -> int:
    if df.empty:
        return 0
    rows = [
        (idx.strftime("%Y-%m-%d"), r["open"], r["high"], r["low"], r["close"], source)
        for idx, r in df.iterrows()
    ]
    with get_conn(db_path) as conn:
        conn.executemany(
            """INSERT INTO india_vix (date, open, high, low, close, source)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(date) DO UPDATE SET
                 open=excluded.open, high=excluded.high, low=excluded.low,
                 close=excluded.close, source=excluded.source""",
            rows,
        )
    return len(rows)


def read_vix(start: str, end: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    with get_conn(db_path) as conn:
        df = pd.read_sql_query(
            """SELECT date, open, high, low, close FROM india_vix
               WHERE date BETWEEN ? AND ? ORDER BY date""",
            conn, params=(start, end),
        )
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


# ── option chain ────────────────────────────────────────────────────────────

def insert_option_chain_snapshot(df: pd.DataFrame, db_path: Path = DB_PATH) -> int:
    """
    df must have columns: instrument, expiry, fetch_timestamp, strike,
    option_type, ltp, bid, ask, oi, oi_day_change, volume, iv, underlying_ltp
    """
    if df.empty:
        return 0
    cols = ["instrument", "expiry", "fetch_timestamp", "strike", "option_type",
            "ltp", "bid", "ask", "oi", "oi_day_change", "volume", "iv", "underlying_ltp"]
    rows = list(df[cols].itertuples(index=False, name=None))
    with get_conn(db_path) as conn:
        conn.executemany(
            f"""INSERT OR IGNORE INTO option_chain_snapshots ({", ".join(cols)})
                VALUES ({", ".join("?" for _ in cols)})""",
            rows,
        )
    return len(rows)


def read_latest_option_chain(instrument: str, expiry: str,
                              db_path: Path = DB_PATH) -> pd.DataFrame:
    """Most recent snapshot on/before now for the given instrument+expiry."""
    with get_conn(db_path) as conn:
        latest_ts = conn.execute(
            """SELECT MAX(fetch_timestamp) FROM option_chain_snapshots
               WHERE instrument = ? AND expiry = ?""",
            (instrument, expiry),
        ).fetchone()[0]
        if latest_ts is None:
            return pd.DataFrame()
        df = pd.read_sql_query(
            """SELECT * FROM option_chain_snapshots
               WHERE instrument = ? AND expiry = ? AND fetch_timestamp = ?
               ORDER BY strike, option_type""",
            conn, params=(instrument, expiry, latest_ts),
        )
    return df


def latest_snapshot_age_seconds(instrument: str, expiry: str,
                                 db_path: Path = DB_PATH) -> float | None:
    """Seconds since the last cached snapshot, or None if there isn't one."""
    with get_conn(db_path) as conn:
        latest_ts = conn.execute(
            """SELECT MAX(fetch_timestamp) FROM option_chain_snapshots
               WHERE instrument = ? AND expiry = ?""",
            (instrument, expiry),
        ).fetchone()[0]
    if latest_ts is None:
        return None
    age = pd.Timestamp.now() - pd.Timestamp(latest_ts)
    return age.total_seconds()


# ── instrument dump freshness ───────────────────────────────────────────────

def get_instrument_dump_fetched_at(exchange: str, db_path: Path = DB_PATH) -> str | None:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT fetched_at FROM instrument_dump_meta WHERE exchange = ?", (exchange,)
        ).fetchone()
    return row[0] if row else None


def set_instrument_dump_fetched_at(exchange: str, fetched_at: str, db_path: Path = DB_PATH) -> None:
    with get_conn(db_path) as conn:
        conn.execute(
            """INSERT INTO instrument_dump_meta (exchange, fetched_at) VALUES (?, ?)
               ON CONFLICT(exchange) DO UPDATE SET fetched_at=excluded.fetched_at""",
            (exchange, fetched_at),
        )
