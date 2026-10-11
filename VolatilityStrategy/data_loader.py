"""data_loader.py — Kite Connect historical data with caching and retry logic."""

from __future__ import annotations

import csv
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from utils import (
    get_logger,
    load_config,
    load_env,
    get_kite_credentials,
    save_access_token_to_env,
    to_date,
    nse_expiry_month_year_pairs,
    nifty_futures_symbol,
    ensure_dirs,
    project_root,
)

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Authentication helper
# ─────────────────────────────────────────────────────────────────────────────

def get_kite_client(cfg: dict[str, Any] | None = None):
    """
    Return an authenticated KiteConnect session.

    Credentials are read from environment variables (loaded from .env):
        KITE_API_KEY      — Zerodha app API key
        KITE_ACCESS_TOKEN — generated daily via `python main.py --auth`

    The `cfg` argument is accepted for backward-compatibility but is not
    used for credentials.
    """
    from kiteconnect import KiteConnect

    api_key, access_token, _ = get_kite_credentials()

    if not access_token:
        raise EnvironmentError(
            "KITE_ACCESS_TOKEN not set. Run the auth flow first:\n"
            "  python main.py --auth --request-token <token>\n"
            "Then add KITE_ACCESS_TOKEN=<token> to your .env file."
        )

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


def generate_access_token(request_token: str, api_key: str | None = None, api_secret: str | None = None) -> str:
    """
    Exchange a request_token for an access_token and persist it to .env.

    Reads KITE_API_KEY and KITE_API_SECRET from .env if not provided.
    """
    from kiteconnect import KiteConnect

    if api_key is None or api_secret is None:
        _key, _, _secret = get_kite_credentials()
        api_key = api_key or _key
        api_secret = api_secret or _secret

    if not api_secret:
        raise EnvironmentError(
            "KITE_API_SECRET not set. Add it to your .env file:\n"
            "  KITE_API_SECRET=your_secret_here"
        )

    kite = KiteConnect(api_key=api_key)
    session = kite.generate_session(request_token, api_secret=api_secret)
    token = session["access_token"]

    env_path = save_access_token_to_env(token)
    logger.info("Access token saved → %s", env_path)
    return token


# ─────────────────────────────────────────────────────────────────────────────
# Low-level fetch with retry
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_historical(
    kite,
    instrument_token: int,
    from_date: date,
    to_date_: date,
    interval: str,
    max_retries: int,
    retry_delay: float,
    rate_delay: float,
    failed_log: list[dict],
) -> list[dict]:
    """Fetch one chunk of historical data, retrying on transient errors."""
    for attempt in range(1, max_retries + 1):
        try:
            time.sleep(rate_delay)
            records = kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date_,
                interval=interval,
            )
            return records
        except Exception as exc:
            logger.warning(
                "Attempt %d/%d failed for token=%s from=%s to=%s: %s",
                attempt, max_retries, instrument_token, from_date, to_date_, exc,
            )
            if attempt == max_retries:
                failed_log.append({
                    "instrument_token": instrument_token,
                    "from_date": str(from_date),
                    "to_date": str(to_date_),
                    "error": str(exc),
                })
                return []
            time.sleep(retry_delay * attempt)
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Records → DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def _records_to_df(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index("date").sort_index()
    df.index = df.index.tz_localize(None)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Index / VIX loader
# ─────────────────────────────────────────────────────────────────────────────

def fetch_index_data(
    kite,
    instrument_token: int,
    start: date,
    end: date,
    interval: str,
    max_retries: int,
    retry_delay: float,
    rate_delay: float,
    failed_log: list[dict],
) -> pd.DataFrame:
    """
    Kite historical_data API returns at most 2000 candles per call.
    For daily data this is ~8 years, which usually fits in one call.
    We chunk by year to be safe.
    """
    all_records: list[dict] = []
    cur = start
    while cur <= end:
        chunk_end = min(date(cur.year, 12, 31), end)
        records = _fetch_historical(
            kite, instrument_token, cur, chunk_end,
            interval, max_retries, retry_delay, rate_delay, failed_log,
        )
        all_records.extend(records)
        cur = date(cur.year + 1, 1, 1)
    return _records_to_df(all_records)


# ─────────────────────────────────────────────────────────────────────────────
# Continuous futures builder
# ─────────────────────────────────────────────────────────────────────────────

def fetch_continuous_futures(
    kite,
    start: date,
    end: date,
    interval: str,
    max_retries: int,
    retry_delay: float,
    rate_delay: float,
    failed_log: list[dict],
) -> pd.DataFrame:
    """
    Build a continuous NIFTY front-month futures series.

    Strategy: for each calendar month in [start, end], look up the NFO
    instrument list, find the token for NIFTY<YY><MON>FUT, then fetch
    historical data. Concatenate month slices to produce a clean daily series.

    We use the near-month contract (lowest expiry date above today in that month).
    """
    logger.info("Fetching NFO instrument list for NIFTY futures tokens …")
    try:
        time.sleep(rate_delay)
        nfo_instruments = kite.instruments("NFO")
    except Exception as exc:
        logger.error("Failed to fetch NFO instruments: %s", exc)
        return pd.DataFrame()

    # Build symbol → token map for NIFTY index futures only
    sym_token_map: dict[str, int] = {}
    for inst in nfo_instruments:
        if (
            inst.get("name") == "NIFTY"
            and inst.get("instrument_type") == "FUT"
            and inst.get("segment") == "NFO-FUT"
        ):
            sym = inst["tradingsymbol"]   # e.g. NIFTY25JULFUT
            sym_token_map[sym] = inst["instrument_token"]

    month_pairs = nse_expiry_month_year_pairs(start, end)
    pieces: list[pd.DataFrame] = []

    for year, month in month_pairs:
        sym = nifty_futures_symbol(year, month)
        token = sym_token_map.get(sym)
        if token is None:
            logger.warning("No token found for %s — skipping month", sym)
            continue

        # Fetch from start of month to end of month (clipped to [start, end])
        m_start = max(start, date(year, month, 1))
        if month == 12:
            m_end = min(end, date(year + 1, 1, 1) - timedelta(days=1))
        else:
            m_end = min(end, date(year, month + 1, 1) - timedelta(days=1))

        logger.debug("Fetching %s  %s → %s", sym, m_start, m_end)
        records = _fetch_historical(
            kite, token, m_start, m_end,
            interval, max_retries, retry_delay, rate_delay, failed_log,
        )
        df = _records_to_df(records)
        if not df.empty:
            pieces.append(df)

    if not pieces:
        return pd.DataFrame()

    fut = pd.concat(pieces).sort_index()
    # De-duplicate: keep last record for any date (handles roll overlap)
    fut = fut[~fut.index.duplicated(keep="last")]
    return fut


# ─────────────────────────────────────────────────────────────────────────────
# Save / load CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    logger.info("Saved %d rows → %s", len(df), path)


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df


def _save_failed_log(failed: list[dict], raw_dir: Path) -> None:
    if not failed:
        return
    path = raw_dir / "failed_requests.csv"
    fieldnames = ["instrument_token", "from_date", "to_date", "error"]
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerows(failed)
    logger.warning("Logged %d failed requests → %s", len(failed), path)


# ─────────────────────────────────────────────────────────────────────────────
# Cache-aware public API
# ─────────────────────────────────────────────────────────────────────────────

class DataLoader:
    """
    Loads NIFTY index, NIFTY continuous futures, and India VIX data.

    On first run: fetches from Kite Connect and saves CSVs to raw_dir.
    Subsequent runs: reads from cache unless force_refresh=True.
    """

    NIFTY_FILE = "nifty_index.csv"
    FUTURES_FILE = "nifty_futures_continuous.csv"
    VIX_FILE = "india_vix.csv"

    def __init__(self, cfg: dict[str, Any] | None = None):
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg
        self.raw_dir = project_root() / cfg["data"]["raw_dir"]
        self.interval = cfg["data"]["interval"]
        self.max_retries = cfg["data"].get("max_retries", 3)
        self.retry_delay = cfg["data"].get("retry_delay_sec", 2.0)
        self.rate_delay = cfg["data"].get("rate_limit_delay_sec", 0.4)
        ensure_dirs(self.raw_dir)

    # ── internal helpers ─────────────────────────────────────────────────────

    def _path(self, filename: str) -> Path:
        return self.raw_dir / filename

    def _needs_refresh(self, path: Path, start: date, end: date) -> bool:
        """True if cache doesn't exist or doesn't cover [start, end]."""
        if not path.exists():
            return True
        try:
            df = load_csv(path)
            if df.empty:
                return True
            cached_start = df.index.min().date()
            cached_end = df.index.max().date()
            return cached_start > start or cached_end < end
        except Exception:
            return True

    # ── public load methods ─────────────────────────────────────────────────

    def load_nifty(
        self,
        start: str | date,
        end: str | date | None = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        start_d, end_d = to_date(start), to_date(end)
        path = self._path(self.NIFTY_FILE)

        if force_refresh or self._needs_refresh(path, start_d, end_d):
            logger.info("Fetching NIFTY index from Kite Connect …")
            kite = get_kite_client(self.cfg)
            failed: list[dict] = []
            df = fetch_index_data(
                kite,
                instrument_token=self.cfg["instruments"]["nifty_index_token"],
                start=start_d,
                end=end_d,
                interval=self.interval,
                max_retries=self.max_retries,
                retry_delay=self.retry_delay,
                rate_delay=self.rate_delay,
                failed_log=failed,
            )
            if not df.empty:
                save_csv(df, path)
            _save_failed_log(failed, self.raw_dir)
        else:
            logger.info("Loading NIFTY index from cache: %s", path)
            df = load_csv(path)

        return df.loc[str(start_d): str(end_d)]

    def load_vix(
        self,
        start: str | date,
        end: str | date | None = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        start_d, end_d = to_date(start), to_date(end)
        path = self._path(self.VIX_FILE)

        if force_refresh or self._needs_refresh(path, start_d, end_d):
            logger.info("Fetching India VIX from Kite Connect …")
            kite = get_kite_client(self.cfg)
            failed: list[dict] = []
            df = fetch_index_data(
                kite,
                instrument_token=self.cfg["instruments"]["india_vix_token"],
                start=start_d,
                end=end_d,
                interval=self.interval,
                max_retries=self.max_retries,
                retry_delay=self.retry_delay,
                rate_delay=self.rate_delay,
                failed_log=failed,
            )
            if not df.empty:
                save_csv(df, path)
            _save_failed_log(failed, self.raw_dir)
        else:
            logger.info("Loading India VIX from cache: %s", path)
            df = load_csv(path)

        return df.loc[str(start_d): str(end_d)]

    def load_futures(
        self,
        start: str | date,
        end: str | date | None = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        start_d, end_d = to_date(start), to_date(end)
        path = self._path(self.FUTURES_FILE)

        if force_refresh or self._needs_refresh(path, start_d, end_d):
            logger.info("Building continuous NIFTY futures series from Kite Connect …")
            kite = get_kite_client(self.cfg)
            failed: list[dict] = []
            df = fetch_continuous_futures(
                kite,
                start=start_d,
                end=end_d,
                interval=self.interval,
                max_retries=self.max_retries,
                retry_delay=self.retry_delay,
                rate_delay=self.rate_delay,
                failed_log=failed,
            )
            if not df.empty:
                save_csv(df, path)
            _save_failed_log(failed, self.raw_dir)
        else:
            logger.info("Loading NIFTY futures from cache: %s", path)
            df = load_csv(path)

        return df.loc[str(start_d): str(end_d)]

    def load_all(
        self,
        start: str | date,
        end: str | date | None = None,
        force_refresh: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Convenience: load NIFTY, futures, and VIX in one call."""
        return {
            "nifty": self.load_nifty(start, end, force_refresh),
            "futures": self.load_futures(start, end, force_refresh),
            "vix": self.load_vix(start, end, force_refresh),
        }
