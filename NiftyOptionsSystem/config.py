"""
config.py
---------
Central configuration: paths, instrument specs, rate limits, and signal
defaults. Nothing in here should require a live Kite connection to import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = CACHE_DIR / "market_data.sqlite"
INSTRUMENTS_PARQUET = CACHE_DIR / "instruments_nfo.parquet"

# .env lives at the repo root (Alpha/.env was consolidated there) — this is
# where KITE_API_KEY / KITE_API_SECRET / KITE_ACCESS_TOKEN are read from.
ENV_PATH = REPO_ROOT / ".env"


@dataclass(frozen=True)
class InstrumentSpec:
    name: str                  # Kite "name" field on NFO option contracts, e.g. "NIFTY"
    spot_symbol: str           # Kite tradingsymbol on NSE indices segment, e.g. "NIFTY 50"
    exchange: str = "NSE"
    option_exchange: str = "NFO"
    strike_step: int = 50
    lot_size: int = 25         # NOTE: NSE revises lot sizes periodically — verify before trading.
    weekly_expiry: bool = True


# Lot sizes / strike steps as of the latest NSE circular we know of (Nov 2024).
# ASSUMPTION FLAG: verify current lot size and expiry-day cadence against the
# live NFO instrument dump before trading — NSE has changed both Nifty's
# weekly expiry day and discontinued Bank Nifty weekly expiry in the past.
# This toolkit resolves expiries dynamically from the instrument dump
# (data/instruments.py) rather than hardcoding a day-of-week, specifically
# to avoid rotting when NSE changes the calendar again.
INSTRUMENTS: dict[str, InstrumentSpec] = {
    "NIFTY": InstrumentSpec(
        name="NIFTY", spot_symbol="NIFTY 50",
        strike_step=50, lot_size=75,  # placeholder — refreshed from instrument dump lot_size at runtime
    ),
    "BANKNIFTY": InstrumentSpec(
        name="BANKNIFTY", spot_symbol="NIFTY BANK",
        strike_step=100, lot_size=30,  # placeholder — refreshed from instrument dump lot_size at runtime
    ),
}

VIX_SPOT_SYMBOL = "INDIA VIX"

RISK_FREE_RATE = 0.068     # 10Y G-sec proxy — flagged assumption, not pulled live
DIVIDEND_YIELD = {"NIFTY": 0.015, "BANKNIFTY": 0.010}


@dataclass(frozen=True)
class RateLimitConfig:
    # Kite Connect published limits (2024): roughly 3 req/sec sustained across
    # quote/historical endpoints, with bursts throttled server-side (429).
    # We stay conservative and configurable rather than hard-coding trust in
    # the documented ceiling.
    requests_per_second: float = 3.0
    quote_chunk_size: int = 200        # kite.quote() historically caps around 500 symbols/call; we stay well under it
    max_retries: int = 3
    backoff_base_seconds: float = 1.5


@dataclass(frozen=True)
class SignalConfig:
    iv_lookback_days: int = 252
    realized_vol_window: int = 20
    supertrend_period: int = 10
    supertrend_multiplier: float = 3.0
    adx_period: int = 14
    adx_range_bound_threshold: float = 20.0   # ADX below this => no clear trend
    adx_strong_trend_threshold: float = 25.0
    oi_concentration_strikes: int = 5          # +/- N strikes around spot for concentration ratio
    iv_rank_no_trade_below: float = 20.0       # premium likely too thin to sell


@dataclass(frozen=True)
class StrikeSelectionConfig:
    target_delta: float = 0.175            # midpoint of the commonly-used 0.15-0.20 short-leg range
    max_probability_of_touch: float = 0.35  # reject candidate short strikes riskier than this
    oi_wall_multiple: float = 3.0           # a strike's OI > this x its LOCAL median OI counts as a "wall"
    oi_wall_local_window: int = 5           # "local" = median over the nearest `window` strikes on each side
    oi_wall_band_strikes: int = 1           # avoid strikes within this many steps of a wall
    spread_width_strikes: int = 4           # long leg sits this many strike-steps beyond the short leg


RATE_LIMIT = RateLimitConfig()
SIGNALS = SignalConfig()
STRIKE_SELECTION = StrikeSelectionConfig()
