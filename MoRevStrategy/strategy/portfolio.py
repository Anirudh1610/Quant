"""
portfolio.py
------------
Portfolio construction and position management for the
cross-sectional momentum + mean-reversion strategy.

Responsibilities
----------------
- Apply NIFTY regime filter to raw entry signals
- Apply volatility filter to suppress high-vol names
- Enforce maximum position limits (long and short sides)
- Optional: sector-neutrality cap
- Equal-weight position sizing within each side
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# REGIME FILTER
# ──────────────────────────────────────────────────────────────────────────────


def apply_regime_filter(
    long_entries: pd.DataFrame,
    short_entries: pd.DataFrame,
    regime: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Suppress entries that are against the NIFTY regime.

    Rules
    -----
    Regime = +1 (BULL) : allow longs, suppress new shorts
    Regime = -1 (BEAR) : allow shorts, suppress new longs

    This keeps the book in alignment with the macro trend, significantly
    reducing the probability of fighting a strong directional move.

    Parameters
    ----------
    long_entries  : pd.DataFrame  Raw boolean long entry signals.
    short_entries : pd.DataFrame  Raw boolean short entry signals.
    regime        : pd.Series     +1 or -1 indexed by date.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Regime-filtered ``(long_entries, short_entries)``.
    """
    # Reindex regime to signal dates (forward-fill any gaps from index holidays)
    regime_aligned = regime.reindex(long_entries.index, method="ffill").fillna(-1)

    bull = (regime_aligned == 1)
    bear = (regime_aligned == -1)

    # Broadcast the scalar regime flag across all ticker columns
    filtered_longs = long_entries.mul(bull, axis=0).astype(bool)
    filtered_shorts = short_entries.mul(bear, axis=0).astype(bool)

    return filtered_longs, filtered_shorts


# ──────────────────────────────────────────────────────────────────────────────
# VOLATILITY FILTER
# ──────────────────────────────────────────────────────────────────────────────


def apply_volatility_filter(
    entries: pd.DataFrame,
    vol_ok: pd.DataFrame,
) -> pd.DataFrame:
    """
    Mask out entry signals for stocks that fail the volatility screen.

    A stock fails the screen if its rolling annualised volatility exceeds
    a multiple of the daily cross-sectional median (computed in main.py).

    Parameters
    ----------
    entries : pd.DataFrame
        Boolean entry signals.
    vol_ok  : pd.DataFrame
        Boolean mask — True where volatility is within acceptable range.

    Returns
    -------
    pd.DataFrame
        Filtered boolean entry signals.
    """
    # Align columns in case some tickers are absent from vol_ok
    vol_ok_aligned = vol_ok.reindex(columns=entries.columns, fill_value=True)
    return (entries & vol_ok_aligned).astype(bool)


# ──────────────────────────────────────────────────────────────────────────────
# POSITION CAP
# ──────────────────────────────────────────────────────────────────────────────


def cap_positions(signal: pd.DataFrame, max_positions: int) -> pd.DataFrame:
    """
    Hard-cap the number of simultaneous positions to ``max_positions`` per day.

    When more stocks qualify than the cap, retain the first ``max_positions``
    in alphabetical (ticker) order. This is deterministic and avoids any
    forward-looking selection criteria that would introduce lookahead bias.

    Implemented with a vectorised cumsum so it runs in O(n×m) instead of
    iterating row-by-row in Python.

    Parameters
    ----------
    signal        : pd.DataFrame  Boolean entry signals.
    max_positions : int           Maximum allowed simultaneous positions.

    Returns
    -------
    pd.DataFrame
        Boolean mask with at most ``max_positions`` True values per row.
    """
    if max_positions <= 0:
        return pd.DataFrame(False, index=signal.index, columns=signal.columns)
    # Sort columns alphabetically for deterministic tie-breaking (same as before)
    sorted_cols = sorted(signal.columns)
    s = signal[sorted_cols].astype(float)
    # cumsum along sorted columns: keep positions where running count ≤ max_positions
    keep = (s.cumsum(axis=1) <= max_positions) & (s > 0)
    return keep.reindex(columns=signal.columns).fillna(False).astype(bool)


# ──────────────────────────────────────────────────────────────────────────────
# SECTOR NEUTRALITY (OPTIONAL)
# ──────────────────────────────────────────────────────────────────────────────


def sector_neutral_selection(
    long_mask: pd.DataFrame,
    short_mask: pd.DataFrame,
    sector_map: dict[str, str],
    max_per_sector: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Enforce a per-sector position cap on each side of the book.

    For each day, iterate through the selected stocks in alphabetical order
    and drop any ticker whose sector already has ``max_per_sector`` names
    on that side. This prevents sector concentration (e.g., 5 IT longs in
    a bull-IT environment inflating returns non-representatively).

    Parameters
    ----------
    long_mask      : pd.DataFrame   Boolean long signals.
    short_mask     : pd.DataFrame   Boolean short signals.
    sector_map     : dict           {ticker: sector_name} mapping.
    max_per_sector : int            Max stocks from the same sector per side.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Sector-neutral ``(long_mask, short_mask)``.
    """

    def _filter_one_side(mask: pd.DataFrame) -> pd.DataFrame:
        out = mask.copy().astype(float)
        for date, row in mask.iterrows():
            active = sorted(row[row.astype(bool)].index.tolist())
            sector_counts: dict[str, int] = {}
            for ticker in active:
                sector = sector_map.get(ticker, "Unknown")
                count = sector_counts.get(sector, 0)
                if count >= max_per_sector:
                    out.loc[date, ticker] = 0.0
                else:
                    sector_counts[sector] = count + 1
        return out.astype(bool)

    return _filter_one_side(long_mask), _filter_one_side(short_mask)


# ──────────────────────────────────────────────────────────────────────────────
# POSITION SIZING
# ──────────────────────────────────────────────────────────────────────────────


def compute_equal_weight_notional(
    n_long: int,
    n_short: int,
    portfolio_value: float,
    max_long: int = 10,
    max_short: int = 10,
) -> tuple[float, float]:
    """
    Compute per-position notional allocation using equal weighting.

    The portfolio is split 50 / 50 between the long and short books.
    Within each book, capital is divided equally across the active
    positions, capped at ``max_long`` / ``max_short`` slots.

    Parameters
    ----------
    n_long         : int   Number of new long positions to open.
    n_short        : int   Number of new short positions to open.
    portfolio_value: float Current NAV of the portfolio.
    max_long       : int   Maximum long slots.
    max_short      : int   Maximum short slots.

    Returns
    -------
    tuple[float, float]
        ``(long_notional_per_stock, short_notional_per_stock)``
    """
    # Each side gets 50% of the portfolio, divided across max slots
    # (not just active count) to keep position sizes stable over time
    long_alloc = portfolio_value * 0.5
    short_alloc = portfolio_value * 0.5

    long_notional = long_alloc / max_long if n_long > 0 else 0.0
    short_notional = short_alloc / max_short if n_short > 0 else 0.0

    return long_notional, short_notional
