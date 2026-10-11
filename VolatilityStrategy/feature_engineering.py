"""feature_engineering.py — All feature construction for the volatility model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils import get_logger, annualized_rv

logger = get_logger(__name__)

TRADING_DAYS = 252


# ─────────────────────────────────────────────────────────────────────────────
# Return features
# ─────────────────────────────────────────────────────────────────────────────

def add_return_features(df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
    c = df[close_col]
    df["ret_1d"] = c.pct_change()
    df["ret_5d"] = c.pct_change(5)
    df["ret_20d"] = c.pct_change(20)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Realized volatility features
# ─────────────────────────────────────────────────────────────────────────────

def add_rv_features(df: pd.DataFrame, ret_col: str = "ret_1d") -> pd.DataFrame:
    r = df[ret_col]
    df["rv5"] = annualized_rv(r, 5)
    df["rv10"] = annualized_rv(r, 10)
    df["rv20"] = annualized_rv(r, 20)

    # Volatility trend ratios
    df["rv5_rv20"] = df["rv5"] / df["rv20"].replace(0, np.nan)
    df["rv10_rv20"] = df["rv10"] / df["rv20"].replace(0, np.nan)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Price / trend features
# ─────────────────────────────────────────────────────────────────────────────

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def add_price_features(
    df: pd.DataFrame,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    atr_window: int = 14,
) -> pd.DataFrame:
    c = df[close_col]

    df["ma20"] = c.rolling(20).mean()
    df["ma50"] = c.rolling(50).mean()
    df["dist_ma20"] = (c - df["ma20"]) / df["ma20"]
    df["dist_ma50"] = (c - df["ma50"]) / df["ma50"]

    if high_col in df.columns and low_col in df.columns:
        df["atr14"] = _atr(df[high_col], df[low_col], c, atr_window)
        df["atr14_pct"] = df["atr14"] / c  # normalised ATR
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Volume features
# ─────────────────────────────────────────────────────────────────────────────

def add_volume_features(df: pd.DataFrame, volume_col: str = "volume") -> pd.DataFrame:
    if volume_col not in df.columns:
        return df
    v = df[volume_col].replace(0, np.nan)
    df["vol_change"] = v.pct_change()
    df["vol_ma20"] = v.rolling(20).mean()
    df["vol_rel_ma20"] = v / df["vol_ma20"].replace(0, np.nan)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Futures features (OI)
# ─────────────────────────────────────────────────────────────────────────────

def add_futures_features(df: pd.DataFrame, oi_col: str = "oi") -> pd.DataFrame:
    if oi_col not in df.columns:
        return df
    oi = df[oi_col].replace(0, np.nan)
    df["oi"] = oi
    df["oi_change"] = oi.diff()
    df["oi_pct_change"] = oi.pct_change()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# VIX features
# ─────────────────────────────────────────────────────────────────────────────

def add_vix_features(df: pd.DataFrame, vix_col: str = "vix") -> pd.DataFrame:
    v = df[vix_col]
    df["vix_change"] = v.pct_change()
    df["vix_ma5"] = v.rolling(5).mean()
    df["vix_ma20"] = v.rolling(20).mean()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Volatility risk premium (VRP) features
# ─────────────────────────────────────────────────────────────────────────────

def add_vrp_features(df: pd.DataFrame, vix_col: str = "vix") -> pd.DataFrame:
    """VIX (annualized %) minus realized vol — measures VRP."""
    vix_ann = df[vix_col] / 100.0   # VIX is quoted as e.g. 18.5 → 0.185
    df["vrp_vix_rv5"] = vix_ann - df["rv5"]
    df["vrp_vix_rv10"] = vix_ann - df["rv10"]
    df["vrp_vix_rv20"] = vix_ann - df["rv20"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Implied 5-day volatility (target benchmark)
# ─────────────────────────────────────────────────────────────────────────────

def compute_implied_5d_vol(vix_series: pd.Series) -> pd.Series:
    """
    Convert annualized India VIX to a 5-trading-day implied vol forecast.

        implied_5d = (VIX / 100) * sqrt(5 / 252)

    This is the fair value of a 5-day ATM straddle's expected move (1 std-dev).
    """
    return (vix_series / 100.0) * np.sqrt(5 / TRADING_DAYS)


# ─────────────────────────────────────────────────────────────────────────────
# Target variable — future 5-day realized vol
# ─────────────────────────────────────────────────────────────────────────────

def compute_target(returns: pd.Series, horizon: int = 5) -> pd.Series:
    """
    Future `horizon`-day realized volatility (annualized), at day t.

    future_rv[t] = std(returns[t+1 .. t+horizon]) * sqrt(252)

    Derivation:
        rolling(horizon).std()[j]  = std(returns[j-horizon+1 .. j])
        shift(-horizon)[t]         = rolling(horizon).std()[t+horizon]
                                   = std(returns[t+1 .. t+horizon])  ✓

    No look-ahead: features at day t are all backward-looking; only the
    LABEL uses future returns, which is correct for supervised learning.
    """
    return (
        returns.rolling(horizon).std().shift(-horizon) * np.sqrt(TRADING_DAYS)
    ).rename("future_rv5")


# ─────────────────────────────────────────────────────────────────────────────
# Master feature builder
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    nifty: pd.DataFrame,
    futures: pd.DataFrame,
    vix: pd.DataFrame,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """
    Join NIFTY, futures, and VIX data then compute all features.

    Returns a single DataFrame indexed by date with all engineered columns
    plus the target `future_rv5` and benchmark `implied_5d_vol`.
    """
    atr_window = 14
    if cfg:
        atr_window = cfg.get("features", {}).get("atr_window", 14)

    # ── 1. Align on common trading dates ────────────────────────────────────
    base_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in nifty.columns]
    df = nifty[base_cols].copy()

    # Volume: prefer nifty, fall back to futures
    if "volume" not in df.columns and "volume" in futures.columns:
        df["volume"] = futures["volume"].reindex(df.index)

    # Merge futures OI and close (for continuity checks)
    for col in ["oi", "volume"]:
        if col not in df.columns and col in futures.columns:
            df[col] = futures[col].reindex(df.index)
    if "oi" in futures.columns:
        df["oi"] = futures["oi"].reindex(df.index)

    # Merge VIX close
    vix_close = vix["close"] if "close" in vix.columns else vix.iloc[:, 0]
    df["vix"] = vix_close.reindex(df.index).ffill()

    # ── 2. Feature engineering ───────────────────────────────────────────────
    df = add_return_features(df)
    df = add_rv_features(df)
    df = add_price_features(df, atr_window=atr_window)
    df = add_volume_features(df)
    df = add_futures_features(df)
    df = add_vix_features(df)
    df = add_vrp_features(df)

    # ── 3. Implied vol (two representations) ────────────────────────────────
    # 5-day absolute: used for visualization and the original benchmark
    df["implied_5d_vol"] = compute_implied_5d_vol(df["vix"])
    # Annualized: used for signal edge — matches the LSTM prediction units
    df["implied_vol_ann"] = df["vix"] / 100.0

    # ── 4. Target ────────────────────────────────────────────────────────────
    df["future_rv5"] = compute_target(df["ret_1d"], horizon=5)

    # ── 5. Ensure every expected column exists (fill missing with NaN) ───────
    for col in FEATURE_COLS + [TARGET_COL, BENCHMARK_COL]:
        if col not in df.columns:
            df[col] = np.nan
            logger.debug("Column '%s' not available — filled with NaN", col)

    missing = [c for c in FEATURE_COLS if df[c].isna().all()]
    if missing:
        logger.warning(
            "These features are all-NaN (data not available from Kite): %s",
            missing,
        )

    logger.info(
        "Feature matrix: %d rows × %d cols  [%s → %s]",
        len(df), df.shape[1],
        df.index.min().date(), df.index.max().date(),
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature column list (excludes raw OHLCV and target/benchmark)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "ret_1d", "ret_5d", "ret_20d",
    "rv5", "rv10", "rv20",
    "rv5_rv20", "rv10_rv20",
    "atr14_pct",
    "ma20", "ma50",
    "dist_ma20", "dist_ma50",
    "vol_change", "vol_rel_ma20",
    "oi_pct_change",
    "vix", "vix_change", "vix_ma5", "vix_ma20",
    "vrp_vix_rv5", "vrp_vix_rv10", "vrp_vix_rv20",
]

TARGET_COL = "future_rv5"
BENCHMARK_COL = "implied_5d_vol"


def get_available_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the subset of FEATURE_COLS that exist and are not entirely NaN in df."""
    return [c for c in FEATURE_COLS if c in df.columns and not df[c].isna().all()]
