"""dataset_builder.py — Sequence creation, train/val/test splitting, normalisation."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from feature_engineering import FEATURE_COLS, TARGET_COL, BENCHMARK_COL
from utils import get_logger, load_config, ensure_dirs, project_root

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data splits
# ─────────────────────────────────────────────────────────────────────────────

class Split(NamedTuple):
    name: str
    df: pd.DataFrame


def make_splits(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
) -> tuple[Split, Split, Split]:
    train = Split("train", df.loc[:train_end])
    val = Split("val", df.loc[train_end:val_end].iloc[1:])
    test = Split("test", df.loc[val_end:].iloc[1:])
    logger.info(
        "Splits — train: %d  val: %d  test: %d",
        len(train.df), len(val.df), len(test.df),
    )
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# Sequence builder
# ─────────────────────────────────────────────────────────────────────────────

def make_sequences(
    feature_array: np.ndarray,
    target_array: np.ndarray,
    dates: pd.DatetimeIndex,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Convert flat arrays to (samples, lookback, features) LSTM sequences.

    Returns
    -------
    X       : (N, lookback, n_features)
    y       : (N,)
    seq_dates : DatetimeIndex aligned to the last day of each sequence
    """
    X, y, seq_dates = [], [], []
    n = len(feature_array)
    for i in range(lookback, n):
        X.append(feature_array[i - lookback: i])
        y.append(target_array[i])
        seq_dates.append(dates[i])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y, pd.DatetimeIndex(seq_dates)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset container
# ─────────────────────────────────────────────────────────────────────────────

class VolatilityDataset:
    """
    Wraps the full train/val/test pipeline.

    Usage
    -----
        ds = VolatilityDataset(feature_df)
        ds.fit_and_transform()
        X_train, y_train = ds.X_train, ds.y_train
        ds.save_scaler("models/scaler.pkl")
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cfg: dict | None = None,
        feature_cols: list[str] | None = None,
    ):
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg
        self.lookback: int = cfg["features"]["lookback"]
        self.train_end: str = cfg["dates"]["train_end"]
        self.val_end: str = cfg["dates"]["val_end"]
        requested = feature_cols or FEATURE_COLS

        # Drop rows where target is NaN
        df = df.copy()
        df = df.dropna(subset=[TARGET_COL])

        # Drop feature columns that are entirely NaN (e.g. OI when not available)
        all_nan = [c for c in requested if c in df.columns and df[c].isna().all()]
        if all_nan:
            logger.warning(
                "Dropping %d all-NaN feature columns (not available in source data): %s",
                len(all_nan), all_nan,
            )
        self.feature_cols: list[str] = [c for c in requested if c in df.columns and c not in all_nan]

        required = self.feature_cols + [TARGET_COL, BENCHMARK_COL]
        df = df[[c for c in required if c in df.columns]].copy()

        # Forward-fill then drop remaining NaNs in features
        df[self.feature_cols] = df[self.feature_cols].ffill().bfill()
        df = df.dropna(subset=self.feature_cols)

        self.df = df
        self.scaler: StandardScaler | None = None

        # Populated by fit_and_transform()
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.X_val: np.ndarray | None = None
        self.y_val: np.ndarray | None = None
        self.X_test: np.ndarray | None = None
        self.y_test: np.ndarray | None = None
        self.dates_train: pd.DatetimeIndex | None = None
        self.dates_val: pd.DatetimeIndex | None = None
        self.dates_test: pd.DatetimeIndex | None = None
        self.implied_val: np.ndarray | None = None
        self.implied_test: np.ndarray | None = None

    # ── pipeline ─────────────────────────────────────────────────────────────

    def fit_and_transform(self) -> "VolatilityDataset":
        train, val, test = make_splits(self.df, self.train_end, self.val_end)

        # Fit scaler only on training features
        self.scaler = StandardScaler()
        train_feats = self.scaler.fit_transform(train.df[self.feature_cols].values)
        val_feats = self.scaler.transform(val.df[self.feature_cols].values)
        test_feats = self.scaler.transform(test.df[self.feature_cols].values)

        train_target = train.df[TARGET_COL].values
        val_target = val.df[TARGET_COL].values
        test_target = test.df[TARGET_COL].values

        # Build sequences — note: the first `lookback` rows in each split are
        # used as context only and don't generate a labelled sample.
        # For val and test we prepend the tail of the previous split to avoid
        # losing a full lookback window.
        def _seq(feats, targets, dates):
            return make_sequences(feats, targets, dates, self.lookback)

        # For contiguous context, prepend tail of train to val, and tail of val to test
        lb = self.lookback

        # val context: append last `lookback` rows of train
        val_f = np.concatenate([train_feats[-lb:], val_feats])
        val_t = np.concatenate([train_target[-lb:], val_target])
        val_d = train.df.index[-lb:].append(val.df.index)

        test_f = np.concatenate([val_feats[-lb:], test_feats])
        test_t = np.concatenate([val_target[-lb:], test_target])
        test_d = val.df.index[-lb:].append(test.df.index)

        self.X_train, self.y_train, self.dates_train = _seq(
            train_feats, train_target, train.df.index
        )
        self.X_val, self.y_val, self.dates_val = _seq(val_f, val_t, val_d)
        self.X_test, self.y_test, self.dates_test = _seq(test_f, test_t, test_d)

        # Store implied vol (annualized) aligned to sequence dates.
        # Using implied_vol_ann (= VIX/100) keeps signal edge in the same
        # units as predicted_rv5 (annualized).  implied_5d_vol is kept
        # separately for plots.
        ann_col = "implied_vol_ann" if "implied_vol_ann" in self.df.columns else BENCHMARK_COL
        self.implied_val = self.df[ann_col].reindex(self.dates_val).values
        self.implied_test = self.df[ann_col].reindex(self.dates_test).values

        # Also expose 5-day vol for visualisation
        bench_5d = "implied_5d_vol"
        self.implied_5d_val = (
            self.df[bench_5d].reindex(self.dates_val).values
            if bench_5d in self.df.columns else self.implied_val
        )
        self.implied_5d_test = (
            self.df[bench_5d].reindex(self.dates_test).values
            if bench_5d in self.df.columns else self.implied_test
        )

        logger.info(
            "Sequences — train: %s  val: %s  test: %s",
            self.X_train.shape, self.X_val.shape, self.X_test.shape,
        )
        return self

    def n_features(self) -> int:
        assert self.X_train is not None, "Call fit_and_transform() first"
        return self.X_train.shape[2]

    # ── persistence ──────────────────────────────────────────────────────────

    def save_scaler(self, path: str | Path | None = None) -> Path:
        assert self.scaler is not None, "Call fit_and_transform() first"
        if path is None:
            path = project_root() / self.cfg["model"]["model_dir"] / self.cfg["model"]["scaler_name"]
        path = Path(path)
        ensure_dirs(path.parent)
        joblib.dump(self.scaler, path)
        logger.info("Scaler saved → %s", path)
        return path

    @staticmethod
    def load_scaler(path: str | Path) -> StandardScaler:
        return joblib.load(path)

    def transform_new(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
    ) -> np.ndarray:
        """Apply fitted scaler to new data and return a sequence array."""
        assert self.scaler is not None, "Call fit_and_transform() first"
        cols = feature_cols or self.feature_cols
        feats = self.scaler.transform(df[cols].ffill().bfill().values)
        if len(feats) < self.lookback:
            raise ValueError(
                f"Need at least {self.lookback} rows for inference, got {len(feats)}"
            )
        X = []
        for i in range(self.lookback, len(feats) + 1):
            X.append(feats[i - self.lookback: i])
        return np.array(X, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward window generator
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_windows(
    df: pd.DataFrame,
    initial_train_years: int = 3,
    retrain_freq: str = "monthly",
) -> list[dict]:
    """
    Generate walk-forward (train_end, test_start, test_end) tuples.

    Parameters
    ----------
    df               : Full feature DataFrame (sorted by date).
    initial_train_years : Minimum training history before first prediction.
    retrain_freq     : "monthly" or "quarterly".

    Returns
    -------
    List of dicts with keys: train_end, test_start, test_end.
    """
    start = df.index.min()
    end = df.index.max()
    train_cutoff = start + pd.DateOffset(years=initial_train_years)

    if retrain_freq == "monthly":
        offsets = pd.date_range(train_cutoff, end, freq="MS")
    elif retrain_freq == "quarterly":
        offsets = pd.date_range(train_cutoff, end, freq="QS")
    else:
        raise ValueError(f"retrain_freq must be 'monthly' or 'quarterly', got '{retrain_freq}'")

    windows = []
    for i, test_start in enumerate(offsets):
        train_end = test_start - pd.Timedelta(days=1)
        test_end = offsets[i + 1] - pd.Timedelta(days=1) if i + 1 < len(offsets) else end
        if train_end >= start and test_start <= end:
            windows.append({
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            })
    logger.info("Walk-forward: %d windows (%s retraining)", len(windows), retrain_freq)
    return windows
