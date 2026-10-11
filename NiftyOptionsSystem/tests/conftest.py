from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def make_gbm_ohlc(n: int, mu: float, sigma: float, s0: float = 100.0,
                   seed: int = 42) -> pd.DataFrame:
    """
    Synthetic daily OHLC via GBM close prices, with high/low widened around
    open/close by an intraday-range fraction of daily vol — good enough for
    testing indicator plumbing, not a substitute for real market data.
    """
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    rets = rng.normal((mu - 0.5 * sigma ** 2) * dt, sigma * np.sqrt(dt), n)
    close = s0 * np.exp(np.cumsum(rets))
    open_ = np.empty(n)
    open_[0] = s0
    open_[1:] = close[:-1]

    intraday = sigma * np.sqrt(dt) * 0.5
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, intraday, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, intraday, n)))

    dates = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=dates)


@pytest.fixture
def trending_up_ohlc() -> pd.DataFrame:
    return make_gbm_ohlc(300, mu=0.60, sigma=0.10, seed=1)


@pytest.fixture
def trending_down_ohlc() -> pd.DataFrame:
    return make_gbm_ohlc(300, mu=-0.60, sigma=0.10, seed=2)


@pytest.fixture
def choppy_ohlc() -> pd.DataFrame:
    return make_gbm_ohlc(300, mu=0.0, sigma=0.12, seed=3)
