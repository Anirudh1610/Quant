"""strategy.py — Volatility trading signal generation from model predictions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

import numpy as np
import pandas as pd

from utils import get_logger, load_config

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Signal enum
# ─────────────────────────────────────────────────────────────────────────────

class Signal(IntEnum):
    SHORT_VOL = -1   # sell volatility (predicted RV < implied: vol overpriced)
    NEUTRAL = 0
    LONG_VOL = 1     # buy volatility  (predicted RV > implied: vol underpriced)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StrategyConfig:
    threshold: float = 0.02   # minimum edge (abs pct) to trigger a trade
    hold_days: int = 5
    transaction_cost: float = 0.0005   # per-side fraction of notional
    slippage: float = 0.0002          # per-side fraction of notional

    @classmethod
    def from_cfg(cls, cfg: dict) -> "StrategyConfig":
        s = cfg.get("strategy", {})
        return cls(
            threshold=s.get("threshold", 0.02),
            hold_days=s.get("hold_days", 5),
            transaction_cost=s.get("transaction_cost", 0.0005),
            slippage=s.get("slippage", 0.0002),
        )

    @property
    def round_trip_cost(self) -> float:
        return 2 * (self.transaction_cost + self.slippage)


# ─────────────────────────────────────────────────────────────────────────────
# Signal generation
# ─────────────────────────────────────────────────────────────────────────────

class SignalRow(NamedTuple):
    date: pd.Timestamp
    predicted_rv5: float
    implied_5d_vol: float
    edge: float
    signal: Signal


def generate_signals(
    predicted_rv5: pd.Series,
    implied_vol_ann: pd.Series,
    cfg: StrategyConfig | None = None,
    implied_5d_vol: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Compare LSTM predictions to annualized implied vol (VIX / 100).

    Both series must be in the same units (annualized fraction):
        predicted_rv5   — LSTM output, e.g. 0.14 = 14% annualized RV
        implied_vol_ann — VIX / 100,   e.g. 0.12 = 12% annualized IV

    edge = predicted_rv5 - implied_vol_ann

    LONG  VOL : edge >  threshold   (realized > implied → vol underpriced)
    SHORT VOL : edge < -threshold   (realized < implied → vol overpriced)
    NEUTRAL   : otherwise

    threshold = 0.02 means a 2-percentage-point annualized vol difference.

    Returns
    -------
    DataFrame with columns: predicted_rv5, implied_vol_ann, implied_5d_vol,
                            edge, signal, signal_name
    """
    if cfg is None:
        cfg = StrategyConfig.from_cfg(load_config())

    idx = predicted_rv5.index.intersection(implied_vol_ann.index)
    pred = predicted_rv5.reindex(idx)
    impl = implied_vol_ann.reindex(idx)

    edge = pred - impl

    signal = pd.Series(Signal.NEUTRAL, index=idx, dtype=int)
    signal[edge > cfg.threshold] = Signal.LONG_VOL
    signal[edge < -cfg.threshold] = Signal.SHORT_VOL

    df = pd.DataFrame({
        "predicted_rv5": pred,
        "implied_vol_ann": impl,
        "implied_5d_vol": implied_5d_vol.reindex(idx) if implied_5d_vol is not None else np.nan,
        "edge": edge,
        "signal": signal,
        "signal_name": signal.map({
            Signal.LONG_VOL: "LONG VOL",
            Signal.NEUTRAL: "NEUTRAL",
            Signal.SHORT_VOL: "SHORT VOL",
        }),
    })

    counts = df["signal_name"].value_counts()
    logger.info(
        "Signals — LONG VOL: %d  NEUTRAL: %d  SHORT VOL: %d",
        counts.get("LONG VOL", 0),
        counts.get("NEUTRAL", 0),
        counts.get("SHORT VOL", 0),
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Position sizing (unit notional; backtest scales by capital separately)
# ─────────────────────────────────────────────────────────────────────────────

def compute_position_sizes(
    signals: pd.DataFrame,
    vol_scalar: bool = True,
) -> pd.Series:
    """
    Return a series of position weights ∈ {-1, 0, +1}.

    vol_scalar: if True, scale position by |edge| (larger edge → larger size,
                capped at 1).  This implements a simple edge-proportional
                sizing rather than fixed unit positions.
    """
    sig = signals["signal"].astype(float)

    if vol_scalar:
        edge_mag = signals["edge"].abs().clip(upper=0.10) / 0.10  # normalise to [0, 1]
        return sig * edge_mag
    return sig
