"""NSE Futures Strategy package."""

from .backtest import FuturesBacktestConfig, FuturesBacktestEngine
from .data import load_futures_dataset
from .signals import (
    calculate_momentum, rank_momentum,
    get_long_candidates, get_short_candidates,
    calculate_rsi, calculate_bollinger_bands,
    calculate_sma, calculate_atr,
    india_vix_regime, vix_position_scalar,
    trend_regime, detect_combined_regime,
    get_entry_signals,
)
from .portfolio import compute_position_size, compute_exposure
from .costs import compute_futures_cost, round_trip_cost_pct
from .universe import DEFAULT_UNIVERSE, LOT_SIZES

__all__ = [
    "FuturesBacktestConfig", "FuturesBacktestEngine",
    "load_futures_dataset",
    "calculate_momentum", "rank_momentum",
    "get_long_candidates", "get_short_candidates",
    "calculate_rsi", "calculate_bollinger_bands",
    "calculate_sma", "calculate_atr",
    "india_vix_regime", "vix_position_scalar",
    "trend_regime", "detect_combined_regime", "get_entry_signals",
    "compute_position_size", "compute_exposure",
    "compute_futures_cost", "round_trip_cost_pct",
    "DEFAULT_UNIVERSE", "LOT_SIZES",
]
