from __future__ import annotations

import pandas as pd
import pytest

from backtest.engine import BacktestConfig
from backtest.sweep import evaluate_is_oos, run_grid_search
from config import StrikeSelectionConfig


def _ohlc(closes: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(closes))
    close = pd.Series(closes, index=dates)
    return pd.DataFrame({"open": close, "high": close * 1.002, "low": close * 0.998, "close": close})


def _vix(values: list[float], start: str = "2024-01-01") -> pd.Series:
    dates = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=dates, name="close")


LOOSE_CFG = StrikeSelectionConfig(max_probability_of_touch=1.0, oi_wall_multiple=999.0,
                                   oi_wall_band_strikes=0, spread_width_strikes=4)
BASE_CFG = BacktestConfig(strike_selection_cfg=LOOSE_CFG)


@pytest.fixture(scope="module")
def small_history():
    n = 20
    vix_values = [30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 9, 8, 8, 8, 8, 8, 8, 8, 8]
    return _ohlc([24300.0] * n), _vix(vix_values)


class TestRunGridSearch:
    def test_row_count_matches_cartesian_product(self, small_history):
        ohlc, vix = small_history
        results = run_grid_search(
            "NIFTY", ohlc, vix,
            tenor_days_grid=[7, 10],
            profit_target_grid=[0.3, 0.5],
            stop_loss_grid=[1.0],
            target_delta_grid=[0.175],
            assumed_range_score_grid=[50],
            base_cfg=BASE_CFG, min_trades=1,
        )
        assert len(results) == 2 * 2 * 1 * 1 * 1

    def test_swept_params_appear_in_output(self, small_history):
        ohlc, vix = small_history
        results = run_grid_search(
            "NIFTY", ohlc, vix,
            tenor_days_grid=[7], profit_target_grid=[0.3], stop_loss_grid=[1.0],
            target_delta_grid=[0.175], assumed_range_score_grid=[50],
            base_cfg=BASE_CFG, min_trades=1,
        )
        row = results.iloc[0]
        assert row["tenor_days"] == 7
        assert row["profit_target_pct"] == 0.3
        assert row["stop_loss_pct"] == 1.0
        assert row["target_delta"] == 0.175
        assert row["assumed_range_score"] == 50

    def test_performance_report_fields_present(self, small_history):
        ohlc, vix = small_history
        results = run_grid_search(
            "NIFTY", ohlc, vix,
            tenor_days_grid=[7], profit_target_grid=[0.3], stop_loss_grid=[1.0],
            target_delta_grid=[0.175], assumed_range_score_grid=[50],
            base_cfg=BASE_CFG, min_trades=1,
        )
        expected = {"n_trades", "win_rate", "total_return_pct", "max_drawdown_pct", "sharpe_ratio"}
        assert expected <= set(results.columns)

    def test_reliable_flag_reflects_min_trades_threshold(self, small_history):
        ohlc, vix = small_history
        results = run_grid_search(
            "NIFTY", ohlc, vix,
            tenor_days_grid=[7], profit_target_grid=[0.3], stop_loss_grid=[1.0],
            target_delta_grid=[0.175], assumed_range_score_grid=[50],
            base_cfg=BASE_CFG, min_trades=10_000,  # impossibly high -> nothing qualifies as reliable
        )
        assert not results["reliable"].any()

    def test_different_target_delta_values_produce_independent_rows(self, small_history):
        ohlc, vix = small_history
        results = run_grid_search(
            "NIFTY", ohlc, vix,
            tenor_days_grid=[7], profit_target_grid=[0.3], stop_loss_grid=[1.0],
            target_delta_grid=[0.10, 0.25], assumed_range_score_grid=[50],
            base_cfg=BASE_CFG, min_trades=1,
        )
        assert sorted(results["target_delta"].tolist()) == [0.10, 0.25]


class TestEvaluateIsOos:
    def test_one_row_per_config(self, small_history):
        ohlc, vix = small_history
        configs = [
            BacktestConfig(tenor_days=7, strike_selection_cfg=LOOSE_CFG),
            BacktestConfig(tenor_days=10, strike_selection_cfg=LOOSE_CFG),
        ]
        result = evaluate_is_oos("NIFTY", ohlc, vix, ohlc, vix, configs, labels=["a", "b"])
        assert len(result) == 2
        assert list(result["label"]) == ["a", "b"]

    def test_has_both_is_and_oos_columns(self, small_history):
        ohlc, vix = small_history
        configs = [BacktestConfig(tenor_days=7, strike_selection_cfg=LOOSE_CFG)]
        result = evaluate_is_oos("NIFTY", ohlc, vix, ohlc, vix, configs)
        for prefix in ("is_", "oos_"):
            assert any(c.startswith(prefix) for c in result.columns)

    def test_identical_is_and_oos_data_gives_identical_results(self, small_history):
        # sanity check: same data in both windows -> IS and OOS metrics must match exactly
        ohlc, vix = small_history
        configs = [BacktestConfig(tenor_days=7, strike_selection_cfg=LOOSE_CFG)]
        result = evaluate_is_oos("NIFTY", ohlc, vix, ohlc, vix, configs).iloc[0]
        assert result["is_n_trades"] == result["oos_n_trades"]
        assert result["is_return_pct"] == pytest.approx(result["oos_return_pct"])
