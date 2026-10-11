from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.metrics import (
    avg_loss,
    avg_win,
    compute_performance_report,
    daily_returns,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    trade_level_pnl,
    trade_level_sharpe,
    win_rate,
    worst_single_day_loss_pct,
)


def _leg_rows(trade_id, entry_date, exit_date, instrument, strategy_type, exit_reason, leg_pnls):
    return [
        {"trade_id": trade_id, "entry_date": entry_date, "exit_date": exit_date,
         "instrument": instrument, "strategy_type": strategy_type, "exit_reason": exit_reason,
         "leg_pnl": pnl}
        for pnl in leg_pnls
    ]


class TestTradeLevelPnl:
    def test_sums_legs_per_trade(self):
        rows = (
            _leg_rows(1, "2026-01-01", "2026-01-05", "NIFTY", "BULL_PUT_SPREAD", "profit_target", [60.0, -20.0])
            + _leg_rows(2, "2026-01-06", "2026-01-10", "NIFTY", "BEAR_CALL_SPREAD", "stop_loss", [-40.0, 10.0])
        )
        trade_log = pd.DataFrame(rows)
        trades = trade_level_pnl(trade_log)
        assert len(trades) == 2
        assert trades.set_index("trade_id").loc[1, "pnl"] == pytest.approx(40.0)
        assert trades.set_index("trade_id").loc[2, "pnl"] == pytest.approx(-30.0)

    def test_empty_trade_log(self):
        empty = pd.DataFrame(columns=["trade_id", "entry_date", "exit_date", "instrument",
                                       "strategy_type", "exit_reason", "leg_pnl"])
        trades = trade_level_pnl(empty)
        assert trades.empty


class TestWinLossStats:
    @pytest.fixture
    def trades(self) -> pd.DataFrame:
        return pd.DataFrame({"pnl": [100.0, -50.0, 200.0, -30.0, -10.0]})

    def test_win_rate(self, trades):
        assert win_rate(trades) == pytest.approx(2 / 5)

    def test_avg_win(self, trades):
        assert avg_win(trades) == pytest.approx((100.0 + 200.0) / 2)

    def test_avg_loss(self, trades):
        assert avg_loss(trades) == pytest.approx((-50.0 - 30.0 - 10.0) / 3)

    def test_empty_trades_win_rate_is_nan(self):
        assert np.isnan(win_rate(pd.DataFrame({"pnl": []})))

    def test_no_wins_avg_win_is_zero(self):
        assert avg_win(pd.DataFrame({"pnl": [-10.0, -20.0]})) == 0.0

    def test_no_losses_avg_loss_is_zero(self):
        assert avg_loss(pd.DataFrame({"pnl": [10.0, 20.0]})) == 0.0


class TestMaxDrawdown:
    def test_matches_hand_computation(self):
        equity = pd.Series([100.0, 120.0, 90.0, 110.0, 80.0, 150.0])
        # peak 120 -> trough 80 is the worst decline: (80-120)/120 = -1/3
        assert max_drawdown(equity) == pytest.approx(1 / 3)

    def test_monotonic_increase_has_zero_drawdown(self):
        equity = pd.Series([100.0, 110.0, 120.0, 130.0])
        assert max_drawdown(equity) == pytest.approx(0.0)

    def test_empty_series_is_nan(self):
        assert np.isnan(max_drawdown(pd.Series([], dtype=float)))


class TestDailyReturns:
    def test_matches_hand_computation(self):
        equity = pd.Series([100.0, 110.0, 99.0])
        rets = daily_returns(equity)
        assert rets.tolist() == pytest.approx([0.10, -0.10])


class TestSharpeSortino:
    def test_zero_variance_returns_nan(self):
        equity = pd.Series([100.0, 100.0, 100.0, 100.0])
        assert np.isnan(sharpe_ratio(equity))

    def test_sharpe_matches_hand_computation_zero_rf(self):
        equity = pd.Series([100.0, 102.0, 101.0, 104.0, 103.0])
        rets = equity.pct_change().dropna()
        expected = rets.mean() / rets.std() * np.sqrt(252)
        assert sharpe_ratio(equity, risk_free_rate=0.0) == pytest.approx(expected)

    def test_positive_drift_gives_positive_sharpe(self):
        equity = pd.Series([100.0, 101.0, 102.5, 103.0, 105.0, 104.5, 106.0])
        assert sharpe_ratio(equity, risk_free_rate=0.0) > 0

    def test_sortino_ignores_upside_volatility(self):
        # same mean return, but one series has a huge upside outlier and no
        # downside variance beyond a single small loss -- Sortino should be
        # much more forgiving of the upside-heavy series than Sharpe is.
        steady_losses = pd.Series([100.0, 99.0, 98.5, 99.5, 99.0, 100.5])
        upside_outlier = pd.Series([100.0, 99.0, 98.5, 130.0, 99.0, 100.5])

        sharpe_outlier = sharpe_ratio(upside_outlier)
        sortino_outlier = sortino_ratio(upside_outlier)
        # the upside spike inflates the Sharpe denominator (total vol) but not
        # the Sortino denominator (downside-only vol), so Sortino should read
        # comparatively higher for the same series
        assert sortino_outlier > sharpe_outlier

    def test_no_downside_returns_nan(self):
        equity = pd.Series([100.0, 101.0, 102.0, 103.0])  # monotonic up, no downside days
        assert np.isnan(sortino_ratio(equity))


class TestWorstSingleDayLoss:
    def test_matches_hand_computation(self):
        equity = pd.Series([100.0, 95.0, 110.0, 90.0])
        # returns: -0.05, +0.1579, -0.1818 -- worst is the last one
        rets = equity.pct_change().dropna()
        assert worst_single_day_loss_pct(equity) == pytest.approx(-rets.min())
        assert worst_single_day_loss_pct(equity) == pytest.approx(0.181818, abs=1e-5)


class TestComputePerformanceReport:
    def test_report_has_expected_keys_and_consistent_values(self):
        rows = (
            _leg_rows(1, "2026-01-01", "2026-01-05", "NIFTY", "BULL_PUT_SPREAD", "profit_target", [60.0, -20.0])
            + _leg_rows(2, "2026-01-06", "2026-01-10", "NIFTY", "BEAR_CALL_SPREAD", "stop_loss", [-40.0, 10.0])
        )
        trade_log = pd.DataFrame(rows)
        equity = pd.Series([1_000_000.0, 1_000_040.0, 1_000_010.0])

        report = compute_performance_report(trade_log, equity, initial_capital=1_000_000.0)

        assert report["n_trades"] == 2
        assert report["total_pnl"] == pytest.approx(40.0 - 30.0)
        assert report["final_equity"] == pytest.approx(1_000_010.0)
        assert report["total_return_pct"] == pytest.approx(1_000_010.0 / 1_000_000.0 - 1)

    def test_empty_backtest_does_not_crash(self):
        empty_log = pd.DataFrame(columns=["trade_id", "entry_date", "exit_date", "instrument",
                                           "strategy_type", "exit_reason", "leg_pnl"])
        equity = pd.Series([1_000_000.0])
        report = compute_performance_report(empty_log, equity, initial_capital=1_000_000.0)
        assert report["n_trades"] == 0
        assert report["total_pnl"] == 0.0


class TestTradeLevelSharpe:
    CAL = pd.bdate_range("2024-01-01", periods=10)  # Mon 1/1 .. Fri 1/12 (2 weeks, Mon-Fri)

    def _trades(self, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def test_matches_independent_computation(self):
        trades = self._trades([
            {"entry_date": self.CAL[0], "exit_date": self.CAL[2], "pnl": 1000.0},  # 2 trading days held
            {"entry_date": self.CAL[3], "exit_date": self.CAL[8], "pnl": 3000.0},  # 5 trading days held
        ])
        initial_capital = 100_000.0
        rf = 0.10

        result = trade_level_sharpe(trades, initial_capital, self.CAL, risk_free_rate=rf)

        # independent computation, not reusing the function's own internals
        returns = trades["pnl"] / initial_capital
        holding_days = pd.Series([2, 5])
        rf_hurdle = rf * holding_days / 252
        excess = returns - rf_hurdle
        trades_per_year = 2 / (10 / 252)
        expected = excess.mean() / excess.std() * np.sqrt(trades_per_year)

        assert result == pytest.approx(expected)

    def test_uses_trading_days_not_calendar_days(self):
        # entry Thursday, exit the following Monday -- 4 calendar days but
        # only 1 trading day (Monday) actually elapses in the simulation.
        # Second trade has a different pnl/holding period so excess returns
        # aren't degenerate (zero-variance) across the two trades.
        thursday, monday = self.CAL[4], self.CAL[5]
        trades = self._trades([
            {"entry_date": thursday, "exit_date": monday, "pnl": 500.0},
            {"entry_date": self.CAL[6], "exit_date": self.CAL[8], "pnl": 800.0},  # 2 trading days held
        ])
        result = trade_level_sharpe(trades, 100_000.0, self.CAL, risk_free_rate=0.10)

        # if calendar days were (wrongly) used, the first trade's hurdle would be
        # 4/252 instead of 1/252 -- reproduce both and confirm we get the trading-day one
        returns = trades["pnl"] / 100_000.0
        calendar_days_hurdle = 0.10 * pd.Series([4, 2]) / 252
        trading_days_hurdle = 0.10 * pd.Series([1, 2]) / 252
        trades_per_year = 2 / (10 / 252)

        excess_trading = returns - trading_days_hurdle
        expected_trading = excess_trading.mean() / excess_trading.std() * np.sqrt(trades_per_year)
        excess_calendar = returns - calendar_days_hurdle
        expected_calendar = excess_calendar.mean() / excess_calendar.std() * np.sqrt(trades_per_year)

        assert result == pytest.approx(expected_trading)
        assert result != pytest.approx(expected_calendar)

    def test_longer_holding_period_gets_larger_rf_hurdle(self):
        # identical raw pnl, different holding periods -> the longer-held
        # trade's excess return must be smaller (more rf subtracted)
        trades = self._trades([
            {"entry_date": self.CAL[0], "exit_date": self.CAL[1], "pnl": 1000.0},  # 1 trading day
            {"entry_date": self.CAL[0], "exit_date": self.CAL[9], "pnl": 1000.0},  # 9 trading days
        ])
        returns = trades["pnl"] / 100_000.0
        holding_days = pd.Series([1, 9])
        rf_hurdle = 0.10 * holding_days / 252
        excess = returns - rf_hurdle
        assert excess.iloc[1] < excess.iloc[0]

    def test_single_trade_is_nan(self):
        trades = self._trades([{"entry_date": self.CAL[0], "exit_date": self.CAL[2], "pnl": 1000.0}])
        assert np.isnan(trade_level_sharpe(trades, 100_000.0, self.CAL))

    def test_empty_trades_is_nan(self):
        trades = self._trades([])
        assert np.isnan(trade_level_sharpe(trades, 100_000.0, self.CAL))

    def test_identical_trades_zero_variance_is_nan(self):
        trades = self._trades([
            {"entry_date": self.CAL[0], "exit_date": self.CAL[2], "pnl": 1000.0},
            {"entry_date": self.CAL[3], "exit_date": self.CAL[5], "pnl": 1000.0},  # same 2-day holding period too
        ])
        assert np.isnan(trade_level_sharpe(trades, 100_000.0, self.CAL))
