from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestConfig, run_backtest
from config import StrikeSelectionConfig


def _ohlc(closes: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(closes))
    close = pd.Series(closes, index=dates)
    # small fixed intraday range around close -- enough for ADX/ATR to not
    # degenerate, doesn't need to be realistic since these tests force
    # strategy_override and don't depend on the trend classification
    return pd.DataFrame({
        "open": close, "high": close * 1.002, "low": close * 0.998, "close": close,
    })


def _vix(values: list[float], start: str = "2024-01-01") -> pd.Series:
    dates = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=dates, name="close")


LOOSE_CFG = StrikeSelectionConfig(target_delta=0.175, max_probability_of_touch=1.0,
                                   oi_wall_multiple=999.0, oi_wall_band_strikes=0, spread_width_strikes=4)


def _assert_no_overlapping_trades(trade_log: pd.DataFrame) -> None:
    if trade_log.empty:
        return
    trades = trade_log.groupby("trade_id").agg(entry_date=("entry_date", "first"),
                                                 exit_date=("exit_date", "first")).sort_values("entry_date")
    for prev_exit, next_entry in zip(trades["exit_date"], trades["entry_date"].iloc[1:]):
        assert next_entry > prev_exit  # no same-day (or overlapping) re-entry


class TestNoTradeWithoutSignal:
    def test_short_history_never_trades(self):
        # too short for iv_rank to warm up -> recommend_strategy_type always
        # returns NO_TRADE, and there's no strategy_override here
        n = 30
        ohlc = _ohlc([24300.0] * n)
        vix = _vix([15.0] * n)
        cfg = BacktestConfig(strike_selection_cfg=LOOSE_CFG)

        result = run_backtest("NIFTY", ohlc, vix, cfg)
        assert result["trade_log"].empty
        # no trades -> every day is "idle", so equity should exactly match
        # pure risk-free compounding from RISK_FREE_RATE, not stay flat
        from config import RISK_FREE_RATE
        from backtest.metrics import TRADING_DAYS_PER_YEAR
        expected = cfg.initial_capital * (1 + RISK_FREE_RATE / TRADING_DAYS_PER_YEAR) ** np.arange(1, n + 1)
        np.testing.assert_allclose(result["equity_curve"].to_numpy(), expected)


class TestProfitTargetExit:
    def test_iv_crush_on_flat_spot_triggers_profit_target(self):
        n = 15
        ohlc = _ohlc([24300.0] * n)
        vix_values = [30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 9, 8, 8, 8]
        vix = _vix(vix_values)

        cfg = BacktestConfig(
            tenor_days=15, min_dte_exit=1, profit_target_pct=0.30, stop_loss_pct=5.0,
            slippage_pct=0.01, strategy_override="BULL_PUT_SPREAD", strike_selection_cfg=LOOSE_CFG,
        )
        result = run_backtest("NIFTY", ohlc, vix, cfg)
        trade_log = result["trade_log"]

        assert not trade_log.empty
        trades = trade_log.groupby("trade_id").agg(pnl=("leg_pnl", "sum"), exit_reason=("exit_reason", "first"))
        assert (trades["exit_reason"] == "profit_target").any()
        profit_trade = trades[trades["exit_reason"] == "profit_target"].iloc[0]
        assert profit_trade["pnl"] > 0
        _assert_no_overlapping_trades(trade_log)

    def test_final_equity_exceeds_initial_capital(self):
        n = 15
        ohlc = _ohlc([24300.0] * n)
        vix = _vix([30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 9, 8, 8, 8])
        cfg = BacktestConfig(
            tenor_days=15, min_dte_exit=1, profit_target_pct=0.30, stop_loss_pct=5.0,
            slippage_pct=0.01, strategy_override="BULL_PUT_SPREAD", strike_selection_cfg=LOOSE_CFG,
        )
        result = run_backtest("NIFTY", ohlc, vix, cfg)
        assert result["equity_curve"].iloc[-1] > cfg.initial_capital


class TestStopLossExit:
    def test_sharp_adverse_move_triggers_stop_loss(self):
        n = 15
        # entry at 24300, then a sharp crash the very next day -- bad for a short put spread
        closes = [24300.0, 24300.0] + [22000.0] * (n - 2)
        ohlc = _ohlc(closes)
        vix = _vix([15, 15, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32])

        cfg = BacktestConfig(
            tenor_days=15, min_dte_exit=1, profit_target_pct=5.0, stop_loss_pct=0.5,
            slippage_pct=0.01, strategy_override="BULL_PUT_SPREAD", strike_selection_cfg=LOOSE_CFG,
        )
        result = run_backtest("NIFTY", ohlc, vix, cfg)
        trade_log = result["trade_log"]

        assert not trade_log.empty
        trades = trade_log.groupby("trade_id").agg(pnl=("leg_pnl", "sum"), exit_reason=("exit_reason", "first"))
        assert (trades["exit_reason"] == "stop_loss").any()
        loss_trade = trades[trades["exit_reason"] == "stop_loss"].iloc[0]
        assert loss_trade["pnl"] < 0
        _assert_no_overlapping_trades(trade_log)


class TestDTEExit:
    def test_flat_market_exits_on_dte_threshold(self):
        n = 20
        ohlc = _ohlc([24300.0] * n)
        vix = _vix([15.0] * n)  # perfectly flat -- nothing should trigger profit/stop

        cfg = BacktestConfig(
            tenor_days=10, min_dte_exit=2, profit_target_pct=5.0, stop_loss_pct=5.0,
            slippage_pct=0.01, strategy_override="BULL_PUT_SPREAD", strike_selection_cfg=LOOSE_CFG,
        )
        result = run_backtest("NIFTY", ohlc, vix, cfg)
        trade_log = result["trade_log"]

        assert not trade_log.empty
        trades = trade_log.groupby("trade_id").agg(exit_reason=("exit_reason", "first")).iloc[0]
        assert trades["exit_reason"] == "dte_exit"


class TestIVSkewOptIn:
    def test_default_is_flat_iv_unchanged(self):
        # use_iv_skew defaults False -- this must reproduce the exact flat-IV
        # trade log/equity curve, i.e. adding the feature must not silently
        # change anyone's existing results
        n = 15
        ohlc = _ohlc([24300.0] * n)
        vix = _vix([30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 9, 8, 8, 8])
        cfg_implicit = BacktestConfig(
            tenor_days=15, min_dte_exit=1, profit_target_pct=0.30, stop_loss_pct=5.0,
            slippage_pct=0.01, strategy_override="BULL_PUT_SPREAD", strike_selection_cfg=LOOSE_CFG,
        )
        cfg_explicit_false = BacktestConfig(
            tenor_days=15, min_dte_exit=1, profit_target_pct=0.30, stop_loss_pct=5.0,
            slippage_pct=0.01, strategy_override="BULL_PUT_SPREAD", strike_selection_cfg=LOOSE_CFG,
            use_iv_skew=False,
        )
        r1 = run_backtest("NIFTY", ohlc, vix, cfg_implicit)
        r2 = run_backtest("NIFTY", ohlc, vix, cfg_explicit_false)
        pd.testing.assert_series_equal(r1["equity_curve"], r2["equity_curve"])
        pd.testing.assert_frame_equal(r1["trade_log"], r2["trade_log"])

    def test_skew_enabled_runs_without_error_and_differs_from_flat(self):
        n = 15
        ohlc = _ohlc([24300.0] * n)
        vix = _vix([30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 9, 8, 8, 8])
        base_kwargs = dict(
            tenor_days=15, min_dte_exit=1, profit_target_pct=0.30, stop_loss_pct=5.0,
            slippage_pct=0.01, strategy_override="BULL_PUT_SPREAD", strike_selection_cfg=LOOSE_CFG,
        )
        flat_result = run_backtest("NIFTY", ohlc, vix, BacktestConfig(**base_kwargs, use_iv_skew=False))
        skewed_result = run_backtest("NIFTY", ohlc, vix, BacktestConfig(**base_kwargs, use_iv_skew=True))

        assert not skewed_result["trade_log"].empty
        assert not flat_result["equity_curve"].equals(skewed_result["equity_curve"])

    def test_skew_changes_which_strikes_get_selected(self):
        # with skew on, OTM puts read higher IV -> different delta at the same
        # strike -> select_short_strike should land on a different strike than
        # the flat-IV path for at least one entry, given a wide enough grid
        n = 15
        ohlc = _ohlc([24300.0] * n)
        vix = _vix([30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 9, 8, 8, 8])
        base_kwargs = dict(
            tenor_days=15, min_dte_exit=1, profit_target_pct=0.30, stop_loss_pct=5.0,
            slippage_pct=0.01, strategy_override="BULL_PUT_SPREAD", strike_selection_cfg=LOOSE_CFG,
        )
        flat_result = run_backtest("NIFTY", ohlc, vix, BacktestConfig(**base_kwargs, use_iv_skew=False))
        skewed_result = run_backtest("NIFTY", ohlc, vix, BacktestConfig(**base_kwargs, use_iv_skew=True))

        flat_strikes = set(flat_result["trade_log"]["strike"])
        skewed_strikes = set(skewed_result["trade_log"]["strike"])
        assert flat_strikes != skewed_strikes


class TestTradeLogSchema:
    def test_bull_put_spread_has_two_legs_per_trade(self):
        n = 15
        ohlc = _ohlc([24300.0] * n)
        vix = _vix([30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 9, 8, 8, 8])
        cfg = BacktestConfig(
            tenor_days=15, min_dte_exit=1, profit_target_pct=0.30, stop_loss_pct=5.0,
            slippage_pct=0.01, strategy_override="BULL_PUT_SPREAD", strike_selection_cfg=LOOSE_CFG,
        )
        result = run_backtest("NIFTY", ohlc, vix, cfg)
        trade_log = result["trade_log"]
        leg_counts = trade_log.groupby("trade_id").size()
        assert (leg_counts == 2).all()
        assert set(trade_log["option_type"].unique()) == {"PE"}

    def test_iron_condor_has_four_legs_per_trade(self):
        n = 20
        ohlc = _ohlc([24300.0] * n)
        vix = _vix([15.0] * n)
        cfg = BacktestConfig(
            tenor_days=10, min_dte_exit=2, profit_target_pct=5.0, stop_loss_pct=5.0,
            slippage_pct=0.01, strategy_override="IRON_CONDOR", strike_selection_cfg=LOOSE_CFG,
        )
        result = run_backtest("NIFTY", ohlc, vix, cfg)
        trade_log = result["trade_log"]
        assert not trade_log.empty
        leg_counts = trade_log.groupby("trade_id").size()
        assert (leg_counts == 4).all()
        assert set(trade_log["option_type"].unique()) == {"CE", "PE"}


class TestRegimeDrivenEntry:
    def test_runs_without_strategy_override(self):
        # Regression test: strategy_override bypasses recommend_strategy_type()
        # entirely (short-circuits via `or`), so every other test in this file
        # -- all of which set strategy_override -- never actually exercises
        # this call path. This test doesn't; it caught a real bug where
        # recommend_strategy_type() was called with the wrong config object
        # (StrikeSelectionConfig instead of SignalConfig), an AttributeError
        # that only surfaced here, and only live (scripts/run_backtest.py).
        import numpy as np
        rng = np.random.default_rng(11)
        n = 300  # long enough for iv_rank's default min_periods (252//4=63) to be satisfied
        closes = 24000 * np.exp(np.cumsum(rng.normal(0, 0.008, n)))
        ohlc = _ohlc(list(closes))
        vix = _vix(list(15 + rng.normal(0, 3, n).clip(min=-10)))

        cfg = BacktestConfig(tenor_days=7, min_dte_exit=1, strike_selection_cfg=LOOSE_CFG)
        result = run_backtest("NIFTY", ohlc, vix, cfg)  # must not raise

        assert len(result["equity_curve"]) == n
        assert not result["equity_curve"].isna().any()
        if not result["trade_log"].empty:
            assert set(result["trade_log"]["strategy_type"].unique()) <= {
                "BULL_PUT_SPREAD", "BEAR_CALL_SPREAD", "IRON_CONDOR",
            }


class TestEquityCurve:
    def test_equity_curve_covers_every_trading_day(self):
        n = 20
        ohlc = _ohlc([24300.0] * n)
        vix = _vix([15.0] * n)
        cfg = BacktestConfig(strike_selection_cfg=LOOSE_CFG)  # no override -> NO_TRADE the whole time
        result = run_backtest("NIFTY", ohlc, vix, cfg)
        assert len(result["equity_curve"]) == n
        assert list(result["equity_curve"].index) == list(ohlc.index)

    def test_equity_never_nan(self):
        n = 15
        ohlc = _ohlc([24300.0] * n)
        vix = _vix([30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 9, 8, 8, 8])
        cfg = BacktestConfig(
            tenor_days=15, min_dte_exit=1, profit_target_pct=0.30, stop_loss_pct=5.0,
            slippage_pct=0.01, strategy_override="BULL_PUT_SPREAD", strike_selection_cfg=LOOSE_CFG,
        )
        result = run_backtest("NIFTY", ohlc, vix, cfg)
        assert not result["equity_curve"].isna().any()


def _concurrency_series(trades: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.Series:
    concurrency = pd.Series(0, index=calendar)
    for _, t in trades.iterrows():
        concurrency.loc[pd.Timestamp(t["entry_date"]):pd.Timestamp(t["exit_date"])] += 1
    return concurrency


class TestMaxConcurrentPositions:
    # loose enough that profit_target/stop_loss essentially never fire --
    # exits are driven purely by min_dte_exit, giving fully deterministic,
    # hand-traceable timing for these structural tests
    _LOOSE_EXITS = dict(profit_target_pct=5.0, stop_loss_pct=5.0)

    def test_default_matches_explicit_max_concurrent_one(self):
        n = 15
        ohlc = _ohlc([24300.0] * n)
        vix = _vix([30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 9, 8, 8, 8])
        base_kwargs = dict(
            tenor_days=15, min_dte_exit=1, profit_target_pct=0.30, stop_loss_pct=5.0,
            slippage_pct=0.01, strategy_override="BULL_PUT_SPREAD", strike_selection_cfg=LOOSE_CFG,
        )
        default_result = run_backtest("NIFTY", ohlc, vix, BacktestConfig(**base_kwargs))
        explicit_result = run_backtest("NIFTY", ohlc, vix, BacktestConfig(**base_kwargs, max_concurrent_positions=1))
        pd.testing.assert_series_equal(default_result["equity_curve"], explicit_result["equity_curve"])
        pd.testing.assert_frame_equal(default_result["trade_log"], explicit_result["trade_log"])

    def test_single_slot_never_same_day_reentry_but_refills_next_trading_day(self):
        n = 40
        ohlc = _ohlc([24300.0] * n)
        vix = _vix([15.0] * n)
        cfg = BacktestConfig(
            tenor_days=6, min_dte_exit=1, slippage_pct=0.01, strategy_override="BULL_PUT_SPREAD",
            strike_selection_cfg=LOOSE_CFG, max_concurrent_positions=1, **self._LOOSE_EXITS,
        )
        result = run_backtest("NIFTY", ohlc, vix, cfg)
        trades = result["trade_log"].groupby("trade_id").agg(
            entry_date=("entry_date", "first"), exit_date=("exit_date", "first")
        ).sort_values("entry_date")
        assert len(trades) >= 3  # enough transitions to be a meaningful check

        calendar = list(ohlc.index)
        for (_, a), (_, b) in zip(trades.iloc[:-1].iterrows(), trades.iloc[1:].iterrows()):
            assert b["entry_date"] != a["exit_date"]  # never same-day
            next_trading_day = calendar[calendar.index(pd.Timestamp(a["exit_date"])) + 1]
            assert pd.Timestamp(b["entry_date"]) == next_trading_day  # refills immediately when signal keeps firing

    def test_multiple_positions_open_concurrently_when_cap_allows(self):
        n = 30
        ohlc = _ohlc([24300.0] * n)
        vix = _vix([15.0] * n)
        cfg = BacktestConfig(
            tenor_days=10, min_dte_exit=1, strategy_override="BULL_PUT_SPREAD",
            strike_selection_cfg=LOOSE_CFG, max_concurrent_positions=3, **self._LOOSE_EXITS,
        )
        result = run_backtest("NIFTY", ohlc, vix, cfg)
        trades = result["trade_log"].groupby("trade_id").agg(
            entry_date=("entry_date", "first"), exit_date=("exit_date", "first")
        )
        assert _concurrency_series(trades, ohlc.index).max() > 1  # laddering actually happened

    @pytest.mark.parametrize("cap", [1, 2, 3, 4])
    def test_concurrency_never_exceeds_cap(self, cap):
        n = 30
        ohlc = _ohlc([24300.0] * n)
        vix = _vix([15.0] * n)
        cfg = BacktestConfig(
            tenor_days=10, min_dte_exit=1, strategy_override="BULL_PUT_SPREAD",
            strike_selection_cfg=LOOSE_CFG, max_concurrent_positions=cap, **self._LOOSE_EXITS,
        )
        result = run_backtest("NIFTY", ohlc, vix, cfg)
        trades = result["trade_log"].groupby("trade_id").agg(
            entry_date=("entry_date", "first"), exit_date=("exit_date", "first")
        )
        assert _concurrency_series(trades, ohlc.index).max() <= cap

    def test_higher_cap_deploys_more_capital(self):
        n = 30
        ohlc = _ohlc([24300.0] * n)
        vix = _vix([15.0] * n)

        def deployed_days(cap):
            cfg = BacktestConfig(
                tenor_days=10, min_dte_exit=1, strategy_override="BULL_PUT_SPREAD",
                strike_selection_cfg=LOOSE_CFG, max_concurrent_positions=cap, **self._LOOSE_EXITS,
            )
            result = run_backtest("NIFTY", ohlc, vix, cfg)
            trades = result["trade_log"].groupby("trade_id").agg(
                entry_date=("entry_date", "first"), exit_date=("exit_date", "first")
            )
            return _concurrency_series(trades, ohlc.index).sum()  # sum of concurrency = total position-days

        assert deployed_days(cap=3) > deployed_days(cap=1)
