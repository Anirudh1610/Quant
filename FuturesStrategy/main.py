"""
main.py — CLI entry point for the NSE Futures Strategy.

Usage
-----
    python main.py                              # run with defaults
    python main.py --capital 2000000            # ₹20 lakh starting capital
    python main.py --start 2019-01-01           # custom start date
    python main.py --no-charts                  # skip chart generation
    python main.py --output results/            # save outputs to custom dir
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from strategy.backtest import FuturesBacktestConfig, FuturesBacktestEngine
from strategy.data import load_futures_dataset
from strategy.metrics import (
    plot_equity_curve, plot_drawdown, plot_monthly_heatmap,
    plot_exposure_analysis, plot_long_short_pnl, plot_rollover_costs,
    plot_stock_contribution, plot_trades_scatter, plot_win_loss_distribution,
    generate_performance_report, generate_comparison_report,
)
from strategy.signals import (
    calculate_momentum, rank_momentum,
    get_long_candidates, get_short_candidates,
    calculate_rsi, calculate_bollinger_bands, calculate_sma, calculate_atr,
    calculate_ema, calculate_adx,
    get_entry_signals, get_entry_signals_v2,
    detect_combined_regime, vix_position_scalar,
    volume_trend_filter,
)
from strategy.universe import DEFAULT_UNIVERSE


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NSE Futures Momentum + Mean-Reversion Strategy")
    p.add_argument("--start",    default="2017-01-01", help="Backtest start date (YYYY-MM-DD)")
    p.add_argument("--end",      default=None,         help="Backtest end date (YYYY-MM-DD); defaults to today")
    p.add_argument("--capital",  default=5_000_000.0,  type=float, help="Initial capital in ₹")
    p.add_argument("--data-dir", default="data/cache",             help="Cache directory for downloaded data")
    p.add_argument("--output",   default="results",                help="Directory for charts + CSV output")
    p.add_argument("--no-charts", action="store_true",             help="Skip generating charts")
    p.add_argument("--max-long",  default=8,  type=int)
    p.add_argument("--max-short", default=8,  type=int)
    p.add_argument("--hold-days", default=5,  type=int)
    p.add_argument("--risk-frac", default=0.01, type=float, help="Risk per trade (fraction of NAV)")
    p.add_argument("--compare",   action="store_true",       help="Run v1 + v2 configs and print side-by-side comparison")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# SIGNAL COMPUTATION
# ──────────────────────────────────────────────────────────────────────────────

def compute_signals(dataset: dict, cfg: FuturesBacktestConfig):
    close  = dataset["futures"]["close"]
    high   = dataset["futures"]["high"]
    low    = dataset["futures"]["low"]
    volume = dataset["spot"].get("volume", None)

    momentum = calculate_momentum(close, window=cfg.momentum_window)
    ranks    = rank_momentum(momentum)
    total    = len(close.columns)

    long_candidates  = get_long_candidates(ranks,  top_n=cfg.rank_entry)
    short_candidates = get_short_candidates(ranks, total_stocks=total, bottom_n=cfg.rank_entry)

    atr   = calculate_atr(high, low, close, period=cfg.atr_period)
    ema20 = calculate_ema(close, window=cfg.ema_period)
    adx   = calculate_adx(high, low, close, period=cfg.adx_period)

    # Volume trend filter as Open Interest proxy (NSE OI not available via yfinance).
    # Restricts entries to periods of elevated volume = rising market participation.
    vol_filter = None
    if volume is not None and not volume.empty:
        vol_aligned = volume.reindex(index=close.index, columns=close.columns)
        vol_filter  = volume_trend_filter(vol_aligned, short_window=5, long_window=20)

    long_entries, short_entries = get_entry_signals_v2(
        long_candidates, short_candidates,
        close, ema20, adx,
        adx_threshold=cfg.adx_threshold,
        ema_tolerance=cfg.ema_tolerance,
        vol_filter=vol_filter,
    )

    # Legacy RSI is passed through for v1-mode compatibility
    rsi  = calculate_rsi(close, period=cfg.rsi_period)

    nifty_close = dataset["spot"]["close"].get("NIFTY")
    vix_series  = dataset["india_vix"]
    regime      = detect_combined_regime(nifty_close, vix_series)

    return long_entries, short_entries, rsi, atr, regime, ranks


def compute_signals_v1(dataset: dict, cfg: FuturesBacktestConfig):
    """Legacy v1 signals for comparison mode."""
    close = dataset["futures"]["close"]
    high  = dataset["futures"]["high"]
    low   = dataset["futures"]["low"]

    momentum = calculate_momentum(close, window=60)
    ranks_v1 = rank_momentum(momentum)
    total    = len(close.columns)

    long_candidates  = get_long_candidates(ranks_v1, top_n=10)
    short_candidates = get_short_candidates(ranks_v1, total_stocks=total, bottom_n=10)

    rsi              = calculate_rsi(close, period=2)
    bb_u, bb_m, bb_l = calculate_bollinger_bands(close, window=20, num_std=2.0)
    ma50             = calculate_sma(close, window=50)
    atr              = calculate_atr(high, low, close, period=14)

    long_entries, short_entries = get_entry_signals(
        long_candidates, short_candidates, rsi, bb_u, bb_l, close, ma50,
        rsi_long_thresh=10.0, rsi_short_thresh=90.0, bb_buffer=0.005,
    )

    nifty_close = dataset["spot"]["close"].get("NIFTY")
    vix_series  = dataset["india_vix"]
    regime      = detect_combined_regime(nifty_close, vix_series)

    return long_entries, short_entries, rsi, atr, regime


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    end_date = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    out_dir  = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== NSE Futures Strategy  [{args.start}  →  {end_date}] ===")
    print(f"    Capital: ₹{args.capital:,.0f}  |  Max Long: {args.max_long}  |  Max Short: {args.max_short}")
    print(f"    Data cache: {args.data_dir}  |  Output: {args.output}\n")

    # 1. Download / load data
    print("[1/4] Loading futures dataset …")
    dataset = load_futures_dataset(
        tickers=DEFAULT_UNIVERSE,
        start=args.start,
        end=end_date,
        data_dir=args.data_dir,
    )
    print(f"      Tickers: {len(dataset['tickers'])}  |  "
          f"Dates: {dataset['futures']['close'].index[0].date()} → "
          f"{dataset['futures']['close'].index[-1].date()}")

    # 2. Compute signals
    print("[2/4] Computing signals …")
    cfg = FuturesBacktestConfig(
        initial_capital=args.capital,
        max_long=args.max_long,
        max_short=args.max_short,
        risk_fraction=args.risk_frac,
    )
    long_entries, short_entries, rsi, atr, regime, ranks = compute_signals(dataset, cfg)

    # 3. Run backtest
    print("[3/4] Running backtest …")
    engine  = FuturesBacktestEngine(cfg)
    results = engine.run(dataset, long_entries, short_entries, rsi, atr, regime, ranks=ranks)

    # 4. Performance report + charts
    print("[4/4] Generating report and charts …")
    generate_performance_report(
        equity=results["equity_curve"],
        daily_returns=results["daily_returns"],
        trade_log=results["trade_log"],
        initial_capital=args.capital,
        output_dir=out_dir,
    )

    # Optional: run v1 for side-by-side comparison
    if args.compare:
        print("\n[COMPARE] Running v1 (original) config for comparison …")
        cfg_v1 = FuturesBacktestConfig(
            initial_capital=args.capital,
            max_long=args.max_long,
            max_short=args.max_short,
            momentum_window=60,
            top_n=10, bottom_n=10,
            hold_days=args.hold_days,
            risk_fraction=args.risk_frac,
            rebalance_weekly=False,
        )
        long_v1, short_v1, rsi_v1, atr_v1, regime_v1 = compute_signals_v1(dataset, cfg_v1)
        engine_v1  = FuturesBacktestEngine(cfg_v1)
        results_v1 = engine_v1.run(dataset, long_v1, short_v1, rsi_v1, atr_v1, regime_v1)

        generate_comparison_report(
            {
                "V1 (Original)": results_v1,
                "V2 (Improved)": results,
            },
            initial_capital=args.capital,
            output_dir=out_dir,
        )

    if not args.no_charts:
        import matplotlib
        matplotlib.use("Agg")

        chart_dir = out_dir / "charts"
        chart_dir.mkdir(exist_ok=True)

        nifty_close = results.get("nifty_close")

        plot_equity_curve(results["equity_curve"], nifty_close, chart_dir)
        plot_drawdown(results["equity_curve"], chart_dir)
        plot_monthly_heatmap(results["daily_returns"], chart_dir)
        plot_exposure_analysis(results["exposure_df"], results["equity_curve"], chart_dir)

        if not results["trade_log"].empty:
            plot_long_short_pnl(results["trade_log"], chart_dir)
            plot_rollover_costs(results["trade_log"], chart_dir)
            plot_stock_contribution(results["trade_log"], chart_dir)
            plot_trades_scatter(results["trade_log"], chart_dir)
            plot_win_loss_distribution(results["trade_log"], chart_dir)

        print(f"\n  Charts saved to: {chart_dir}")

    # Save trade log
    if not results["trade_log"].empty:
        tl_path = out_dir / "trade_log.csv"
        results["trade_log"].to_csv(tl_path, index=False)
        print(f"  Trade log saved to: {tl_path}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
