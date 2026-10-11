"""main.py — CLI entry point for NIFTY Volatility Forecasting.

Usage
-----
    # First-time Kite Connect authentication
    python main.py --auth --api-secret YOUR_SECRET --request-token TOKEN

    # Full pipeline (download → features → train → backtest → report)
    python main.py

    # Skip data download if cache exists
    python main.py --no-refresh

    # Only benchmark models, no LSTM training
    python main.py --benchmarks-only

    # Walk-forward validation
    python main.py --walk-forward --retrain-freq monthly

    # Custom date range
    python main.py --start 2018-01-01 --train-end 2022-12-31 --val-end 2024-06-30

    # Tune threshold
    python main.py --threshold 0.015
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd

from utils import get_logger, load_config, load_env, get_kite_credentials, set_seed, ensure_dirs
from data_loader import DataLoader, generate_access_token

load_env()  # load .env before anything else accesses os.environ
from feature_engineering import build_feature_matrix, FEATURE_COLS, TARGET_COL, BENCHMARK_COL, get_available_feature_cols
from dataset_builder import VolatilityDataset, walk_forward_windows
from train_lstm import LSTMTrainer, train_model, walk_forward_train
from benchmark_models import run_benchmarks
from strategy import StrategyConfig, generate_signals
from backtest import BacktestEngine, BacktestConfig, compute_performance
from evaluate import (
    plot_actual_vs_predicted,
    plot_residuals,
    plot_prediction_error_distribution,
    plot_training_loss,
    plot_feature_correlation,
    plot_signal_distribution,
    plot_equity_curve,
    plot_drawdown,
    plot_rolling_sharpe,
    permutation_importance,
    shap_analysis,
    generate_performance_report,
)

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NIFTY Volatility Forecasting — LSTM + Kite Connect"
    )

    # Auth — credentials come from .env; only request-token is needed here
    p.add_argument("--auth", action="store_true",
                   help="Exchange a request_token for an access_token and save to .env")
    p.add_argument("--request-token", default=None,
                   help="Request token from Kite login redirect URL")

    # Data
    p.add_argument("--start", default=None,
                   help="Data start date (default: from config)")
    p.add_argument("--end", default=None)
    p.add_argument("--refresh", action="store_true",
                   help="Force re-download from Kite Connect")
    p.add_argument("--no-refresh", action="store_true",
                   help="Use cached data only (never hit Kite API)")

    # Training splits
    p.add_argument("--train-end", default=None)
    p.add_argument("--val-end", default=None)

    # Strategy
    p.add_argument("--threshold", type=float, default=None,
                   help="Edge threshold for signal (default: from config)")

    # Pipeline flags
    p.add_argument("--benchmarks-only", action="store_true",
                   help="Skip LSTM training; run benchmarks only")
    p.add_argument("--no-train", action="store_true",
                   help="Load existing LSTM model from disk, skip training")
    p.add_argument("--no-backtest", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--walk-forward", action="store_true")
    p.add_argument("--retrain-freq", default=None, choices=["monthly", "quarterly"])
    p.add_argument("--no-shap", action="store_true",
                   help="Skip SHAP analysis (slow on large datasets)")

    # Config path
    p.add_argument("--config", default=None, help="Path to config YAML")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Auth flow
# ─────────────────────────────────────────────────────────────────────────────

def run_auth(args) -> None:
    from utils import get_kite_credentials

    api_key, _, api_secret = get_kite_credentials()

    if not args.request_token:
        login_url = f"https://kite.zerodha.com/connect/login?api_key={api_key}&v=3"
        print(f"\nOpen this URL in your browser and log in:\n  {login_url}")
        print("\nAfter redirect, copy the 'request_token' from the URL parameters.")
        print("Then run:\n  python main.py --auth --request-token <TOKEN>\n")
        print("KITE_ACCESS_TOKEN will be written automatically to your .env file.")
        sys.exit(0)

    token = generate_access_token(args.request_token)
    print(f"\naccess_token: {token}")
    print("KITE_ACCESS_TOKEN has been saved to your .env file automatically.\n")
    sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # Override config with CLI args
    if args.start:
        cfg["dates"]["start"] = args.start
    if args.end:
        cfg["dates"]["end"] = args.end
    if args.train_end:
        cfg["dates"]["train_end"] = args.train_end
    if args.val_end:
        cfg["dates"]["val_end"] = args.val_end
    if args.threshold is not None:
        cfg["strategy"]["threshold"] = args.threshold
    if args.retrain_freq:
        cfg["walk_forward"]["retrain_freq"] = args.retrain_freq

    # Auth flow
    if args.auth:
        run_auth(args)

    set_seed(cfg["model"]["seed"])
    plots_dir = BASE_DIR / cfg["reports"]["plots_dir"]
    reports_dir = BASE_DIR / cfg["reports"]["output_dir"]
    ensure_dirs(plots_dir, reports_dir)

    start = cfg["dates"]["start"]
    end = cfg["dates"].get("end") or str(pd.Timestamp.today().date())

    print(f"\n{'='*60}")
    print(f"  NIFTY Volatility Forecasting Pipeline")
    print(f"  Period: {start} → {end}")
    print(f"{'='*60}\n")

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("[1/7] Loading data …")
    loader = DataLoader(cfg)
    force_refresh = args.refresh and not args.no_refresh
    raw = loader.load_all(start, end, force_refresh=force_refresh)
    nifty, futures, vix = raw["nifty"], raw["futures"], raw["vix"]
    print(f"      NIFTY: {len(nifty)} days  Futures: {len(futures)} days  VIX: {len(vix)} days")

    # ── 2. Feature engineering ────────────────────────────────────────────────
    print("[2/7] Engineering features …")
    df = build_feature_matrix(nifty, futures, vix, cfg)
    active_features = get_available_feature_cols(df)
    print(f"      {df.shape[0]} rows × {df.shape[1]} columns  |  active features: {len(active_features)}/{len(FEATURE_COLS)}")

    if not args.no_plots:
        plot_feature_correlation(df, active_features, plots_dir)

    # ── 3. Dataset preparation ────────────────────────────────────────────────
    print("[3/7] Preparing dataset …")
    ds = VolatilityDataset(df, cfg=cfg, feature_cols=active_features)
    ds.fit_and_transform()
    ds.save_scaler()

    # ── 4. LSTM training ──────────────────────────────────────────────────────
    trainer = LSTMTrainer(cfg)
    lstm_val_preds: np.ndarray | None = None
    lstm_test_preds: np.ndarray | None = None

    if not args.benchmarks_only:
        if args.no_train:
            print("[4/7] Loading existing LSTM model …")
            try:
                trainer.load(input_size=ds.n_features())
                lstm_val_preds = trainer.predict(ds.X_val)
                lstm_test_preds = trainer.predict(ds.X_test)
            except FileNotFoundError:
                print("      No saved model found — will train from scratch.")
                args.no_train = False

        if not args.no_train:
            print("[4/7] Training LSTM …")
            trainer.build(input_size=ds.n_features())
            trainer.fit(ds.X_train, ds.y_train, ds.X_val, ds.y_val)
            lstm_val_preds = trainer.predict(ds.X_val)
            lstm_test_preds = trainer.predict(ds.X_test)

        if not args.no_plots and trainer.history.get("train_loss"):
            plot_training_loss(trainer.history, plots_dir)
    else:
        print("[4/7] Skipping LSTM (--benchmarks-only)")

    # ── 5. Benchmark comparison ───────────────────────────────────────────────
    print("[5/7] Running benchmark models …")
    train_mask = df.index <= cfg["dates"]["train_end"]
    bench_table = run_benchmarks(
        returns=df["ret_1d"],
        actual_rv5=df[TARGET_COL],
        train_mask=train_mask,
        lstm_preds=lstm_test_preds,
        lstm_dates=ds.dates_test,
    )
    bench_table.to_csv(reports_dir / "benchmark_comparison.csv")
    print("\n" + bench_table.to_string() + "\n")

    # ── 6. Strategy + backtest ────────────────────────────────────────────────
    print("[6/7] Running strategy + backtest …")

    if not args.no_backtest and lstm_test_preds is not None:
        pred_series = pd.Series(lstm_test_preds, index=ds.dates_test, name="predicted_rv5")
        # Annualized implied vol (VIX/100) — matches LSTM output units for signal edge
        impl_ann_series = pd.Series(ds.implied_test, index=ds.dates_test, name="implied_vol_ann")
        # 5-day implied vol — used only for visualization (absolute scale)
        impl_5d_series = pd.Series(ds.implied_5d_test, index=ds.dates_test, name="implied_5d_vol")

        strat_cfg = StrategyConfig.from_cfg(cfg)
        signals_df = generate_signals(pred_series, impl_ann_series, strat_cfg, implied_5d_vol=impl_5d_series)
        signals_df.to_csv(reports_dir / "signals.csv")

        if not args.no_plots:
            plot_signal_distribution(signals_df, plots_dir)

        # actual_rv5 aligned to signal dates: after compute_target fix,
        # actual_rv5[t] = annualized std of returns[t+1..t+5] (the hold-period vol)
        actual_rv_test = df[TARGET_COL].reindex(ds.dates_test)

        bt_cfg = BacktestConfig.from_cfg(cfg)
        engine = BacktestEngine(bt_cfg)
        results = engine.run(signals_df, actual_rv_test)

        if not args.no_plots:
            plot_equity_curve(results["equity_curve"], plots_dir)
            plot_drawdown(results["equity_curve"], plots_dir)
            plot_rolling_sharpe(results["daily_returns"], plots_dir=plots_dir)

        generate_performance_report(
            equity=results["equity_curve"],
            daily_returns=results["daily_returns"],
            trade_log=results["trade_log"],
            initial_capital=bt_cfg.initial_capital,
            benchmark_table=bench_table,
            output_dir=reports_dir,
        )

        # LSTM forecast visualizations
        actual_test_series = df[TARGET_COL].reindex(ds.dates_test)
        if not args.no_plots:
            plot_actual_vs_predicted(
                actual=actual_test_series,
                predicted=pred_series,
                implied=impl_5d_series,
                plots_dir=plots_dir,
            )
            plot_residuals(actual_test_series, pred_series, plots_dir)
            plot_prediction_error_distribution(actual_test_series, pred_series, plots_dir)

    # ── 7. Walk-forward validation ────────────────────────────────────────────
    if args.walk_forward:
        print("[7/7] Walk-forward validation …")
        windows = walk_forward_windows(
            df,
            initial_train_years=cfg["walk_forward"]["initial_train_years"],
            retrain_freq=cfg["walk_forward"]["retrain_freq"],
        )
        wf_results = walk_forward_train(df, windows, cfg, FEATURE_COLS)

        # Aggregate all walk-forward predictions
        all_preds, all_actual, all_dates = [], [], []
        for r in wf_results:
            all_preds.extend(r["predictions"].tolist())
            all_actual.extend(r["actuals"].tolist())
            all_dates.extend(r["dates"].tolist())

        if all_preds:
            wf_pred = pd.Series(all_preds, index=pd.DatetimeIndex(all_dates))
            wf_actual = pd.Series(all_actual, index=pd.DatetimeIndex(all_dates))
            from benchmark_models import compute_metrics
            wf_metrics = compute_metrics("LSTM Walk-Forward", np.array(all_actual), np.array(all_preds))
            print(f"\nWalk-forward LSTM: {wf_metrics}")

            wf_pred.to_csv(reports_dir / "walk_forward_predictions.csv", header=["prediction"])
            if not args.no_plots:
                plot_actual_vs_predicted(wf_actual, wf_pred, plots_dir=plots_dir)
    else:
        print("[7/7] Skipping walk-forward (pass --walk-forward to enable)")

    # ── Explainability ────────────────────────────────────────────────────────
    if not args.benchmarks_only and not args.no_plots and trainer.model is not None:
        print("\n[+] Permutation feature importance …")
        permutation_importance(
            model_predict_fn=trainer.predict,
            X=ds.X_test,
            y=ds.y_test,
            feature_names=ds.feature_cols,
            plots_dir=plots_dir,
        )

        if not args.no_shap:
            print("[+] SHAP analysis …")
            shap_analysis(trainer.model, ds.X_test, ds.feature_cols, plots_dir)

    print(f"\nDone.  Reports → {reports_dir}  |  Plots → {plots_dir}\n")


if __name__ == "__main__":
    main()
