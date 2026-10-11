"""
main.py
-------
Entry point for the Cross-Sectional Momentum + Mean Reversion strategy.

Full pipeline
-------------
1. Download / load cached NIFTY 50 OHLCV data from Yahoo Finance
2. Download NIFTY 50 index for regime filter and benchmark comparison
3. Compute cross-sectional momentum, RSI, regime, and volatility signals
4. Run the event-driven backtest engine
5. Print performance summary table
6. Save equity curve, drawdown, monthly heatmap, stock contribution charts
7. Export trade log to CSV
8. Optionally run walk-forward parameter optimisation

Usage
-----
    python main.py                         # default settings
    python main.py --start 2020-01-01      # custom date range
    python main.py --walk-forward          # enable walk-forward test
    python main.py --sector-neutral        # enable sector neutrality
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Make sure the strategy package is importable when run directly ─────────
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from strategy.signals import (
    calculate_momentum,
    rank_momentum,
    get_long_candidates,
    get_short_candidates,
    calculate_rsi,
    get_entry_signals,
    calculate_volatility_filter,
    nifty_regime_filter,
    calculate_atr,
)
from strategy.portfolio import (
    apply_regime_filter,
    apply_volatility_filter,
    cap_positions,
    sector_neutral_selection,
)
from strategy.backtest import BacktestEngine, BacktestConfig
from strategy.metrics import (
    generate_performance_summary,
    plot_equity_curve,
    plot_drawdown,
    plot_monthly_heatmap,
    plot_stock_contribution,
)


# ──────────────────────────────────────────────────────────────────────────────
# NIFTY 50 UNIVERSE  (Yahoo Finance tickers with .NS suffix)
# ──────────────────────────────────────────────────────────────────────────────

NIFTY50_TICKERS: list[str] = [
    "RELIANCE.NS", "TCS.NS",        "HDFCBANK.NS",   "INFY.NS",      "ICICIBANK.NS",
    "HINDUNILVR.NS","ITC.NS",       "SBIN.NS",        "BHARTIARTL.NS","KOTAKBANK.NS",
    "LT.NS",        "AXISBANK.NS",  "ASIANPAINT.NS",  "MARUTI.NS",    "WIPRO.NS",
    "HCLTECH.NS",   "ULTRACEMCO.NS","TITAN.NS",        "POWERGRID.NS", "NESTLEIND.NS",
    "BAJFINANCE.NS","SUNPHARMA.NS", "TECHM.NS",        "ONGC.NS",      "NTPC.NS",
    "ADANIENT.NS",  "ADANIPORTS.NS","BAJAJFINSV.NS",   "GRASIM.NS",    "JSWSTEEL.NS",
    "TATAMOTORS.NS","TATACONSUM.NS","TATASTEEL.NS",    "COALINDIA.NS", "CIPLA.NS",
    "BRITANNIA.NS", "HINDALCO.NS",  "DRREDDY.NS",      "DIVISLAB.NS",  "EICHERMOT.NS",
    "HEROMOTOCO.NS","INDUSINDBK.NS","BPCL.NS",         "APOLLOHOSP.NS","BAJAJ-AUTO.NS",
    "M&M.NS",       "SBILIFE.NS",   "HDFCLIFE.NS",     "UPL.NS",       "LTIM.NS",
]

NIFTY_INDEX_TICKER = "^NSEI"

# GICS-inspired sector mapping for sector-neutrality feature
NIFTY50_SECTORS: dict[str, str] = {
    "RELIANCE.NS": "Energy",    "ONGC.NS": "Energy",       "BPCL.NS": "Energy",
    "COALINDIA.NS": "Energy",   "NTPC.NS": "Utilities",    "POWERGRID.NS": "Utilities",
    "TCS.NS": "IT",             "INFY.NS": "IT",            "WIPRO.NS": "IT",
    "HCLTECH.NS": "IT",         "TECHM.NS": "IT",           "LTIM.NS": "IT",
    "HDFCBANK.NS": "Financials","ICICIBANK.NS": "Financials","SBIN.NS": "Financials",
    "KOTAKBANK.NS": "Financials","AXISBANK.NS": "Financials","BAJFINANCE.NS": "Financials",
    "BAJAJFINSV.NS": "Financials","INDUSINDBK.NS": "Financials","SBILIFE.NS": "Financials",
    "HDFCLIFE.NS": "Financials",
    "HINDUNILVR.NS": "FMCG",   "ITC.NS": "FMCG",          "NESTLEIND.NS": "FMCG",
    "BRITANNIA.NS": "FMCG",    "TATACONSUM.NS": "FMCG",
    "SUNPHARMA.NS": "Pharma",  "CIPLA.NS": "Pharma",       "DRREDDY.NS": "Pharma",
    "DIVISLAB.NS": "Pharma",   "APOLLOHOSP.NS": "Healthcare",
    "TATAMOTORS.NS": "Auto",   "MARUTI.NS": "Auto",        "EICHERMOT.NS": "Auto",
    "HEROMOTOCO.NS": "Auto",   "BAJAJ-AUTO.NS": "Auto",    "M&M.NS": "Auto",
    "LT.NS": "Industrials",    "ADANIPORTS.NS": "Industrials",
    "ADANIENT.NS": "Conglomerate","GRASIM.NS": "Conglomerate",
    "ASIANPAINT.NS": "Chemicals","UPL.NS": "Chemicals",
    "ULTRACEMCO.NS": "Cement",
    "JSWSTEEL.NS": "Metals",   "TATASTEEL.NS": "Metals",   "HINDALCO.NS": "Metals",
    "TITAN.NS": "Consumer Discretionary",
    "BHARTIARTL.NS": "Telecom",
}


# ──────────────────────────────────────────────────────────────────────────────
# DATA LAYER
# ──────────────────────────────────────────────────────────────────────────────


def download_data(
    tickers: list[str],
    start: str,
    end: str,
    data_dir: Path,
) -> dict[str, pd.DataFrame]:
    """
    Download OHLCV data from Yahoo Finance and cache as Parquet.

    On subsequent runs the Parquet cache is loaded directly, making
    repeated experiments significantly faster.

    Parameters
    ----------
    tickers  : Yahoo Finance ticker list.
    start    : Start date string "YYYY-MM-DD".
    end      : End date string  "YYYY-MM-DD".
    data_dir : Directory to store / read the Parquet cache.

    Returns
    -------
    dict with keys "close", "open", "high", "low", "volume".
    Each value is a pd.DataFrame (rows = dates, cols = tickers).
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = data_dir / f"stocks_{start}_{end}.pkl"

    if cache.exists():
        print(f"[DATA] Loading cached data  → {cache}")
        data_dict = pd.read_pickle(cache)
        return data_dict

    print(f"[DATA] Downloading {len(tickers)} tickers from Yahoo Finance …")
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,   # adjusts for splits and dividends
        progress=True,
    )

    def _extract(field: str) -> pd.DataFrame:
        """Pull a single price field from the yfinance MultiIndex DataFrame."""
        if isinstance(raw.columns, pd.MultiIndex):
            # yfinance MultiIndex: level 0 = field, level 1 = ticker
            if field in raw.columns.get_level_values(0):
                df = raw[field].copy()
                df.columns.name = None          # drop the 'Ticker' axis name
                return df.dropna(how="all", axis=1)
            return pd.DataFrame()
        # Single-ticker fallback (scalar columns)
        return raw[[field]].dropna(how="all", axis=1) if field in raw.columns else pd.DataFrame()

    data_dict = {
        "close":  _extract("Close"),
        "open":   _extract("Open"),
        "high":   _extract("High"),
        "low":    _extract("Low"),
        "volume": _extract("Volume"),
    }
    # Pickle preserves exact DataFrame structure with no serialisation issues
    pd.to_pickle(data_dict, cache)
    print(f"[DATA] Saved to {cache}")
    return data_dict


def download_nifty_index(start: str, end: str, data_dir: Path) -> pd.Series:
    """
    Download NIFTY 50 index close prices and cache to Parquet.

    Returns
    -------
    pd.Series indexed by date.
    """
    cache = data_dir / f"nifty_index_{start}_{end}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    print("[DATA] Downloading NIFTY 50 index …")
    df = yf.download(NIFTY_INDEX_TICKER, start=start, end=end, auto_adjust=True, progress=False)
    # Handle MultiIndex columns returned by newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        close_series = df["Close"].squeeze()
    else:
        close_series = df["Close"].squeeze()
    close_series.name = "Close"
    pd.to_pickle(close_series, cache)
    return close_series


# ──────────────────────────────────────────────────────────────────────────────
# SIGNAL PIPELINE
# ──────────────────────────────────────────────────────────────────────────────


def build_signals(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    nifty_close: pd.Series,
    cfg: BacktestConfig,
    use_sector_neutrality: bool = False,
    sector_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full signal pipeline.

    Steps
    -----
    1. Momentum ranking     → cross-sectional returns ranked daily (JT skip optional)
    2. Min-momentum gate    → optional threshold on raw momentum strength
    3. RSI filter           → only enter on RSI(2) extremes
    4. Regime filter        → align with NIFTY 50 trend (200-day MA)
    5. Volatility filter    → exclude stocks > vol_threshold_multiplier × median vol
    6. Sector neutrality    → optional, cap per-sector exposure
    7. Position cap         → hard limit of max_long / max_short per day
    8. ATR                  → computed for adaptive stop-loss in the engine

    A signal on day T means *execute at open of day T+1* (handled in engine).

    Parameters
    ----------
    close                : Daily adjusted close prices.
    high                 : Daily high prices (needed for ATR).
    low                  : Daily low prices (needed for ATR).
    nifty_close          : NIFTY 50 index close for regime filter.
    cfg                  : BacktestConfig parameters.
    use_sector_neutrality: Enable sector-cap filter.
    sector_map           : {ticker: sector} mapping.

    Returns
    -------
    (long_entries, short_entries, rsi, atr)  — all pd.DataFrame on same index as close.
    """
    n_stocks = close.shape[1]

    # ── 1. Momentum ────────────────────────────────────────────────────────
    momentum = calculate_momentum(close, window=cfg.momentum_window,
                                  skip_days=cfg.momentum_skip_days)
    ranks    = rank_momentum(momentum)
    long_c   = get_long_candidates(ranks, top_n=cfg.max_long)
    short_c  = get_short_candidates(ranks, total_stocks=n_stocks, bottom_n=cfg.max_short)

    # ── 2. Minimum momentum gate (optional) ───────────────────────────────
    if cfg.min_momentum_pct > 0:
        long_c  = long_c  & (momentum >= cfg.min_momentum_pct)
        short_c = short_c & (momentum <= -cfg.min_momentum_pct)

    # ── 3. RSI entry timing ────────────────────────────────────────────────
    rsi = calculate_rsi(close, period=cfg.rsi_period)
    long_e, short_e = get_entry_signals(
        long_c, short_c, rsi,
        rsi_long_thresh=cfg.rsi_long_entry,
        rsi_short_thresh=cfg.rsi_short_entry,
    )

    # ── 4. NIFTY regime filter ─────────────────────────────────────────────
    regime  = nifty_regime_filter(nifty_close, ma_window=cfg.regime_ma_window)
    long_e, short_e = apply_regime_filter(long_e, short_e, regime)

    # ── 5. Volatility filter ───────────────────────────────────────────────
    vol          = calculate_volatility_filter(close, window=cfg.vol_filter_window)
    median_vol   = vol.median(axis=1)
    threshold    = median_vol * cfg.vol_threshold_multiplier
    vol_ok       = vol.lt(threshold, axis=0).fillna(True)
    long_e       = apply_volatility_filter(long_e, vol_ok)
    short_e      = apply_volatility_filter(short_e, vol_ok)

    # ── 6. Sector neutrality (optional) ───────────────────────────────────
    if use_sector_neutrality and sector_map:
        long_e, short_e = sector_neutral_selection(
            long_e, short_e, sector_map, max_per_sector=2
        )

    # ── 7. Hard position cap (safety belt) ────────────────────────────────
    long_e  = cap_positions(long_e,  cfg.max_long)
    short_e = cap_positions(short_e, cfg.max_short)

    # ── 8. ATR for adaptive stop-loss ─────────────────────────────────────
    atr = calculate_atr(high, low, close, period=14)

    return long_e, short_e, rsi, atr


# ──────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD OPTIMISATION
# ──────────────────────────────────────────────────────────────────────────────


def walk_forward_test(
    close: pd.DataFrame,
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    nifty_close: pd.Series,
    param_grid: list[dict],
    train_months: int = 12,
    test_months:  int = 3,
    eval_start: str | None = None,
) -> pd.DataFrame:
    """
    Anchored walk-forward optimisation and out-of-sample validation.

    For each rolling window:
      1. Select the parameter set with the highest in-sample Sharpe.
      2. Apply those parameters to the held-out test window.
      3. Record in-sample and out-of-sample Sharpe.

    Stitching together OOS windows gives an unbiased equity curve.

    Parameters
    ----------
    param_grid    : List of dicts; each dict is passed as **kwargs to BacktestConfig.
    train_months  : Training window length in calendar months.
    test_months   : Out-of-sample test window length.

    Returns
    -------
    pd.DataFrame  One row per walk-forward window with Sharpe statistics.
    """
    try:
        from dateutil.relativedelta import relativedelta
    except ImportError:
        print("[WF] dateutil not installed — skipping walk-forward test.")
        return pd.DataFrame()

    from strategy.metrics import calculate_sharpe as _sharpe

    # ── Pre-compute signals on the FULL dataset for each param set ───────
    # All indicators (MA, RSI, momentum, Bollinger) are purely look-back;
    # pre-computing on full history removes the warmup/cold-start problem
    # that arises when slicing a short window (e.g. 200-day MA needs 200 days).
    precomputed: dict[str, tuple] = {}
    for params in param_grid:
        key = str(params)
        try:
            cfg = BacktestConfig(**params)
            le, se, rsi_df, atr_df = build_signals(close, high, low, nifty_close, cfg)
            precomputed[key] = (le, se, rsi_df, atr_df, cfg)
        except Exception as exc:
            print(f"[WF] Signal pre-compute failed for {params}: {exc}")

    if not precomputed:
        print("[WF] No param sets produced valid signals — aborting.")
        return pd.DataFrame()

    results = []
    all_dates = close.index
    # Skip the warmup period — windows start at eval_start (or first date)
    if eval_start:
        first_eval = pd.Timestamp(eval_start)
        valid_starts = all_dates[all_dates >= first_eval]
        window_start = valid_starts[0] if len(valid_starts) else all_dates[0]
    else:
        window_start = all_dates[0]

    while True:
        train_end = window_start + relativedelta(months=train_months)
        test_end  = train_end   + relativedelta(months=test_months)

        if test_end > all_dates[-1]:
            break

        t_mask   = (close.index >= window_start) & (close.index < train_end)
        oos_mask = (close.index >= train_end)    & (close.index < test_end)

        if t_mask.sum() < 60 or oos_mask.sum() < 15:
            window_start = train_end
            continue

        # ── Grid search on training window (using pre-computed signals) ──
        best_sharpe = -np.inf
        best_params = param_grid[0]

        for params in param_grid:
            key = str(params)
            if key not in precomputed:
                continue
            le, se, rsi_df, atr_df, cfg = precomputed[key]
            try:
                eng = BacktestEngine(cfg)
                res = eng.run(
                    close.loc[t_mask],  open_.loc[t_mask],
                    high.loc[t_mask],   low.loc[t_mask],
                    le.loc[t_mask],     se.loc[t_mask],    rsi_df.loc[t_mask],
                    atr=atr_df.loc[t_mask],
                )
                sh = _sharpe(res["daily_returns"])
                if sh > best_sharpe:
                    best_sharpe = sh
                    best_params = params
            except Exception:
                continue

        # ── Apply best params to OOS window (using pre-computed signals) ─
        key = str(best_params)
        if key not in precomputed:
            window_start = train_end
            continue
        try:
            le, se, rsi_df, atr_df, cfg = precomputed[key]
            eng = BacktestEngine(cfg)
            oos = eng.run(
                close.loc[oos_mask],  open_.loc[oos_mask],
                high.loc[oos_mask],   low.loc[oos_mask],
                le.loc[oos_mask],     se.loc[oos_mask],   rsi_df.loc[oos_mask],
                atr=atr_df.loc[oos_mask],
            )
            oos_sharpe = _sharpe(oos["daily_returns"])
            results.append({
                "window_start":  window_start.date(),
                "train_end":     train_end.date(),
                "test_end":      test_end.date(),
                "best_params":   str(best_params),
                "train_sharpe":  round(best_sharpe, 3),
                "oos_sharpe":    round(oos_sharpe, 3),
            })
        except Exception as exc:
            print(f"[WF] OOS window {train_end.date()} failed: {exc}")

        window_start = train_end

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────


def main(
    start_date: str = "2019-01-01",
    end_date: str   = "2024-12-31",
    use_sector_neutrality: bool = False,
    run_walk_forward: bool = False,
    capital: float = 1_000_000.0,
    compare: bool = False,
    max_short: int = 0,
    use_atr_stop: bool = False,
    trailing_stop_pct: float = 0.0,
    momentum_skip_days: int = 0,
    min_momentum_pct: float = 0.0,
) -> dict:
    """
    Orchestrates the full backtest pipeline.

    Returns the results dict from BacktestEngine.run() for programmatic use
    (e.g. from the research notebook).
    """
    output_dir = BASE_DIR / "outputs"
    data_dir   = BASE_DIR / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("  NIFTY 50 Cross-Sectional Momentum + Mean Reversion")
    print(f"  Period : {start_date}  →  {end_date}")
    print("=" * 62)

    # ── 1. Download data (+ 1-year warmup so the 200-day MA is valid
    #    from the very first day of start_date) ─────────────────────
    try:
        from dateutil.relativedelta import relativedelta as _rd
        warmup_start = (pd.Timestamp(start_date) - _rd(years=1)).strftime("%Y-%m-%d")
    except ImportError:
        warmup_start = start_date
    price_data  = download_data(NIFTY50_TICKERS, warmup_start, end_date, data_dir)
    close       = price_data["close"]
    open_       = price_data["open"]
    high        = price_data["high"]
    low         = price_data["low"]
    nifty_close = download_nifty_index(warmup_start, end_date, data_dir)

    # Align all series to a common trading-day index
    common_idx = (
        close.index
        .intersection(open_.index)
        .intersection(nifty_close.index)
    )
    close, open_, high, low = (
        df.loc[common_idx] for df in [close, open_, high, low]
    )
    nifty_close = nifty_close.loc[common_idx]

    # Drop tickers with >20% missing observations (avoids survivorship issues
    # for stocks that listed mid-period; they simply won't appear early)
    min_obs = int(len(common_idx) * 0.80)
    close   = close.dropna(thresh=min_obs, axis=1)
    for df in [open_, high, low]:
        df.drop(columns=[c for c in df.columns if c not in close.columns],
                errors="ignore", inplace=True)

    # Forward-fill residual NaNs (public holidays, exchange halts)
    close  = close.ffill()
    open_  = open_.reindex(columns=close.columns).ffill()
    high   = high.reindex(columns=close.columns).ffill()
    low    = low.reindex(columns=close.columns).ffill()

    print(f"\n[DATA] {len(close)} trading days  ×  {len(close.columns)} stocks")
    print(f"[DATA] warmup start   : {close.index[0].date()}")
    print(f"[DATA] eval start     : {start_date}  →  {close.index[-1].date()}")

    # ── 2. Build signals on full data (including warmup year) ───────────
    cfg = BacktestConfig(
        initial_capital=capital,
        max_short=max_short,
        use_atr_stop=use_atr_stop,
        trailing_stop_pct=trailing_stop_pct,
        momentum_skip_days=momentum_skip_days,
        min_momentum_pct=min_momentum_pct,
    )
    print("\n[SIGNAL] Computing momentum, RSI, regime, and volatility signals …")
    long_e, short_e, rsi, atr = build_signals(
        close, high, low, nifty_close, cfg,
        use_sector_neutrality=use_sector_neutrality,
        sector_map=NIFTY50_SECTORS,
    )
    print(f"[SIGNAL] LONG  entry days × tickers : {long_e.sum().sum():,}")
    print(f"[SIGNAL] SHORT entry days × tickers : {short_e.sum().sum():,}")

    # Trim to the user-requested start_date (warmup rows were only for indicator
    # initialisation and must not appear in the backtest equity curve / charts)
    eval_mask = close.index >= pd.Timestamp(start_date)
    close_bt  = close.loc[eval_mask]
    open_bt   = open_.loc[eval_mask]
    high_bt   = high.loc[eval_mask]
    low_bt    = low.loc[eval_mask]
    nc_bt     = nifty_close.loc[eval_mask]
    le_bt     = long_e.loc[eval_mask]
    se_bt     = short_e.loc[eval_mask]
    rsi_bt    = rsi.loc[eval_mask]
    atr_bt    = atr.loc[eval_mask]

    # ── 3. Run backtest ──────────────────────────────────────────────────
    print("\n[BACKTEST] Running simulation …")
    engine  = BacktestEngine(cfg)
    results = engine.run(close_bt, open_bt, high_bt, low_bt, le_bt, se_bt, rsi_bt, nc_bt, atr=atr_bt)

    equity_curve  = results["equity_curve"]
    daily_returns = results["daily_returns"]
    trade_log     = results["trade_log"]

    # ── 4. Performance metrics ───────────────────────────────────────────
    print("\n[METRICS] Computing performance statistics …")
    summary = generate_performance_summary(
        equity_curve, daily_returns, trade_log, cfg.initial_capital
    )
    print("\n" + "=" * 52)
    print("  PERFORMANCE SUMMARY")
    print("=" * 52)
    print(summary.to_string(index=False))

    summary.to_csv(output_dir / "performance_summary.csv", index=False)
    print(f"\n[OUTPUT] Summary saved  → {output_dir / 'performance_summary.csv'}")

    # ── 5. Charts ────────────────────────────────────────────────────────
    print("[OUTPUT] Generating charts …")

    nifty_bm = nc_bt.reindex(equity_curve.index, method="ffill")

    plot_equity_curve(
        equity_curve, benchmark=nifty_bm,
        output_path=str(output_dir / "equity_curve.png"),
    )
    plot_drawdown(
        equity_curve,
        output_path=str(output_dir / "drawdown.png"),
    )
    plot_monthly_heatmap(
        equity_curve,
        output_path=str(output_dir / "monthly_heatmap.png"),
    )
    plot_stock_contribution(
        trade_log,
        output_path=str(output_dir / "stock_contribution.png"),
    )
    print(f"[OUTPUT] Charts saved   → {output_dir}/")

    # ── 6. Trade log ─────────────────────────────────────────────────────
    tl_path = output_dir / "trade_log.csv"
    trade_log.to_csv(tl_path, index=False)
    print(f"[OUTPUT] Trade log      → {tl_path}")

    # ── 7. Walk-forward optimisation (optional) ──────────────────────────
    if compare:
        print("\n[COMPARE] Running short-term config (20d momentum) for comparison …")
        cfg_v1 = BacktestConfig(
            initial_capital=capital,
            momentum_window=20,
            rsi_long_entry=10.0,
            hold_days=3,
            max_short=max_short,
        )
        le_v1, se_v1, rsi_v1, atr_v1 = build_signals(
            close, high, low, nifty_close, cfg_v1,
            use_sector_neutrality=use_sector_neutrality,
            sector_map=NIFTY50_SECTORS,
        )
        le_v1_bt  = le_v1.loc[eval_mask]
        se_v1_bt  = se_v1.loc[eval_mask]
        rsi_v1_bt = rsi_v1.loc[eval_mask]
        atr_v1_bt = atr_v1.loc[eval_mask]

        engine_v1  = BacktestEngine(cfg_v1)
        results_v1 = engine_v1.run(
            close_bt, open_bt, high_bt, low_bt,
            le_v1_bt, se_v1_bt, rsi_v1_bt, nc_bt,
            atr=atr_v1_bt,
        )

        from strategy.metrics import (
            calculate_cagr, calculate_sharpe, calculate_sortino,
            calculate_max_drawdown, calculate_calmar,
            calculate_win_rate, calculate_profit_factor,
            calculate_avg_holding_period,
        )

        def _cstats(res: dict, lbl: str) -> dict:
            eq  = res["equity_curve"]
            dr  = res["daily_returns"]
            tl  = res["trade_log"]
            nav = eq.dropna().iloc[-1] if not eq.dropna().empty else capital
            tot_cost = (
                tl["entry_cost"].sum() + tl["exit_cost"].sum()
                if not tl.empty else 0.0
            )
            return {
                "Label":          lbl,
                "CAGR (%)": f"{calculate_cagr(eq) * 100:.2f}%",
                "Sharpe":    f"{calculate_sharpe(dr):.3f}",
                "Sortino":   f"{calculate_sortino(dr):.3f}",
                "Max DD (%)": f"{calculate_max_drawdown(eq)[0] * 100:.2f}%",
                "Calmar":    f"{calculate_calmar(eq):.3f}",
                "Trades":    len(tl),
                "Win Rate (%)": f"{calculate_win_rate(tl) * 100:.1f}%",
                "Profit Factor": f"{calculate_profit_factor(tl):.2f}",
                "Avg Hold (d)": f"{calculate_avg_holding_period(tl):.1f}",
                "Costs (\u20b9L)": f"{tot_cost / 1e5:.1f}",
                "Final NAV (\u20b9L)": f"{nav / 1e5:.1f}",
            }

        s1 = _cstats(results_v1, f"Short-term (20d mom, entry RSI<10, hold≤3d)")
        s2 = _cstats(results,    f"Current    (60d mom, entry RSI<15, hold≤5d)")
        comp_df = pd.DataFrame([s1, s2]).set_index("Label").T

        print("\n" + "=" * 72)
        print("  CONFIGURATION COMPARISON")
        print("=" * 72)
        print(comp_df.to_string())
        print("=" * 72 + "\n")

        comp_path = output_dir / "comparison_report.csv"
        comp_df.to_csv(comp_path)
        print(f"[OUTPUT] Comparison saved  \u2192 {comp_path}")

    if run_walk_forward:
        print("\n[WF] Running walk-forward optimisation …")
        param_grid = [
            # Vary momentum window
            {"momentum_window": 20, "rsi_long_entry": 20.0, "hold_days": 5, "max_short": 0},
            {"momentum_window": 25, "rsi_long_entry": 20.0, "hold_days": 5, "max_short": 0},
            {"momentum_window": 30, "rsi_long_entry": 20.0, "hold_days": 5, "max_short": 0},
            # With JT momentum skip (avoid short-term reversal)
            {"momentum_window": 20, "rsi_long_entry": 20.0, "hold_days": 5, "momentum_skip_days": 5, "max_short": 0},
            {"momentum_window": 25, "rsi_long_entry": 20.0, "hold_days": 5, "momentum_skip_days": 5, "max_short": 0},
            # Vary RSI entry threshold
            {"momentum_window": 25, "rsi_long_entry": 15.0, "hold_days": 5, "max_short": 0},
            {"momentum_window": 25, "rsi_long_entry": 25.0, "hold_days": 5, "max_short": 0},
            # Vary hold period
            {"momentum_window": 25, "rsi_long_entry": 20.0, "hold_days": 7, "max_short": 0},
            {"momentum_window": 25, "rsi_long_entry": 20.0, "hold_days": 10, "max_short": 0},
        ]
        wf = walk_forward_test(
            close, open_, high, low, nifty_close, param_grid,
            train_months=12, test_months=3,
            eval_start=start_date,
        )
        if not wf.empty:
            wf_path = output_dir / "walk_forward_results.csv"
            wf.to_csv(wf_path, index=False)
            print(f"[WF] Results saved      → {wf_path}")
            print(wf.to_string())

    print("\n[DONE] Backtest complete.\n")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NIFTY 50 Cross-Sectional Momentum + Mean Reversion Backtest"
    )
    parser.add_argument("--start",          default="2019-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",            default="2024-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--capital",        default=1_000_000.0,  type=float, help="Initial capital in ₹")
    parser.add_argument("--sector-neutral", action="store_true",  help="Enable sector neutrality")
    parser.add_argument("--walk-forward",   action="store_true",  help="Run walk-forward optimisation")
    parser.add_argument("--compare",        action="store_true",  help="Compare short-term vs current momentum config")
    parser.add_argument("--max-short",      default=0,            type=int,   help="Max simultaneous short positions (0 = long-only)")
    parser.add_argument("--atr-stop",       action="store_true",  help="Use ATR-based adaptive stops instead of fixed %%")
    parser.add_argument("--trail-stop",     default=0.0,          type=float, help="Trailing stop as decimal from peak (e.g. 0.03 = 3%%)")
    parser.add_argument("--mom-skip",       default=0,            type=int,   help="JT momentum skip days (try 5 to avoid short-term reversal)")
    parser.add_argument("--min-mom",        default=0.0,          type=float, help="Min momentum threshold to enter (e.g. 0.03 = 3%%)")
    args = parser.parse_args()

    main(
        start_date=args.start,
        end_date=args.end,
        use_sector_neutrality=args.sector_neutral,
        run_walk_forward=args.walk_forward,
        capital=args.capital,
        compare=args.compare,
        max_short=args.max_short,
        use_atr_stop=args.atr_stop,
        trailing_stop_pct=args.trail_stop,
        momentum_skip_days=args.mom_skip,
        min_momentum_pct=args.min_mom,
    )
