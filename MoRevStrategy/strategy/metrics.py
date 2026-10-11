"""
metrics.py
----------
Performance analytics and visualisation for the backtest results.

Metrics computed
----------------
- CAGR               (Compound Annual Growth Rate)
- Sharpe Ratio       (annualised, risk-free = RBI repo rate proxy 6.5%)
- Sortino Ratio      (uses downside deviation only)
- Max Drawdown       (peak-to-trough; also returns full drawdown series)
- Calmar Ratio       (CAGR / |Max Drawdown|)
- Win Rate           (% of trades with positive net PnL)
- Profit Factor      (sum of wins / |sum of losses|)
- Average Holding Period
- Monthly Returns    (pivot for heatmap)

Plots produced
--------------
1. Equity curve (vs optional NIFTY 50 benchmark)
2. Underwater / drawdown chart
3. Monthly returns heatmap
4. Per-stock PnL contribution bar chart
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive — safe in all environments
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

RISK_FREE_ANNUAL = 0.065    # ~6.5% (RBI repo rate proxy)
TRADING_DAYS     = 252


# ──────────────────────────────────────────────────────────────────────────────
# SCALAR METRICS
# ──────────────────────────────────────────────────────────────────────────────


def calculate_cagr(equity_curve: pd.Series) -> float:
    """
    Compound Annual Growth Rate.

    CAGR = (end_NAV / start_NAV)^(252 / n_trading_days) − 1

    Parameters
    ----------
    equity_curve : pd.Series  Portfolio NAV time series.

    Returns
    -------
    float   CAGR as a decimal (e.g. 0.18 = 18% p.a.).
    """
    clean = equity_curve.dropna()
    if len(clean) < 2 or clean.iloc[0] <= 0:
        return 0.0
    years = len(clean) / TRADING_DAYS
    return (clean.iloc[-1] / clean.iloc[0]) ** (1.0 / years) - 1.0


def calculate_sharpe(daily_returns: pd.Series) -> float:
    """
    Annualised Sharpe Ratio.

    Sharpe = (mean(excess_daily_return) / std(daily_return)) × √252

    Parameters
    ----------
    daily_returns : pd.Series  Daily percentage returns.

    Returns
    -------
    float   Sharpe ratio. 0.0 if std = 0.
    """
    r = daily_returns.dropna().replace([np.inf, -np.inf], 0.0)
    if r.std() == 0.0:
        return 0.0
    rfr_daily = RISK_FREE_ANNUAL / TRADING_DAYS
    return ((r - rfr_daily).mean() / r.std()) * np.sqrt(TRADING_DAYS)


def calculate_sortino(daily_returns: pd.Series) -> float:
    """
    Annualised Sortino Ratio — penalises only downside volatility.

    Sortino = mean(excess_daily_return) / downside_std × √252
    """
    r = daily_returns.dropna().replace([np.inf, -np.inf], 0.0)
    rfr_daily = RISK_FREE_ANNUAL / TRADING_DAYS
    excess = r - rfr_daily
    downside = excess[excess < 0.0]
    if len(downside) == 0 or downside.std() == 0.0:
        return 0.0
    return (excess.mean() / downside.std()) * np.sqrt(TRADING_DAYS)


def calculate_max_drawdown(equity_curve: pd.Series) -> tuple[float, pd.Series]:
    """
    Maximum peak-to-trough drawdown and the full drawdown time series.

    drawdown[t] = (equity[t] − running_max[t]) / running_max[t]

    Returns
    -------
    tuple[float, pd.Series]
        (max_drawdown_decimal, drawdown_series).
        max_drawdown is a non-positive number (e.g. -0.25 = -25%).
    """
    clean = equity_curve.dropna()
    running_max = clean.cummax()
    drawdown = (clean - running_max) / running_max
    return float(drawdown.min()), drawdown


def calculate_calmar(equity_curve: pd.Series) -> float:
    """Calmar Ratio = CAGR / |Max Drawdown|."""
    cagr = calculate_cagr(equity_curve)
    mdd, _ = calculate_max_drawdown(equity_curve)
    return cagr / abs(mdd) if mdd != 0.0 else 0.0


def calculate_win_rate(trade_log: pd.DataFrame) -> float:
    """Win rate = fraction of trades with net_pnl > 0."""
    if trade_log.empty or "net_pnl" not in trade_log.columns:
        return 0.0
    return (trade_log["net_pnl"] > 0).mean()


def calculate_profit_factor(trade_log: pd.DataFrame) -> float:
    """
    Profit Factor = total winning PnL / |total losing PnL|.

    Values > 1 indicate a profitable strategy.
    """
    if trade_log.empty:
        return 0.0
    wins   = trade_log.loc[trade_log["net_pnl"] > 0,  "net_pnl"].sum()
    losses = trade_log.loc[trade_log["net_pnl"] <= 0, "net_pnl"].sum()
    return wins / abs(losses) if losses != 0.0 else float("inf")


def calculate_avg_holding_period(trade_log: pd.DataFrame) -> float:
    """Average calendar days a position is held (entry → exit)."""
    if trade_log.empty or "holding_days" not in trade_log.columns:
        return 0.0
    return trade_log["holding_days"].dropna().mean()


# ──────────────────────────────────────────────────────────────────────────────
# MONTHLY RETURNS
# ──────────────────────────────────────────────────────────────────────────────


def calculate_monthly_returns(equity_curve: pd.Series) -> pd.DataFrame:
    """
    Resample the equity curve to end-of-month NAV and compute monthly returns.

    Returns
    -------
    pd.DataFrame
        Rows = calendar year, columns = month abbreviations,
        values = monthly return as decimal.
    """
    monthly = equity_curve.resample("ME").last().pct_change().dropna()
    monthly.index = pd.to_datetime(monthly.index)

    pivot = pd.DataFrame({
        "year":   monthly.index.year,
        "month":  monthly.index.month,
        "return": monthly.values,
    }).pivot(index="year", columns="month", values="return")

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot.columns = [month_labels[m - 1] for m in pivot.columns]
    return pivot


# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ──────────────────────────────────────────────────────────────────────────────


def generate_performance_summary(
    equity_curve: pd.Series,
    daily_returns: pd.Series,
    trade_log: pd.DataFrame,
    initial_capital: float = 1_000_000.0,
) -> pd.DataFrame:
    """
    Compile a human-readable performance summary table.

    Returns
    -------
    pd.DataFrame with columns ["Metric", "Value"].
    """
    cagr         = calculate_cagr(equity_curve)
    sharpe       = calculate_sharpe(daily_returns)
    sortino      = calculate_sortino(daily_returns)
    mdd, _       = calculate_max_drawdown(equity_curve)
    calmar       = calculate_calmar(equity_curve)
    win_rate     = calculate_win_rate(trade_log)
    pf           = calculate_profit_factor(trade_log)
    avg_hold     = calculate_avg_holding_period(trade_log)

    final_nav    = equity_curve.dropna().iloc[-1] if not equity_curve.dropna().empty else initial_capital
    total_return = (final_nav - initial_capital) / initial_capital

    n_total  = len(trade_log)
    n_long   = int((trade_log["direction"] == "LONG").sum())  if not trade_log.empty else 0
    n_short  = int((trade_log["direction"] == "SHORT").sum()) if not trade_log.empty else 0
    tot_cost = trade_log["entry_cost"].sum() + trade_log["exit_cost"].sum() if not trade_log.empty else 0.0

    # Exit reason breakdown
    exit_counts = trade_log["exit_reason"].value_counts().to_dict() if not trade_log.empty else {}

    rows = [
        ("Initial Capital (₹)",            f"₹{initial_capital:>12,.0f}"),
        ("Final Capital (₹)",              f"₹{final_nav:>12,.0f}"),
        ("Total Return (%)",               f"{total_return * 100:>8.2f}%"),
        ("CAGR (%)",                       f"{cagr * 100:>8.2f}%"),
        ("Sharpe Ratio",                   f"{sharpe:>8.3f}"),
        ("Sortino Ratio",                  f"{sortino:>8.3f}"),
        ("Calmar Ratio",                   f"{calmar:>8.3f}"),
        ("Max Drawdown (%)",               f"{mdd * 100:>8.2f}%"),
        ("Win Rate (%)",                   f"{win_rate * 100:>8.2f}%"),
        ("Profit Factor",                  f"{pf:>8.3f}"),
        ("Total Trades",                   f"{n_total:>8,}"),
        ("Long Trades",                    f"{n_long:>8,}"),
        ("Short Trades",                   f"{n_short:>8,}"),
        ("Avg Holding Period (days)",      f"{avg_hold:>8.1f}"),
        ("Total Transaction Costs (₹)",   f"₹{tot_cost:>12,.0f}"),
        ("Exits — TIME",                   f"{exit_counts.get('TIME', 0):>8,}"),
        ("Exits — STOP_LOSS",             f"{exit_counts.get('STOP_LOSS', 0):>8,}"),
        ("Exits — TRAIL_STOP",            f"{exit_counts.get('TRAIL_STOP', 0):>8,}"),
        ("Exits — RSI_EXIT",              f"{exit_counts.get('RSI_EXIT', 0):>8,}"),
    ]

    return pd.DataFrame(rows, columns=["Metric", "Value"])


# ──────────────────────────────────────────────────────────────────────────────
# PLOTS
# ──────────────────────────────────────────────────────────────────────────────


def plot_equity_curve(
    equity_curve: pd.Series,
    benchmark: pd.Series | None = None,
    title: str = "Equity Curve — Cross-Sectional Momentum + Mean Reversion",
    output_path: str | None = None,
) -> plt.Figure:
    """
    Plot indexed portfolio NAV vs optional benchmark (both rebased to 100).

    Parameters
    ----------
    equity_curve : pd.Series      Strategy NAV.
    benchmark    : pd.Series      Optional index (e.g. NIFTY 50).
    output_path  : str            If given, saves the figure to disk.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    base = equity_curve.dropna().iloc[0]
    ax.plot(equity_curve / base * 100, label="Strategy", color="#1565C0", linewidth=1.8)

    if benchmark is not None:
        bm = benchmark.reindex(equity_curve.index, method="ffill").dropna()
        if not bm.empty:
            ax.plot(bm / bm.iloc[0] * 100, label="NIFTY 50", color="#EF6C00",
                    linewidth=1.4, alpha=0.85, linestyle="--")

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Indexed NAV (Base = 100)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=":")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_drawdown(
    equity_curve: pd.Series,
    output_path: str | None = None,
) -> plt.Figure:
    """
    Underwater (drawdown) chart — shaded area from 0 down to drawdown level.

    Parameters
    ----------
    equity_curve : pd.Series  Portfolio NAV.
    output_path  : str        Optional save path.

    Returns
    -------
    plt.Figure
    """
    _, dd = calculate_max_drawdown(equity_curve)
    dd_pct = dd * 100.0

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(dd_pct.index, dd_pct, 0.0,
                    color="#C62828", alpha=0.55, label="Drawdown")
    ax.plot(dd_pct, color="#B71C1C", linewidth=0.9)
    ax.set_title("Portfolio Drawdown", fontsize=14, pad=12)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Drawdown (%)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=":")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_monthly_heatmap(
    equity_curve: pd.Series,
    output_path: str | None = None,
) -> plt.Figure | None:
    """
    Colour-coded monthly returns heatmap (green = positive, red = negative).

    Parameters
    ----------
    equity_curve : pd.Series  Portfolio NAV.
    output_path  : str        Optional save path.

    Returns
    -------
    plt.Figure or None (if fewer than 2 months of data).
    """
    monthly_df = calculate_monthly_returns(equity_curve)
    if monthly_df.empty:
        return None

    values = monthly_df.values.astype(float)
    finite_vals = values[np.isfinite(values)]
    if len(finite_vals) == 0:
        return None

    vmax = max(abs(finite_vals).max(), 0.005)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    nrows = len(monthly_df)
    fig, ax = plt.subplots(figsize=(14, max(3, nrows * 0.55 + 1.5)))
    im = ax.imshow(values, cmap="RdYlGn", norm=norm, aspect="auto")

    ax.set_xticks(range(len(monthly_df.columns)))
    ax.set_xticklabels(monthly_df.columns, fontsize=9)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(monthly_df.index, fontsize=9)

    # Annotate cells with return %
    for r in range(nrows):
        for c in range(len(monthly_df.columns)):
            val = values[r, c]
            if np.isfinite(val):
                txt_color = "white" if abs(val) > vmax * 0.55 else "black"
                ax.text(c, r, f"{val * 100:.1f}%",
                        ha="center", va="center", fontsize=7.5, color=txt_color)

    plt.colorbar(im, ax=ax, label="Monthly Return", fraction=0.025, pad=0.02)
    ax.set_title("Monthly Returns Heatmap", fontsize=14, pad=12)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_stock_contribution(
    trade_log: pd.DataFrame,
    output_path: str | None = None,
    top_n: int = 30,
) -> plt.Figure | None:
    """
    Horizontal bar chart of per-stock net PnL contribution.

    Shows the ``top_n`` names by absolute PnL impact (positive + negative).

    Parameters
    ----------
    trade_log : pd.DataFrame  Output of BacktestEngine.run().
    output_path : str         Optional save path.
    top_n : int               Maximum tickers to display.

    Returns
    -------
    plt.Figure or None.
    """
    if trade_log.empty or "net_pnl" not in trade_log.columns:
        return None

    contrib = (
        trade_log.groupby("ticker")["net_pnl"]
        .sum()
        .reindex(
            trade_log.groupby("ticker")["net_pnl"].sum()
            .abs().nlargest(top_n).index
        )
        .sort_values()
    )

    colors = ["#388E3C" if v >= 0 else "#D32F2F" for v in contrib.values]
    fig, ax = plt.subplots(figsize=(13, max(5, len(contrib) * 0.38)))
    ax.barh(contrib.index, contrib.values, color=colors, edgecolor="none")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Cumulative Net PnL (₹)", fontsize=11)
    ax.set_title(f"Per-Stock PnL Contribution (top {top_n} by impact)", fontsize=14, pad=12)
    ax.grid(True, axis="x", alpha=0.3, linestyle=":")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
