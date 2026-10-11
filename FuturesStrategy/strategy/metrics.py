"""
metrics.py
----------
Performance analytics and visualisation for the NSE Futures strategy.

Ten chart functions
-------------------
1.  plot_equity_curve          — NAV vs NIFTY benchmark
2.  plot_drawdown              — Underwater drawdown chart
3.  plot_monthly_heatmap       — Calendar heatmap of monthly returns
4.  plot_exposure_analysis     — Gross / net notional and margin utilisation
5.  plot_long_short_pnl        — Cumulative P&L split by direction
6.  plot_rollover_costs        — Cumulative rollover cost vs total cost
7.  plot_stock_contribution    — Per-ticker realised P&L bar chart
8.  plot_trades_scatter        — Hold-days vs net P&L scatter
9.  plot_win_loss_distribution — Histogram of trade returns
10. generate_performance_report — Print + save CSV summary with all KPIs
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as mcolors


# ──────────────────────────────────────────────────────────────────────────────
# HELPER STATISTICS
# ──────────────────────────────────────────────────────────────────────────────

def _annualised_return(equity: pd.Series) -> float:
    """CAGR assuming 252 trading days per year."""
    n = len(equity.dropna())
    if n < 2:
        return np.nan
    total = equity.iloc[-1] / equity.iloc[0]
    years = n / 252
    return total ** (1.0 / years) - 1.0


def _max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative fraction)."""
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return float(dd.min())


def _drawdown_series(equity: pd.Series) -> pd.Series:
    roll_max = equity.cummax()
    return (equity - roll_max) / roll_max


def _sharpe(returns: pd.Series, rf: float = 0.065) -> float:
    daily_rf = (1 + rf) ** (1 / 252) - 1
    excess   = returns - daily_rf
    if excess.std() == 0:
        return np.nan
    return float((excess.mean() / excess.std()) * np.sqrt(252))


def _sortino(returns: pd.Series, rf: float = 0.065) -> float:
    daily_rf  = (1 + rf) ** (1 / 252) - 1
    excess    = returns - daily_rf
    downside  = excess[excess < 0].std()
    if downside == 0:
        return np.nan
    return float((excess.mean() / downside) * np.sqrt(252))


def _calmar(equity: pd.Series) -> float:
    ann_ret = _annualised_return(equity)
    mdd     = _max_drawdown(equity)
    if mdd == 0:
        return np.nan
    return -ann_ret / mdd


def _trade_stats(trade_log: pd.DataFrame) -> dict:
    if trade_log.empty:
        return {}
    pnl   = trade_log["net_pnl"]
    wins  = pnl[pnl > 0]
    losses= pnl[pnl < 0]
    return {
        "total_trades":     len(pnl),
        "win_rate":         len(wins) / len(pnl),
        "avg_win":          wins.mean()   if len(wins)  else 0.0,
        "avg_loss":         losses.mean() if len(losses)else 0.0,
        "profit_factor":    wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf,
        "avg_hold_days":    trade_log["hold_days"].mean() if "hold_days" in trade_log else np.nan,
        "total_pnl":        pnl.sum(),
        "total_entry_cost": trade_log["entry_cost"].sum(),
        "total_exit_cost":  trade_log["exit_cost"].sum(),
        "total_roll_cost":  trade_log["rollover_costs"].sum(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 1. EQUITY CURVE
# ──────────────────────────────────────────────────────────────────────────────

def plot_equity_curve(
    equity: pd.Series,
    nifty_close: pd.Series | None = None,
    output_dir: str | Path | None = None,
) -> plt.Figure:
    """
    NAV curve normalised to 100 at start, vs NIFTY benchmark (if provided).
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    eq_norm = equity / equity.iloc[0] * 100
    ax.plot(eq_norm.index, eq_norm, label="Strategy", color="steelblue", lw=1.5)

    if nifty_close is not None and not nifty_close.empty:
        aligned = nifty_close.reindex(equity.index, method="ffill").dropna()
        if not aligned.empty:
            nifty_norm = aligned / aligned.iloc[0] * 100
            ax.plot(nifty_norm.index, nifty_norm, label="NIFTY 50", color="darkorange",
                    lw=1.2, linestyle="--", alpha=0.8)

    ax.set_title("Equity Curve (Indexed to 100)", fontsize=13)
    ax.set_ylabel("Indexed NAV")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if output_dir:
        fig.savefig(Path(output_dir) / "equity_curve.png", dpi=150)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 2. DRAWDOWN
# ──────────────────────────────────────────────────────────────────────────────

def plot_drawdown(
    equity: pd.Series,
    output_dir: str | Path | None = None,
) -> plt.Figure:
    dd = _drawdown_series(equity) * 100
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(dd.index, dd, 0, color="crimson", alpha=0.45, label="Drawdown")
    ax.plot(dd.index, dd, color="crimson", lw=0.7)
    ax.set_title("Drawdown (%)", fontsize=13)
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if output_dir:
        fig.savefig(Path(output_dir) / "drawdown.png", dpi=150)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 3. MONTHLY RETURN HEATMAP
# ──────────────────────────────────────────────────────────────────────────────

def plot_monthly_heatmap(
    daily_returns: pd.Series,
    output_dir: str | Path | None = None,
) -> plt.Figure:
    """Calendar heatmap of monthly returns (%)."""
    monthly = (1 + daily_returns).resample("ME").prod() - 1
    monthly.index = monthly.index.to_period("M")
    pivot = monthly.groupby([monthly.index.year, monthly.index.month]).first().unstack()
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.55)))
    cmap = mcolors.TwoSlopeNorm(vmin=-0.10, vcenter=0, vmax=0.10)
    im   = ax.imshow(pivot.values.astype(float), cmap="RdYlGn", norm=cmap, aspect="auto")

    ax.set_xticks(range(12))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index)

    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            v = pivot.iloc[r, c]
            if not np.isnan(v):
                ax.text(c, r, f"{v*100:.1f}%", ha="center", va="center", fontsize=7,
                        color="black" if abs(v) < 0.06 else "white")

    plt.colorbar(im, ax=ax, format=mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_title("Monthly Returns Heatmap", fontsize=13)
    fig.tight_layout()
    if output_dir:
        fig.savefig(Path(output_dir) / "monthly_heatmap.png", dpi=150)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 4. EXPOSURE ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def plot_exposure_analysis(
    exposure_df: pd.DataFrame,
    equity: pd.Series,
    output_dir: str | Path | None = None,
) -> plt.Figure:
    """
    Three panels:
    (a) Gross and net notional (₹ cr)
    (b) Long vs short notional
    (c) Margin utilisation as % of NAV
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    cr = 1e7  # 1 crore

    # Panel (a) — gross / net
    axes[0].plot(exposure_df.index, exposure_df["gross_notional"] / cr, label="Gross", color="steelblue")
    axes[0].plot(exposure_df.index, exposure_df["net_notional"]   / cr, label="Net",   color="seagreen", linestyle="--")
    axes[0].set_ylabel("Notional (₹ Cr)")
    axes[0].set_title("Portfolio Exposure")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.3)

    # Panel (b) — long / short
    axes[1].fill_between(exposure_df.index, exposure_df["long_notional"]  / cr, label="Long",  color="limegreen", alpha=0.5)
    axes[1].fill_between(exposure_df.index, -exposure_df["short_notional"]/ cr, label="Short", color="tomato",    alpha=0.5)
    axes[1].set_ylabel("Notional (₹ Cr)")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.3)

    # Panel (c) — margin utilisation
    margin_pct = exposure_df.get("margin_used", pd.Series(dtype=float)) / equity.reindex(exposure_df.index).fillna(method="ffill") * 100
    axes[2].plot(margin_pct.index, margin_pct, color="darkorchid", lw=1.2)
    axes[2].axhline(100, color="crimson", linestyle="--", lw=0.8, label="100% NAV")
    axes[2].set_ylabel("Margin / NAV (%)")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    if output_dir:
        fig.savefig(Path(output_dir) / "exposure_analysis.png", dpi=150)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 5. LONG / SHORT CUMULATIVE P&L
# ──────────────────────────────────────────────────────────────────────────────

def plot_long_short_pnl(
    trade_log: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> plt.Figure:
    if trade_log.empty:
        return plt.figure()

    fig, ax = plt.subplots(figsize=(12, 5))
    for direction, color in [("LONG", "limegreen"), ("SHORT", "tomato")]:
        sub = trade_log[trade_log["direction"] == direction].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("exit_date")
        cum = sub["net_pnl"].cumsum()
        ax.plot(sub["exit_date"], cum / 1e5, label=f"{direction} cumP&L (₹ L)", color=color, lw=1.5)

    ax.set_title("Cumulative P&L by Direction (₹ Lakhs)")
    ax.set_ylabel("Cumulative P&L (₹ L)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if output_dir:
        fig.savefig(Path(output_dir) / "long_short_pnl.png", dpi=150)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 6. ROLLOVER COSTS
# ──────────────────────────────────────────────────────────────────────────────

def plot_rollover_costs(
    trade_log: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> plt.Figure:
    if trade_log.empty:
        return plt.figure()

    tl = trade_log.sort_values("exit_date").copy()
    fig, ax = plt.subplots(figsize=(12, 4))

    cum_total = (tl["entry_cost"] + tl["exit_cost"] + tl["rollover_costs"]).cumsum()
    cum_roll  = tl["rollover_costs"].cumsum()

    ax.plot(tl["exit_date"], cum_total / 1e5, label="Total Costs",    color="steelblue", lw=1.5)
    ax.plot(tl["exit_date"], cum_roll  / 1e5, label="Rollover Costs", color="darkorange", lw=1.2, linestyle="--")

    ax.set_title("Cumulative Transaction + Rollover Costs (₹ Lakhs)")
    ax.set_ylabel("Cumulative Cost (₹ L)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if output_dir:
        fig.savefig(Path(output_dir) / "rollover_costs.png", dpi=150)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 7. PER-STOCK CONTRIBUTION
# ──────────────────────────────────────────────────────────────────────────────

def plot_stock_contribution(
    trade_log: pd.DataFrame,
    output_dir: str | Path | None = None,
    top_n: int = 20,
) -> plt.Figure:
    if trade_log.empty:
        return plt.figure()

    by_ticker = trade_log.groupby("ticker")["net_pnl"].sum().sort_values()
    if len(by_ticker) > top_n:
        keep_neg = by_ticker.head(top_n // 2)
        keep_pos = by_ticker.tail(top_n // 2)
        by_ticker = pd.concat([keep_neg, keep_pos]).sort_values()

    colors = ["tomato" if v < 0 else "limegreen" for v in by_ticker]

    fig, ax = plt.subplots(figsize=(10, max(4, len(by_ticker) * 0.4)))
    ax.barh(by_ticker.index, by_ticker / 1e5, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Net P&L (₹ Lakhs)")
    ax.set_title(f"Per-Ticker Net P&L Contribution (Top/Bottom {top_n // 2})")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    if output_dir:
        fig.savefig(Path(output_dir) / "stock_contribution.png", dpi=150)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 8. TRADES SCATTER
# ──────────────────────────────────────────────────────────────────────────────

def plot_trades_scatter(
    trade_log: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> plt.Figure:
    """Hold days vs net P&L scatter, coloured by direction."""
    if trade_log.empty or "hold_days" not in trade_log.columns:
        return plt.figure()

    fig, ax = plt.subplots(figsize=(9, 5))
    for direction, color in [("LONG", "steelblue"), ("SHORT", "darkorange")]:
        sub = trade_log[trade_log["direction"] == direction]
        ax.scatter(sub["hold_days"], sub["net_pnl"] / 1e5, color=color,
                   alpha=0.5, s=25, label=direction)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("Hold Days")
    ax.set_ylabel("Net P&L (₹ L)")
    ax.set_title("Trade P&L vs Hold Duration")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if output_dir:
        fig.savefig(Path(output_dir) / "trades_scatter.png", dpi=150)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 9. WIN / LOSS DISTRIBUTION
# ──────────────────────────────────────────────────────────────────────────────

def plot_win_loss_distribution(
    trade_log: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> plt.Figure:
    if trade_log.empty:
        return plt.figure()

    pnl = trade_log["net_pnl"] / 1e5
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(pnl[pnl >= 0], bins=30, color="limegreen", alpha=0.7, label="Wins")
    ax.hist(pnl[pnl <  0], bins=30, color="tomato",    alpha=0.7, label="Losses")
    ax.axvline(0, color="black", lw=1)
    ax.axvline(pnl.mean(), color="steelblue", lw=1.2, linestyle="--", label=f"Mean={pnl.mean():.2f}L")
    ax.set_xlabel("Trade Net P&L (₹ Lakhs)")
    ax.set_ylabel("Count")
    ax.set_title("Win / Loss Distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if output_dir:
        fig.savefig(Path(output_dir) / "win_loss_distribution.png", dpi=150)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 10. PERFORMANCE REPORT
# ──────────────────────────────────────────────────────────────────────────────

def generate_performance_report(
    equity: pd.Series,
    daily_returns: pd.Series,
    trade_log: pd.DataFrame,
    initial_capital: float,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Compute and print all key performance indicators; optionally save to CSV.

    Returns
    -------
    pd.DataFrame  single-row summary table.
    """
    tr_stats = _trade_stats(trade_log)

    start_date = equity.index[0]
    end_date   = equity.index[-1]
    years      = (end_date - start_date).days / 365

    kpis = {
        "start_date":       start_date.date(),
        "end_date":         end_date.date(),
        "initial_capital":  f"₹{initial_capital:,.0f}",
        "final_nav":        f"₹{equity.iloc[-1]:,.0f}",
        "total_return_pct": f"{(equity.iloc[-1]/initial_capital - 1)*100:.2f}%",
        "cagr_pct":         f"{_annualised_return(equity)*100:.2f}%",
        "volatility_ann":   f"{daily_returns.std() * np.sqrt(252) * 100:.2f}%",
        "sharpe":           f"{_sharpe(daily_returns):.3f}",
        "sortino":          f"{_sortino(daily_returns):.3f}",
        "calmar":           f"{_calmar(equity):.3f}",
        "max_drawdown":     f"{_max_drawdown(equity)*100:.2f}%",
        "total_trades":     tr_stats.get("total_trades", 0),
        "win_rate":         f"{tr_stats.get('win_rate', 0)*100:.1f}%",
        "profit_factor":    f"{tr_stats.get('profit_factor', 0):.2f}",
        "avg_win":          f"₹{tr_stats.get('avg_win', 0):,.0f}",
        "avg_loss":         f"₹{tr_stats.get('avg_loss', 0):,.0f}",
        "avg_hold_days":    f"{tr_stats.get('avg_hold_days', 0):.1f}",
        "total_pnl":        f"₹{tr_stats.get('total_pnl', 0):,.0f}",
        "total_costs":      f"₹{(tr_stats.get('total_entry_cost',0)+tr_stats.get('total_exit_cost',0)+tr_stats.get('total_roll_cost',0)):,.0f}",
        "rollover_costs":   f"₹{tr_stats.get('total_roll_cost', 0):,.0f}",
        "years_simulated":  f"{years:.1f}",
    }

    # Pretty print
    pad = max(len(k) for k in kpis)
    print("\n" + "="*60)
    print("  FUTURES STRATEGY — PERFORMANCE REPORT")
    print("="*60)
    for k, v in kpis.items():
        print(f"  {k.replace('_',' ').title():<{pad+2}}  {v}")
    print("="*60 + "\n")

    df = pd.DataFrame([kpis])
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "performance_summary.csv", index=False)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# TURNOVER CALCULATION
# ──────────────────────────────────────────────────────────────────────────────

def calculate_turnover(trade_log: pd.DataFrame, equity: pd.Series) -> float:
    """
    Annualised portfolio turnover as a multiple of average NAV.

    turnover = total_notional_traded / average_NAV / years

    A turnover of 2.0 means the portfolio turns over its entire NAV twice
    per year; lower is better for cost reduction.

    Parameters
    ----------
    trade_log : DataFrame with a ``notional`` column (from engine.run()).
    equity    : Daily equity curve (pd.Series, index = dates).

    Returns
    -------
    float  Annualised turnover ratio.
    """
    if trade_log.empty or equity.empty:
        return np.nan

    years = max((equity.index[-1] - equity.index[0]).days / 365, 1e-9)
    avg_nav = equity.mean()
    if avg_nav <= 0:
        return np.nan

    # Each trade contributes its entry notional (one-way; entry + exit = 2×)
    total_notional = trade_log["notional"].abs().sum() if "notional" in trade_log.columns else 0.0
    return total_notional / avg_nav / years


# ──────────────────────────────────────────────────────────────────────────────
# SIDE-BY-SIDE COMPARISON REPORT
# ──────────────────────────────────────────────────────────────────────────────

def generate_comparison_report(
    results_dict: dict[str, dict],
    initial_capital: float = 5_000_000.0,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Generate a side-by-side performance comparison for two or more backtest runs.

    Parameters
    ----------
    results_dict    : Mapping of label → results dict (output of engine.run()).
                      E.g. {"V1 (Original)": results_v1, "V2 (Improved)": results_v2}
    initial_capital : Starting capital for return calculations.
    output_dir      : If given, save comparison CSV and print to file.

    Returns
    -------
    pd.DataFrame  Metrics × Strategies comparison table.
    """
    rows = {}

    for label, res in results_dict.items():
        equity   = res.get("equity_curve", pd.Series(dtype=float)).dropna()
        returns  = res.get("daily_returns", pd.Series(dtype=float)).fillna(0.0)
        tl       = res.get("trade_log", pd.DataFrame())
        exp_df   = res.get("exposure_df", pd.DataFrame())

        tr_stats = _trade_stats(tl)

        years = max((equity.index[-1] - equity.index[0]).days / 365, 1e-9) if len(equity) > 1 else 1.0
        total_cost = (
            tr_stats.get("total_entry_cost", 0)
            + tr_stats.get("total_exit_cost", 0)
            + tr_stats.get("total_roll_cost", 0)
        )

        # Average gross exposure as % of NAV (on days when at least one position is open)
        avg_gross_exp_pct = np.nan
        if not exp_df.empty and "gross_notional" in exp_df.columns and len(equity) > 1:
            eq_aligned = equity.reindex(exp_df.index).ffill().bfill()
            gross_pct = (exp_df["gross_notional"] / eq_aligned.replace(0, np.nan)).dropna()
            invested = gross_pct[gross_pct > 0]
            avg_gross_exp_pct = round(invested.mean() * 100, 1) if len(invested) > 0 else 0.0

        # Long / short trade split
        long_trades  = len(tl[tl["direction"] == "LONG"])  if not tl.empty and "direction" in tl.columns else 0
        short_trades = len(tl[tl["direction"] == "SHORT"]) if not tl.empty and "direction" in tl.columns else 0

        rows[label] = {
            "CAGR (%)":           round(_annualised_return(equity) * 100, 2),
            "Sharpe":             round(_sharpe(returns), 3),
            "Sortino":            round(_sortino(returns), 3),
            "Max DD (%)":         round(_max_drawdown(equity) * 100, 2),
            "Calmar":             round(_calmar(equity), 3),
            "Ann Volatility (%)": round(returns.std() * np.sqrt(252) * 100, 2),
            "Total Trades":       tr_stats.get("total_trades", 0),
            "  Long Trades":      long_trades,
            "  Short Trades":     short_trades,
            "Win Rate (%)":       round(tr_stats.get("win_rate", 0) * 100, 1),
            "Profit Factor":      round(tr_stats.get("profit_factor", 0), 2),
            "Avg Hold Days":      round(tr_stats.get("avg_hold_days", 0), 1),
            "Turnover (×/yr)":    round(calculate_turnover(tl, equity), 2),
            "Total Costs (₹L)":   round(total_cost / 1e5, 1),
            "Avg Gross Exp (%)":  avg_gross_exp_pct,
            "Final NAV (₹L)":     round(equity.iloc[-1] / 1e5, 1) if len(equity) > 0 else 0,
        }

    df = pd.DataFrame(rows).T

    # Pretty print
    print("\n" + "=" * 70)
    print("  BACKTEST COMPARISON REPORT")
    print("=" * 70)
    print(df.to_string())
    print("=" * 70 + "\n")

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "comparison_report.csv")
        print(f"  Comparison saved to: {out / 'comparison_report.csv'}\n")

    return df
