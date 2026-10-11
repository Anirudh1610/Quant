"""
metrics.py
----------
Performance statistics and charts for the options selling strategy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ──────────────────────────────────────────────────────────────────────────────

def cagr(equity: pd.Series) -> float:
    eq = equity.dropna()
    if len(eq) < 2 or eq.iloc[0] == 0:
        return 0.0
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    return (eq.iloc[-1] / eq.iloc[0]) ** (1 / max(years, 1e-6)) - 1


def sharpe(returns: pd.Series, ann: int = 252) -> float:
    r = returns.dropna()
    return float(r.mean() / r.std() * np.sqrt(ann)) if r.std() > 0 else 0.0


def sortino(returns: pd.Series, ann: int = 252) -> float:
    r = returns.dropna()
    neg = r[r < 0]
    downside_std = neg.std() if len(neg) > 1 else 1e-9
    return float(r.mean() / downside_std * np.sqrt(ann))


def max_drawdown(equity: pd.Series) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    eq   = equity.dropna()
    peak = eq.cummax()
    dd   = (eq - peak) / peak
    worst     = dd.idxmin()
    peak_date = eq.loc[:worst].idxmax()
    return float(dd.min()), peak_date, worst


def win_rate(tl: pd.DataFrame) -> float:
    return float((tl["net_pnl"] > 0).mean()) if not tl.empty else 0.0


def profit_factor(tl: pd.DataFrame) -> float:
    if tl.empty:
        return 0.0
    wins   = tl.loc[tl["net_pnl"] > 0, "net_pnl"].sum()
    losses = -tl.loc[tl["net_pnl"] < 0, "net_pnl"].sum()
    return float(wins / losses) if losses > 0 else float("inf")


def avg_premium(tl: pd.DataFrame) -> float:
    return float(tl["total_premium"].mean()) if not tl.empty else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# REPORT
# ──────────────────────────────────────────────────────────────────────────────

def generate_performance_report(
    equity: pd.Series,
    returns: pd.Series,
    trade_log: pd.DataFrame,
    initial_capital: float,
    output_dir: Path,
) -> pd.DataFrame:
    mdd, mdd_start, mdd_end = max_drawdown(equity)

    stop_pct = (
        f"{(trade_log['exit_reason'] == 'STOP_LOSS').mean() * 100:.1f}%"
        if not trade_log.empty else "N/A"
    )

    rows = [
        ("Initial Capital (₹)",      f"₹{initial_capital:,.0f}"),
        ("Final Equity (₹)",          f"₹{equity.dropna().iloc[-1]:,.0f}"),
        ("CAGR",                       f"{cagr(equity) * 100:.2f}%"),
        ("Sharpe Ratio",               f"{sharpe(returns):.3f}"),
        ("Sortino Ratio",              f"{sortino(returns):.3f}"),
        ("Max Drawdown",               f"{mdd * 100:.2f}%"),
        ("Max DD Period",              f"{mdd_start.date()} → {mdd_end.date()}"),
        ("Total Trades",               str(len(trade_log))),
        ("Win Rate",                   f"{win_rate(trade_log) * 100:.1f}%"),
        ("Profit Factor",              f"{profit_factor(trade_log):.2f}"),
        ("Avg Premium Collected (pts)",f"{avg_premium(trade_log):.1f}"),
        ("Total Costs (₹)",            f"₹{trade_log['costs'].sum():,.0f}" if not trade_log.empty else "₹0"),
        ("Stop-Loss Exits",            stop_pct),
    ]

    df = pd.DataFrame(rows, columns=["Metric", "Value"])

    print("\n" + "=" * 56)
    print("  OPTIONS STRATEGY  —  PERFORMANCE SUMMARY")
    print("=" * 56)
    print(df.to_string(index=False))
    print("=" * 56 + "\n")

    df.to_csv(output_dir / "performance_summary.csv", index=False)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# CHARTS
# ──────────────────────────────────────────────────────────────────────────────

def _save(fig, output_dir: Path, name: str) -> None:
    path = output_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  → {path}")


def plot_equity_curve(equity: pd.Series, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(equity.index, equity.values / 1e5, linewidth=1.5, color="steelblue")
    ax.set_title("Equity Curve  (₹ lakh)", fontsize=13)
    ax.set_ylabel("₹ lakh")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, "equity_curve.png")
    plt.close(fig)


def plot_drawdown(equity: pd.Series, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    peak = equity.cummax()
    dd   = (equity - peak) / peak * 100

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.55)
    ax.set_title("Drawdown  (%)", fontsize=13)
    ax.set_ylabel("Drawdown %")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, "drawdown.png")
    plt.close(fig)


def plot_monthly_heatmap(trade_log: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    if trade_log.empty:
        return

    tl = trade_log.copy()
    tl["year"]  = pd.to_datetime(tl["exit_date"]).dt.year
    tl["month"] = pd.to_datetime(tl["exit_date"]).dt.month
    pivot = tl.groupby(["year", "month"])["net_pnl"].sum().unstack(fill_value=0) / 1e3

    vmax = max(abs(pivot.values.max()), abs(pivot.values.min()), 1)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(14, max(3, len(pivot) * 0.75)))
    im = ax.imshow(pivot.values, cmap="RdYlGn", norm=norm, aspect="auto")

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([month_labels[m - 1] for m in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=7, color="black")

    ax.set_title("Monthly Net P&L  (₹k)", fontsize=13)
    fig.colorbar(im, ax=ax, label="₹k")
    fig.tight_layout()
    _save(fig, output_dir, "monthly_pnl_heatmap.png")
    plt.close(fig)


def plot_pnl_by_signal(trade_log: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if trade_log.empty:
        return

    grp = trade_log.groupby("trend_signal").agg(
        total_pnl=("net_pnl", "sum"),
        trades=("net_pnl", "count"),
        avg_pnl=("net_pnl", "mean"),
        win_rate=("net_pnl", lambda x: (x > 0).mean() * 100),
    )

    colors = {"NEUTRAL": "#4472C4", "BULL": "#70AD47", "BEAR": "#FF0000",
              "STRANGLE": "#ED7D31"}
    c = [colors.get(s, "#A5A5A5") for s in grp.index]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].bar(grp.index, grp["total_pnl"] / 1e3, color=c)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Total P&L by Signal (₹k)")
    axes[0].set_ylabel("₹k")

    axes[1].bar(grp.index, grp["avg_pnl"], color=c)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Avg P&L per Trade (₹)")

    axes[2].bar(grp.index, grp["win_rate"], color=c)
    axes[2].set_ylim(0, 100)
    axes[2].axhline(50, color="black", linewidth=0.8, linestyle="--")
    axes[2].set_title("Win Rate by Signal (%)")

    for ax in axes:
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Performance by Trend Signal", fontsize=13)
    fig.tight_layout()
    _save(fig, output_dir, "pnl_by_signal.png")
    plt.close(fig)


def plot_pnl_distribution(trade_log: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if trade_log.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(trade_log["net_pnl"], bins=40, color="steelblue", edgecolor="white")
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1.2)
    axes[0].set_title("P&L Distribution per Trade  (₹)")
    axes[0].set_xlabel("₹")

    axes[1].hist(trade_log["total_premium"], bins=30, color="seagreen", edgecolor="white")
    axes[1].set_title("Total Premium Collected per Trade  (index points)")
    axes[1].set_xlabel("Index points")

    fig.tight_layout()
    _save(fig, output_dir, "pnl_distribution.png")
    plt.close(fig)


def plot_vix_vs_pnl(trade_log: pd.DataFrame, output_dir: Path) -> None:
    """Scatter: VIX at entry vs net P&L — reveals how VIX level affects outcomes."""
    import matplotlib.pyplot as plt

    if trade_log.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = trade_log["net_pnl"].apply(lambda x: "seagreen" if x > 0 else "crimson")
    ax.scatter(trade_log["vix_entry"], trade_log["net_pnl"], c=colors, alpha=0.5, s=20)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("India VIX at entry")
    ax.set_ylabel("Net P&L (₹)")
    ax.set_title("VIX at Entry vs Trade P&L")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, "vix_vs_pnl.png")
    plt.close(fig)
