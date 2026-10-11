"""evaluate.py — Metrics, visualisations, reports, and explainability."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

from benchmark_models import ForecastMetrics, compute_metrics
from backtest import compute_performance
from utils import get_logger, ensure_dirs, project_root, load_config

logger = get_logger(__name__)

STYLE = "seaborn-v0_8-darkgrid"
FIGSIZE_WIDE = (14, 5)
FIGSIZE_SQUARE = (10, 8)
DPI = 150


def _savefig(fig: plt.Figure, path: Path) -> None:
    ensure_dirs(path.parent)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot → %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Forecast plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_actual_vs_predicted(
    actual: pd.Series,
    predicted: pd.Series,
    implied: pd.Series | None = None,
    title: str = "Actual vs Predicted Volatility",
    plots_dir: str | Path = "reports/plots",
) -> None:
    plots_dir = Path(plots_dir)
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        ax.plot(actual.index, actual.values * 100, label="Actual RV5", lw=1.5, color="#2196F3")
        ax.plot(predicted.index, predicted.values * 100, label="LSTM Predicted", lw=1.2,
                color="#FF5722", alpha=0.85, linestyle="--")
        if implied is not None:
            ax.plot(implied.index, implied.values * 100, label="Implied (VIX-derived)",
                    lw=1.0, color="#4CAF50", alpha=0.7, linestyle=":")
        ax.set_title(title)
        ax.set_ylabel("Volatility (%)")
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()
    _savefig(fig, plots_dir / "actual_vs_predicted.png")


def plot_residuals(
    actual: pd.Series,
    predicted: pd.Series,
    plots_dir: str | Path = "reports/plots",
) -> None:
    residuals = actual - predicted
    plots_dir = Path(plots_dir)
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(residuals.index, residuals.values * 100, lw=0.8, color="#9C27B0")
        axes[0].axhline(0, color="red", lw=1, linestyle="--")
        axes[0].set_title("Residuals over Time")
        axes[0].set_ylabel("Residual (%)")
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()

        axes[1].hist(residuals.dropna() * 100, bins=50, color="#9C27B0", edgecolor="white", alpha=0.8)
        axes[1].set_title("Residual Distribution")
        axes[1].set_xlabel("Residual (%)")
        axes[1].set_ylabel("Count")

        fig.suptitle("Residual Analysis")
    _savefig(fig, plots_dir / "residual_analysis.png")


def plot_prediction_error_distribution(
    actual: pd.Series,
    predicted: pd.Series,
    plots_dir: str | Path = "reports/plots",
) -> None:
    errors = (actual - predicted).dropna()
    plots_dir = Path(plots_dir)
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(errors * 100, bins=60, color="#3F51B5", edgecolor="white", alpha=0.85)
        ax.axvline(0, color="red", lw=1.5, linestyle="--")
        ax.axvline(errors.mean() * 100, color="orange", lw=1.5,
                   linestyle="--", label=f"Mean = {errors.mean()*100:.3f}%")
        ax.set_title("Prediction Error Distribution")
        ax.set_xlabel("Error (Actual − Predicted) %")
        ax.set_ylabel("Frequency")
        ax.legend()
    _savefig(fig, plots_dir / "prediction_error_distribution.png")


def plot_training_loss(
    history: dict[str, list[float]],
    plots_dir: str | Path = "reports/plots",
) -> None:
    plots_dir = Path(plots_dir)
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history["train_loss"], label="Train Loss", color="#2196F3")
        ax.plot(history["val_loss"], label="Val Loss", color="#FF5722")
        ax.set_title("LSTM Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
    _savefig(fig, plots_dir / "training_loss.png")


# ─────────────────────────────────────────────────────────────────────────────
# Feature analysis plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_correlation(
    df: pd.DataFrame,
    feature_cols: list[str],
    plots_dir: str | Path = "reports/plots",
) -> None:
    plots_dir = Path(plots_dir)
    corr = df[feature_cols].corr()
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, annot=False, fmt=".2f",
            cmap="RdYlGn", center=0, ax=ax,
            linewidths=0.3,
        )
        ax.set_title("Feature Correlation Heatmap")
        ax.tick_params(axis="x", rotation=45)
    _savefig(fig, plots_dir / "feature_correlation_heatmap.png")


def plot_signal_distribution(
    signals_df: pd.DataFrame,
    plots_dir: str | Path = "reports/plots",
) -> None:
    plots_dir = Path(plots_dir)
    counts = signals_df["signal_name"].value_counts()
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = {"LONG VOL": "#4CAF50", "NEUTRAL": "#9E9E9E", "SHORT VOL": "#F44336"}
        bars = ax.bar(counts.index, counts.values,
                      color=[colors.get(k, "#9E9E9E") for k in counts.index])
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    str(val), ha="center", va="bottom", fontsize=11)
        ax.set_title("Signal Distribution")
        ax.set_ylabel("Count")
    _savefig(fig, plots_dir / "signal_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# Backtest plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_equity_curve(
    equity: pd.Series,
    plots_dir: str | Path = "reports/plots",
) -> None:
    plots_dir = Path(plots_dir)
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        ax.plot(equity.index, equity.values / equity.iloc[0] * 100,
                label="Strategy NAV", color="#2196F3", lw=1.5)
        ax.axhline(100, color="grey", lw=0.8, linestyle="--")
        ax.set_title("Equity Curve (Normalized)")
        ax.set_ylabel("NAV (100 = Start)")
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()
    _savefig(fig, plots_dir / "equity_curve.png")


def plot_drawdown(
    equity: pd.Series,
    plots_dir: str | Path = "reports/plots",
) -> None:
    plots_dir = Path(plots_dir)
    peak = equity.cummax()
    dd = (equity - peak) / peak * 100
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        ax.fill_between(dd.index, dd.values, 0, alpha=0.6, color="#F44336")
        ax.plot(dd.index, dd.values, color="#F44336", lw=0.8)
        ax.set_title(f"Drawdown Curve  (Max DD: {dd.min():.2f}%)")
        ax.set_ylabel("Drawdown (%)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()
    _savefig(fig, plots_dir / "drawdown_curve.png")


def plot_rolling_sharpe(
    daily_returns: pd.Series,
    window: int = 63,
    plots_dir: str | Path = "reports/plots",
) -> None:
    plots_dir = Path(plots_dir)
    roll = daily_returns.rolling(window)
    sharpe = roll.mean() / roll.std() * np.sqrt(252)
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        ax.plot(sharpe.index, sharpe.values, color="#9C27B0", lw=1.2)
        ax.axhline(0, color="grey", lw=0.8, linestyle="--")
        ax.axhline(1, color="#4CAF50", lw=0.8, linestyle=":", label="Sharpe = 1")
        ax.set_title(f"Rolling {window}-day Sharpe Ratio")
        ax.set_ylabel("Sharpe")
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()
    _savefig(fig, plots_dir / "rolling_sharpe.png")


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance
# ─────────────────────────────────────────────────────────────────────────────

def permutation_importance(
    model_predict_fn,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 10,
    plots_dir: str | Path = "reports/plots",
) -> pd.DataFrame:
    """
    Permutation feature importance: for each feature, measure how much
    MSE increases when that feature is randomly shuffled.
    """
    from sklearn.metrics import mean_squared_error

    plots_dir = Path(plots_dir)
    baseline_preds = model_predict_fn(X)
    mask = ~np.isnan(y) & ~np.isnan(baseline_preds)
    baseline_mse = mean_squared_error(y[mask], baseline_preds[mask])

    importances = np.zeros((len(feature_names), n_repeats))
    rng = np.random.default_rng(42)

    for j, feat_name in enumerate(feature_names):
        for r in range(n_repeats):
            X_perm = X.copy()
            shuffle_idx = rng.permutation(len(X_perm))
            X_perm[:, :, j] = X_perm[shuffle_idx, :, j]
            preds = model_predict_fn(X_perm)
            mse = mean_squared_error(y[mask], preds[mask])
            importances[j, r] = mse - baseline_mse

    mean_imp = importances.mean(axis=1)
    std_imp = importances.std(axis=1)

    result = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": mean_imp,
        "importance_std": std_imp,
    }).sort_values("importance_mean", ascending=False)

    # Plot
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(9, max(6, len(feature_names) * 0.35)))
        n_show = min(20, len(result))
        top = result.head(n_show)
        ax.barh(top["feature"][::-1], top["importance_mean"][::-1],
                xerr=top["importance_std"][::-1], color="#FF9800", alpha=0.85)
        ax.set_title("Permutation Feature Importance (LSTM)")
        ax.set_xlabel("Increase in MSE when shuffled")
    _savefig(fig, plots_dir / "permutation_importance.png")

    result.to_csv(Path(plots_dir).parent / "permutation_importance.csv", index=False)
    return result


def shap_analysis(
    model,
    X: np.ndarray,
    feature_names: list[str],
    plots_dir: str | Path = "reports/plots",
    max_samples: int = 200,
) -> None:
    """SHAP DeepExplainer analysis for the LSTM model."""
    try:
        import shap
        import torch

        plots_dir = Path(plots_dir)
        ensure_dirs(plots_dir)

        device = next(model.parameters()).device
        X_sample = X[:max_samples]
        X_tensor = torch.tensor(X_sample, dtype=torch.float32).to(device)

        background = X_tensor[:50]
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(X_tensor[:100])

        # shap_values: (samples, seq_len, features) — average over time
        if isinstance(shap_values, list):
            sv = np.array(shap_values[0])
        else:
            sv = np.array(shap_values)

        if sv.ndim == 3:
            sv_mean = sv.mean(axis=1)  # (samples, features)
        else:
            sv_mean = sv

        with plt.style.context(STYLE):
            fig, ax = plt.subplots(figsize=(9, max(6, len(feature_names) * 0.35)))
            mean_abs = np.abs(sv_mean).mean(axis=0)
            order = np.argsort(mean_abs)
            n_show = min(20, len(feature_names))
            top_idx = order[-n_show:]
            ax.barh(
                [feature_names[i] for i in top_idx],
                mean_abs[top_idx],
                color="#E91E63", alpha=0.85,
            )
            ax.set_title("SHAP Feature Importance (LSTM)")
            ax.set_xlabel("|SHAP value|")
        _savefig(fig, plots_dir / "shap_importance.png")
        logger.info("SHAP analysis complete")

    except Exception as exc:
        logger.warning("SHAP analysis failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Full performance report
# ─────────────────────────────────────────────────────────────────────────────

def generate_performance_report(
    equity: pd.Series,
    daily_returns: pd.Series,
    trade_log: pd.DataFrame,
    initial_capital: float,
    benchmark_table: pd.DataFrame | None = None,
    output_dir: str | Path = "reports",
) -> None:
    output_dir = Path(output_dir)
    ensure_dirs(output_dir)

    perf = compute_performance(equity, daily_returns, trade_log, initial_capital)

    lines = [
        "=" * 60,
        "       NIFTY VOLATILITY FORECASTING — PERFORMANCE REPORT",
        "=" * 60,
        f"  Period            : {equity.index.min().date()} → {equity.index.max().date()}",
        f"  Initial Capital   : ₹{initial_capital:,.0f}",
        f"  Final Capital     : ₹{equity.iloc[-1]:,.0f}",
        "-" * 60,
        f"  Total Return      : {perf.get('total_return', 0)*100:.2f}%",
        f"  CAGR              : {perf.get('cagr', 0)*100:.2f}%",
        f"  Ann. Volatility   : {perf.get('ann_volatility', 0)*100:.2f}%",
        f"  Sharpe Ratio      : {perf.get('sharpe_ratio', 0):.3f}",
        f"  Sortino Ratio     : {perf.get('sortino_ratio', 0):.3f}",
        f"  Calmar Ratio      : {perf.get('calmar_ratio', 0):.3f}",
        f"  Max Drawdown      : {perf.get('max_drawdown', 0)*100:.2f}%",
        "-" * 60,
        f"  # Trades          : {perf.get('n_trades', 0)}",
        f"  Win Rate          : {perf.get('win_rate', 0)*100:.1f}%",
        f"  Profit Factor     : {perf.get('profit_factor', 0):.3f}",
        "=" * 60,
    ]
    report_str = "\n".join(lines)
    print(report_str)

    report_path = output_dir / "performance_report.txt"
    report_path.write_text(report_str)

    if not trade_log.empty:
        trade_log.to_csv(output_dir / "trade_log.csv", index=False)

    if benchmark_table is not None:
        benchmark_table.to_csv(output_dir / "benchmark_comparison.csv")
        logger.info("Benchmark comparison saved")

    logger.info("Performance report saved → %s", report_path)
