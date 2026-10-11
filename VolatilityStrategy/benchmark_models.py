"""benchmark_models.py — Historical vol, rolling mean, EWMA, and GARCH forecasters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import get_logger

logger = get_logger(__name__)

TRADING_DAYS = 252


# ─────────────────────────────────────────────────────────────────────────────
# Protocol
# ─────────────────────────────────────────────────────────────────────────────

class VolForecastModel(Protocol):
    """Any model that can forecast volatility from a returns series."""

    name: str

    def fit(self, returns: pd.Series) -> "VolForecastModel": ...
    def predict(self, returns: pd.Series) -> np.ndarray: ...


# ─────────────────────────────────────────────────────────────────────────────
# 1. Historical Volatility (naive: use current RV as next-period forecast)
# ─────────────────────────────────────────────────────────────────────────────

class HistoricalVolForecast:
    """Forecast next 5-day RV as current 5-day realized vol (random-walk baseline)."""

    name = "Historical Vol (RW)"

    def __init__(self, window: int = 5) -> None:
        self.window = window

    def fit(self, returns: pd.Series) -> "HistoricalVolForecast":
        return self

    def predict(self, returns: pd.Series) -> np.ndarray:
        rv = returns.rolling(self.window).std() * np.sqrt(TRADING_DAYS)
        return rv.values


# ─────────────────────────────────────────────────────────────────────────────
# 2. Rolling Mean of Historical RV
# ─────────────────────────────────────────────────────────────────────────────

class RollingMeanForecast:
    """Forecast next 5-day RV as rolling mean of past 20-day RV."""

    name = "Rolling Mean (20d)"

    def __init__(self, rv_window: int = 5, mean_window: int = 20) -> None:
        self.rv_window = rv_window
        self.mean_window = mean_window

    def fit(self, returns: pd.Series) -> "RollingMeanForecast":
        return self

    def predict(self, returns: pd.Series) -> np.ndarray:
        rv = returns.rolling(self.rv_window).std() * np.sqrt(TRADING_DAYS)
        forecast = rv.rolling(self.mean_window).mean()
        return forecast.values


# ─────────────────────────────────────────────────────────────────────────────
# 3. EWMA
# ─────────────────────────────────────────────────────────────────────────────

class EWMAForecast:
    """
    RiskMetrics EWMA volatility model.

    σ²_t = λ σ²_{t-1} + (1-λ) r²_{t-1}
    """

    name = "EWMA (λ=0.94)"

    def __init__(self, lam: float = 0.94) -> None:
        self.lam = lam

    def fit(self, returns: pd.Series) -> "EWMAForecast":
        return self

    def predict(self, returns: pd.Series) -> np.ndarray:
        r = returns.fillna(0).values
        n = len(r)
        var = np.zeros(n)
        var[0] = r[0] ** 2
        lam = self.lam
        for t in range(1, n):
            var[t] = lam * var[t - 1] + (1 - lam) * r[t - 1] ** 2
        sigma = np.sqrt(var) * np.sqrt(TRADING_DAYS)
        return sigma


# ─────────────────────────────────────────────────────────────────────────────
# 4. GARCH(1,1)
# ─────────────────────────────────────────────────────────────────────────────

class GARCHForecast:
    """
    GARCH(1,1) fitted via the `arch` library.

    Predicts the conditional 5-day-ahead volatility (annualized) using
    multi-step simulation from the GARCH variance equation.
    """

    name = "GARCH(1,1)"

    def __init__(self, horizon: int = 5) -> None:
        self.horizon = horizon
        self._fitted = None

    def fit(self, returns: pd.Series) -> "GARCHForecast":
        try:
            from arch import arch_model

            r = returns.dropna() * 100  # scale for numerical stability
            model = arch_model(r, vol="Garch", p=1, q=1, mean="Zero", dist="normal")
            self._fitted = model.fit(disp="off", show_warning=False)
            logger.info("GARCH(1,1) fitted: ω=%.4f α=%.4f β=%.4f",
                        self._fitted.params["omega"],
                        self._fitted.params["alpha[1]"],
                        self._fitted.params["beta[1]"])
        except Exception as exc:
            logger.warning("GARCH fitting failed: %s", exc)
            self._fitted = None
        return self

    def predict(self, returns: pd.Series) -> np.ndarray:
        if self._fitted is None:
            # Fallback to EWMA if GARCH failed
            return EWMAForecast().predict(returns)

        try:
            from arch import arch_model

            r = returns.dropna() * 100
            model = arch_model(r, vol="Garch", p=1, q=1, mean="Zero", dist="normal")

            n = len(returns)
            forecasts = np.full(n, np.nan)

            # Rolling one-step-ahead forecasts for each date
            params = self._fitted.params
            omega = params["omega"]
            alpha = params["alpha[1]"]
            beta = params["beta[1]"]

            r_vals = r.values
            var_t = r_vals[0] ** 2  # initial variance

            for t in range(1, n):
                # 1-step conditional var
                var_t = omega + alpha * r_vals[t - 1] ** 2 + beta * var_t
                # h-step ahead variance: sum of geometric series for GARCH
                # E[σ²_{t+h}] = ω*(1-φ^(h-1))/(1-φ) + φ^(h-1)*σ²_{t+1}
                # where φ = α + β
                phi = alpha + beta
                if abs(phi) < 1.0:
                    h = self.horizon
                    sum_var = var_t * (1 - phi ** h) / (1 - phi) + omega * sum(
                        phi ** i for i in range(h - 1)
                    )
                    h_step_var = sum_var / h
                else:
                    h_step_var = var_t
                # Annualize: was in (return*100)² space, convert back
                forecasts[t] = np.sqrt(h_step_var / 10000) * np.sqrt(TRADING_DAYS)

            return forecasts

        except Exception as exc:
            logger.warning("GARCH prediction failed: %s — falling back to EWMA", exc)
            return EWMAForecast().predict(returns)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ForecastMetrics:
    name: str
    rmse: float
    mae: float
    mape: float
    correlation: float
    r2: float

    def to_dict(self) -> dict:
        return {
            "Model": self.name,
            "RMSE": round(self.rmse, 6),
            "MAE": round(self.mae, 6),
            "MAPE (%)": round(self.mape, 2),
            "Correlation": round(self.correlation, 4),
            "R²": round(self.r2, 4),
        }


def compute_metrics(name: str, actual: np.ndarray, predicted: np.ndarray) -> ForecastMetrics:
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    a, p = actual[mask], predicted[mask]

    rmse = float(np.sqrt(mean_squared_error(a, p)))
    mae = float(mean_absolute_error(a, p))
    mape = float(np.mean(np.abs((a - p) / np.where(a == 0, 1e-8, a))) * 100)
    corr = float(np.corrcoef(a, p)[0, 1]) if len(a) > 1 else 0.0
    r2 = float(r2_score(a, p))
    return ForecastMetrics(name=name, rmse=rmse, mae=mae, mape=mape, correlation=corr, r2=r2)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmarks(
    returns: pd.Series,
    actual_rv5: pd.Series,
    train_mask: pd.Series,
    lstm_preds: np.ndarray | None = None,
    lstm_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """
    Fit benchmarks on training data, predict on full series, return metrics table.

    Parameters
    ----------
    returns      : Daily log/arithmetic returns for NIFTY.
    actual_rv5   : True future 5-day realized vol (target).
    train_mask   : Boolean Series — True where data is training.
    lstm_preds   : LSTM predictions (optional, aligned to lstm_dates).
    lstm_dates   : Dates corresponding to lstm_preds.

    Returns
    -------
    pd.DataFrame with one row per model.
    """
    train_returns = returns[train_mask]

    models: list[tuple[str, np.ndarray]] = []

    def _run(model, name):
        model.fit(train_returns)
        preds = model.predict(returns)
        models.append((name, preds))

    _run(HistoricalVolForecast(), "Historical Vol (RW)")
    _run(RollingMeanForecast(), "Rolling Mean (20d)")
    _run(EWMAForecast(), "EWMA (λ=0.94)")
    _run(GARCHForecast(), "GARCH(1,1)")

    # Test period only
    test_mask = ~train_mask
    actual_test = actual_rv5[test_mask].values

    rows = []
    for name, preds in models:
        pred_test = pd.Series(preds, index=returns.index)[test_mask].values
        rows.append(compute_metrics(name, actual_test, pred_test).to_dict())

    # LSTM metrics
    if lstm_preds is not None and lstm_dates is not None:
        lstm_series = pd.Series(lstm_preds, index=lstm_dates)
        actual_lstm = actual_rv5.reindex(lstm_dates).values
        rows.append(compute_metrics("LSTM", actual_lstm, lstm_preds).to_dict())

    table = pd.DataFrame(rows).set_index("Model").sort_values("RMSE")
    logger.info("\n%s", table.to_string())
    return table
