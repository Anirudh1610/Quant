# Nifty 50 Weekly Alpha Strategy — Documentation

**File:** `alpha.ipynb`  
**Last updated:** May 2026  
**Language:** Python 3.14 · Jupyter Notebook

---

## Table of Contents

1. [Strategy Overview](#strategy-overview)
2. [Universe & Data](#universe--data)
3. [Alpha Signal — Jensen's Alpha](#alpha-signal--jensens-alpha)
4. [Volatility & Regime Filter](#volatility--regime-filter)
5. [Execution Layer — Position Sizing & Risk Management](#execution-layer--position-sizing--risk-management)
6. [Backtest Engine](#backtest-engine)
7. [Bias Corrections](#bias-corrections)
8. [Performance Metrics](#performance-metrics)
9. [Configuration Reference](#configuration-reference)
10. [Notebook Cell Map](#notebook-cell-map)
11. [Known Limitations](#known-limitations)

---

## Strategy Overview

A systematic, weekly-rebalanced long-only equity strategy on the **Nifty 50 universe**.

Each Friday, the strategy:
1. Runs CAPM OLS regression on each stock over the trailing 52 weeks
2. Ranks stocks by Jensen's Alpha (intercept)
3. Filters out stocks with vol spikes and checks the India VIX regime
4. Enters the **top 5** stocks, sized using inverse-volatility weighting scaled to a 12% annualised vol target
5. Applies stop-losses, portfolio drawdown limits, and a VIX-gated exposure multiplier

Returns are compared against the Nifty 50 index (`^NSEI`) on both a gross and net-of-cost basis.

---

## Universe & Data

### Ticker Universe

Static snapshot of **Nifty 50 constituents circa January 2022** (50 tickers, `.NS` suffix for NSE).

```
ADANIENT, ADANIPORTS, APOLLOHOSP, ASIANPAINT, AXISBANK,
BAJAJ-AUTO, BAJAJFINSV, BAJFINANCE, BHARTIARTL, BPCL,
BRITANNIA, CIPLA, COALINDIA, DIVISLAB, DRREDDY,
EICHERMOT, GRASIM, HCLTECH, HDFCBANK, HDFC,
HDFCLIFE, HEROMOTOCO, HINDALCO, HINDUNILVR, ICICIBANK,
INDUSINDBK, INFY, ITC, JSWSTEEL, KOTAKBANK,
LT, M&M, MARUTI, NESTLEIND, NTPC,
ONGC, POWERGRID, RELIANCE, SBIN, SBILIFE,
SHREECEM, SUNPHARMA, TATACONSUM, TATAMOTORS, TATASTEEL,
TCS, TECHM, TITAN, ULTRACEMCO, WIPRO
```

**Why static?** Using live constituents introduces survivorship bias — underperformers dropped mid-period are silently excluded. A point-in-time snapshot from the start of the backtest avoids this. Note: HDFC.NS merged into HDFCBANK in mid-2023 and is naturally dropped by the ≥90% data completeness filter.

### Data Source

- **Library:** `yfinance` (auto-adjusted close prices)
- **Frequency:** Daily → resampled to **weekly Friday close**
- **Benchmark:** `^NSEI` (Nifty 50 index)
- **VIX:** `^INDIAVIX` (India VIX fear gauge)
- **Period:** `today − (3 years + 52 weeks + 30 days buffer)` → `today`

### Data Quality

- Stocks with > 10% missing daily data are dropped
- Forward-fill applied after resampling
- Weekly returns computed as `pct_change()` on Friday closes

---

## Alpha Signal — Jensen's Alpha

### Model

Each week, for every stock in the universe, estimate CAPM via OLS on the trailing 52-week window:

$$r_{i,t} - r_f = \alpha_i + \beta_i (r_{m,t} - r_f) + \varepsilon_{i,t}$$

Where:
- $r_{i,t}$ = weekly return of stock $i$ at time $t$
- $r_f$ = weekly risk-free rate (`RISK_FREE_ANNUAL / 52 = 6.5\% / 52`)
- $r_{m,t}$ = weekly Nifty 50 return
- $\alpha_i$ = **Jensen's Alpha** (intercept) — the signal used for ranking
- $\beta_i$ = market beta (estimated but not used for sizing)

### Stock Selection

- **Rank** all valid stocks by $\alpha_i$ (descending)
- **Select top `TOP_N = 5`** stocks
- Stocks failing the vol-spike filter are excluded before ranking

### Caveat

Weekly CAPM alpha over a 52-week rolling window is highly correlated with trailing 12-month momentum. The alpha signal should be interpreted as a blend of factor alpha and price momentum.

---

## Volatility & Regime Filter

Two filters are applied every week before the alpha ranking.

### 1. Market Regime — India VIX Gate

India VIX is downloaded weekly and mapped to an **exposure multiplier**:

| VIX Level | Exposure Scale | Interpretation |
|-----------|---------------|----------------|
| VIX < 15 | 1.0 (100%) | Full investment |
| 15 ≤ VIX < 20 | 0.5 (50%) | De-risk, half position |
| VIX ≥ 20 | 0.0 (0%) | Flat — all cash |

The uninvested fraction earns the risk-free rate. Thresholds were calibrated to India VIX's typical range (12–18 in calm markets, >20 during stress events).

**Key function:** `vix_scale(week_date)` → `{0.0, 0.5, 1.0}`

### 2. Stock-Level Vol Spike Filter

A stock is excluded from the candidate pool if its short-term volatility has spiked relative to its long-term average:

$$\sigma_{4\text{-week}} > 2.0 \times \sigma_{52\text{-week}}$$

This avoids entering stocks after a volatility event (earnings miss, news shock) when risk is temporarily elevated.

**Key function:** `is_vol_spike(ticker, i)` → `bool`

**Parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `VIX_LOW` | 15.0 | Full exposure threshold |
| `VIX_HIGH` | 20.0 | Flat / cash threshold |
| `STOCK_VOL_MULT` | 2.0 | Vol spike multiplier |
| `VOL_SPIKE_WINDOW` | 4 | Short-term vol lookback (weeks) |

---

## Execution Layer — Position Sizing & Risk Management

### Position Sizing — Inverse-Vol + Vol Targeting

Rather than equal-weighting, each stock is sized so its contribution to portfolio volatility is equal, then the whole basket is scaled to target a fixed portfolio volatility.

**Step 1 — Inverse-vol allocation:**

$$w_i^{\text{raw}} = \frac{1/\sigma_i}{\sum_j 1/\sigma_j}$$

**Step 2 — Concentration cap:** Cap each weight at `MAX_WEIGHT = 15%`, re-normalise.

**Step 3 — Portfolio vol scaling:**

$$\sigma_{\text{port}} = \sqrt{\sum_i (w_i \cdot \sigma_i)^2}$$

$$\text{exposure} = \min\left(\frac{\sigma_{\text{target,weekly}}}{\sigma_{\text{port}}}, 1.0\right)$$

$$w_i^{\text{final}} = w_i \cdot \text{exposure}$$

The sum of weights is ≤ 1 (no leverage). When the portfolio is more volatile than the target, the basket automatically de-levers and the remainder earns the risk-free rate.

**Key function:** `vol_target_weights(top_tickers, window)` → `dict {ticker: weight}`

### Risk Management Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TARGET_ANNUAL_VOL` | 12% | Target annualised portfolio volatility |
| `MAX_WEIGHT` | 15% | Max allocation to any single stock |
| `STOP_LOSS_PCT` | 6% | Per-stock stop-loss below entry price |
| `PORTFOLIO_DD_LIMIT` | 10% | Max tolerated portfolio drawdown |
| `COOL_OFF_WEEKS` | 4 | Weeks to stay flat after DD limit hit |

### Stop-Loss (Per Stock)

Entry prices are recorded when a stock enters the portfolio. Each week, if any stock has fallen more than `STOP_LOSS_PCT = 6%` below its entry price, it is excluded from the current week's effective holdings (forced exit at next open in live trading).

> **Note:** This backtest uses weekly closes only. A stop triggered intra-week is applied at Friday's close, which is conservative. Intraday stops require daily/intraday data.

### Portfolio Drawdown Cool-Off

The running cumulative return is tracked each week. If the portfolio drawdown from its peak exceeds `PORTFOLIO_DD_LIMIT = 10%`:
- The portfolio goes **flat for 4 weeks** (earns risk-free rate)
- All positions are cleared (`prev_holdings` reset)
- Re-entry resumes after `COOL_OFF_WEEKS = 4` weeks

### Futures Rollover Calendar

NSE stock futures expire on the **last Thursday of each month**. To avoid end-of-contract illiquidity, roll **2 calendar days before expiry**.

**Key function:** `next_rollover_date(ref_date)` → roll date

---

## Backtest Engine

### Loop Logic (Weekly, Friday to Friday)

For each week `i` from week 52 to the end of the data:

```
1. Check cool-off: if cool_off_remaining > 0 → earn risk-free, skip
2. Get VIX scale = vix_scale(week_date)
3. Filter candidates: remove stocks with is_vol_spike(ticker, i)
4. Run CAPM OLS on each remaining stock (52-week window)
5. Select top TOP_N=5 by alpha
6. Compute weights = vol_target_weights(top_tickers, window)
7. Calculate turnover vs prev_holdings → transaction cost
8. Record entry prices for new entrants; flag stop-loss breaches
9. Compute gross return = Σ weight[t] * next_week_return[t]
10. Apply VIX scale: gross = gross * scale + risk_free * (1 - scale)
11. Net return = gross - cost * scale
12. Check portfolio DD: if DD > 10% → set cool_off = 4, go flat
```

### Transaction Costs

```
cost = (stocks_entering + stocks_exiting) / TOP_N × 0.15%
```

Only deployed capital (VIX scale × portfolio) pays costs — cash portions are free.

Components included in 0.15%/leg:
- STT (Securities Transaction Tax)
- Brokerage
- Stamp duty
- Estimated slippage

### Output DataFrame

`results_df` columns:

| Column | Description |
|--------|-------------|
| `Portfolio` | Net-of-cost weekly return |
| `Portfolio_Gross` | Gross weekly return (no costs) |
| `Benchmark` | Nifty 50 weekly return |
| `Turnover` | Fraction of portfolio traded |
| `VIX_Scale` | Exposure multiplier (0 / 0.5 / 1.0) |

---

## Bias Corrections

Four explicit corrections vs a naïve equal-weight strategy:

### 1. Survivorship Bias
Static Jan-2022 universe rather than live constituents. Stocks subsequently dropped from the index (historically underperformers) remain in the backtest universe, preventing the bias of only backtesting on today's winners.

### 2. Transaction Costs
0.15% per leg applied on every stock added or removed from the portfolio each week. This is the primary drag; the net Sharpe is typically ~0.15–0.20 below the gross Sharpe.

### 3. Sharpe Confidence Intervals
A 10,000-sample IID bootstrap produces 95% confidence intervals and a one-sided p-value for the Sharpe ratio. With ~150 weekly observations, a high point estimate can arise by chance; statistical testing validates whether the edge is real.

### 4. Momentum Conflation Caveat
The alpha signal is explicitly noted to be correlated with 12-month trailing momentum. Any alpha decomposition should account for this.

---

## Performance Metrics

The notebook computes and displays the following:

### Summary Table (Backtest cell)
| Metric | Gross | Net | Benchmark |
|--------|-------|-----|-----------|
| Ann. Return | | | |
| Ann. Volatility | | | |
| Sharpe Ratio | | | |
| Max Drawdown | | | |
| Total Return | | | |

### Bootstrap Sharpe CI
- 10,000 resamples
- 95% confidence interval for Gross, Net, and Benchmark
- One-sided p-value (H₀: Sharpe ≤ 0)

### Accuracy Metrics
- **Win rate** — fraction of weeks with positive net return
- **Hit rate** — fraction of weeks portfolio beats benchmark
- **Profit factor** — gross gains / gross losses
- **Max consecutive wins / losses**
- **Avg win vs avg loss**

### Visualisations (4-panel chart)
1. Equity curve — Gross vs Net vs Benchmark (log scale)
2. Rolling 26-week Sharpe ratio
3. Portfolio drawdown
4. Current holdings bar chart (weights)

### Monthly Return Heatmap
Year × Month table of portfolio vs benchmark monthly returns.

---

## Configuration Reference

All tunable parameters in one place:

| Parameter | Value | Cell | Description |
|-----------|-------|------|-------------|
| `TOP_N` | 5 | Universe Config | Number of stocks to hold |
| `ROLLING_WEEKS` | 52 | Universe Config | OLS estimation window |
| `BACKTEST_YEARS` | 3 | Universe Config | Backtest length |
| `RISK_FREE_ANNUAL` | 6.5% | Universe Config | India 91-day T-bill proxy |
| `VIX_LOW` | 15.0 | Vol Filter | Full-exposure VIX threshold |
| `VIX_HIGH` | 20.0 | Vol Filter | Flat/cash VIX threshold |
| `STOCK_VOL_MULT` | 2.0 | Vol Filter | Vol spike multiplier |
| `VOL_SPIKE_WINDOW` | 4 | Vol Filter | Short-term vol lookback (weeks) |
| `TARGET_ANNUAL_VOL` | 12% | Execution Layer | Portfolio vol target |
| `MAX_WEIGHT` | 15% | Execution Layer | Max single-stock weight |
| `STOP_LOSS_PCT` | 6% | Execution Layer | Per-stock stop-loss |
| `PORTFOLIO_DD_LIMIT` | 10% | Execution Layer | Portfolio drawdown limit |
| `COOL_OFF_WEEKS` | 4 | Execution Layer | Post-DD cool-off period |
| `COST_ONE_WAY` | 0.15% | Backtest | Transaction cost per leg |

---

## Notebook Cell Map

| Cell | ID | Description |
|------|----|-------------|
| 1 | `#VSC-158c7fec` | Markdown — title, methodology, bias corrections |
| 2 | `#VSC-71f715b0` | Imports (`yfinance`, `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`) |
| 3 | `#VSC-41353a67` | Universe config — 50 tickers, strategy parameters |
| 4 | `#VSC-72031641` | Data download & resample to weekly Friday close |
| 5 | `#VSC-0b169598` | **Volatility & Regime Filter** — `vix_scale()`, `is_vol_spike()` |
| 6 | `#VSC-399be184` | **Execution Layer** — `vol_target_weights()`, `next_rollover_date()`, risk params |
| 7 | `#VSC-0c0715c5` | **Backtest engine** — weekly loop, all filters wired in, `results_df` |
| 8 | `#VSC-fdc04857` | Bootstrap Sharpe confidence intervals + distribution plot |
| 9 | `#VSC-8ca5f869` | Current holdings — live top-5 signal for next Monday |
| 10 | `#VSC-44b8114a` | Visualisations — 4-panel chart |
| 11 | `#VSC-720a13d4` | Accuracy metrics — win rate, hit rate, profit factor, streaks |
| 12 | `#VSC-447aa871` | Monthly return heatmap |

**Execution order:** Run cells 1 → 12 in sequence. Cells 5–12 depend on cell 4 having run first.

---

## Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Static Jan-2022 universe | Partial survivorship bias remains (mid-period changes not tracked) | Use NSE's official historical membership data for production |
| Weekly resolution stop-loss | Intra-week stop breaches reflected at Friday close only | Use daily/intraday data for tighter stops |
| Diagonal vol estimate in sizing | Ignores cross-correlations between stocks | Add correlation matrix for more accurate portfolio vol |
| IID bootstrap Sharpe CI | Assumes uncorrelated weekly returns; momentum violates this | Use block bootstrap for more accurate CI |
| No short positions | Long-only; cannot profit from negative alpha stocks | Add short leg via index futures hedge for market-neutral exposure |
| No slippage model for large orders | 0.15% cost assumes small account / high-liquidity stocks | Scale cost with order size for larger capital |
| Forward-fill on missing data | Stale prices may distort returns around data gaps | Flag and exclude stocks with gaps > N consecutive days |
