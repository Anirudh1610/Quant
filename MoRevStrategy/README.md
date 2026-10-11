# NIFTY 50 Cross-Sectional Momentum + Mean Reversion

A production-quality quantitative trading research system for the Indian equity market that combines **cross-sectional momentum** with **short-term mean reversion** timing.

---

## Strategy Logic

### Core Idea
Strong stocks tend to keep outperforming (momentum), but they occasionally dip short-term (mean reversion). This strategy exploits both effects:

- **Go LONG** on the top-10 momentum stocks *only after* a short-term pullback (RSI(2) < 10)
- **Go SHORT** on the bottom-10 momentum stocks *only after* a short-term rally (RSI(2) > 90)
- **Regime filter**: only take longs when NIFTY 50 is above its 50-day MA; only take shorts when it is below

### Signal Pipeline

```
Daily close prices
       │
       ▼
┌─────────────────────┐
│  20-day Momentum    │  → rank all 50 stocks
│  cross-sectionally  │
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│   RSI(2) filter     │  Long: RSI < 10  │  Short: RSI > 90
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  NIFTY Regime       │  50-day MA filter
│  + Vol filter       │  2× median rolling vol
└─────────────────────┘
       │
       ▼
   Entry signals
```

### Exit Rules
| Condition | Action |
|-----------|--------|
| 3 trading days elapsed | Time exit |
| RSI(2) crosses above 50 (for longs) | RSI exit |
| RSI(2) crosses below 50 (for shorts) | RSI exit |
| Intraday low/high breaches 3% stop | Stop-loss exit |

---

## Project Structure

```
MoRevStrategy/
│
├── data/                   ← Parquet price cache (auto-created)
├── outputs/                ← Generated charts and CSV reports
│
├── strategy/
│   ├── __init__.py
│   ├── signals.py          ← Momentum, RSI, regime, vol filter
│   ├── portfolio.py        ← Position construction & sizing
│   ├── backtest.py         ← Event-driven backtest engine
│   └── metrics.py          ← Performance analytics + plots
│
├── notebooks/
│   └── research.ipynb      ← Interactive research notebook
│
└── main.py                 ← CLI entry point
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install pandas numpy matplotlib scipy pandas-ta statsmodels yfinance python-dateutil pyarrow
```

### 2. Run backtest

```bash
cd MoRevStrategy
python main.py
```

### 3. Custom date range

```bash
python main.py --start 2020-01-01 --end 2024-12-31
```

### 4. Enable sector neutrality + walk-forward test

```bash
python main.py --sector-neutral --walk-forward
```

### 5. Open the research notebook

```bash
jupyter notebook notebooks/research.ipynb
```

---

## Outputs

| File | Description |
|------|-------------|
| `outputs/equity_curve.png` | Portfolio NAV vs NIFTY 50 benchmark |
| `outputs/drawdown.png` | Underwater equity curve |
| `outputs/monthly_heatmap.png` | Monthly returns (green/red colour map) |
| `outputs/stock_contribution.png` | Per-ticker cumulative PnL bar chart |
| `outputs/trade_log.csv` | Full trade-by-trade record |
| `outputs/performance_summary.csv` | CAGR, Sharpe, MDD, win rate, etc. |
| `outputs/walk_forward_results.csv` | Walk-forward in-sample vs OOS Sharpe |
| `outputs/sensitivity_results.csv` | Parameter sweep results |

---

## Transaction Costs (Indian Market)

| Cost | Value |
|------|-------|
| Brokerage | 0.03% per side |
| STT (sell) | 0.10% on sell side |
| SEBI / exchange charges | 0.005% per side |
| Slippage | 0.05% per side |
| **Total buy cost** | **~0.086%** |
| **Total sell cost** | **~0.156%** |

---

## Risk Management

- Equal-weight positions (each slot = 5% of NAV per side)
- Max 10 long + 10 short simultaneously (50/50 capital split)
- 3% intraday stop-loss (uses high/low as proxy)
- NIFTY regime filter prevents longs in bear markets
- Volatility filter excludes stocks > 2× universe median vol

---

## Extra Features

| Feature | How to use |
|---------|-----------|
| **Sector neutrality** | `--sector-neutral` flag or `use_sector_neutrality=True` in `main()` |
| **Walk-forward optimisation** | `--walk-forward` flag |
| **Parameter sweep** | Section 7 of the research notebook |
| **Volatility filter** | Always on (2× median vol threshold) |
| **NIFTY regime filter** | Always on (50-day MA) |

---

## Important Notes

- **No lookahead bias**: signals from day T → execution at open of day T+1
- **Survivorship bias**: stocks listed mid-period simply don't appear in early ranks (minor residual bias from using current NIFTY 50 list)
- **Data source**: Yahoo Finance via `yfinance` (adjusted for splits and dividends)
- **Short positions**: modelled as margin-funded futures-style shorts (full notional posted as margin)
- **Risk-free rate**: 6.5% p.a. (RBI repo rate proxy) used in Sharpe/Sortino calculations
