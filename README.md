# Quant

Systematic trading research on Indian markets (NSE equities, futures and
index options). Seven self-contained projects, each in its own folder,
written in Python. Everything here is research and backtesting — **nothing
places live orders**.

The honest summary: the infrastructure works, and no strategy in this repo
has yet shown a risk-adjusted edge after costs. The results table below
gives the numbers.

## Projects

| Folder | Idea | Data | Form | State |
|---|---|---|---|---|
| [Alpha/](Alpha/) | Weekly long-only: top 10 Nifty 200 stocks by rolling 52-week Jensen's alpha | yfinance | Notebook | Runs; no saved results |
| [Momentum/](Momentum/) | Daily long/short: 1 long + 1 short Nifty 50 stock future on a composite momentum score, ₹3L | yfinance | Notebook | Runs; no saved results |
| [MoRevStrategy/](MoRevStrategy/) | Nifty 50 cross-sectional momentum, entered on RSI(2) pullbacks | yfinance | Package + CLI | Backtested |
| [FuturesStrategy/](FuturesStrategy/) | NSE stock futures: 120-day momentum, ADX filter, EMA pullback entry, ATR sizing | yfinance (synthetic futures) | Package + CLI | Backtested |
| [OptionsStrategy/](OptionsStrategy/) | Short straddle/strangle on Nifty and Bank Nifty, chosen by a daily trend signal | yfinance (Black-Scholes prices) | Package + CLI | Legacy; superseded by NiftyOptionsSystem |
| [VolatilityStrategy/](VolatilityStrategy/) | LSTM forecast of Nifty realized vol, traded against India VIX | Kite Connect | Scripts + CLI | Model trained; no saved backtest report |
| [NiftyOptionsSystem/](NiftyOptionsSystem/) | Regime-driven credit spreads (bull put, bear call, iron condor) on Nifty / Bank Nifty | Kite Connect | Package, CLI scripts, 210 tests | Phases 1–4 of 5 built — the active project |

## Results so far

Only runs with saved or reproducible output are listed.

| Project | Period | Trades | Win rate | Return | Sharpe | Max drawdown |
|---|---|---|---|---|---|---|
| MoRevStrategy | 2019 – 2024 | 910 | 58.8% | +8.2% total, 1.4% CAGR | −0.47 | −20.4% |
| FuturesStrategy | 2018 – 2024 | 29 | 58.6% | +0.9% total, 0.1% CAGR | −6.83 | −1.4% |
| NiftyOptionsSystem | Jan 2023 – Aug 2026 | 98 | 81.6% | +2.0% from trades (₹40,790 on ₹20L) | −2.43 | −0.6% |

How to read these:

- **Every Sharpe is negative** because each strategy earned less than the
  6.5–6.8% Indian risk-free rate it is measured against. Win rates above
  55% did not translate into returns: average losses are larger than
  average wins, and costs take most of the rest (MoRevStrategy paid ₹3.5L
  in costs on ₹10L capital to net ₹82k).
- **MoRevStrategy** took 910 long trades and zero shorts, so the result is
  a long-only one despite the long/short design.
- **FuturesStrategy** made only 29 trades in seven years; the filters are
  too tight for the result to mean much either way.
- **NiftyOptionsSystem** reports a 21.6% total return, but that includes
  risk-free interest the backtester accrues on idle cash. Trade P&L alone
  is 2.0%; cash for the whole period would have returned about 27%. Its
  backtest also runs on a synthetic option chain, since no historical
  option-chain data is available. Details in its
  [README](NiftyOptionsSystem/README.md).

## What each project does

### Alpha — weekly Jensen's alpha
[Alpha/alpha.ipynb](Alpha/alpha.ipynb). Each Friday, regress every stock's
weekly excess return on Nifty 50's over the trailing 52 weeks and hold the
10 highest intercepts. Exposure is scaled by India VIX (100% below 15, 50%
below 20, cash above), stocks whose 4-week vol is more than twice their
52-week vol are skipped, and weights are inverse-volatility with a 12%
portfolio vol target. Includes a 6% per-stock stop, a 10% drawdown
cool-off, 0.15% per-leg costs and a bootstrap confidence interval on the
Sharpe ratio. Uses a fixed January 2022 universe to limit survivorship
bias.

[Alpha/README.md](Alpha/README.md) documents an earlier Nifty 50 / top-5
version; the notebook has since moved to Nifty 200 / top-10.

### Momentum — daily two-position futures
[Momentum/momentum.ipynb](Momentum/momentum.ipynb). Ranks Nifty 50 stocks
daily on a z-scored blend of momentum indicators, buys the top-ranked
stock's future and sells the bottom-ranked one's, ₹1.5L per leg. Positions
are only refreshed on predicted "high-conviction days" (when the score
spread between best and worst is in its top 30% historically); otherwise
they are held, which saves costs. India VIX regime filter and bootstrap
Sharpe intervals as in Alpha.

### MoRevStrategy — momentum with mean-reversion timing
Long the top 10 momentum stocks after an RSI(2) dip below 10; short the
bottom 10 after a spike above 90. Longs only when Nifty is above its
50-day average, shorts only below. Exits after 3 days, on RSI crossing 50,
or on a 3% stop. Signals at close of day T execute at the open of T+1.
Supports sector neutrality, walk-forward testing and a parameter sweep.
Full write-up in [MoRevStrategy/README.md](MoRevStrategy/README.md);
charts and CSVs in [MoRevStrategy/outputs/](MoRevStrategy/outputs/).

```bash
cd MoRevStrategy && python main.py --start 2020-01-01 --end 2024-12-31
```

### FuturesStrategy — NSE stock futures
Event-driven futures backtester with daily mark-to-market, monthly
rollover, a full Zerodha cost model (brokerage, STT, exchange, SEBI, stamp
duty, GST on traded notional) and ATR-based sizing at 1% of NAV risk per
trade. Futures prices are synthetic, built from spot with a cost-of-carry
model, because free historical NSE futures data does not exist.

```bash
cd FuturesStrategy && python main.py --capital 5000000 --compare
```

### OptionsStrategy — short straddles and strangles (legacy)
Classifies each day as neutral, bullish or bearish from the opening gap,
previous candle, RSI, EMA and ADX, then sells an at-the-money straddle or
a skewed strangle the day before weekly expiry. Option prices come from
Black-Scholes with India VIX as the volatility input, not from real
quotes. Kept for reference; NiftyOptionsSystem replaces it.

### VolatilityStrategy — LSTM volatility forecast
Pulls Nifty, Nifty futures and India VIX history from Kite, builds
realized-vol, moving-average and ATR features over a 60-day lookback, and
trains a two-layer PyTorch LSTM (64 → 32 units) to forecast realized
volatility. Goes long or short vol when the forecast differs from implied
by more than a 2% threshold. Benchmarked against historical, rolling-mean,
EWMA and GARCH forecasts, with walk-forward retraining, permutation
importance and SHAP. A trained model is saved in
[VolatilityStrategy/models/](VolatilityStrategy/models/).

```bash
cd VolatilityStrategy && python main.py --no-refresh
```

### NiftyOptionsSystem — regime-driven credit spreads
The most developed project, built in phases:

1. **Data** — Kite login and daily token refresh, rate-limited client,
   spot and VIX history, live option-chain snapshots cached in SQLite.
   Kite does not supply implied volatility, so it is computed per strike
   by inverting Black-76 against a forward implied from put-call parity.
2. **Regime** — realized vol, IV rank and percentile, Supertrend and ADX,
   max pain, put-call ratio and OI concentration, combined into one
   recommendation: bull put spread, bear call spread, iron condor, or no
   trade.
3. **Strike selection** — picks the short strike nearest a 0.175 delta
   after rejecting strikes with high probability of touch or sitting on an
   open-interest wall, adds the long leg four strikes further out, and
   returns credit, max loss, breakevens and live SPAN margin.
4. **Backtest** — day-by-day simulation with slippage, profit-target /
   stop-loss / expiry exits, and a parameter sweep with an in-sample /
   out-of-sample split. The sweep showed most in-sample "improvements" do
   not survive out of sample.
5. **Execution** — not built. Planned as paper trading by default with a
   manual confirmation before any live order.

Three real bugs were caught only by running against live data, after unit
tests had passed. They are written up in its
[README](NiftyOptionsSystem/README.md), along with the data schema, every
flagged assumption and the full backtest and sweep results.

```bash
cd NiftyOptionsSystem
python scripts/generate_access_token.py          # each trading morning
python scripts/pull_data.py --start 2023-01-01
python scripts/show_regime.py --instrument NIFTY
python scripts/select_spread.py --instrument NIFTY --margin
python scripts/run_backtest.py --instrument NIFTY --start 2023-01-01
pytest tests/
```

## Setup

Python 3.14, one virtual environment at the repo root.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r NiftyOptionsSystem/requirements.txt
```

The other projects have no requirements file. Between them they need
`pandas numpy scipy matplotlib seaborn yfinance python-dateutil
python-dotenv pyyaml kiteconnect torch scikit-learn joblib arch shap`,
plus `jupyter` for the notebooks.

The Kite-based projects (VolatilityStrategy, NiftyOptionsSystem) read
credentials from a `.env` at the repo root:

```
KITE_API_KEY=...
KITE_API_SECRET=...
KITE_ACCESS_TOKEN=...   # expires daily; regenerate with generate_access_token.py
```

`.env`, data caches, SQLite files and `results/` folders are git-ignored.
Historical spot data from Kite needs Zerodha's paid Historical Data
add-on. The yfinance projects need no credentials.

## Caveats that apply across the repo

- **Derivative prices are mostly modelled, not observed.** Futures are
  built from spot by cost of carry; historical option prices come from
  Black-Scholes with India VIX as the vol input. Only NiftyOptionsSystem's
  live pipeline uses real option quotes.
- **Survivorship bias is reduced, not removed.** The equity strategies use
  a fixed index membership list rather than point-in-time constituents.
- **Lot sizes and expiry days change.** NSE has revised both several
  times. NiftyOptionsSystem reads them from the live instrument list; the
  older projects hardcode 2024 values.
- **Backtests are single-path.** Where out-of-sample checks exist
  (MoRevStrategy walk-forward, NiftyOptionsSystem sweep) they use one
  split on one history.

## What's next

- NiftyOptionsSystem Phase 5: paper-trading execution scaffold with a
  manual confirmation gate.
- Rerun the NiftyOptionsSystem parameter sweep on the current engine and
  rank by trade P&L, since total return now includes interest on idle
  cash.
- Start collecting live option-chain snapshots daily, so future backtests
  can use real quotes and open interest instead of a synthetic chain.
