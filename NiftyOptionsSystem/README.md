# NiftyOptionsSystem

Systematic options toolkit for NIFTY / BANKNIFTY index options on Zerodha
(Kite Connect). It sells defined-risk credit spreads — bull put, bear call,
iron condor — chosen by a volatility/trend/open-interest regime signal.

| Phase | What it does | Status |
|---|---|---|
| 1. Data | Kite session, spot/VIX history, live option chain with computed IV, SQLite cache | Built, verified live |
| 2. Regime | Realized vol, IV rank, Supertrend/ADX, max pain/PCR → one recommended strategy type | Built, verified live |
| 3. Strike selection | Delta-targeted short strike, OI-wall and probability-of-touch filters, full spread + SPAN margin | Built, verified live |
| 4. Backtest | Day-by-day simulation on a synthetic chain, metrics, IS/OOS parameter sweep | Built; **no validated edge found** |
| 5. Execution | Paper-trading default, manual confirmation before any live order | Not built |

**Where it stands** (NIFTY, 2023-01-01 → 2026-08-11, default config): 98
trades, 81.6% win rate, ₹40,790 trade P&L on ₹20L — about 2% over 3.6
years. Reported total return is 21.6%, but almost all of that is the 6.8%
risk-free rate the engine now accrues on idle cash; holding cash the whole
time would have returned about 27%, which is why Sharpe is negative. See
[Phase 4](#phase-4--backtesting-engine) for the detail and the caveats on
the backtest data. Nothing here places orders.

The legacy `../OptionsStrategy/` folder (a synthetic short-straddle/strangle
backtester on yfinance data) is untouched and unrelated to this project.
The other strategies in this repo are summarised in `../README.md`.

## Setup

```bash
cd NiftyOptionsSystem
pip install -r requirements.txt
```

Credentials live in the **repo-root** `.env` (`../env` relative to this
folder) — `KITE_API_KEY`, `KITE_API_SECRET`, `KITE_ACCESS_TOKEN`. See
`.env.example` for the expected keys.

Kite access tokens expire daily. Each trading morning, run:

```bash
python scripts/generate_access_token.py
```

This opens the Kite login page, prompts you to paste back the redirected
URL (or just the `request_token`), exchanges it for an access token, and
rewrites `KITE_ACCESS_TOKEN` in `.env` in place.

## Phase 1 — Data layer

```bash
python scripts/pull_data.py --start 2023-01-01 --end 2024-12-31
python scripts/pull_data.py --instrument NIFTY --chain          # + live option chain, nearest expiry
python scripts/pull_data.py --instrument BANKNIFTY --chain --expiry 2025-01-02
```

Modules:

| Module | Responsibility |
|---|---|
| `core/session.py` | Kite login flow, access-token persistence |
| `core/kite_client.py` | Rate-limited, retrying wrapper around `KiteConnect` |
| `core/pricing.py` | Black-Scholes price/Greeks/IV inversion |
| `data/instruments.py` | Resolves instrument tokens, option contracts, expiries dynamically from the NFO/NSE dumps (cached daily as parquet) |
| `data/spot.py` | Live + historical spot OHLC (SQLite-cached) |
| `data/vix.py` | Live + historical India VIX (SQLite-cached) |
| `data/option_chain.py` | Live option chain snapshot with computed IV (SQLite-cached, append-only) |
| `data/cache.py` | SQLite schema + read/write helpers |

### Data schema (SQLite, `cache/market_data.sqlite`)

```
spot_ohlc(instrument, date, open, high, low, close, volume, source)
    PRIMARY KEY (instrument, date)

india_vix(date, open, high, low, close, source)
    PRIMARY KEY (date)

option_chain_snapshots(
    id, instrument, expiry, fetch_timestamp, strike, option_type,
    ltp, bid, ask, oi, oi_day_change, volume, iv, underlying_ltp
)
    Append-only. UNIQUE (instrument, expiry, strike, option_type, fetch_timestamp)
    -- every fetch adds a new timestamped batch rather than overwriting,
    -- so intraday chain history accumulates locally over time.

instrument_dump_meta(exchange, fetched_at)
    -- tracks when the NFO/NSE instrument master was last refreshed
```

`data/instruments.py` also caches the full NFO/NSE instrument dumps as
parquet files (`cache/instruments_nfo.parquet`, `instruments_nse.parquet`),
refreshed at most once/day.

### Flagged assumptions / limitations (Phase 1)

- **IV is not provided by Kite.** `kite.quote()` returns OI, volume, LTP,
  and 5-level depth — no implied volatility field. IV is computed by
  inverting Black-Scholes from the mid price (bid/ask average, falling back
  to LTP) via Brent's method (`core/pricing.py`). This means IV quality
  depends on quote liquidity: illiquid far-OTM/deep-ITM strikes can produce
  noisy or `None` IV (arbitrage-violating stale quotes are explicitly
  rejected rather than fabricated — verified live: a 6-days-to-expiry deep
  ITM NIFTY call showed a spurious 46% IV against an ~12% VIX, purely from
  thin trading amplifying pricing noise on a low-vega contract).
- **IV is priced off a market-implied forward, not spot + a static
  dividend yield.** `data/option_chain.py:_estimate_forward` backs out the
  forward from put-call parity on the ~8 most liquid near-ATM strikes each
  snapshot (median'd for robustness; falls back to the static
  `S*e^{(r-q)T}` approximation if fewer than 2 liquid two-sided strikes
  exist). This replaced an earlier version that priced off spot with a
  static `config.DIVIDEND_YIELD` guess — caught during a live sanity check:
  the static guess was off by ~44 points (0.18%) vs. the market-implied
  forward, which produced a *spurious* call/put IV skew (calls read
  ~12-14%, puts ~8-10% at the *same* strikes — backwards from real index
  skew). Re-verified live post-fix: call/put IV now agree to within ~0.03
  points at every near-ATM strike. See `core/pricing.py`'s module docstring
  for the Black-76 (S=forward, q=r) mechanics, and
  `tests/test_pricing.py::TestBlack76ForwardPricing` /
  `tests/test_option_chain.py` for the regression coverage.
- **OI day-change is not available from `kite.quote()`.** True OI change
  vs. the prior day's close would require diffing against
  `historical_data(..., oi=True)`; `oi_day_change` is left `NULL` for now
  rather than approximated from the wrong field.
- **Lot sizes / strike steps in `config.py` are placeholders** — the code
  path that matters (`data/instruments.py:get_current_lot_size`) resolves
  the *current* lot size from the live NFO dump instead of trusting the
  hardcoded config value, specifically because NSE revises these
  periodically.
- **Expiry resolution is fully dynamic**, not hardcoded to a weekday
  (NSE has changed Nifty's weekly expiry day and discontinued Bank Nifty's
  weekly expiry before — hardcoding would silently rot).
- **The risk-free rate is still a static config value** (`RISK_FREE_RATE =
  6.8%`) — it only affects discounting (a small effect over weekly/monthly
  tenors), not the forward level itself anymore, since the forward is now
  market-implied rather than carry-implied. The per-instrument
  `DIVIDEND_YIELD` config values are now only a fallback path (illiquid
  session with <2 usable near-ATM strikes), not the primary pricing input.
- Rate limiting is a conservative client-side sleep (default 3 req/sec,
  configurable in `config.py`) — not derived from Kite's actual documented
  limits at request time, since Kite doesn't expose remaining-quota headers.

## Phase 2 — Signal/regime layer

```bash
python scripts/show_regime.py --instrument NIFTY
```

Modules:

| Module | Responsibility |
|---|---|
| `signals/volatility.py` | Realized vol (close-to-close, Parkinson, Yang-Zhang), IV Rank, IV Percentile, IV-RV spread |
| `signals/trend.py` | Supertrend + ADX trend/regime classifier |
| `signals/oi.py` | Max pain, PCR, OI concentration ratio, range-bound conviction score |
| `signals/regime.py` | Combines the above into one regime object |

### Signal output schema

`signals.regime.RegimeSnapshot` (`.to_dict()` / JSON via `show_regime.py`):

```json
{
  "timestamp": "2025-01-02T09:20:00",
  "instrument": "NIFTY",
  "spot": 23540.25,
  "trend": "BULLISH",
  "adx": 27.4,
  "iv": 0.132,
  "iv_rank": 41.2,
  "iv_percentile": 55.0,
  "realized_vol_20d": 0.101,
  "iv_rv_spread": 0.031,
  "pcr": 1.14,
  "max_pain_strike": 23500.0,
  "oi_concentration_ratio": 0.62,
  "range_score": 48.3,
  "recommended_strategy_type": "BULL_PUT_SPREAD"
}
```

`trend` ∈ `{BULLISH, BEARISH, RANGE_BOUND}`.
`recommended_strategy_type` ∈ `{BULL_PUT_SPREAD, BEAR_CALL_SPREAD, IRON_CONDOR, NO_TRADE}`,
via the heuristic decision table in `signals/regime.py:recommend_strategy_type`:

```
IV rank < iv_rank_no_trade_below (default 20)   -> NO_TRADE   (premium too thin)
RANGE_BOUND + range_score >= 60                 -> IRON_CONDOR
BULLISH                                          -> BULL_PUT_SPREAD
BEARISH                                          -> BEAR_CALL_SPREAD
RANGE_BOUND + range_score < 60                   -> NO_TRADE   (ambiguous)
```

**This mapping is a documented starting heuristic, not a validated
model** — thresholds (60, 20, ADX bands, the equal-weighted range-score
formula) are meant to be revisited once the Phase 4 backtester exists to
actually test them against outcomes.

### Flagged assumptions / limitations (Phase 2)

- **IV Rank/Percentile use India VIX as the IV proxy for both NIFTY and
  BANKNIFTY.** VIX is Nifty-specific (30-day forward implied vol on Nifty
  options); using it for Bank Nifty is a standard-but-imperfect
  approximation, since Bank Nifty can diverge in vol regime around
  sector-specific events. `data/option_chain.py`'s per-strike computed IV
  could eventually replace this for BANKNIFTY once enough local chain-
  snapshot history accumulates to build a trailing window — there's no
  historical option-chain data source to backfill this retroactively.
- **`signals.oi` functions require a live/cached option chain snapshot**,
  so `build_regime_snapshot(...)`'s OI-derived fields (`pcr`,
  `max_pain_strike`, `oi_concentration_ratio`, `range_score`) can only be
  computed for "now" (or whatever chain-snapshot history accumulates going
  forward) — never retroactively for arbitrary historical dates, since Kite
  has no historical option-chain endpoint. This is the same limitation the
  Phase 4 spec anticipated ("reconstruct via Black-Scholes... flag this
  limitation clearly").
  `signals.regime.build_trend_vol_frame(...)`, by contrast, *is* fully
  backtestable — it only needs historical spot OHLC + VIX, both cached
  locally.
- `range_bound_conviction_score`'s three components (max-pain proximity,
  PCR balance, OI concentration) are equal-weighted by construction — an
  explicit simplification, not a fitted model.

## Phase 3 — Strike selection

```bash
python scripts/select_spread.py --instrument NIFTY
python scripts/select_spread.py --instrument NIFTY --strategy IRON_CONDOR --margin  # override regime + fetch live SPAN margin
python scripts/select_spread.py --instrument BANKNIFTY --lots 2
```

Given a live option chain snapshot and a target strategy type (from
`signals.regime.RegimeSnapshot.recommended_strategy_type`, or an explicit
`--strategy` override), builds a full spread definition. Does not place
orders — Phase 5's manual-confirmation gate is what turns a selected
spread into an order.

| Module | Responsibility |
|---|---|
| `core/probability.py` | Probability-of-touch (GBM reflection principle) |
| `strikes/selection.py` | Chain Greeks, OI-wall detection, delta-targeted short-strike selection, width-based long-strike selection |
| `strikes/spreads.py` | Builds the full spread definition from selected legs |
| `strikes/margin.py` | Live SPAN+exposure margin via `kite.basket_order_margins()`, + a documented fallback |
| `strikes/orchestrator.py` | Ties selection + spread-building together per strategy type |

Selection logic per short leg (`strikes/selection.py:select_short_strike`):
1. Restrict to OTM strikes on the relevant side (puts below forward, calls above).
2. Reject strikes with probability-of-touch > `cfg.max_probability_of_touch` (default 0.35).
3. Reject strikes inside/adjacent to an OI wall (`cfg.oi_wall_multiple`× the
   *local* median OI, `cfg.oi_wall_band_strikes` steps of padding).
4. Of what survives, pick the strike whose |delta| is closest to `cfg.target_delta` (default 0.175).

The long leg (`select_long_strike`) is simply `cfg.spread_width_strikes`
strike-steps further OTM than the short leg — no delta/probability
filtering, since its job is capping risk at a defined width, not hitting a
target.

### Spread output schema

`strikes.spreads.SpreadDefinition.to_dict()`:

```json
{
  "strategy_type": "IRON_CONDOR",
  "instrument": "NIFTY",
  "expiry": "2026-08-18",
  "legs": [
    {"option_type": "PE", "strike": 24100.0, "action": "SELL", "price": 23.68, "delta": -0.141, "iv": 0.1046},
    {"option_type": "PE", "strike": 23900.0, "action": "BUY", "price": 8.83, "delta": -0.058, "iv": 0.1131},
    {"option_type": "CE", "strike": 24750.0, "action": "SELL", "price": 26.08, "delta": 0.163, "iv": 0.0975},
    {"option_type": "CE", "strike": 24950.0, "action": "BUY", "price": 7.85, "delta": 0.058, "iv": 0.1012}
  ],
  "lot_size": 75,
  "n_lots": 1,
  "net_credit": 33.075,
  "max_profit": 2480.625,
  "max_loss": 12519.375,
  "breakevens": [24066.925, 24783.075],
  "margin_estimate": 84311,
  "margin_source": "kite_basket_margin"
}
```

(Real output from a live run against NIFTY, 6 DTE — see the two bugs this
surfaced, below.)

### Two bugs found and fixed during live verification (not caught by unit tests alone)

Both were caught by running the actual pipeline against a real Kite
session and sanity-checking the output — synthetic-chain unit tests alone
had passed against both bugs, since the synthetic chains used didn't
reproduce the specific real-world condition that triggered them. Left as
a reminder to verify against live data before trusting any new signal.

1. **OI-wall filter blocked the entire tradeable region.** `find_oi_walls`
   originally compared each strike's OI to the *whole chain's* median. In a
   real chain, OI decays sharply away from spot, so the whole-chain median
   is dragged down by dozens of near-zero-OI far strikes — a 3x threshold
   against that median flagged 42 of 105 live NIFTY PE strikes as "walls",
   including every strike within 450 points of spot. That left only
   strikes far enough out to have near-zero OI, which also have near-zero
   delta — so the "closest to target delta" pick came out at delta≈0.007
   against a 0.175 target. Fixed by comparing each strike to a *local*
   rolling median (±5 strikes by default — `cfg.oi_wall_local_window`)
   instead; re-verified live, real walls (round strikes like 22500/23000/
   25000 sitting at 5-20x their local neighborhood) are still caught, while
   the near-ATM tradeable region is no longer blanket-flagged.
2. **Deep-ITM IV inversion could reject valid quotes.** Found while adding
   Black-76 forward-pricing tests (see Phase 1/2 section above) —
   `implied_volatility`'s arbitrage-rejection check compared price against
   undiscounted `S-K` instead of the discounted forward intrinsic,
   mis-rejecting borderline-valid deep ITM/OTM quotes. Fixed alongside the
   forward-pricing change.

### Flagged assumptions / limitations (Phase 3)

- **Margin defaults to Kite's live `basket_order_margins()`** when
  `--margin`/`compute_margin=True` is used — this is the exchange's actual
  SPAN+exposure figure, not an approximation. Verified live: for a NIFTY
  bull put spread, the naive per-leg total (`initial.total`, ignoring hedge
  benefit) was ~5x the correct hedge-aware total (`final.total`) — this
  module always reads `final`. Separately, on the live iron condor above,
  margin (₹84,311) came out ~6.7x max_loss (₹12,519) — real SPAN scanning
  doesn't net the put and call sides down to max-loss the way intuition
  might suggest; that gap is real, not a bug, and matters for capital
  sizing. Without `--margin`, `strikes.margin.estimate_margin_fallback`
  (a crude `max_loss × multiple` approximation, NOT SPAN) is used instead —
  needed for Phase 4 backtesting, where there's no live margin calculator
  to query for historical dates.
- **`target_delta` and `max_probability_of_touch` can conflict** for
  near-week expiries: a strike close to the 0.175 delta target may also
  have a probability-of-touch above the 0.35 default threshold (observed
  live at 6 DTE: the strike closest to 0.175 delta had a 0.357
  probability-of-touch, just over the threshold, so the next strike out
  was chosen instead). This is a real tradeoff, not a bug — tightening one
  constraint costs the other — and the defaults haven't been tuned against
  backtested outcomes yet (that's a Phase 4 job).
- **Spread width is a fixed strike-count** (`cfg.spread_width_strikes`,
  default 4), not delta- or premium-targeted. A fixed width is simple and
  predictable but means the long leg's cost (and therefore net credit)
  varies with local vol/skew rather than hitting a target credit-to-width
  ratio — no such ratio target is enforced in this phase.
- **Leg fills are theoretical mid prices** (bid/ask average, LTP fallback),
  the same convention `data/option_chain.py` uses for IV. Phase 4's
  backtester is where a configurable slippage/bid-ask assumption on top of
  this belongs, per the original spec — not modeled here.

## Phase 4 — Backtesting engine

```bash
python scripts/run_backtest.py --instrument NIFTY --start 2023-01-01 --end 2024-12-31
python scripts/run_backtest.py --instrument NIFTY --strategy IRON_CONDOR --tenor-days 7
python scripts/run_backtest.py --instrument BANKNIFTY --profit-target 0.5 --stop-loss 1.0 --slippage 0.02
```

Runs entirely off cached historical spot OHLC + India VIX (Phase 1) — no
live Kite session needed, and no real historical option chain data exists
to run it against (flagged in Phase 1/3 already). Simulates day-by-day:
regime signal → strategy selection → strike selection (Phase 2/3 code,
unchanged) → theoretical entry fill with slippage → daily full-reprice
mark-to-market → exit on profit target / stop loss / DTE threshold.

| Module | Responsibility |
|---|---|
| `backtest/reconstruction.py` | Synthetic option chain from a historical date's spot+VIX (Black-76, same schema as `data/option_chain.py`) |
| `backtest/engine.py` | The day-by-day simulation loop, `BacktestConfig` |
| `backtest/metrics.py` | Win rate, avg win/loss, max drawdown, Sharpe/Sortino, worst single-day loss |

**Why the synthetic chain matters less than it sounds**: `strikes/selection.py`,
`strikes/spreads.py`, and `strikes/orchestrator.py` — all of Phase 3 — run
completely unchanged against a reconstructed chain, because
`reconstruct_chain()` outputs the exact same schema `fetch_option_chain_snapshot()`
does. That schema match was deliberate specifically so Phase 3 wouldn't
need a backtest-specific version.

### Real backtest run against cached live data (NIFTY, 2023-01-01 → 2026-08-11, 894 trading days)

Regime-driven (no strategy override), default config, rerun 2026-10-11 on
the current engine:

```json
{
  "n_trades": 98,
  "win_rate": 0.816,
  "avg_win": 1306.85,
  "avg_loss": -3542.11,
  "total_pnl": 40790.06,
  "final_equity": 2432085.10,
  "total_return_pct": 0.2160,
  "max_drawdown_pct": 0.0056,
  "sharpe_ratio": -2.43,
  "sortino_ratio": -2.51,
  "worst_single_day_loss_pct": 0.0030
}
```

**`total_return_pct` is not the strategy's return.** Since the risk-free
accrual change (see "Engine changes after the first backtest" below), the
equity curve includes 6.8% p.a. interest on every day with no open
position. The trades themselves made `total_pnl` = ₹40,790, i.e. 2.04% of
the ₹20L starting capital; the remaining ~19.6 points are cash interest.
Cash held for all 894 days at the same rate would have returned 27.2%, so
the strategy finished *behind* cash — that is what the negative Sharpe now
says. Use `total_pnl` to compare configurations: with
`--assumed-range-score 70` trade P&L rises to ₹1,11,328 (140 trades, 49 of
them iron condors) while `total_return_pct` *falls* to 20.1%, because more
days in a position means fewer days accruing interest.

All 98 trades were `BULL_PUT_SPREAD`/`BEAR_CALL_SPREAD` (55/43) — zero
`IRON_CONDOR` trades, correctly, since the default `assumed_range_score`
(50.0) sits below the 60 threshold `recommend_strategy_type` requires for
an iron condor (see Phase 2's decision table) — RANGE_BOUND days default
to `NO_TRADE` unless you raise `--assumed-range-score`. Exit reasons: 79
profit_target, 16 stop_loss, 3 dte_exit.

**Why Sharpe/Sortino were −10.9/−13.1 in the first run and are −2.4/−2.5
now.** The strategy holds a position on only ~26% of trading days. The
original engine left capital flat on the other ~74% while the Sharpe
formula still charged the 6.8% risk-free hurdle on every day, so most of
the −10.9 was that mismatch rather than the trades. The engine now accrues
the risk-free rate on idle days, which removes the mismatch. What is left
is the real result: trade P&L is smaller than the interest forgone on the
days capital is tied up. Sharpe/Sortino still use simple daily returns on
a jump-heavy curve (see `backtest/metrics.py`'s docstring), so treat them
as comparative between configs rather than literal.

### Engine changes after the first backtest (2026-08-17)

Three additions to `backtest/`, all covered by tests and all defaulting to
the original behaviour except the first:

- **Risk-free accrual on idle capital** (always on). Any day that starts
  with no open position compounds `capital` at `RISK_FREE_RATE / 252`.
  This changes `total_return_pct` and `max_drawdown_pct`, not just Sharpe;
  `total_pnl` is summed from the trade log and is unaffected.
- **Opt-in IV skew** (`BacktestConfig.use_iv_skew`, default `False`).
  `reconstruct_chain_with_skew` / `reprice_leg_with_skew` replace the flat
  per-day IV with `iv_atm * (1 - skew_slope*x + smile_curvature*x²)`, where
  `x` is log-moneyness in ATM standard deviations. The coefficients (0.08,
  0.02) are illustrative, not calibrated — there is no historical skew to
  calibrate against.
- **Laddered positions** (`BacktestConfig.max_concurrent_positions`,
  default `1`). Up to N positions open at once, at most one new entry per
  day, no same-day re-entry into a freed slot. Capital is one shared pool
  with no per-position margin check.

`backtest/metrics.py` also gained `trade_level_sharpe`, which prorates the
risk-free hurdle over each trade's actual holding period; it is not part
of `compute_performance_report`'s output.

`use_iv_skew` and `max_concurrent_positions` have no CLI flag in
`scripts/run_backtest.py` yet and are not swept by `backtest/sweep.py` —
set them on `BacktestConfig` directly.

**A bug this surfaced**: `recommend_strategy_type` was originally called
with `cfg.strike_selection_cfg` (wrong config type — `StrikeSelectionConfig`
has no `iv_rank_no_trade_below` attribute) instead of `signals.SIGNALS`.
Every engine unit test had `strategy_override` set, which short-circuits
past that call entirely (`cfg.strategy_override or recommend_strategy_type(...)`
never evaluates the right side when the left is truthy) — so all tests
passed while the regime-driven path was actually broken. Caught only by
running `scripts/run_backtest.py` for real. Fixed, and a regression test
(`TestRegimeDrivenEntry`, no override) now exercises that path directly.

### Parameter sweep (`backtest/sweep.py`, `scripts/run_parameter_sweep.py`)

The default config's weak backtested return (see above) raised the
obvious question — can tuning fix it? Grid-searches
`tenor_days × profit_target_pct × stop_loss_pct × target_delta ×
assumed_range_score`, ranks by in-sample (IS) return, then re-runs the
top N on a held-out out-of-sample (OOS) window — the actual test for
whether a config's IS performance is real or just overfit to that period.

```bash
python scripts/run_parameter_sweep.py --instrument NIFTY \
    --is-start 2023-01-01 --is-end 2025-08-11 \
    --oos-start 2025-08-12 --oos-end 2026-08-11
```

**Real run, NIFTY, 162 combos, IS=648 days / OOS=246 days** (run
2026-08-12, before the risk-free accrual change — the return and Sharpe
figures below are trade-P&L-only and are not comparable to what the sweep
prints today; it has not been rerun):

The top 5 IS configs (4.0-4.8% IS return) mostly did **not** hold up OOS —
returns fell to 0.2-2.0%, a 2-22x degradation. That's the sweep doing its
job: exposing that most of the apparent IS "improvement" was overfitting
to one period's noise, not a real edge. One config held up better than
the rest (tenor=7d, profit_target=50%, stop_loss=150%, assumed_range_score=70:
IS 4.26% → OOS 2.01%, roughly halving rather than collapsing) — the most
defensible starting point of the five, not a validated one.

One finding **did** hold up as a robust, grid-wide average (not
cherry-picked from the top 5): `assumed_range_score >= 70` (enabling
`IRON_CONDOR` on range-bound days, vs. the default 50 which always
resolves to `NO_TRADE` there) more than tripled mean IS return across all
81 matching combos (0.72% → 2.27%), consistent with the earlier isolated
diagnostic (2.04% → 5.57% total return). `profit_target_pct=0.5` and
`tenor_days∈{5,7}` also beat their alternatives on average, more mildly;
`stop_loss_pct` showed only a weak trend.

**Sharpe/Sortino were negative for every single one of the 162 configs**
(mean Sharpe -10.45, best -5.11) — no combination in this grid produced
good risk-adjusted daily returns, only sometimes-decent absolute ones.
That's a structural property of this strategy family's P&L shape under
this metric (see the Sharpe/Sortino note above), not something tuning
escaped anywhere in the grid — worth weighing against the absolute-return
numbers, not ignoring.

**Bottom line**: enabling range-bound entries is the one lever here worth
trusting; the specific tenor/profit-target/stop-loss combination that
looked best in-sample should not be — and even the OOS-durable config's
~2%/year is still weak, likely capped by the Phase 4 data-quality
limitations below (flat IV, static forward, synthetic tenor, no real
historical OI) as much as by strategy logic. A single IS/OOS split on one
instrument's history is also not a rigorous validation on its own — see
`backtest/sweep.py`'s docstring.

### Flagged assumptions / limitations (Phase 4)

- **No real historical option chain data exists** (same root cause as
  Phase 1/3's live-only OI signals) — `backtest/reconstruction.py`
  rebuilds a synthetic chain from spot+VIX each entry day. This means,
  relative to the live pipeline:
  - **Flat IV per day by default** — every strike gets that day's VIX/100,
    no skew (the opt-in `use_iv_skew` path adds a synthetic, uncalibrated
    one). Backtested delta-targeted selection is systematically less
    accurate for OTM strikes than live selection (which computes true
    per-strike IV).
  - **No OI data** — `find_oi_walls` degrades to a no-op automatically;
    OI-wall avoidance is not modeled in backtests at all.
  - **Static-carry forward** (`S*e^{(r-q)T}`), not market-implied — the
    same approximation the live pipeline moved away from in Phase 1/2
    because it measurably diverges from the true forward. There's no live
    chain to imply a forward from on a historical date.
  - **Synthetic expiry tenor** (`tenor_days` calendar days from each entry,
    default 7) instead of the real historical NSE expiry calendar, which
    has changed over time (see `data/instruments.py`'s docstring) and has
    no historical dataset to align to. `min_dte_exit` counts down against
    this synthetic date.
- **`assumed_range_score` (default 50.0) stands in for the OI-based
  range_score** Phase 2 can only compute live. This is the single biggest
  lever on how often `IRON_CONDOR` gets selected in a backtest — the
  default deliberately sits below the 60-point threshold so backtests
  don't silently assume an edge on a signal they can't actually compute
  historically.
- **One instrument per run, one position at a time by default**
  (`max_concurrent_positions` raises the cap) — no margin-based position
  sizing (`n_lots` is fixed by config, not derived from available capital
  via `strikes/margin.py`), no margin check across concurrent positions,
  and no multi-instrument portfolio simulation.
- **"Leg risk" is limited to independent per-leg slippage.** Each leg's
  fill is the theoretical price adjusted by `slippage_pct` independently —
  sold legs fill worse (lower), bought legs fill worse (higher). True
  non-atomic execution risk (only some legs filling, or the market moving
  in the gap between leg fills) is not modeled — every leg's fill is still
  computed from the same day's theoretical price.
- **Sharpe/Sortino use simple daily returns** on the equity curve with the
  standard `×√252` annualization — see the real example above for why the
  sign can mislead on a jump-heavy options P&L curve.

## Tests

```bash
pytest tests/ -v
```

210 tests, all passing (last run 2026-10-11). Coverage focus, per the correctness-over-speed
priority for pricing/probability math:

- `test_pricing.py` — Black-Scholes put-call parity, Greeks vs. finite-
  difference derivatives, implied-volatility round-trip recovery (price a
  known σ, invert, recover σ) across strikes/moneyness/option types,
  degenerate-input handling (stale/arbitrage-violating quotes correctly
  return `None`), and the Black-76 forward-pricing equivalence (including a
  test that reproduces the exact call/put IV skew mechanism found live).
- `test_volatility.py` — realized-vol estimators against hand-computed
  values on deterministic series (not just statistical GBM checks), IV
  Rank/Percentile against hand-computed expected values for every index in
  a 10-point series.
- `test_trend.py` — ADX/Supertrend sanity (bounds, uptrend line stays
  below price, downtrend line stays above), regime classification on
  synthetic trending/choppy series.
- `test_oi.py` — max pain, PCR, OI concentration, and range-score against
  hand-computed values on constructed option chains.
- `test_probability.py` — probability-of-touch edge cases, monotonicity,
  driftless up/down symmetry, and cross-validation against Monte Carlo
  simulation of the underlying GBM (not just the closed-form formula
  checking itself).
- `test_option_chain.py` — the forward-estimation helper (`_estimate_forward`):
  recovers a known forward from synthetic parity-consistent quotes, is
  robust to one corrupted strike, and falls back correctly when the chain
  is illiquid.
- `test_strike_selection.py` — chain Greeks computation, local OI-wall
  detection (including the regression case for the whole-chain-median
  bug), delta targeting, and the probability-of-touch filter.
- `test_spreads.py` — bull put / bear call / iron condor construction
  against hand-computed credit/max-profit/max-loss/breakevens, including
  the asymmetric-width iron condor case.
- `test_orchestrator.py` — end-to-end strike selection → spread
  construction per strategy type, and failure paths (unavailable long leg,
  NO_TRADE rejection, missing client for margin).
- `test_reconstruction.py` — synthetic chain matches the live schema,
  satisfies put-call parity, matches direct `bs_price` calls, and
  `reprice_leg` agrees with `reconstruct_chain` for the same inputs.
- `test_backtest_metrics.py` — win rate/avg win/avg loss/max drawdown
  against hand-computed values, Sharpe zero-variance edge case, Sortino
  correctly ignoring upside volatility.
- `test_backtest_engine.py` — scenario tests engineered to trigger each
  exit condition (IV-crush → profit_target, sharp adverse move →
  stop_loss, flat market → dte_exit), leg-count-per-trade schema checks,
  no-same-day-re-entry, and the regime-driven (non-override) path that
  caught the config-mismatch bug described above.
- `test_sweep.py` — grid size matches the cartesian product, swept params
  round-trip into the output, the `reliable` flag reflects the min-trades
  threshold, and IS/OOS evaluation gives identical results when fed
  identical data (a sanity check on the split logic itself).

## What's next (not built yet)

- **Phase 5** — execution scaffold, paper-trading default, manual
  confirmation gate before any live order.
- Rerunning the parameter sweep on the current engine, ranked by trade
  P&L rather than `total_return_pct`, and exposing `use_iv_skew` /
  `max_concurrent_positions` to the CLI and the sweep.
- Validating/tuning Phase 2 regime thresholds and Phase 3 selection
  defaults (target_delta, max_probability_of_touch, spread_width_strikes,
  assumed_range_score) against actual backtested outcomes — Phase 4 built
  the mechanism to do this, not a validated edge.
