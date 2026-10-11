#!/usr/bin/env python3
"""
scripts/select_spread.py
--------------------------
Phase 3 CLI: build today's regime (Phase 2), then select and print a full
spread definition for it — strikes, theoretical credit, max profit/loss,
breakevens, and (optionally) live SPAN+exposure margin.

Does not place orders.

Usage:
    python scripts/select_spread.py --instrument NIFTY
    python scripts/select_spread.py --instrument BANKNIFTY --lots 2 --margin
    python scripts/select_spread.py --instrument NIFTY --strategy BULL_PUT_SPREAD  # override the regime's recommendation
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import INSTRUMENTS, STRIKE_SELECTION
from core.kite_client import KiteRateLimited
from core.session import KiteAuthError
from data.cache import init_db
from data.instruments import get_nearest_expiry
from data.option_chain import estimate_forward_from_chain, fetch_option_chain_snapshot, time_to_expiry_years
from data.spot import get_historical_spot, get_live_spot
from data.vix import get_historical_vix
from signals.regime import NO_TRADE, build_regime_snapshot, build_trend_vol_frame
from strikes.margin import estimate_margin_fallback, get_live_margin
from strikes.orchestrator import select_spread


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select a candidate spread for today's regime (Phase 3)")
    p.add_argument("--instrument", choices=list(INSTRUMENTS), required=True)
    p.add_argument("--expiry", default=None, help="YYYY-MM-DD; default: nearest listed expiry")
    p.add_argument("--strategy", default=None,
                    choices=["BULL_PUT_SPREAD", "BEAR_CALL_SPREAD", "IRON_CONDOR"],
                    help="Override the regime's recommended_strategy_type")
    p.add_argument("--lots", type=int, default=1)
    p.add_argument("--margin", action="store_true", help="Fetch live SPAN+exposure margin via Kite")
    p.add_argument("--history-days", type=int, default=400)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    init_db()

    try:
        client = KiteRateLimited()
    except KiteAuthError as e:
        print(f"[ERROR] {e}\nRun scripts/generate_access_token.py first.")
        sys.exit(1)

    end = date.today()
    start = end - timedelta(days=args.history_days)
    spot_hist = get_historical_spot(client, args.instrument, start, end)
    vix_hist = get_historical_vix(client, start, end)
    if spot_hist.empty or vix_hist.empty:
        print("[ERROR] Not enough cached/pulled history — run scripts/pull_data.py first.")
        sys.exit(1)

    trend_vol = build_trend_vol_frame(spot_hist, vix_hist["close"])
    latest_row = trend_vol.iloc[-1]

    expiry = (datetime.strptime(args.expiry, "%Y-%m-%d").date() if args.expiry
              else get_nearest_expiry(client, args.instrument))
    chain = fetch_option_chain_snapshot(client, args.instrument, expiry)
    live_spot = get_live_spot(client, args.instrument)["last_price"]

    regime = build_regime_snapshot(args.instrument, latest_row, live_spot, chain)
    strategy_type = args.strategy or regime.recommended_strategy_type

    print("=== Regime ===")
    print(json.dumps(regime.to_dict(), indent=2, default=str))

    if strategy_type == NO_TRADE:
        print("\nRegime recommends NO_TRADE and no --strategy override was given. Exiting.")
        sys.exit(0)

    # recompute the same market-implied forward the chain's IVs were priced
    # off of (data/option_chain.py), for consistent Greeks/probability-of-touch
    from config import DIVIDEND_YIELD, RISK_FREE_RATE
    T = time_to_expiry_years(expiry)
    r = RISK_FREE_RATE
    q = DIVIDEND_YIELD.get(args.instrument, 0.0)
    forward = estimate_forward_from_chain(chain, live_spot, T, r, q)

    spec = INSTRUMENTS[args.instrument]
    spread = select_spread(chain, strategy_type, forward, T, r, args.instrument, expiry,
                            spec.strike_step, lot_size=spec.lot_size, n_lots=args.lots,
                            client=client, compute_margin=False)

    if spread is None:
        print(f"\nCould not build a {strategy_type} — no strike survived selection filters "
              f"(probability-of-touch / OI-wall / listed-strike constraints).")
        sys.exit(1)

    if args.margin:
        get_live_margin(client, args.instrument, expiry, spread)
    else:
        estimate_margin_fallback(spread)

    print("\n=== Spread ===")
    print(json.dumps(spread.to_dict(), indent=2, default=str))


if __name__ == "__main__":
    main()
