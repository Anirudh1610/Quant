#!/usr/bin/env python3
"""
scripts/show_regime.py
------------------------
Phase 2 CLI: build and print today's regime object for NIFTY/BANKNIFTY,
using cached spot/VIX history (for trend + IV rank) plus a live option
chain snapshot (for OI-derived signals). Run scripts/pull_data.py first to
populate history if the cache is empty.

Usage:
    python scripts/show_regime.py --instrument NIFTY
    python scripts/show_regime.py --instrument BANKNIFTY --expiry 2025-01-02
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import INSTRUMENTS, SIGNALS
from core.kite_client import KiteRateLimited
from core.session import KiteAuthError
from data.cache import init_db
from data.instruments import get_nearest_expiry
from data.option_chain import fetch_option_chain_snapshot
from data.spot import get_historical_spot, get_live_spot
from data.vix import get_historical_vix
from signals.regime import build_regime_snapshot, build_trend_vol_frame


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Print today's regime object (Phase 2)")
    p.add_argument("--instrument", choices=list(INSTRUMENTS), required=True)
    p.add_argument("--expiry", default=None, help="YYYY-MM-DD; default: nearest listed expiry")
    p.add_argument("--history-days", type=int, default=400,
                    help="Calendar days of history to pull for trend/IV-rank calc")
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
        print("[ERROR] Not enough cached/pulled history to compute trend and IV rank.")
        sys.exit(1)

    trend_vol = build_trend_vol_frame(spot_hist, vix_hist["close"])
    latest_row = trend_vol.iloc[-1]

    expiry = (datetime.strptime(args.expiry, "%Y-%m-%d").date() if args.expiry
              else get_nearest_expiry(client, args.instrument))
    chain = fetch_option_chain_snapshot(client, args.instrument, expiry)

    live_spot = get_live_spot(client, args.instrument)["last_price"]

    snapshot = build_regime_snapshot(args.instrument, latest_row, live_spot, chain)

    print(json.dumps(snapshot.to_dict(), indent=2, default=str))


if __name__ == "__main__":
    main()
