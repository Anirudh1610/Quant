#!/usr/bin/env python3
"""
scripts/pull_data.py
---------------------
Phase 1 CLI: pull and cache historical spot + India VIX for NIFTY/BANKNIFTY,
and optionally a live option chain snapshot. Run scripts/generate_access_token.py
first if today's access token hasn't been generated yet.

Usage:
    python scripts/pull_data.py --start 2024-01-01 --end 2024-12-31
    python scripts/pull_data.py --instrument NIFTY --chain --expiry 2025-01-02
    python scripts/pull_data.py --instrument BANKNIFTY --chain   # nearest expiry
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import INSTRUMENTS
from core.kite_client import KiteRateLimited
from core.session import KiteAuthError
from data.cache import init_db
from data.instruments import get_nearest_expiry
from data.option_chain import fetch_option_chain_snapshot
from data.spot import get_historical_spot
from data.vix import get_historical_vix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pull and cache Kite market data (Phase 1)")
    p.add_argument("--instrument", choices=list(INSTRUMENTS), default=None,
                    help="Default: pull both NIFTY and BANKNIFTY")
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default=date.today().isoformat())
    p.add_argument("--refresh", action="store_true", help="Re-fetch even if cached")
    p.add_argument("--chain", action="store_true", help="Also pull a live option chain snapshot")
    p.add_argument("--expiry", default=None, help="YYYY-MM-DD; default: nearest listed expiry")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    init_db()

    try:
        client = KiteRateLimited()
    except KiteAuthError as e:
        print(f"[ERROR] {e}\nRun scripts/generate_access_token.py first.")
        sys.exit(1)

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    instruments = [args.instrument] if args.instrument else list(INSTRUMENTS)

    print("[1/2] Pulling India VIX history...")
    vix = get_historical_vix(client, start, end, refresh=args.refresh)
    print(f"      {len(vix)} rows cached")

    for instrument in instruments:
        print(f"[2/2] Pulling {instrument} spot history...")
        spot = get_historical_spot(client, instrument, start, end, refresh=args.refresh)
        print(f"      {len(spot)} rows cached")

        if args.chain:
            expiry = (datetime.strptime(args.expiry, "%Y-%m-%d").date() if args.expiry
                      else get_nearest_expiry(client, instrument))
            print(f"      Pulling {instrument} option chain for expiry {expiry}...")
            chain = fetch_option_chain_snapshot(client, instrument, expiry, force_refresh=args.refresh)
            print(f"      {len(chain)} contracts snapshotted")

    print("\nDone.")


if __name__ == "__main__":
    main()
