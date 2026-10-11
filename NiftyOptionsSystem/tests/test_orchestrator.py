from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from config import StrikeSelectionConfig
from core.pricing import bs_price
from signals.regime import BEAR_CALL_SPREAD, BULL_PUT_SPREAD, IRON_CONDOR
from strikes.orchestrator import select_spread

T, R = 6 / 365, 0.068
FORWARD = 24339.5
TRUE_SIGMA = 0.13
STRIKE_STEP = 50
EXPIRY = date(2026, 8, 18)


def _synthetic_chain(strikes: list[float]) -> pd.DataFrame:
    rows = []
    for k in strikes:
        for opt_type, bs_type in (("CE", "call"), ("PE", "put")):
            price = bs_price(FORWARD, k, T, R, TRUE_SIGMA, R, bs_type)
            rows.append({
                "strike": k, "option_type": opt_type,
                "ltp": price, "bid": price - 0.25, "ask": price + 0.25,
                "oi": 1000.0, "volume": 5000.0, "iv": TRUE_SIGMA, "underlying_ltp": FORWARD,
            })
    return pd.DataFrame(rows)


WIDE_STRIKES = list(range(23600, 25100, 50))
LOOSE_CFG = StrikeSelectionConfig(target_delta=0.175, max_probability_of_touch=1.0,
                                   oi_wall_multiple=999.0, oi_wall_band_strikes=0, spread_width_strikes=4)


class TestSelectSpread:
    def test_bull_put_spread(self):
        chain = _synthetic_chain(WIDE_STRIKES)
        spread = select_spread(chain, BULL_PUT_SPREAD, FORWARD, T, R, "NIFTY", EXPIRY,
                                STRIKE_STEP, lot_size=75, n_lots=1, cfg=LOOSE_CFG)
        assert spread is not None
        assert spread.strategy_type == "BULL_PUT_SPREAD"
        assert len(spread.legs) == 2
        assert all(leg.option_type == "PE" for leg in spread.legs)
        assert spread.legs[0].action == "SELL"
        assert spread.legs[1].action == "BUY"
        assert spread.legs[1].strike < spread.legs[0].strike  # long leg further OTM (lower)

    def test_bear_call_spread(self):
        chain = _synthetic_chain(WIDE_STRIKES)
        spread = select_spread(chain, BEAR_CALL_SPREAD, FORWARD, T, R, "NIFTY", EXPIRY,
                                STRIKE_STEP, lot_size=75, n_lots=1, cfg=LOOSE_CFG)
        assert spread is not None
        assert spread.strategy_type == "BEAR_CALL_SPREAD"
        assert all(leg.option_type == "CE" for leg in spread.legs)
        assert spread.legs[1].strike > spread.legs[0].strike  # long leg further OTM (higher)

    def test_iron_condor(self):
        chain = _synthetic_chain(WIDE_STRIKES)
        spread = select_spread(chain, IRON_CONDOR, FORWARD, T, R, "NIFTY", EXPIRY,
                                STRIKE_STEP, lot_size=75, n_lots=1, cfg=LOOSE_CFG)
        assert spread is not None
        assert spread.strategy_type == "IRON_CONDOR"
        assert len(spread.legs) == 4
        assert len(spread.breakevens) == 2
        assert spread.breakevens[0] < FORWARD < spread.breakevens[1]

    def test_returns_none_when_long_leg_unavailable(self):
        # chain too narrow for a 4-strike-wide long leg to exist
        narrow_chain = _synthetic_chain([24300, 24350, 24400])
        spread = select_spread(narrow_chain, BULL_PUT_SPREAD, FORWARD, T, R, "NIFTY", EXPIRY,
                                STRIKE_STEP, lot_size=75, n_lots=1, cfg=LOOSE_CFG)
        assert spread is None

    def test_rejects_no_trade_strategy_type(self):
        chain = _synthetic_chain(WIDE_STRIKES)
        with pytest.raises(ValueError):
            select_spread(chain, "NO_TRADE", FORWARD, T, R, "NIFTY", EXPIRY,
                           STRIKE_STEP, lot_size=75, n_lots=1, cfg=LOOSE_CFG)

    def test_compute_margin_without_client_raises(self):
        chain = _synthetic_chain(WIDE_STRIKES)
        with pytest.raises(ValueError):
            select_spread(chain, BULL_PUT_SPREAD, FORWARD, T, R, "NIFTY", EXPIRY,
                           STRIKE_STEP, lot_size=75, n_lots=1, cfg=LOOSE_CFG, compute_margin=True)
