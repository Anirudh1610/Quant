from __future__ import annotations

import pytest

from strikes.spreads import build_bear_call_spread, build_bull_put_spread, build_iron_condor


def _row(strike, bid=None, ask=None, ltp=None, delta=None, iv=None, option_type="PE"):
    return {"strike": strike, "option_type": option_type, "bid": bid, "ask": ask,
            "ltp": ltp, "delta": delta, "iv": iv}


class TestLegPricing:
    def test_uses_mid_when_bid_ask_present(self):
        short_put = _row(24300, bid=99, ask=101, ltp=95)
        long_put = _row(24100, bid=39, ask=41, ltp=38)
        spread = build_bull_put_spread(short_put, long_put, "NIFTY", "2026-08-18", lot_size=75, n_lots=1)
        # mid(99,101)=100, mid(39,41)=40 -> net_credit = 60, not the ltp-based 57
        assert spread.net_credit == pytest.approx(60.0)

    def test_falls_back_to_ltp_when_no_bid_ask(self):
        short_put = _row(24300, bid=None, ask=None, ltp=100)
        long_put = _row(24100, bid=0, ask=0, ltp=40)
        spread = build_bull_put_spread(short_put, long_put, "NIFTY", "2026-08-18", lot_size=75, n_lots=1)
        assert spread.net_credit == pytest.approx(60.0)


class TestBullPutSpread:
    def test_matches_hand_computation(self):
        short_put = _row(24300, bid=99, ask=101)  # mid=100
        long_put = _row(24100, bid=39, ask=41)     # mid=40
        spread = build_bull_put_spread(short_put, long_put, "NIFTY", "2026-08-18", lot_size=75, n_lots=2)

        assert spread.net_credit == pytest.approx(60.0)
        assert spread.max_profit == pytest.approx(60.0 * 75 * 2)
        assert spread.max_loss == pytest.approx((200 - 60.0) * 75 * 2)
        assert spread.breakevens == [pytest.approx(24240.0)]
        assert spread.strategy_type == "BULL_PUT_SPREAD"
        assert [leg.action for leg in spread.legs] == ["SELL", "BUY"]

    def test_rejects_inverted_strikes(self):
        short_put = _row(24100, bid=39, ask=41)
        long_put = _row(24300, bid=99, ask=101)  # long strike above short -- invalid for a bull put spread
        with pytest.raises(ValueError):
            build_bull_put_spread(short_put, long_put, "NIFTY", "2026-08-18", lot_size=75)


class TestBearCallSpread:
    def test_matches_hand_computation(self):
        short_call = _row(24700, bid=89, ask=91, option_type="CE")   # mid=90
        long_call = _row(24900, bid=29, ask=31, option_type="CE")     # mid=30
        spread = build_bear_call_spread(short_call, long_call, "NIFTY", "2026-08-18", lot_size=75, n_lots=1)

        assert spread.net_credit == pytest.approx(60.0)
        assert spread.max_profit == pytest.approx(60.0 * 75)
        assert spread.max_loss == pytest.approx((200 - 60.0) * 75)
        assert spread.breakevens == [pytest.approx(24760.0)]
        assert spread.strategy_type == "BEAR_CALL_SPREAD"

    def test_rejects_inverted_strikes(self):
        short_call = _row(24900, bid=29, ask=31, option_type="CE")
        long_call = _row(24700, bid=89, ask=91, option_type="CE")
        with pytest.raises(ValueError):
            build_bear_call_spread(short_call, long_call, "NIFTY", "2026-08-18", lot_size=75)


class TestIronCondor:
    def test_matches_hand_computation_equal_widths(self):
        short_put = _row(24300, bid=99, ask=101)
        long_put = _row(24100, bid=39, ask=41)
        short_call = _row(24700, bid=89, ask=91, option_type="CE")
        long_call = _row(24900, bid=29, ask=31, option_type="CE")

        spread = build_iron_condor(short_put, long_put, short_call, long_call,
                                    "NIFTY", "2026-08-18", lot_size=75, n_lots=1)

        assert spread.net_credit == pytest.approx(120.0)
        assert spread.max_profit == pytest.approx(120.0 * 75)
        # equal 200-point widths -> max_loss driven by either side, same result
        assert spread.max_loss == pytest.approx((200 - 120.0) * 75)
        assert spread.breakevens == [pytest.approx(24180.0), pytest.approx(24820.0)]
        assert len(spread.legs) == 4

    def test_wider_side_drives_max_loss(self):
        short_put = _row(24300, bid=99, ask=101)
        long_put = _row(24100, bid=39, ask=41)     # put width = 200
        short_call = _row(24700, bid=89, ask=91, option_type="CE")
        long_call = _row(25000, bid=19, ask=21, option_type="CE")  # call width = 300

        spread = build_iron_condor(short_put, long_put, short_call, long_call,
                                    "NIFTY", "2026-08-18", lot_size=75, n_lots=1)
        net_credit = 60.0 + 70.0  # put mid diff (100-40) + call mid diff (90-20)
        assert spread.net_credit == pytest.approx(net_credit)
        assert spread.max_loss == pytest.approx((300 - net_credit) * 75)  # call side (wider) dominates
