"""Scenario: Bull Open Breakout.

NIFTY gaps up +0.6% at open, sustains above prior-day high.  Strong FII
buying overnight. ORB / VWAP breakout strategies should trigger a BUY signal
within the first 15 minutes.
"""

from __future__ import annotations

from datetime import datetime

from tests.fixtures.historical_bars import make_warmup_bars
from tests.fixtures.macro_data import make_fii_dii_flows, make_vix_series
from tests.fixtures.news_rss import sample_articles_positive_for
from tests.fixtures.option_chain import make_option_chain
from tests.fixtures.upstox_ws_payloads import stream_ticks

_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"
_SPOT_RUPEES = 24_650.0          # +0.6% gap-up from prev close of 24,500
_EXPIRY = "2026-05-15"


def setup() -> dict:
    """Build the bull open breakout scenario bundle.

    Returns:
        Scenario dict with keys: name, description, ticks, warmup_bars,
        option_chains, news, macro_state, expected_outcome.
    """
    # 30 upward-biased ticks over first 15 minutes
    ticks = stream_ticks(
        instrument_key=_INSTRUMENT_KEY,
        start_paisa=int(_SPOT_RUPEES * 100),
        count=30,
        drift=0.0015,   # strong bullish drift
        seed=101,
    )

    warmup_bars = make_warmup_bars(
        instrument_keys=[_INSTRUMENT_KEY, "NSE_EQ|INE002A01018"],
        days=30,
        timeframe_minutes=5,
        seed=101,
    )

    option_chains = {
        "NIFTY": make_option_chain(
            underlying_symbol="NIFTY",
            spot_price_rupees=_SPOT_RUPEES,
            expiry_date=_EXPIRY,
            num_strikes_each_side=5,
            seed=101,
        )
    }

    news = sample_articles_positive_for("NIFTY", count=3, seed=101)

    macro_state = {
        "vix": 13.2,
        "usdinr": 83.80,
        "crude_usd": 74.50,
        "fii_net_crore": 2_150.0,
        "dii_net_crore": 430.0,
        "regime": "bullish",
        "market_open_gap_pct": 0.60,
    }

    start = datetime(2026, 4, 1)
    end = datetime(2026, 4, 28)
    macro_state["vix_series"] = make_vix_series(start, end, regime="low", seed=101).to_dict("records")
    macro_state["fii_dii_series"] = make_fii_dii_flows(start, end, regime="bullish", seed=101).to_dict("records")

    return {
        "name": "bull_open_breakout",
        "description": "NIFTY gaps up 0.6%; ORB/VWAP breakout buy expected within 15 min.",
        "ticks": ticks,
        "warmup_bars": warmup_bars,
        "option_chains": option_chains,
        "news": news,
        "macro_state": macro_state,
        "expected_outcome": {
            "signal_count_min": 1,
            "orders_placed_min": 1,
            "rejection_reason": None,
        },
    }
