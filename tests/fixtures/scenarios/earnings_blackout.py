"""Scenario: Earnings Blackout.

INFY is in a 48-hour pre-earnings blackout window.  The strategy engine
should detect the upcoming earnings event and REJECT any new order on INFY,
regardless of the price signal.

Expectation: signal may fire but order is rejected with reason
``"earnings_blackout"``.
"""

from __future__ import annotations

from datetime import datetime

from tests.fixtures.historical_bars import make_warmup_bars
from tests.fixtures.macro_data import make_vix_series
from tests.fixtures.news_rss import sample_articles_neutral
from tests.fixtures.option_chain import make_option_chain
from tests.fixtures.upstox_ws_payloads import stream_ticks

_INSTRUMENT_KEY = "NSE_EQ|INE009A01021"   # INFY
_SPOT_RUPEES = 1_820.0
_EXPIRY = "2026-05-15"


def setup() -> dict:
    """Build the earnings blackout scenario bundle.

    Returns:
        Scenario dict with keys: name, description, ticks, warmup_bars,
        option_chains, news, macro_state, expected_outcome.
    """
    # Mildly bullish ticks — would normally trigger buy
    ticks = stream_ticks(
        instrument_key=_INSTRUMENT_KEY,
        start_paisa=int(_SPOT_RUPEES * 100),
        count=20,
        drift=0.0008,
        seed=202,
    )

    warmup_bars = make_warmup_bars(
        instrument_keys=[_INSTRUMENT_KEY],
        days=30,
        timeframe_minutes=5,
        seed=202,
    )

    option_chains = {
        "INFY": make_option_chain(
            underlying_symbol="INFY",
            spot_price_rupees=_SPOT_RUPEES,
            expiry_date=_EXPIRY,
            num_strikes_each_side=5,
            seed=202,
        )
    }

    # Neutral news — no directional catalyst
    news = sample_articles_neutral(count=3, seed=202)

    macro_state = {
        "vix": 15.8,
        "usdinr": 84.10,
        "crude_usd": 76.00,
        "fii_net_crore": -120.0,
        "dii_net_crore": 300.0,
        "regime": "neutral",
        # Earnings blackout flag — strategy engine checks this
        "blackout_instruments": [_INSTRUMENT_KEY],
        "earnings_date_iso": "2026-05-10",
        "blackout_hours_before": 48,
    }

    start = datetime(2026, 4, 1)
    end = datetime(2026, 4, 28)
    macro_state["vix_series"] = make_vix_series(start, end, regime="normal", seed=202).to_dict("records")

    return {
        "name": "earnings_blackout",
        "description": "INFY pre-earnings 48h blackout; orders on INFY must be rejected.",
        "ticks": ticks,
        "warmup_bars": warmup_bars,
        "option_chains": option_chains,
        "news": news,
        "macro_state": macro_state,
        "expected_outcome": {
            "signal_count_min": 0,   # signals may or may not fire, orders must not
            "orders_placed_min": 0,
            "rejection_reason": "earnings_blackout",
        },
    }
