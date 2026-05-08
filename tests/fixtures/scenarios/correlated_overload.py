"""Scenario: Correlated Portfolio Overload.

All five instruments in the watchlist (RELIANCE, HDFCBANK, ICICIBANK, TCS,
INFY) simultaneously emit BUY signals due to a broad market surge.  The risk
manager should enforce the max-open-positions limit (default: 5) and the
max-capital-deployment limit (default: 80%).  New orders beyond the cap must
be queued or rejected with ``"max_positions_reached"``.
"""

from __future__ import annotations

from datetime import datetime

from tests.fixtures.historical_bars import make_warmup_bars
from tests.fixtures.macro_data import make_fii_dii_flows, make_vix_series
from tests.fixtures.news_rss import sample_articles_positive_for
from tests.fixtures.option_chain import make_option_chain
from tests.fixtures.upstox_ws_payloads import stream_ticks

_INSTRUMENTS = [
    ("NSE_EQ|INE002A01018", "RELIANCE", 290_000, 250),
    ("NSE_EQ|INE040A01034", "HDFCBANK",  160_000, 550),
    ("NSE_EQ|INE090A01021", "ICICIBANK", 130_000, 1375),
    ("NSE_EQ|INE467B01029", "TCS",       370_000, 150),
    ("NSE_EQ|INE009A01021", "INFY",      180_000, 300),
]
_EXPIRY = "2026-05-15"


def setup() -> dict:
    """Build the correlated overload scenario bundle.

    Returns:
        Scenario dict with keys: name, description, ticks, warmup_bars,
        option_chains, news, macro_state, expected_outcome.
    """
    all_ticks: list[dict] = []
    for i, (key, sym, base_paisa, _lot) in enumerate(_INSTRUMENTS):
        ticks = stream_ticks(
            instrument_key=key,
            start_paisa=base_paisa,
            count=10,
            drift=0.0012,
            seed=400 + i,
        )
        all_ticks.extend(ticks)

    instrument_keys = [row[0] for row in _INSTRUMENTS]
    warmup_bars = make_warmup_bars(
        instrument_keys=instrument_keys,
        days=20,
        timeframe_minutes=5,
        seed=400,
    )

    option_chains = {
        sym: make_option_chain(
            underlying_symbol=sym,
            spot_price_rupees=base_paisa / 100.0,
            expiry_date=_EXPIRY,
            num_strikes_each_side=3,
            seed=400 + i,
        )
        for i, (_key, sym, base_paisa, _lot) in enumerate(_INSTRUMENTS)
    }

    news = []
    for _key, sym, _p, _l in _INSTRUMENTS:
        news.extend(sample_articles_positive_for(sym, count=1, seed=400))

    macro_state = {
        "vix": 12.8,
        "usdinr": 83.60,
        "crude_usd": 73.80,
        "fii_net_crore": 3_200.0,
        "dii_net_crore": 800.0,
        "regime": "bullish",
        "simultaneous_signals": len(_INSTRUMENTS),
        "max_positions_config": 5,
        "max_capital_deployment_pct": 80,
    }

    start = datetime(2026, 4, 1)
    end = datetime(2026, 4, 28)
    macro_state["vix_series"] = make_vix_series(start, end, regime="low", seed=400).to_dict("records")
    macro_state["fii_dii_series"] = make_fii_dii_flows(start, end, regime="bullish", seed=400).to_dict("records")

    return {
        "name": "correlated_overload",
        "description": "5 simultaneous BUY signals; max-positions cap must block excess orders.",
        "ticks": all_ticks,
        "warmup_bars": warmup_bars,
        "option_chains": option_chains,
        "news": news,
        "macro_state": macro_state,
        "expected_outcome": {
            "signal_count_min": 5,
            "orders_placed_min": 1,   # at least 1, but no more than 5
            "rejection_reason": "max_positions_reached",
        },
    }
