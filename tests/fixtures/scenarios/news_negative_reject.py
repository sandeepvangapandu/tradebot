"""Scenario: Negative News — Order Rejection.

A negative news article about HDFCBANK (audit findings, regulatory probe)
hits the feed during the trading session.  The news sentiment filter should:
  1. Classify the article as negative (sentiment_score < -0.5).
  2. Suppress any pending BUY signals for HDFCBANK.
  3. If a BUY order is already queued, cancel it.
  4. Optionally trigger a SELL/exit if an open position exists.

Expected rejection reason: ``"negative_news_filter"``.
"""

from __future__ import annotations

from datetime import datetime

from tests.fixtures.historical_bars import make_warmup_bars
from tests.fixtures.macro_data import make_vix_series
from tests.fixtures.news_rss import make_rss_xml, sample_articles_negative_for, sample_articles_neutral
from tests.fixtures.option_chain import make_option_chain
from tests.fixtures.upstox_ws_payloads import stream_ticks

_INSTRUMENT_KEY = "NSE_EQ|INE040A01034"   # HDFCBANK
_SPOT_RUPEES = 1_610.0
_EXPIRY = "2026-05-15"


def setup() -> dict:
    """Build the negative news rejection scenario bundle.

    Returns:
        Scenario dict with keys: name, description, ticks, warmup_bars,
        option_chains, news, macro_state, expected_outcome.
    """
    # Price action is modestly bullish — would normally trigger a BUY signal
    ticks = stream_ticks(
        instrument_key=_INSTRUMENT_KEY,
        start_paisa=int(_SPOT_RUPEES * 100),
        count=20,
        drift=0.0006,
        seed=901,
    )

    warmup_bars = make_warmup_bars(
        instrument_keys=[_INSTRUMENT_KEY],
        days=25,
        timeframe_minutes=5,
        seed=901,
    )

    option_chains = {
        "HDFCBANK": make_option_chain(
            underlying_symbol="HDFCBANK",
            spot_price_rupees=_SPOT_RUPEES,
            expiry_date=_EXPIRY,
            num_strikes_each_side=5,
            seed=901,
        )
    }

    # The trigger: 3 strongly negative articles about HDFCBANK
    negative_articles = sample_articles_negative_for("HDFCBANK", count=3, seed=901)
    # Plus neutral background noise
    neutral_articles = sample_articles_neutral(count=2, seed=902)
    all_articles = negative_articles + neutral_articles

    # Also expose the RSS XML for feed parsers
    rss_xml = make_rss_xml(all_articles)

    macro_state = {
        "vix": 17.3,
        "usdinr": 84.30,
        "crude_usd": 76.80,
        "fii_net_crore": -320.0,
        "dii_net_crore": 600.0,
        "regime": "neutral",
        "negative_news_symbols": ["HDFCBANK"],
        "news_sentiment_score": -0.72,   # strongly negative
        "news_rss_xml": rss_xml,
    }

    start = datetime(2026, 4, 1)
    end = datetime(2026, 4, 28)
    macro_state["vix_series"] = make_vix_series(start, end, regime="normal", seed=901).to_dict("records")

    return {
        "name": "news_negative_reject",
        "description": "Negative HDFCBANK news (sentiment -0.72) must suppress BUY orders.",
        "ticks": ticks,
        "warmup_bars": warmup_bars,
        "option_chains": option_chains,
        "news": all_articles,
        "macro_state": macro_state,
        "expected_outcome": {
            "signal_count_min": 0,
            "orders_placed_min": 0,
            "rejection_reason": "negative_news_filter",
        },
    }
