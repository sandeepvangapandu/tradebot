"""Scenario: VIX Spike Regime Change.

India VIX surges from 17 to 38 intraday following a global risk-off event
(e.g. Fed emergency meeting).  The strategy engine should detect the regime
shift and:
  1. Cancel all pending limit orders.
  2. Tighten stop-losses on open positions.
  3. Halt new entries until VIX stabilises below the configured threshold.
"""

from __future__ import annotations

from datetime import datetime

from tests.fixtures.historical_bars import make_warmup_bars
from tests.fixtures.macro_data import make_fii_dii_flows, make_vix_series
from tests.fixtures.news_rss import sample_articles_negative_for
from tests.fixtures.option_chain import make_option_chain
from tests.fixtures.upstox_ws_payloads import stream_ticks

_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"
_SPOT_RUPEES_PRE_SPIKE = 24_400.0
_EXPIRY = "2026-05-15"


def setup() -> dict:
    """Build the VIX spike regime change scenario bundle.

    Returns:
        Scenario dict with keys: name, description, ticks, warmup_bars,
        option_chains, news, macro_state, expected_outcome.
    """
    # Phase 1: calm market (10 ticks)
    calm_ticks = stream_ticks(
        instrument_key=_INSTRUMENT_KEY,
        start_paisa=int(_SPOT_RUPEES_PRE_SPIKE * 100),
        count=10,
        drift=0.0002,
        seed=501,
    )

    # Phase 2: spike — 15 ticks with strong downward drift
    spike_ticks = stream_ticks(
        instrument_key=_INSTRUMENT_KEY,
        start_paisa=int(_SPOT_RUPEES_PRE_SPIKE * 100 * 0.985),
        count=15,
        drift=-0.0035,   # aggressive sell-off
        seed=502,
    )

    all_ticks = calm_ticks + spike_ticks

    warmup_bars = make_warmup_bars(
        instrument_keys=[_INSTRUMENT_KEY],
        days=30,
        timeframe_minutes=5,
        seed=501,
    )

    option_chains = {
        "NIFTY": make_option_chain(
            underlying_symbol="NIFTY",
            spot_price_rupees=_SPOT_RUPEES_PRE_SPIKE * 0.982,
            expiry_date=_EXPIRY,
            num_strikes_each_side=7,
            seed=501,
        )
    }

    # Negative news headlines driving the selloff
    news = (
        sample_articles_negative_for("NIFTY", count=3, seed=501)
        + sample_articles_negative_for("BANKNIFTY", count=2, seed=502)
    )

    macro_state = {
        "vix": 38.4,               # current VIX (post-spike value for risk checks)
        "vix_pre_spike": 17.2,
        "vix_post_spike": 38.4,
        "vix_threshold_halt": 30.0,   # strategy halts new entries above this
        "usdinr": 85.10,
        "crude_usd": 81.50,
        "fii_net_crore": -4_200.0,
        "dii_net_crore": 1_800.0,
        "regime": "spike",
        "halt_new_entries": True,
        "tighten_stops": True,
    }

    start = datetime(2026, 4, 1)
    end = datetime(2026, 4, 28)
    macro_state["vix_series"] = make_vix_series(start, end, regime="spike", seed=501).to_dict("records")
    macro_state["fii_dii_series"] = make_fii_dii_flows(start, end, regime="panic", seed=501).to_dict("records")

    return {
        "name": "vix_spike_regime",
        "description": "VIX spikes 17→38; strategy must halt new entries and tighten stops.",
        "ticks": all_ticks,
        "warmup_bars": warmup_bars,
        "option_chains": option_chains,
        "news": news,
        "macro_state": macro_state,
        "expected_outcome": {
            "signal_count_min": 0,
            "orders_placed_min": 0,
            "rejection_reason": "vix_above_threshold",
        },
    }
