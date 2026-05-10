"""Scenario: Sector Rotation — IT Outperforms.

Broad market is flat but IT sector (INFY, TCS) is outperforming NIFTY by
+1.5%.  FII flows show selective buying in IT.  The strategy should:
  1. Detect relative strength of IT vs. benchmark.
  2. Generate BUY signals for INFY and/or TCS.
  3. NOT generate signals for lagging sectors (SBIN, ITC, AXISBANK).
"""

from __future__ import annotations

from datetime import datetime

from tests.fixtures.historical_bars import make_ohlcv_bars, make_warmup_bars
from tests.fixtures.macro_data import make_fii_dii_flows, make_vix_series
from tests.fixtures.news_rss import sample_articles_positive_for, sample_articles_neutral
from tests.fixtures.option_chain import make_option_chain
from tests.fixtures.upstox_ws_payloads import stream_ticks

# IT winners
_INFY_KEY = "NSE_EQ|INE009A01021"
_TCS_KEY  = "NSE_EQ|INE467B01029"

# Laggards
_SBIN_KEY = "NSE_EQ|INE062A01020"
_ITC_KEY  = "NSE_EQ|INE154A01025"
_AXISBANK_KEY = "NSE_EQ|INE238A01034"

# Benchmark
_NIFTY_KEY = "NSE_INDEX|Nifty 50"

_EXPIRY = "2026-05-15"


def setup() -> dict:
    """Build the sector rotation winner scenario bundle.

    Returns:
        Scenario dict with keys: name, description, ticks, warmup_bars,
        option_chains, news, macro_state, expected_outcome.
    """
    # INFY and TCS rip higher (+1.5% drift), benchmark flat, laggards slightly down
    infy_ticks = stream_ticks(_INFY_KEY, 183_000_00, 20, drift=0.0010, seed=1001)  # note: paisa
    tcs_ticks  = stream_ticks(_TCS_KEY,  372_000_00, 20, drift=0.0009, seed=1002)
    nifty_ticks = stream_ticks(_NIFTY_KEY, 2_444_000, 20, drift=0.0001, seed=1003)
    sbin_ticks  = stream_ticks(_SBIN_KEY,   85_000,   20, drift=-0.0003, seed=1004)
    itc_ticks   = stream_ticks(_ITC_KEY,    44_500,   20, drift=-0.0002, seed=1005)

    # Fix: INFY/TCS start_paisa should be in paisa (not double-paisa)
    infy_ticks = stream_ticks(_INFY_KEY, 183_000, 20, drift=0.0010, seed=1001)
    tcs_ticks  = stream_ticks(_TCS_KEY,  372_000, 20, drift=0.0009, seed=1002)

    all_ticks = infy_ticks + tcs_ticks + nifty_ticks + sbin_ticks + itc_ticks

    warmup_bars = make_warmup_bars(
        instrument_keys=[
            _INFY_KEY, _TCS_KEY, _NIFTY_KEY,
            _SBIN_KEY, _ITC_KEY, _AXISBANK_KEY,
        ],
        days=30,
        timeframe_minutes=5,
        seed=1001,
    )

    option_chains = {
        "INFY": make_option_chain("INFY", 1_830.0, _EXPIRY, num_strikes_each_side=5, seed=1001),
        "TCS":  make_option_chain("TCS",  3_720.0, _EXPIRY, num_strikes_each_side=5, seed=1002),
    }

    # Positive news for IT, neutral for rest
    news = (
        sample_articles_positive_for("INFY", count=2, seed=1001)
        + sample_articles_positive_for("TCS",  count=2, seed=1002)
        + sample_articles_neutral(count=3, seed=1003)
    )

    macro_state = {
        "vix": 13.5,
        "usdinr": 83.70,
        "crude_usd": 74.20,
        "fii_net_crore": 1_800.0,
        "dii_net_crore": 200.0,
        "regime": "bullish",
        "sector_performance": {
            "IT": +1.52,        # pct vs benchmark
            "PSU_BANKING": -0.45,
            "FMCG": -0.18,
            "PRIVATE_BANKING": +0.10,
        },
        "relative_strength_leaders": ["INFY", "TCS"],
        "relative_strength_laggards": ["SBIN", "ITC", "AXISBANK"],
    }

    start = datetime(2026, 4, 1)
    end = datetime(2026, 4, 28)
    macro_state["vix_series"] = make_vix_series(start, end, regime="low", seed=1001).to_dict("records")
    macro_state["fii_dii_series"] = make_fii_dii_flows(start, end, regime="bullish", seed=1001).to_dict("records")

    return {
        "name": "sector_rotation_winner",
        "description": "IT outperforms +1.5%; BUY INFY/TCS expected, laggards skipped.",
        "ticks": all_ticks,
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
