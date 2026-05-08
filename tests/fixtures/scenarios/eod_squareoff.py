"""Scenario: End-of-Day Square-Off (MIS positions).

At 3:14 PM IST, the bot has 2 open intraday (MIS) positions — one in NIFTY
futures and one in ICICIBANK equity.  The EOD square-off routine must:
  1. Detect remaining MIS positions at 3:14 PM.
  2. Place market SELL orders to close both positions before 3:15 PM.
  3. Mark all positions as CLOSED.
  4. NOT enter any new positions after 3:10 PM.
"""

from __future__ import annotations

from datetime import datetime

from tests.fixtures.historical_bars import make_warmup_bars
from tests.fixtures.macro_data import make_vix_series
from tests.fixtures.news_rss import sample_articles_neutral
from tests.fixtures.upstox_ws_payloads import make_ltpc_tick

_NIFTY_KEY = "NSE_INDEX|Nifty 50"
_ICICI_KEY = "NSE_EQ|INE090A01021"

# Simulate ticks arriving at 3:14 PM IST
# epoch ms for 2026-04-28 15:14:00 IST = 2026-04-28 09:44:00 UTC
_EOD_TS_MS = 1_745_825_040_000   # approximate, fixed for determinism


def setup() -> dict:
    """Build the EOD square-off scenario bundle.

    Returns:
        Scenario dict with keys: name, description, ticks, warmup_bars,
        option_chains, news, macro_state, expected_outcome.
    """
    # A handful of ticks at 3:14 PM — market is winding down
    ticks = []
    for i in range(6):
        ts = _EOD_TS_MS + i * 5_000  # 5-second spacing
        # NIFTY slowly drifting down as participants square off
        ticks.append(make_ltpc_tick(_NIFTY_KEY, 2_445_000 - i * 200, ts, seed=801))
        # ICICI similarly drifting
        ticks.append(make_ltpc_tick(_ICICI_KEY, 128_000 - i * 100, ts + 100, seed=802))

    warmup_bars = make_warmup_bars(
        instrument_keys=[_NIFTY_KEY, _ICICI_KEY],
        days=15,
        timeframe_minutes=5,
        seed=801,
    )

    news = sample_articles_neutral(count=2, seed=801)

    macro_state = {
        "vix": 14.5,
        "usdinr": 83.90,
        "crude_usd": 74.80,
        "fii_net_crore": 350.0,
        "dii_net_crore": 180.0,
        "regime": "neutral",
        "time_ist": "15:14:00",
        "eod_squareoff_trigger_time": "15:15:00",
        "open_mis_positions": [
            {
                "instrument_key": _NIFTY_KEY,
                "qty": 50,
                "side": "BUY",
                "avg_price_paisa": 2_448_000,
                "product": "MIS",
            },
            {
                "instrument_key": _ICICI_KEY,
                "qty": 1_375,
                "side": "BUY",
                "avg_price_paisa": 128_500,
                "product": "MIS",
            },
        ],
    }

    start = datetime(2026, 4, 1)
    end = datetime(2026, 4, 28)
    macro_state["vix_series"] = make_vix_series(start, end, regime="normal", seed=801).to_dict("records")

    return {
        "name": "eod_squareoff",
        "description": "3:14 PM IST — 2 open MIS positions must be squared off before 3:15 PM.",
        "ticks": ticks,
        "warmup_bars": warmup_bars,
        "option_chains": {},
        "news": news,
        "macro_state": macro_state,
        "expected_outcome": {
            "signal_count_min": 0,
            "orders_placed_min": 2,   # 2 market SELL orders for squareoff
            "rejection_reason": None,
        },
    }
