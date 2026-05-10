"""Scenario: WebSocket Connection Drop.

The market data feed drops mid-session (simulated by a gap in tick
timestamps of >30 seconds).  The feed handler must:
  1. Detect the stale-data condition.
  2. Trigger a reconnection attempt.
  3. Re-subscribe to the same instruments.
  4. NOT emit any trading signal during the gap period.
  5. Resume normal operation after reconnect.

The scenario provides pre-drop ticks, a simulated drop gap, and post-reconnect
ticks so test harnesses can validate reconnect logic end-to-end.
"""

from __future__ import annotations

import time
from datetime import datetime

from tests.fixtures.historical_bars import make_warmup_bars
from tests.fixtures.macro_data import make_vix_series
from tests.fixtures.news_rss import sample_articles_neutral
from tests.fixtures.upstox_ws_payloads import make_ltpc_tick, stream_ticks

_INSTRUMENT_KEY = "NSE_EQ|INE238A01034"   # AXISBANK
_SPOT_PAISA = 120_000   # ₹1,200


def setup() -> dict:
    """Build the connection drop scenario bundle.

    Returns:
        Scenario dict with keys: name, description, ticks, warmup_bars,
        option_chains, news, macro_state, expected_outcome.

        ``ticks`` is a list where a sentinel dict
        ``{"event": "connection_drop", "duration_ms": 45_000}``
        separates pre-drop and post-reconnect ticks.
    """
    base_ts = 1_746_600_000_000

    # Pre-drop: 8 ticks at 500 ms intervals
    pre_drop_ticks = []
    price = _SPOT_PAISA
    for i in range(8):
        price = price + 50 * (1 if i % 3 != 0 else -1)
        pre_drop_ticks.append(
            make_ltpc_tick(_INSTRUMENT_KEY, price, base_ts + i * 500, seed=701)
        )

    # Sentinel marking the simulated connection drop
    drop_sentinel = {
        "event": "connection_drop",
        "duration_ms": 45_000,  # 45-second outage
        "ts_start": base_ts + 8 * 500,
        "ts_end": base_ts + 8 * 500 + 45_000,
    }

    # Post-reconnect: 10 ticks resuming after the gap
    reconnect_base_ts = base_ts + 8 * 500 + 45_000
    post_reconnect_ticks = stream_ticks(
        instrument_key=_INSTRUMENT_KEY,
        start_paisa=price,
        count=10,
        drift=0.0004,
        seed=702,
    )
    # Patch timestamps to be sequential after the gap
    for i, tick in enumerate(post_reconnect_ticks):
        tick["currentTs"] = reconnect_base_ts + i * 500
        for v in tick["feeds"].values():
            if "ltpc" in v:
                v["ltpc"]["ltt"] = reconnect_base_ts + i * 500

    all_ticks = pre_drop_ticks + [drop_sentinel] + post_reconnect_ticks

    warmup_bars = make_warmup_bars(
        instrument_keys=[_INSTRUMENT_KEY],
        days=15,
        timeframe_minutes=5,
        seed=701,
    )

    news = sample_articles_neutral(count=2, seed=701)

    macro_state = {
        "vix": 14.9,
        "usdinr": 83.95,
        "crude_usd": 75.00,
        "fii_net_crore": 120.0,
        "dii_net_crore": 350.0,
        "regime": "normal",
        "connection_drop_duration_ms": 45_000,
        "reconnect_expected": True,
        "stale_data_threshold_ms": 30_000,
    }

    start = datetime(2026, 4, 1)
    end = datetime(2026, 4, 28)
    macro_state["vix_series"] = make_vix_series(start, end, regime="normal", seed=701).to_dict("records")

    return {
        "name": "connection_drop",
        "description": "WS feed drops for 45s mid-session; handler must reconnect and resume.",
        "ticks": all_ticks,
        "warmup_bars": warmup_bars,
        "option_chains": {},
        "news": news,
        "macro_state": macro_state,
        "expected_outcome": {
            "signal_count_min": 0,
            "orders_placed_min": 0,
            "rejection_reason": "stale_feed",
        },
    }
