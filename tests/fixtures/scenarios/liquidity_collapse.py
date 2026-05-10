"""Scenario: Liquidity Collapse.

A small-cap stock (simulated via thin depth) sees bid-ask spread widen to
>2% and order book depth collapse to near-zero on the buy side.  The risk
manager should reject any new BUY orders due to insufficient liquidity /
extreme spread filter.
"""

from __future__ import annotations

from datetime import datetime

from tests.fixtures.historical_bars import make_warmup_bars
from tests.fixtures.macro_data import make_vix_series
from tests.fixtures.news_rss import sample_articles_neutral
from tests.fixtures.upstox_ws_payloads import make_full_tick

_INSTRUMENT_KEY = "NSE_EQ|INE999X99999"   # synthetic illiquid instrument
_SPOT_PAISA = 54_000    # ₹540 — small-cap range
_EXPIRY = "2026-05-15"


def _make_thin_ticks(count: int = 15) -> list[dict]:
    """Return ticks with extremely wide spreads and minimal depth."""
    ticks = []
    base_ts = 1_746_600_000_000  # fixed epoch ms for determinism
    for i in range(count):
        spread_step = int(_SPOT_PAISA * 0.015)  # 1.5% spread
        depth = {
            "buy": [
                {
                    "price": (_SPOT_PAISA - spread_step * (j + 1)) / 100.0,
                    "quantity": max(1, 10 - j * 3),  # thin — only 1–10 lots
                    "orders": 1,
                }
                for j in range(5)
            ],
            "sell": [
                {
                    "price": (_SPOT_PAISA + spread_step * (j + 1)) / 100.0,
                    "quantity": max(1, 10 - j * 3),
                    "orders": 1,
                }
                for j in range(5)
            ],
        }
        tick = make_full_tick(
            instrument_key=_INSTRUMENT_KEY,
            ltp_paisa=_SPOT_PAISA - (i * 50),  # slowly falling
            depth_5_levels=depth,
            ts_epoch_ms=base_ts + i * 2000,
            seed=303,
        )
        ticks.append(tick)
    return ticks


def setup() -> dict:
    """Build the liquidity collapse scenario bundle.

    Returns:
        Scenario dict with keys: name, description, ticks, warmup_bars,
        option_chains, news, macro_state, expected_outcome.
    """
    ticks = _make_thin_ticks(count=15)

    warmup_bars = make_warmup_bars(
        instrument_keys=[_INSTRUMENT_KEY],
        days=10,
        timeframe_minutes=5,
        seed=303,
    )

    macro_state = {
        "vix": 21.5,
        "usdinr": 84.55,
        "crude_usd": 78.00,
        "fii_net_crore": -850.0,
        "dii_net_crore": 200.0,
        "regime": "high",
        "bid_ask_spread_pct": 2.1,
        "top_5_bid_depth_lots": 15,
        "liquidity_score": 0.08,   # 0=illiquid, 1=highly liquid
    }

    start = datetime(2026, 4, 1)
    end = datetime(2026, 4, 28)
    macro_state["vix_series"] = make_vix_series(start, end, regime="high", seed=303).to_dict("records")

    news = sample_articles_neutral(count=2, seed=303)

    return {
        "name": "liquidity_collapse",
        "description": "Bid-ask spread >2% and depth collapse; buy orders must be rejected.",
        "ticks": ticks,
        "warmup_bars": warmup_bars,
        "option_chains": {},
        "news": news,
        "macro_state": macro_state,
        "expected_outcome": {
            "signal_count_min": 0,
            "orders_placed_min": 0,
            "rejection_reason": "insufficient_liquidity",
        },
    }
