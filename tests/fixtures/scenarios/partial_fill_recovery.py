"""Scenario: Partial Fill Recovery.

A limit buy order for SBIN (1,500 shares = 1 lot) is partially filled at
600 shares due to thin liquidity at the limit price.  The order manager
should:
  1. Mark the order as ``PARTIAL``.
  2. Re-queue the remaining 900 shares.
  3. Track open position size correctly (only 600 shares are live).
  4. Eventually fill the remainder or cancel-replace at a slightly higher limit.
"""

from __future__ import annotations

from datetime import datetime

from tests.fixtures.historical_bars import make_warmup_bars
from tests.fixtures.macro_data import make_vix_series
from tests.fixtures.news_rss import sample_articles_neutral
from tests.fixtures.upstox_ws_payloads import make_full_tick, stream_ticks

_INSTRUMENT_KEY = "NSE_EQ|INE062A01020"  # SBIN
_LIMIT_PRICE_PAISA = 85_000              # ₹850.00
_TOTAL_QTY = 1_500
_FILLED_QTY = 600
_REMAINING_QTY = _TOTAL_QTY - _FILLED_QTY
_EXPIRY = "2026-05-15"


def _make_partial_depth_ticks(count: int = 12) -> list[dict]:
    """Return ticks where depth at limit price is only ~600 shares."""
    ticks = []
    base_ts = 1_746_600_000_000
    for i in range(count):
        # Depth: only 200 shares available at each of the first 3 levels
        depth = {
            "buy": [
                {
                    "price": (_LIMIT_PRICE_PAISA - 50 * (j + 1)) / 100.0,
                    "quantity": 200,
                    "orders": 2,
                }
                for j in range(5)
            ],
            "sell": [
                {
                    "price": (_LIMIT_PRICE_PAISA + 50 * (j + 1)) / 100.0,
                    "quantity": 200 + 100 * j,
                    "orders": 3 + j,
                }
                for j in range(5)
            ],
        }
        tick = make_full_tick(
            instrument_key=_INSTRUMENT_KEY,
            ltp_paisa=_LIMIT_PRICE_PAISA + (i * 10),  # price drifts up slowly
            depth_5_levels=depth,
            ts_epoch_ms=base_ts + i * 3_000,
            seed=601,
        )
        ticks.append(tick)
    return ticks


def setup() -> dict:
    """Build the partial fill recovery scenario bundle.

    Returns:
        Scenario dict with keys: name, description, ticks, warmup_bars,
        option_chains, news, macro_state, expected_outcome.
    """
    partial_ticks = _make_partial_depth_ticks(count=12)

    # After partial fill, price recovers — more ticks to eventually complete fill
    recovery_ticks = stream_ticks(
        instrument_key=_INSTRUMENT_KEY,
        start_paisa=_LIMIT_PRICE_PAISA + 120,
        count=10,
        drift=0.0003,
        seed=602,
    )

    all_ticks = partial_ticks + recovery_ticks

    warmup_bars = make_warmup_bars(
        instrument_keys=[_INSTRUMENT_KEY],
        days=20,
        timeframe_minutes=5,
        seed=601,
    )

    news = sample_articles_neutral(count=2, seed=601)

    macro_state = {
        "vix": 16.4,
        "usdinr": 84.20,
        "crude_usd": 75.60,
        "fii_net_crore": 280.0,
        "dii_net_crore": 420.0,
        "regime": "neutral",
        "order_status": "PARTIAL",
        "filled_qty": _FILLED_QTY,
        "remaining_qty": _REMAINING_QTY,
        "limit_price_paisa": _LIMIT_PRICE_PAISA,
    }

    start = datetime(2026, 4, 1)
    end = datetime(2026, 4, 28)
    macro_state["vix_series"] = make_vix_series(start, end, regime="normal", seed=601).to_dict("records")

    return {
        "name": "partial_fill_recovery",
        "description": "Limit order partially filled (600/1500); remainder re-queued for recovery.",
        "ticks": all_ticks,
        "warmup_bars": warmup_bars,
        "option_chains": {},
        "news": news,
        "macro_state": macro_state,
        "expected_outcome": {
            "signal_count_min": 0,
            "orders_placed_min": 1,
            "rejection_reason": None,
        },
    }
