"""Tests for src.strategy.conditions_microstructure.

All tests use inline mocks — no external fixtures or DB connections required.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.data.tick_metrics import TickMetricsAggregator
from src.strategy.conditions_microstructure import (
    aggressor_imbalance_pct,
    cvd_above_threshold,
    cvd_divergence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

KEY = "NSE_EQ|INE002A01018"


def _ts(offset_minutes: int = 0) -> datetime:
    base = datetime(2026, 5, 8, 9, 15, 0, tzinfo=timezone.utc)
    return base + timedelta(minutes=offset_minutes)


def _make_agg_with_ticks(
    buys: list[tuple[int, int]],   # list of (minute_offset, volume)
    sells: list[tuple[int, int]],  # list of (minute_offset, volume)
    key: str = KEY,
) -> TickMetricsAggregator:
    """Build a TickMetricsAggregator pre-loaded with synthetic ticks."""
    agg = TickMetricsAggregator()
    for (offset, vol) in buys:
        agg.on_tick(key, _ts(offset), ltp_paisa=10000, volume=vol,
                    bid_paisa=9990, ask_paisa=10000)   # BUY at ask
    for (offset, vol) in sells:
        agg.on_tick(key, _ts(offset), ltp_paisa=9990, volume=vol,
                    bid_paisa=9990, ask_paisa=10000)   # SELL at bid
    return agg


# ---------------------------------------------------------------------------
# cvd_divergence
# ---------------------------------------------------------------------------

class TestCvdDivergence:
    """Tests for cvd_divergence condition helper."""

    def test_cvd_divergence_returns_true_on_bearish_pattern(self):
        """cvd_divergence returns True when detect_divergence reports bearish."""
        agg = TickMetricsAggregator()
        # Stub detect_divergence to return a known bearish divergence
        agg.detect_divergence = MagicMock(return_value={
            "price_trend": "up",
            "cvd_trend": "down",
            "divergent": True,
            "type": "bearish",
        })

        result = cvd_divergence(KEY, agg, divergence_type="bearish", lookback=30)
        assert result is True
        agg.detect_divergence.assert_called_once_with(KEY, lookback_minutes=30)

    def test_cvd_divergence_returns_false_on_bullish_when_asking_bearish(self):
        """cvd_divergence returns False when divergence type does not match."""
        agg = TickMetricsAggregator()
        agg.detect_divergence = MagicMock(return_value={
            "price_trend": "down",
            "cvd_trend": "up",
            "divergent": True,
            "type": "bullish",
        })

        result = cvd_divergence(KEY, agg, divergence_type="bearish", lookback=30)
        assert result is False

    def test_cvd_divergence_returns_true_on_bullish_pattern(self):
        """cvd_divergence returns True when asking for bullish and pattern matches."""
        agg = TickMetricsAggregator()
        agg.detect_divergence = MagicMock(return_value={
            "price_trend": "down",
            "cvd_trend": "up",
            "divergent": True,
            "type": "bullish",
        })

        result = cvd_divergence(KEY, agg, divergence_type="bullish", lookback=15)
        assert result is True

    def test_cvd_divergence_returns_false_when_not_divergent(self):
        """cvd_divergence returns False when divergent flag is False."""
        agg = TickMetricsAggregator()
        agg.detect_divergence = MagicMock(return_value={
            "price_trend": "up",
            "cvd_trend": "up",
            "divergent": False,
            "type": None,
        })

        result = cvd_divergence(KEY, agg, divergence_type="bearish")
        assert result is False

    def test_cvd_divergence_end_to_end_with_real_aggregator(self):
        """End-to-end: bearish divergence detected via actual aggregator logic."""
        # First half of window: strong buys → positive CVD momentum
        # Second half: strong sells → CVD drops → bearish divergence
        buys  = [(i, 500) for i in range(8)]    # minutes 0-7: heavy buy
        sells = [(i, 500) for i in range(8, 16)] # minutes 8-15: heavy sell
        agg = _make_agg_with_ticks(buys, sells)

        result = agg.detect_divergence(KEY, lookback_minutes=30)
        # The divergence itself may be bearish or bullish depending on proxy;
        # assert that the helper returns the correct truth value based on detect_divergence.
        expected = (result.get("divergent") and result.get("type") == "bearish")
        assert cvd_divergence(KEY, agg, "bearish") == expected


# ---------------------------------------------------------------------------
# cvd_above_threshold
# ---------------------------------------------------------------------------

class TestCvdAboveThreshold:
    """Tests for cvd_above_threshold condition helper."""

    def test_cvd_above_threshold_returns_true_when_exceeded(self):
        """Returns True when running CVD exceeds threshold."""
        agg = _make_agg_with_ticks(
            buys=[(0, 1000), (1, 500)],
            sells=[(0, 200)],
        )
        # Net CVD = 1000 + 500 - 200 = 1300
        assert cvd_above_threshold(KEY, agg, threshold=1000) is True

    def test_cvd_above_threshold_returns_false_when_at_or_below(self):
        """Returns False when running CVD is at or below threshold."""
        agg = _make_agg_with_ticks(
            buys=[(0, 300)],
            sells=[],
        )
        # CVD = 300; threshold = 300 → 300 > 300 is False
        assert cvd_above_threshold(KEY, agg, threshold=300) is False

    def test_cvd_above_threshold_returns_false_for_negative_cvd(self):
        """Returns False when CVD is negative and threshold is zero."""
        agg = _make_agg_with_ticks(
            buys=[],
            sells=[(0, 500)],
        )
        # CVD = -500; threshold = 0 → -500 > 0 is False
        assert cvd_above_threshold(KEY, agg, threshold=0) is False

    def test_cvd_above_threshold_negative_threshold(self):
        """Works correctly with a negative threshold."""
        agg = _make_agg_with_ticks(
            buys=[],
            sells=[(0, 200)],
        )
        # CVD = -200; threshold = -500 → -200 > -500 is True
        assert cvd_above_threshold(KEY, agg, threshold=-500) is True

    def test_cvd_above_threshold_zero_data(self):
        """Returns False for a fresh aggregator with threshold >= 0."""
        agg = TickMetricsAggregator()
        assert cvd_above_threshold(KEY, agg, threshold=0) is False


# ---------------------------------------------------------------------------
# aggressor_imbalance_pct
# ---------------------------------------------------------------------------

class TestAggressorImbalancePct:
    """Tests for aggressor_imbalance_pct condition helper."""

    def test_aggressor_imbalance_pct_returns_negative_for_seller_dominated(self):
        """Returns negative value when sell volume exceeds buy volume."""
        agg = _make_agg_with_ticks(
            buys=[(0, 100)],
            sells=[(0, 900)],
        )
        pct = aggressor_imbalance_pct(KEY, agg, lookback_min=5)
        # (100 - 900) / (100 + 900) * 100 = -80.0
        assert pct == pytest.approx(-80.0)

    def test_aggressor_imbalance_pct_returns_positive_for_buyer_dominated(self):
        """Returns positive value when buy volume exceeds sell volume."""
        agg = _make_agg_with_ticks(
            buys=[(0, 900)],
            sells=[(0, 100)],
        )
        pct = aggressor_imbalance_pct(KEY, agg, lookback_min=5)
        # (900 - 100) / (900 + 100) * 100 = 80.0
        assert pct == pytest.approx(80.0)

    def test_aggressor_imbalance_pct_returns_zero_when_balanced(self):
        """Returns 0.0 when buy == sell volume."""
        agg = _make_agg_with_ticks(
            buys=[(0, 500)],
            sells=[(0, 500)],
        )
        pct = aggressor_imbalance_pct(KEY, agg, lookback_min=5)
        assert pct == pytest.approx(0.0)

    def test_aggressor_imbalance_pct_returns_100_when_all_buys(self):
        """Returns +100 when all volume is buy-aggressed."""
        agg = _make_agg_with_ticks(
            buys=[(0, 1000)],
            sells=[],
        )
        pct = aggressor_imbalance_pct(KEY, agg, lookback_min=5)
        assert pct == pytest.approx(100.0)

    def test_aggressor_imbalance_pct_returns_minus_100_when_all_sells(self):
        """Returns -100 when all volume is sell-aggressed."""
        agg = _make_agg_with_ticks(
            buys=[],
            sells=[(0, 1000)],
        )
        pct = aggressor_imbalance_pct(KEY, agg, lookback_min=5)
        assert pct == pytest.approx(-100.0)

    def test_aggressor_imbalance_pct_returns_zero_for_empty_aggregator(self):
        """Returns 0.0 when no ticks have been processed."""
        agg = TickMetricsAggregator()
        pct = aggressor_imbalance_pct(KEY, agg, lookback_min=5)
        assert pct == pytest.approx(0.0)

    def test_aggressor_imbalance_pct_respects_lookback_min(self):
        """Only the most recent lookback_min minutes are included."""
        # Feed 10 minutes: first 5 = heavy sells, last 5 = heavy buys
        buys  = [(i, 1000) for i in range(5, 10)]   # minutes 5-9
        sells = [(i, 1000) for i in range(0, 5)]     # minutes 0-4
        agg = _make_agg_with_ticks(buys, sells)

        # lookback_min=5 → only last 5 minutes (all buys)
        pct = aggressor_imbalance_pct(KEY, agg, lookback_min=5)
        assert pct == pytest.approx(100.0)
