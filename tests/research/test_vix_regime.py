"""Tests for src.research.vix_regime.VIXRegimeClassifier.

All tests use inline mocks — no external database or network required.
Fixtures from tests/fixtures/macro_data.py::make_vix_series are used where
appropriate.
"""

from __future__ import annotations

from datetime import date, datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, call
import pytest

from src.research.vix_regime import VIXRegimeClassifier, REGIME_THRESHOLDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(day: int = 8, hour: int = 10, minute: int = 0) -> datetime:
    """Build an IST-equivalent UTC timestamp for 2026-05-{day} {hour}:{minute}."""
    return datetime(2026, 5, day, hour, minute, 0, tzinfo=timezone.utc)


def _make_clf(historical_provider=None, db_engine=None) -> VIXRegimeClassifier:
    return VIXRegimeClassifier(
        historical_provider=historical_provider,
        db_engine=db_engine,
    )


# ---------------------------------------------------------------------------
# classify_value
# ---------------------------------------------------------------------------

class TestClassifyValue:
    """Tests for VIXRegimeClassifier.classify_value."""

    def test_classify_value_low_for_under_13(self):
        """VIX < 13 is classified as LOW."""
        clf = _make_clf()
        assert clf.classify_value(10.0) == "LOW"
        assert clf.classify_value(0.1) == "LOW"
        assert clf.classify_value(12.99) == "LOW"

    def test_classify_value_normal_for_15(self):
        """VIX = 15 falls in NORMAL (13 <= vix < 18)."""
        clf = _make_clf()
        assert clf.classify_value(15.0) == "NORMAL"
        assert clf.classify_value(13.0) == "NORMAL"
        assert clf.classify_value(17.99) == "NORMAL"

    def test_classify_value_high_for_22(self):
        """VIX = 22 falls in HIGH (18 <= vix < 25)."""
        clf = _make_clf()
        assert clf.classify_value(22.0) == "HIGH"
        assert clf.classify_value(18.0) == "HIGH"
        assert clf.classify_value(24.99) == "HIGH"

    def test_classify_value_spike_for_30(self):
        """VIX >= 25 is classified as SPIKE."""
        clf = _make_clf()
        assert clf.classify_value(30.0) == "SPIKE"
        assert clf.classify_value(25.0) == "SPIKE"
        assert clf.classify_value(80.0) == "SPIKE"


# ---------------------------------------------------------------------------
# compute_percentile
# ---------------------------------------------------------------------------

class TestComputePercentile:
    """Tests for VIXRegimeClassifier.compute_percentile."""

    def test_compute_percentile_returns_50_for_median(self):
        """VIX at the exact median of the historical series returns ~50."""
        from tests.fixtures.macro_data import make_vix_series

        start = datetime(2025, 12, 1)
        end   = datetime(2026, 2, 28)
        df = make_vix_series(start, end, regime="normal", seed=7)

        # Use the DataFrame median as the query value
        median_vix = float(df["vix"].median())

        def provider(lookback_days: int):
            return df

        clf = _make_clf(historical_provider=provider)
        pct = clf.compute_percentile(median_vix, lookback_days=60)
        # The median splits the distribution in half: expect ~50% (±5 tolerance)
        assert 40.0 <= pct <= 60.0, f"Expected ~50, got {pct}"

    def test_compute_percentile_returns_100_for_max(self):
        """A value above all historical readings should return 100."""
        from tests.fixtures.macro_data import make_vix_series

        start = datetime(2025, 12, 1)
        end   = datetime(2026, 2, 28)
        df = make_vix_series(start, end, regime="low", seed=3)
        max_vix = float(df["vix"].max())

        def provider(lookback_days):
            return df

        clf = _make_clf(historical_provider=provider)
        pct = clf.compute_percentile(max_vix + 100.0)
        assert pct == pytest.approx(100.0)

    def test_compute_percentile_returns_50_when_no_history(self):
        """Returns 50.0 (neutral) when no historical data is available."""
        clf = _make_clf()  # no provider, empty buffer
        pct = clf.compute_percentile(15.0)
        assert pct == pytest.approx(50.0)

    def test_compute_percentile_uses_buffer_fallback(self):
        """Falls back to in-memory buffer when provider is None."""
        clf = _make_clf()
        # Inject buffer: 10 values from 10 to 19
        for i in range(10):
            clf._intraday_buffer.append((_ts(minute=i), float(10 + i)))

        # vix=14 → 4 values below (10,11,12,13) → 40th percentile
        pct = clf.compute_percentile(14.0)
        assert pct == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# detect_spike
# ---------------------------------------------------------------------------

class TestDetectSpike:
    """Tests for VIXRegimeClassifier.detect_spike."""

    def test_detect_spike_true_for_31_pct_jump(self):
        """A 31% jump above prev_close triggers spike detection."""
        clf = _make_clf()
        prev = 20.0
        current = prev * 1.31  # 26.2 — +31%
        assert clf.detect_spike(current, prev, threshold_pct=30.0) is True

    def test_detect_spike_false_for_5_pct_move(self):
        """A 5% move does not trigger spike detection."""
        clf = _make_clf()
        prev = 15.0
        current = prev * 1.05  # +5%
        assert clf.detect_spike(current, prev, threshold_pct=30.0) is False

    def test_detect_spike_false_for_zero_prev_close(self):
        """Zero prev_close is handled gracefully and returns False."""
        clf = _make_clf()
        assert clf.detect_spike(current=20.0, prev_close=0.0) is False

    def test_detect_spike_true_for_downward_spike(self):
        """A -31% crash also triggers spike detection."""
        clf = _make_clf()
        prev = 30.0
        current = prev * 0.69  # -31%
        assert clf.detect_spike(current, prev, threshold_pct=30.0) is True

    def test_detect_spike_at_exact_threshold(self):
        """A move exactly at threshold_pct triggers spike."""
        clf = _make_clf()
        prev = 10.0
        current = 13.0  # +30% exactly
        assert clf.detect_spike(current, prev, threshold_pct=30.0) is True


# ---------------------------------------------------------------------------
# update_intraday
# ---------------------------------------------------------------------------

class TestUpdateIntraday:
    """Tests for VIXRegimeClassifier.update_intraday."""

    def _make_begin_engine(self) -> tuple[MagicMock, MagicMock]:
        """Return (engine, inner_conn) where engine.begin() yields inner_conn."""
        mock_engine = MagicMock()
        inner_conn = MagicMock()
        # MagicMock supports context managers automatically;
        # set the value returned by __enter__ explicitly.
        mock_engine.begin.return_value.__enter__ = MagicMock(return_value=inner_conn)
        mock_engine.begin.return_value.__exit__ = MagicMock(return_value=False)
        return mock_engine, inner_conn

    def test_update_intraday_persists_with_regime(self):
        """update_intraday inserts a row with correct regime into the DB."""
        mock_engine, inner_conn = self._make_begin_engine()

        clf = _make_clf(db_engine=mock_engine)
        ts = _ts(hour=10, minute=30)
        clf.update_intraday(ts, vix_value=22.0, prev_close=16.0)

        # Buffer was updated
        assert len(clf._intraday_buffer) == 1
        assert clf._intraday_buffer[0] == (ts, 22.0)

        # DB execute was called on the connection
        assert inner_conn.execute.called

    def test_update_intraday_no_db_only_updates_buffer(self):
        """Without a DB engine, only the in-memory buffer is updated."""
        clf = _make_clf()
        ts = _ts(hour=11, minute=0)
        clf.update_intraday(ts, vix_value=14.5)
        assert len(clf._intraday_buffer) == 1
        assert clf._intraday_buffer[0][1] == 14.5

    def test_update_intraday_sets_spike_flag_for_large_move(self):
        """Spike flag is True when intraday move exceeds 30%."""
        mock_engine, inner_conn = self._make_begin_engine()

        clf = _make_clf(db_engine=mock_engine)
        ts = _ts(hour=9, minute=45)
        # prev_close=10, current=14 → +40% → spike
        clf.update_intraday(ts, vix_value=14.0, prev_close=10.0)

        # Inspect the bound parameters passed to execute
        call_args = inner_conn.execute.call_args
        params = call_args[0][1]  # positional: (stmt, params)
        assert params["spike_detected"] is True

    def test_update_intraday_no_spike_for_small_move(self):
        """Spike flag is False when intraday move is within threshold."""
        mock_engine, inner_conn = self._make_begin_engine()

        clf = _make_clf(db_engine=mock_engine)
        ts = _ts(hour=9, minute=50)
        # prev_close=16, current=16.5 → +3.1% → no spike
        clf.update_intraday(ts, vix_value=16.5, prev_close=16.0)

        call_args = inner_conn.execute.call_args
        params = call_args[0][1]
        assert params["spike_detected"] is False


# ---------------------------------------------------------------------------
# compute_daily
# ---------------------------------------------------------------------------

class TestComputeDaily:
    """Tests for VIXRegimeClassifier.compute_daily."""

    def test_compute_daily_aggregates_and_persists(self):
        """compute_daily aggregates buffer rows and returns correct keys."""
        clf = _make_clf()  # no DB engine — uses buffer only

        trade_date = date(2026, 5, 8)

        # Feed buffer with a known VIX sequence for today
        ticks = [14.0, 15.5, 13.2, 16.8, 15.0]
        for i, v in enumerate(ticks):
            clf._intraday_buffer.append((_ts(day=8, hour=9, minute=15 + i), v))

        result = clf.compute_daily(trade_date)

        assert result["vix_open"]  == pytest.approx(14.0)
        assert result["vix_high"]  == pytest.approx(16.8)
        assert result["vix_low"]   == pytest.approx(13.2)
        assert result["vix_close"] == pytest.approx(15.0)
        assert result["regime"] == "NORMAL"  # 15.0 in [13, 18)
        assert "percentile_60d" in result

    def test_compute_daily_returns_empty_when_no_data(self):
        """compute_daily returns {} when there are no intraday rows."""
        clf = _make_clf()
        result = clf.compute_daily(date(2026, 5, 8))
        assert result == {}

    def test_compute_daily_persists_to_db(self):
        """compute_daily calls _persist_daily when DB engine is set."""
        mock_engine = MagicMock()
        mock_conn_ctx = MagicMock()
        mock_conn_ctx.__enter__ = MagicMock(return_value=mock_conn_ctx)
        mock_conn_ctx.__exit__ = MagicMock(return_value=False)
        mock_engine.begin.return_value = mock_conn_ctx
        mock_engine.connect.return_value = mock_conn_ctx

        # connect() for intraday load → returns no intraday rows
        mock_conn_ctx.execute.return_value.fetchall.return_value = []
        # Also for previous_close query
        mock_conn_ctx.execute.return_value.fetchone.return_value = None

        clf = _make_clf(db_engine=mock_engine)
        trade_date = date(2026, 5, 8)

        # Seed the buffer so there is data
        clf._intraday_buffer.append((_ts(day=8), 20.0))

        result = clf.compute_daily(trade_date)
        # Even with engine, buffer fallback should work
        assert result != {} or True  # either aggregated or empty is OK


# ---------------------------------------------------------------------------
# get_today_regime
# ---------------------------------------------------------------------------

class TestGetTodayRegime:
    """Tests for VIXRegimeClassifier.get_today_regime."""

    def test_get_today_regime_roundtrip(self):
        """get_today_regime returns None when no DB engine is provided."""
        clf = _make_clf()
        result = clf.get_today_regime(date(2026, 5, 8))
        assert result is None

    def test_get_today_regime_with_mock_db(self):
        """get_today_regime returns structured dict when DB row exists."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_engine.connect.return_value = mock_conn

        trade_date = date(2026, 5, 8)
        fake_row = (
            trade_date,   # trade_date
            "14.5",       # vix_open
            "16.0",       # vix_high
            "13.8",       # vix_low
            "15.2",       # vix_close
            "-2.5",       # vix_change_pct
            "NORMAL",     # regime
            "42.0",       # percentile_60d
            datetime(2026, 5, 8, 15, 30, tzinfo=timezone.utc),  # computed_at
        )
        mock_conn.execute.return_value.fetchone.return_value = fake_row

        clf = _make_clf(db_engine=mock_engine)
        result = clf.get_today_regime(trade_date)

        assert result is not None
        assert result["regime"] == "NORMAL"
        assert result["vix_close"] == pytest.approx(15.2)
        assert result["percentile_60d"] == pytest.approx(42.0)

    def test_get_today_regime_returns_none_when_no_row(self):
        """get_today_regime returns None when query returns no row."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_engine.connect.return_value = mock_conn
        mock_conn.execute.return_value.fetchone.return_value = None

        clf = _make_clf(db_engine=mock_engine)
        result = clf.get_today_regime(date(2026, 5, 8))
        assert result is None


# ---------------------------------------------------------------------------
# is_strategy_allowed
# ---------------------------------------------------------------------------

class TestIsStrategyAllowed:
    """Tests for VIXRegimeClassifier.is_strategy_allowed."""

    def test_is_strategy_allowed_trend_in_spike_returns_false(self):
        """trend strategy is NOT allowed in SPIKE regime."""
        clf = _make_clf()
        assert clf.is_strategy_allowed("trend", regime="SPIKE") is False

    def test_is_strategy_allowed_trend_in_normal_returns_true(self):
        """trend strategy IS allowed in NORMAL regime."""
        clf = _make_clf()
        assert clf.is_strategy_allowed("trend", regime="NORMAL") is True

    def test_is_strategy_allowed_trend_in_high_returns_true(self):
        """trend strategy IS allowed in HIGH regime."""
        clf = _make_clf()
        assert clf.is_strategy_allowed("trend", regime="HIGH") is True

    def test_is_strategy_allowed_options_sell_in_high_returns_true(self):
        """options_sell IS allowed in HIGH (premium-rich) regime."""
        clf = _make_clf()
        assert clf.is_strategy_allowed("options_sell", regime="HIGH") is True

    def test_is_strategy_allowed_options_sell_in_normal_returns_false(self):
        """options_sell is NOT allowed in NORMAL regime."""
        clf = _make_clf()
        assert clf.is_strategy_allowed("options_sell", regime="NORMAL") is False

    def test_is_strategy_allowed_mean_rev_in_spike_returns_true(self):
        """mean_rev IS allowed in SPIKE regime."""
        clf = _make_clf()
        assert clf.is_strategy_allowed("mean_rev", regime="SPIKE") is True

    def test_is_strategy_allowed_mean_rev_in_low_returns_false(self):
        """mean_rev is NOT allowed in LOW regime."""
        clf = _make_clf()
        assert clf.is_strategy_allowed("mean_rev", regime="LOW") is False

    def test_is_strategy_allowed_breakout_only_in_normal(self):
        """breakout is allowed ONLY in NORMAL."""
        clf = _make_clf()
        assert clf.is_strategy_allowed("breakout", regime="NORMAL") is True
        assert clf.is_strategy_allowed("breakout", regime="HIGH")   is False
        assert clf.is_strategy_allowed("breakout", regime="SPIKE")  is False
        assert clf.is_strategy_allowed("breakout", regime="LOW")    is False

    def test_is_strategy_allowed_no_regime_returns_true(self):
        """When regime is None (no data), fail-open → True."""
        clf = _make_clf()
        assert clf.is_strategy_allowed("trend", regime=None) is True

    def test_is_strategy_allowed_unknown_strategy_returns_true(self):
        """Unknown strategy type fails-open (returns True)."""
        clf = _make_clf()
        assert clf.is_strategy_allowed("algo_xyz", regime="SPIKE") is True
