"""Tests for src.execution.slippage_monitor.SlippageMonitor.

All tests use inline mocks — no external database or network required.
DB interactions are validated through mock engine/connection stubs.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, call
import pytest

from src.execution.slippage_monitor import SlippageConfig, SlippageMonitor

IST = timezone(timedelta(hours=5, minutes=30))


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------


def _monitor(
    db_engine=None,
    drift_threshold_pct: float = 50.0,
    halt_threshold_pct: float = 100.0,
    drift_min_sample_size: int = 30,
    rolling_window_days: int = 7,
) -> SlippageMonitor:
    cfg = SlippageConfig(
        drift_threshold_pct=drift_threshold_pct,
        halt_threshold_pct=halt_threshold_pct,
        drift_min_sample_size=drift_min_sample_size,
        rolling_window_days=rolling_window_days,
    )
    return SlippageMonitor(db_engine=db_engine, config=cfg)


def _make_engine_with_scalar(scalar_value) -> MagicMock:
    """Return a mock engine whose connection returns scalar_value from execute."""
    conn = MagicMock()
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.scalar.return_value = scalar_value
    engine = MagicMock()
    engine.connect.return_value = conn
    return engine, conn


def _make_engine_with_rows(rows) -> tuple[MagicMock, MagicMock]:
    """Return a mock engine whose execute().fetchall() returns rows."""
    conn = MagicMock()
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.fetchall.return_value = rows
    engine = MagicMock()
    engine.connect.return_value = conn
    return engine, conn


def _make_engine_returning_id(row_id: int) -> tuple[MagicMock, MagicMock]:
    """Return a mock engine whose execute().fetchone() returns (row_id,)."""
    conn = MagicMock()
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.fetchone.return_value = (row_id,)
    engine = MagicMock()
    engine.connect.return_value = conn
    return engine, conn


# ===========================================================================
# compute_slippage — no DB required
# ===========================================================================


class TestComputeSlippageBuyPaidMorePositive:
    """BUY fill above intended price produces positive slippage."""

    def test_compute_slippage_buy_paid_more_positive(self):
        m = _monitor()
        # Intended 1000 paisa, actually paid 1010 paisa
        result = m.compute_slippage(
            intended_price=1000, actual_price=1010, side="BUY", quantity=10
        )
        assert result["slippage_paisa"] == 10
        assert result["slippage_bps"] == pytest.approx(100.0)   # 10/1000 * 10000
        assert result["slippage_cost_paisa"] == 100              # 10 * 10

    def test_compute_slippage_buy_paid_less_negative(self):
        """BUY fill below intended → negative slippage (favourable)."""
        m = _monitor()
        result = m.compute_slippage(
            intended_price=1000, actual_price=990, side="BUY", quantity=5
        )
        assert result["slippage_paisa"] == -10
        assert result["slippage_bps"] == pytest.approx(-100.0)
        assert result["slippage_cost_paisa"] == -50


class TestComputeSlippageSellReceivedLessPositive:
    """SELL fill below intended price produces positive slippage."""

    def test_compute_slippage_sell_received_less_positive(self):
        m = _monitor()
        # Intended 2000 paisa, actually received 1980 paisa
        result = m.compute_slippage(
            intended_price=2000, actual_price=1980, side="SELL", quantity=10
        )
        assert result["slippage_paisa"] == 20
        assert result["slippage_bps"] == pytest.approx(100.0)   # 20/2000 * 10000
        assert result["slippage_cost_paisa"] == 200

    def test_compute_slippage_sell_received_more_negative(self):
        """SELL fill above intended → negative slippage (favourable)."""
        m = _monitor()
        result = m.compute_slippage(
            intended_price=2000, actual_price=2020, side="SELL", quantity=5
        )
        assert result["slippage_paisa"] == -20
        assert result["slippage_cost_paisa"] == -100


class TestComputeSlippageZeroWhenFilledAtIntended:
    """Fill at exactly the intended price → zero slippage."""

    def test_compute_slippage_zero_when_filled_at_intended_buy(self):
        m = _monitor()
        result = m.compute_slippage(
            intended_price=500, actual_price=500, side="BUY", quantity=100
        )
        assert result["slippage_paisa"] == 0
        assert result["slippage_bps"] == pytest.approx(0.0)
        assert result["slippage_cost_paisa"] == 0

    def test_compute_slippage_zero_when_filled_at_intended_sell(self):
        m = _monitor()
        result = m.compute_slippage(
            intended_price=500, actual_price=500, side="SELL", quantity=100
        )
        assert result["slippage_paisa"] == 0
        assert result["slippage_bps"] == pytest.approx(0.0)
        assert result["slippage_cost_paisa"] == 0


class TestComputeSlippageBpsNormalized:
    """Basis-point calculation is normalized by intended_price."""

    def test_compute_slippage_bps_normalized(self):
        """1 paisa slip on 100 paisa intended = 100 bps."""
        m = _monitor()
        result = m.compute_slippage(
            intended_price=100, actual_price=101, side="BUY", quantity=1
        )
        assert result["slippage_bps"] == pytest.approx(100.0)

    def test_compute_slippage_bps_normalized_larger_price(self):
        """10 paisa slip on 10_000 paisa intended = 10 bps."""
        m = _monitor()
        result = m.compute_slippage(
            intended_price=10_000, actual_price=10_010, side="BUY", quantity=1
        )
        assert result["slippage_bps"] == pytest.approx(10.0)

    def test_compute_slippage_side_case_insensitive(self):
        """'buy' (lowercase) should work the same as 'BUY'."""
        m = _monitor()
        result = m.compute_slippage(
            intended_price=1000, actual_price=1010, side="buy", quantity=1
        )
        assert result["slippage_paisa"] == 10

    def test_compute_slippage_invalid_side_raises(self):
        """An unrecognised side value should raise ValueError."""
        m = _monitor()
        with pytest.raises(ValueError, match="BUY or SELL"):
            m.compute_slippage(
                intended_price=1000, actual_price=1010, side="LONG", quantity=1
            )

    def test_compute_slippage_zero_intended_price_returns_zero_bps(self):
        """If intended_price is 0 bps is set to 0 (avoid division by zero)."""
        m = _monitor()
        result = m.compute_slippage(
            intended_price=0, actual_price=10, side="BUY", quantity=1
        )
        assert result["slippage_bps"] == pytest.approx(0.0)


# ===========================================================================
# record_fill — requires mocked DB
# ===========================================================================


class TestRecordFillPersistsWithAllFields:
    """record_fill() should execute INSERT and return the new row id."""

    def test_record_fill_persists_with_all_fields(self):
        engine, conn = _make_engine_returning_id(42)
        m = _monitor(db_engine=engine)
        row_id = m.record_fill(
            fill_id="FILL-001",
            order_id="ORD-001",
            instrument_key="NSE_EQ|RELIANCE",
            side="BUY",
            intended_price=2500_00,  # 2500 INR in paisa
            actual_fill_price=2501_00,
            quantity=10,
            mode="paper",
        )
        assert row_id == 42
        conn.execute.assert_called_once()
        conn.commit.assert_called_once()

    def test_record_fill_no_engine_raises(self):
        m = _monitor(db_engine=None)
        with pytest.raises(RuntimeError, match="db_engine is required"):
            m.record_fill(
                fill_id="F",
                order_id="O",
                instrument_key="NSE_EQ|TCS",
                side="BUY",
                intended_price=3000_00,
                actual_fill_price=3001_00,
                quantity=5,
                mode="paper",
            )

    def test_record_fill_passes_correct_slippage_to_db(self):
        """The INSERT params must include pre-computed slippage fields."""
        engine, conn = _make_engine_returning_id(99)
        m = _monitor(db_engine=engine)
        m.record_fill(
            fill_id="FILL-X",
            order_id="ORD-X",
            instrument_key="NSE_EQ|HDFCBANK",
            side="SELL",
            intended_price=1600_00,
            actual_fill_price=1598_00,   # 2 INR = 200 paisa below intended
            quantity=20,
            mode="live",
        )
        call_args = conn.execute.call_args
        params = call_args[0][1]  # second positional arg is the params dict
        assert params["slippage_paisa"] == 200       # positive for SELL
        assert params["slippage_cost_paisa"] == 4000  # 200 * 20
        assert params["mode"] == "live"


# ===========================================================================
# get_avg_slippage_bps — requires mocked DB
# ===========================================================================


class TestGetAvgSlippageBpsOverWindow:
    """get_avg_slippage_bps() should query and return the avg correctly."""

    def test_get_avg_slippage_bps_over_window(self):
        engine, conn = _make_engine_with_scalar(12.5)
        m = _monitor(db_engine=engine)
        result = m.get_avg_slippage_bps(days=7)
        assert result == pytest.approx(12.5)

    def test_get_avg_slippage_bps_returns_zero_when_no_rows(self):
        engine, conn = _make_engine_with_scalar(None)
        m = _monitor(db_engine=engine)
        result = m.get_avg_slippage_bps(days=7)
        assert result == pytest.approx(0.0)

    def test_get_avg_slippage_bps_no_engine_raises(self):
        m = _monitor(db_engine=None)
        with pytest.raises(RuntimeError, match="db_engine is required"):
            m.get_avg_slippage_bps()


class TestGetAvgSlippageBpsFilteredByMode:
    """get_avg_slippage_bps() mode filter passes mode to DB query."""

    def test_get_avg_slippage_bps_filtered_by_mode(self):
        engine, conn = _make_engine_with_scalar(8.0)
        m = _monitor(db_engine=engine)
        result = m.get_avg_slippage_bps(mode="live", days=7)
        assert result == pytest.approx(8.0)
        # Verify mode param was passed through
        call_args = conn.execute.call_args
        params = call_args[0][1]
        assert params.get("mode") == "live"

    def test_get_avg_slippage_bps_filtered_by_instrument(self):
        engine, conn = _make_engine_with_scalar(5.0)
        m = _monitor(db_engine=engine)
        result = m.get_avg_slippage_bps(
            instrument_key="NSE_EQ|INFY", mode="paper", days=7
        )
        assert result == pytest.approx(5.0)
        call_args = conn.execute.call_args
        params = call_args[0][1]
        assert params.get("instrument_key") == "NSE_EQ|INFY"
        assert params.get("mode") == "paper"


# ===========================================================================
# detect_drift — requires mocked DB with fetchall
# ===========================================================================


def _make_drift_engine(paper_count, paper_bps, live_count, live_bps):
    """Build an engine whose fetchall returns paper + live mode rows."""
    rows = []
    if paper_count > 0:
        rows.append(("paper", paper_count, paper_bps))
    if live_count > 0:
        rows.append(("live", live_count, live_bps))

    engine, conn = _make_engine_with_rows(rows)
    return engine, conn


class TestDetectDriftReturnsNoneWhenInsufficientSamples:
    """detect_drift() must return None when either mode has <min_sample fills."""

    def test_detect_drift_returns_none_when_insufficient_samples(self):
        # paper=5, live=5 — both below default min_sample_size=30
        engine, _ = _make_drift_engine(
            paper_count=5, paper_bps=10.0,
            live_count=5, live_bps=20.0,
        )
        m = _monitor(db_engine=engine, drift_min_sample_size=30)
        result = m.detect_drift()
        assert result is None

    def test_detect_drift_returns_none_when_live_insufficient(self):
        # paper OK, live too few
        engine, _ = _make_drift_engine(
            paper_count=35, paper_bps=10.0,
            live_count=10, live_bps=25.0,
        )
        m = _monitor(db_engine=engine, drift_min_sample_size=30)
        result = m.detect_drift()
        assert result is None

    def test_detect_drift_returns_none_when_paper_insufficient(self):
        engine, _ = _make_drift_engine(
            paper_count=10, paper_bps=10.0,
            live_count=35, live_bps=25.0,
        )
        m = _monitor(db_engine=engine, drift_min_sample_size=30)
        result = m.detect_drift()
        assert result is None


class TestDetectDriftReturnsHighAlertAboveThreshold:
    """When live slippage > drift_threshold_pct% worse, alert_type=DRIFT_HIGH."""

    def test_detect_drift_returns_high_alert_above_threshold(self):
        # paper=5 bps, live=8 bps → drift = (8-5)/5 * 100 = 60% → above 50% threshold
        engine, _ = _make_drift_engine(
            paper_count=31, paper_bps=5.0,
            live_count=31, live_bps=8.0,
        )
        m = _monitor(
            db_engine=engine,
            drift_threshold_pct=50.0,
            halt_threshold_pct=100.0,
            drift_min_sample_size=30,
        )
        result = m.detect_drift()
        assert result is not None
        assert result["alert_type"] == "DRIFT_HIGH"
        assert result["paper_avg_bps"] == pytest.approx(5.0)
        assert result["live_avg_bps"] == pytest.approx(8.0)
        assert result["drift_pct"] == pytest.approx(60.0)
        assert "recommendation" in result


class TestDetectDriftReturnsBlockAboveHaltThreshold:
    """When live slippage > halt_threshold_pct% worse, alert_type=DRIFT_BLOCK."""

    def test_detect_drift_returns_block_above_halt_threshold(self):
        # paper=5 bps, live=12 bps → drift = (12-5)/5 * 100 = 140% → above 100% halt
        engine, _ = _make_drift_engine(
            paper_count=31, paper_bps=5.0,
            live_count=31, live_bps=12.0,
        )
        m = _monitor(
            db_engine=engine,
            drift_threshold_pct=50.0,
            halt_threshold_pct=100.0,
            drift_min_sample_size=30,
        )
        result = m.detect_drift()
        assert result is not None
        assert result["alert_type"] == "DRIFT_BLOCK"
        assert result["drift_pct"] == pytest.approx(140.0)


class TestDetectDriftReturnsNoneWhenWithinTolerance:
    """When drift_pct <= drift_threshold, alert_type should be None."""

    def test_detect_drift_returns_none_when_within_tolerance(self):
        # paper=10 bps, live=14 bps → drift = 40% → below 50% threshold
        engine, _ = _make_drift_engine(
            paper_count=31, paper_bps=10.0,
            live_count=31, live_bps=14.0,
        )
        m = _monitor(
            db_engine=engine,
            drift_threshold_pct=50.0,
            halt_threshold_pct=100.0,
            drift_min_sample_size=30,
        )
        result = m.detect_drift()
        assert result is not None
        assert result["alert_type"] is None
        assert "acceptable" in result["recommendation"].lower()

    def test_detect_drift_identical_paper_live_no_alert(self):
        engine, _ = _make_drift_engine(
            paper_count=50, paper_bps=5.0,
            live_count=50, live_bps=5.0,
        )
        m = _monitor(db_engine=engine, drift_min_sample_size=30)
        result = m.detect_drift()
        assert result is not None
        assert result["alert_type"] is None
        assert result["drift_pct"] == pytest.approx(0.0)


# ===========================================================================
# log_drift_alert — requires mocked DB
# ===========================================================================


class TestLogDriftAlertPersistsToDB:
    """log_drift_alert() should execute INSERT and return the alert row id."""

    def test_log_drift_alert_persists_to_db(self):
        engine, conn = _make_engine_returning_id(7)
        m = _monitor(db_engine=engine)
        drift = {
            "paper_avg_bps": 5.0,
            "live_avg_bps": 12.0,
            "drift_pct": 140.0,
            "alert_type": "DRIFT_BLOCK",
            "recommendation": "Halt live orders.",
            "instrument_key": "NSE_EQ|RELIANCE",
        }
        alert_id = m.log_drift_alert(drift)
        assert alert_id == 7
        conn.execute.assert_called_once()
        conn.commit.assert_called_once()

    def test_log_drift_alert_no_engine_raises(self):
        m = _monitor(db_engine=None)
        with pytest.raises(RuntimeError, match="db_engine is required"):
            m.log_drift_alert({"alert_type": "DRIFT_HIGH", "drift_pct": 60.0})

    def test_log_drift_alert_uses_halt_threshold_for_block(self):
        """DRIFT_BLOCK alerts should record halt_threshold_pct, not drift_threshold."""
        engine, conn = _make_engine_returning_id(1)
        m = _monitor(
            db_engine=engine,
            drift_threshold_pct=50.0,
            halt_threshold_pct=100.0,
        )
        m.log_drift_alert(
            {
                "alert_type": "DRIFT_BLOCK",
                "paper_avg_bps": 5.0,
                "live_avg_bps": 12.0,
                "drift_pct": 140.0,
                "recommendation": "Halt.",
            }
        )
        call_args = conn.execute.call_args
        params = call_args[0][1]
        assert params["threshold_pct"] == 100.0

    def test_log_drift_alert_uses_drift_threshold_for_high(self):
        """DRIFT_HIGH alerts should record drift_threshold_pct."""
        engine, conn = _make_engine_returning_id(2)
        m = _monitor(
            db_engine=engine,
            drift_threshold_pct=50.0,
            halt_threshold_pct=100.0,
        )
        m.log_drift_alert(
            {
                "alert_type": "DRIFT_HIGH",
                "paper_avg_bps": 5.0,
                "live_avg_bps": 8.0,
                "drift_pct": 60.0,
                "recommendation": "Review routing.",
            }
        )
        call_args = conn.execute.call_args
        params = call_args[0][1]
        assert params["threshold_pct"] == 50.0


# ===========================================================================
# get_drift_alerts
# ===========================================================================


class TestGetDriftAlerts:
    def test_get_drift_alerts_no_engine_raises(self):
        m = _monitor(db_engine=None)
        with pytest.raises(RuntimeError, match="db_engine is required"):
            m.get_drift_alerts()

    def test_get_drift_alerts_returns_list_of_dicts(self):
        ts = datetime.now(IST)
        rows = [
            (1, ts, "NSE_EQ|RELIANCE", "DRIFT_HIGH", 5.0, 8.0, 60.0, 50.0, {"recommendation": "ok"}),
            (2, ts, None, "DRIFT_BLOCK", 3.0, 9.0, 200.0, 100.0, None),
        ]
        engine, conn = _make_engine_with_rows(rows)
        m = _monitor(db_engine=engine)
        alerts = m.get_drift_alerts(days=7)
        assert len(alerts) == 2
        assert alerts[0]["alert_type"] == "DRIFT_HIGH"
        assert alerts[0]["paper_avg_bps"] == pytest.approx(5.0)
        assert alerts[1]["instrument_key"] is None


# ===========================================================================
# get_summary — requires mocked DB with sequential execute calls
# ===========================================================================


class TestGetSummaryAggregatesCorrectly:
    """get_summary() aggregates totals and per-mode/instrument breakdowns."""

    def test_get_summary_aggregates_correctly(self):
        # We need to mock 3 sequential execute() calls:
        # 1) overall: total_fills, total_cost
        # 2) by mode: mode, avg_bps
        # 3) by instrument: instrument_key, mode, fills, avg_bps, total_cost

        conn = MagicMock()
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=False)

        overall_result = MagicMock()
        overall_result.fetchone.return_value = (150, 75_000)

        mode_result = MagicMock()
        mode_result.fetchall.return_value = [
            ("paper", 8.5),
            ("live", 12.0),
        ]

        instrument_result = MagicMock()
        instrument_result.fetchall.return_value = [
            ("NSE_EQ|RELIANCE", "paper", 80, 8.0, 40_000),
            ("NSE_EQ|RELIANCE", "live", 70, 12.5, 35_000),
        ]

        conn.execute.side_effect = [overall_result, mode_result, instrument_result]

        engine = MagicMock()
        engine.connect.return_value = conn

        m = _monitor(db_engine=engine)
        summary = m.get_summary(days=7)

        assert summary["total_fills"] == 150
        assert summary["total_slippage_paisa"] == 75_000
        assert summary["avg_bps_by_mode"]["paper"] == pytest.approx(8.5)
        assert summary["avg_bps_by_mode"]["live"] == pytest.approx(12.0)
        assert len(summary["by_instrument"]) == 2
        assert summary["by_instrument"][0]["instrument_key"] == "NSE_EQ|RELIANCE"
        assert summary["by_instrument"][0]["mode"] == "paper"
        assert summary["by_instrument"][0]["fills"] == 80

    def test_get_summary_no_engine_raises(self):
        m = _monitor(db_engine=None)
        with pytest.raises(RuntimeError, match="db_engine is required"):
            m.get_summary()

    def test_get_summary_empty_db_returns_zeros(self):
        """When no rows exist, totals should be 0 and lists empty."""
        conn = MagicMock()
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=False)

        overall_result = MagicMock()
        overall_result.fetchone.return_value = (0, 0)

        mode_result = MagicMock()
        mode_result.fetchall.return_value = []

        instrument_result = MagicMock()
        instrument_result.fetchall.return_value = []

        conn.execute.side_effect = [overall_result, mode_result, instrument_result]

        engine = MagicMock()
        engine.connect.return_value = conn

        m = _monitor(db_engine=engine)
        summary = m.get_summary(days=7)

        assert summary["total_fills"] == 0
        assert summary["total_slippage_paisa"] == 0
        assert summary["avg_bps_by_mode"] == {}
        assert summary["by_instrument"] == []
