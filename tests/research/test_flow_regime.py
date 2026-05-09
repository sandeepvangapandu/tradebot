"""Tests for src.research.flow_regime — FII/DII regime classification.

All tests use inline mocks.  No live DB connection required.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, call

import pytest

from src.research.flow_regime import FlowRegimeAnalyzer


TRADE_DATE = date(2026, 5, 8)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_analyzer(db_engine=None) -> FlowRegimeAnalyzer:
    return FlowRegimeAnalyzer(db_engine=db_engine)


def _make_mock_engine_with_flow(fii_net: float, dii_net: float):
    """Return a mock engine whose flow query returns (fii_net, dii_net)."""
    mock_row = (fii_net, dii_net)

    # connect() context manager for compute_regime
    mock_conn_ro = MagicMock()
    mock_conn_ro.__enter__ = MagicMock(return_value=mock_conn_ro)
    mock_conn_ro.__exit__ = MagicMock(return_value=False)

    flow_result = MagicMock()
    flow_result.fetchone.return_value = mock_row
    mock_conn_ro.execute.return_value = flow_result

    # begin() context manager for insert
    mock_conn_rw = MagicMock()
    mock_conn_rw.__enter__ = MagicMock(return_value=mock_conn_rw)
    mock_conn_rw.__exit__ = MagicMock(return_value=False)

    # streak queries (connect calls for compute_streak → separate from above)
    streak_result = MagicMock()
    streak_result.fetchall.return_value = [(fii_net,)] * 5  # 5 days same direction

    mock_engine = MagicMock()

    # connect() is called 3 times: flow query + 2 streak queries
    # We build a side_effect list for each connect() call
    flow_conn = MagicMock()
    flow_conn.__enter__ = MagicMock(return_value=flow_conn)
    flow_conn.__exit__ = MagicMock(return_value=False)
    flow_conn.execute.return_value = flow_result

    streak_conn1 = MagicMock()
    streak_conn1.__enter__ = MagicMock(return_value=streak_conn1)
    streak_conn1.__exit__ = MagicMock(return_value=False)
    streak_conn1.execute.return_value = streak_result

    streak_conn2 = MagicMock()
    streak_conn2.__enter__ = MagicMock(return_value=streak_conn2)
    streak_conn2.__exit__ = MagicMock(return_value=False)
    streak_conn2.execute.return_value = streak_result

    mock_engine.connect.side_effect = [flow_conn, streak_conn1, streak_conn2]
    mock_engine.begin.return_value = mock_conn_rw

    return mock_engine


# ---------------------------------------------------------------------------
# classify_fii_regime tests
# ---------------------------------------------------------------------------

class TestClassifyFiiRegime:
    """Unit tests for FlowRegimeAnalyzer.classify_fii_regime."""

    def test_classify_fii_regime_strong_buy_above_2000(self):
        """FII net > 2000 cr should return STRONG_BUY."""
        analyzer = _make_analyzer()
        assert analyzer.classify_fii_regime(2500.0) == "STRONG_BUY"

    def test_classify_fii_regime_strong_buy_exactly_2000(self):
        """FII net exactly 2000 cr should return STRONG_BUY (>= threshold)."""
        analyzer = _make_analyzer()
        assert analyzer.classify_fii_regime(2000.0) == "STRONG_BUY"

    def test_classify_fii_regime_buy_between_500_and_2000(self):
        """FII net in 500-2000 range should return BUY."""
        analyzer = _make_analyzer()
        assert analyzer.classify_fii_regime(1000.0) == "BUY"
        assert analyzer.classify_fii_regime(500.0) == "BUY"

    def test_classify_fii_regime_neutral_in_band(self):
        """FII net in (-500, 500) open range should return NEUTRAL."""
        analyzer = _make_analyzer()
        assert analyzer.classify_fii_regime(0.0) == "NEUTRAL"
        assert analyzer.classify_fii_regime(-499.0) == "NEUTRAL"
        assert analyzer.classify_fii_regime(499.0) == "NEUTRAL"

    def test_classify_fii_regime_sell_between_minus_2000_and_minus_500(self):
        """FII net in [-2000, -500] should return SELL.

        Boundary -500 is inclusive for SELL (NEUTRAL threshold is strictly > -500).
        """
        analyzer = _make_analyzer()
        assert analyzer.classify_fii_regime(-1000.0) == "SELL"
        assert analyzer.classify_fii_regime(-500.0) == "SELL"   # boundary belongs to SELL

    def test_classify_fii_regime_strong_sell_below_minus_2000(self):
        """FII net < -2000 cr should return STRONG_SELL."""
        analyzer = _make_analyzer()
        assert analyzer.classify_fii_regime(-3000.0) == "STRONG_SELL"

    def test_classify_fii_regime_strong_sell_exactly_minus_2000(self):
        """FII net exactly -2000 cr should return STRONG_SELL."""
        analyzer = _make_analyzer()
        assert analyzer.classify_fii_regime(-2000.0) == "STRONG_SELL"


# ---------------------------------------------------------------------------
# classify_dii_regime (same thresholds as FII)
# ---------------------------------------------------------------------------

class TestClassifyDiiRegime:
    """Spot-check that DII uses same threshold scale as FII."""

    def test_classify_dii_regime_strong_buy_above_2000(self):
        analyzer = _make_analyzer()
        assert analyzer.classify_dii_regime(2500.0) == "STRONG_BUY"

    def test_classify_fii_regime_neutral_in_band(self):
        analyzer = _make_analyzer()
        assert analyzer.classify_dii_regime(-200.0) == "NEUTRAL"

    def test_classify_fii_regime_strong_sell_below_minus_2000(self):
        analyzer = _make_analyzer()
        assert analyzer.classify_dii_regime(-2500.0) == "STRONG_SELL"


# ---------------------------------------------------------------------------
# compute_streak tests
# ---------------------------------------------------------------------------

class TestComputeStreak:
    """Tests for FlowRegimeAnalyzer.compute_streak."""

    def _make_streak_engine(self, values: list[float], side: str = "fii") -> MagicMock:
        """Create a mock engine that returns *values* as streak rows."""
        result_rows = [(v,) for v in values]
        mock_result = MagicMock()
        mock_result.fetchall.return_value = result_rows

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value = mock_result

        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_conn
        return mock_engine

    def test_compute_streak_fii_3_days_buying_returns_3(self):
        """3 consecutive positive FII days → streak = +3."""
        engine = self._make_streak_engine([1500.0, 1200.0, 800.0], side="fii")
        analyzer = _make_analyzer(db_engine=engine)
        streak = analyzer.compute_streak("fii")
        assert streak == 3

    def test_compute_streak_fii_3_days_selling_returns_minus_3(self):
        """3 consecutive negative FII days → streak = -3."""
        engine = self._make_streak_engine([-1500.0, -1200.0, -800.0], side="fii")
        analyzer = _make_analyzer(db_engine=engine)
        streak = analyzer.compute_streak("fii")
        assert streak == -3

    def test_compute_streak_fii_alternating_returns_1(self):
        """Alternating buy/sell days → streak = ±1 (only first day counted)."""
        engine = self._make_streak_engine([1500.0, -1200.0, 800.0], side="fii")
        analyzer = _make_analyzer(db_engine=engine)
        streak = analyzer.compute_streak("fii")
        # First value is positive → streak = +1 (second value breaks the run)
        assert streak == 1

    def test_compute_streak_dii_5_days_buying_returns_5(self):
        """5 consecutive positive DII days → streak = +5."""
        engine = self._make_streak_engine([500.0, 600.0, 700.0, 800.0, 900.0], side="dii")
        analyzer = _make_analyzer(db_engine=engine)
        streak = analyzer.compute_streak("dii")
        assert streak == 5

    def test_compute_streak_returns_zero_when_no_engine(self):
        """Without a DB engine, compute_streak should return 0."""
        analyzer = _make_analyzer(db_engine=None)
        assert analyzer.compute_streak("fii") == 0

    def test_compute_streak_returns_zero_when_no_data(self):
        """Empty DB rows should return 0."""
        engine = self._make_streak_engine([], side="fii")
        analyzer = _make_analyzer(db_engine=engine)
        streak = analyzer.compute_streak("fii")
        assert streak == 0


# ---------------------------------------------------------------------------
# combine_signals tests
# ---------------------------------------------------------------------------

class TestCombineSignals:
    """Tests for FlowRegimeAnalyzer.combine_signals."""

    def test_combine_signals_both_buy_returns_tailwind(self):
        analyzer = _make_analyzer()
        assert analyzer.combine_signals("BUY", "BUY") == "TAILWIND"

    def test_combine_signals_strong_buy_both_returns_tailwind(self):
        analyzer = _make_analyzer()
        assert analyzer.combine_signals("STRONG_BUY", "STRONG_BUY") == "TAILWIND"

    def test_combine_signals_mixed_buy_returns_tailwind(self):
        """One BUY + one STRONG_BUY → TAILWIND."""
        analyzer = _make_analyzer()
        assert analyzer.combine_signals("BUY", "STRONG_BUY") == "TAILWIND"

    def test_combine_signals_both_sell_returns_headwind(self):
        analyzer = _make_analyzer()
        assert analyzer.combine_signals("SELL", "SELL") == "HEADWIND"

    def test_combine_signals_strong_sell_both_returns_headwind(self):
        analyzer = _make_analyzer()
        assert analyzer.combine_signals("STRONG_SELL", "STRONG_SELL") == "HEADWIND"

    def test_combine_signals_fii_buy_dii_sell_returns_mixed(self):
        analyzer = _make_analyzer()
        assert analyzer.combine_signals("BUY", "SELL") == "MIXED"

    def test_combine_signals_fii_neutral_returns_mixed(self):
        analyzer = _make_analyzer()
        assert analyzer.combine_signals("NEUTRAL", "BUY") == "MIXED"

    def test_combine_signals_both_neutral_returns_mixed(self):
        analyzer = _make_analyzer()
        assert analyzer.combine_signals("NEUTRAL", "NEUTRAL") == "MIXED"


# ---------------------------------------------------------------------------
# compute_regime integration test
# ---------------------------------------------------------------------------

class TestComputeRegime:
    """Tests for FlowRegimeAnalyzer.compute_regime."""

    def test_compute_regime_persists_to_flow_regime_daily(self):
        """compute_regime should read flows and write to flow_regime_daily."""
        engine = _make_mock_engine_with_flow(fii_net=3000.0, dii_net=2500.0)
        analyzer = _make_analyzer(db_engine=engine)

        regime = analyzer.compute_regime(TRADE_DATE)

        assert regime is not None
        assert regime["fii_regime"] == "STRONG_BUY"
        assert regime["dii_regime"] == "STRONG_BUY"
        assert regime["combined_signal"] == "TAILWIND"
        # Verify that begin() (write transaction) was called
        engine.begin.assert_called_once()

    def test_compute_regime_returns_none_when_no_flow_data(self):
        """When no flow data exists for the date, compute_regime should return None."""
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        no_data_result = MagicMock()
        no_data_result.fetchone.return_value = None
        mock_conn.execute.return_value = no_data_result

        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_conn

        analyzer = _make_analyzer(db_engine=mock_engine)
        result = analyzer.compute_regime(TRADE_DATE)

        assert result is None

    def test_compute_regime_headwind_when_both_selling(self):
        """Both FII and DII STRONG_SELL → HEADWIND combined signal."""
        engine = _make_mock_engine_with_flow(fii_net=-3000.0, dii_net=-2500.0)
        analyzer = _make_analyzer(db_engine=engine)

        regime = analyzer.compute_regime(TRADE_DATE)

        assert regime is not None
        assert regime["fii_regime"] == "STRONG_SELL"
        assert regime["dii_regime"] == "STRONG_SELL"
        assert regime["combined_signal"] == "HEADWIND"

    def test_compute_regime_mixed_signal(self):
        """FII buying + DII selling → MIXED combined signal."""
        engine = _make_mock_engine_with_flow(fii_net=2500.0, dii_net=-2500.0)
        analyzer = _make_analyzer(db_engine=engine)

        regime = analyzer.compute_regime(TRADE_DATE)

        assert regime is not None
        assert regime["combined_signal"] == "MIXED"
