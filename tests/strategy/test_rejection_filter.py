"""Tests for src/strategy/rejection_filter.py — anti-signal veto layer.

All tests use inline mocks (unittest.mock) and do NOT require a live database.
The DB engine is always replaced with a MagicMock that returns pre-configured
row data via its context-manager protocol.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta, timezone
from unittest.mock import MagicMock, call, patch

import pytest

from src.strategy.rejection_filter import (
    RejectionConfig,
    RejectionFilter,
    RejectionReason,
)

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

IST = timezone(timedelta(hours=5, minutes=30))

SYMBOL = "RELIANCE"
INSTRUMENT_KEY = "NSE_EQ|RELIANCE"
STRATEGY = "CPR_VWAP_Bounce_BankNifty"

# A trading time comfortably before 15:15 IST
_SAFE_TS = datetime(2026, 5, 8, 10, 0, 0, tzinfo=IST)
# A trading time after the 15-minute buffer (after 15:15 IST)
_LATE_TS = datetime(2026, 5, 8, 15, 20, 0, tzinfo=IST)

_TRADE_DATE = date(2026, 5, 8)


def _make_engine_with_scalar(value) -> MagicMock:
    """Build a mock DB engine whose scalar query returns *value*."""
    row = MagicMock()
    row.__getitem__ = MagicMock(side_effect=lambda i: value if i == 0 else None)
    result = MagicMock()
    result.fetchone.return_value = (value,) if value is not None else None

    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value = result

    engine = MagicMock()
    engine.connect.return_value = mock_conn
    return engine


def _make_engine_for_rejection_log() -> MagicMock:
    """Build a mock DB engine that returns id=99 from RETURNING clause."""
    row = MagicMock()
    row.__getitem__ = MagicMock(return_value=99)
    result = MagicMock()
    result.fetchone.return_value = row

    txn_conn = MagicMock()
    txn_conn.__enter__ = MagicMock(return_value=txn_conn)
    txn_conn.__exit__ = MagicMock(return_value=False)
    txn_conn.execute.return_value = result

    engine = MagicMock()
    engine.begin.return_value = txn_conn
    return engine


def _base_signal(
    signal_type: str = "BUY",
    symbol: str = SYMBOL,
    strategy: str = STRATEGY,
) -> dict:
    return {
        "strategy_name": strategy,
        "instrument_key": INSTRUMENT_KEY,
        "signal_type": signal_type,
        "symbol": symbol,
    }


def _base_context(
    spread_bps: float = 10.0,
    open_positions: list | None = None,
    ts: datetime | None = None,
    trade_date: date | None = None,
) -> dict:
    return {
        "current_spread_bps": spread_bps,
        "open_positions": open_positions or [],
        "current_ts": ts or _SAFE_TS,
        "trade_date": trade_date or _TRADE_DATE,
    }


# ---------------------------------------------------------------------------
# TestCheckSpread
# ---------------------------------------------------------------------------

class TestCheckSpread:
    def test_check_spread_rejects_above_threshold(self):
        filt = RejectionFilter(config=RejectionConfig(max_spread_bps=30.0))
        result = filt.check_spread(INSTRUMENT_KEY, current_spread_bps=50.0)
        assert result == RejectionReason.SPREAD_TOO_WIDE

    def test_check_spread_passes_below_threshold(self):
        filt = RejectionFilter(config=RejectionConfig(max_spread_bps=30.0))
        result = filt.check_spread(INSTRUMENT_KEY, current_spread_bps=15.0)
        assert result is None

    def test_check_spread_passes_at_exact_threshold(self):
        filt = RejectionFilter(config=RejectionConfig(max_spread_bps=30.0))
        result = filt.check_spread(INSTRUMENT_KEY, current_spread_bps=30.0)
        assert result is None  # threshold is exclusive (>)

    def test_check_spread_no_db_required(self):
        """Spread check is purely numeric — no DB needed."""
        filt = RejectionFilter(db_engine=None)
        assert filt.check_spread(INSTRUMENT_KEY, 100.0) == RejectionReason.SPREAD_TOO_WIDE


# ---------------------------------------------------------------------------
# TestCheckEarningsBlackout
# ---------------------------------------------------------------------------

class TestCheckEarningsBlackout:
    def test_check_earnings_blackout_rejects_in_window(self):
        with patch(
            "src.strategy.rejection_filter.EarningsCalendar",
            autospec=False,
        ) as MockCal:
            MockCal.return_value.is_blackout.return_value = (True, "earnings")
            filt = RejectionFilter(db_engine=MagicMock())
            result = filt.check_earnings_blackout(SYMBOL, _TRADE_DATE)
        assert result == RejectionReason.EARNINGS_BLACKOUT

    def test_check_earnings_blackout_passes_outside_window(self):
        with patch(
            "src.strategy.rejection_filter.EarningsCalendar",
            autospec=False,
        ) as MockCal:
            MockCal.return_value.is_blackout.return_value = (False, None)
            filt = RejectionFilter(db_engine=MagicMock())
            result = filt.check_earnings_blackout(SYMBOL, _TRADE_DATE)
        assert result is None

    def test_check_earnings_blackout_returns_none_without_db(self):
        filt = RejectionFilter(db_engine=None)
        result = filt.check_earnings_blackout(SYMBOL, _TRADE_DATE)
        assert result is None


# ---------------------------------------------------------------------------
# TestCheckCorporateBlackout
# ---------------------------------------------------------------------------

class TestCheckCorporateBlackout:
    def test_check_corporate_blackout_rejects_in_window(self):
        with patch(
            "src.strategy.rejection_filter.CorporateCalendar",
            autospec=False,
        ) as MockCal:
            MockCal.return_value.is_blackout.return_value = (True, "dividend ex-date")
            filt = RejectionFilter(db_engine=MagicMock())
            result = filt.check_corporate_blackout(SYMBOL, _TRADE_DATE)
        assert result == RejectionReason.CORPORATE_BLACKOUT

    def test_check_corporate_blackout_passes_outside_window(self):
        with patch(
            "src.strategy.rejection_filter.CorporateCalendar",
            autospec=False,
        ) as MockCal:
            MockCal.return_value.is_blackout.return_value = (False, None)
            filt = RejectionFilter(db_engine=MagicMock())
            result = filt.check_corporate_blackout(SYMBOL, _TRADE_DATE)
        assert result is None

    def test_check_corporate_blackout_returns_none_without_db(self):
        filt = RejectionFilter(db_engine=None)
        result = filt.check_corporate_blackout(SYMBOL, _TRADE_DATE)
        assert result is None


# ---------------------------------------------------------------------------
# TestCheckNews
# ---------------------------------------------------------------------------

class TestCheckNews:
    def test_check_news_rejects_long_on_negative(self):
        with patch(
            "src.strategy.rejection_filter.NewsQuery",
            autospec=False,
        ) as MockNQ:
            MockNQ.return_value.has_negative_news.return_value = True
            MockNQ.return_value.has_positive_news.return_value = False
            filt = RejectionFilter(db_engine=MagicMock())
            result = filt.check_news(SYMBOL, "BUY")
        assert result == RejectionReason.NEGATIVE_NEWS_RECENT

    def test_check_news_rejects_short_on_positive(self):
        with patch(
            "src.strategy.rejection_filter.NewsQuery",
            autospec=False,
        ) as MockNQ:
            MockNQ.return_value.has_negative_news.return_value = False
            MockNQ.return_value.has_positive_news.return_value = True
            filt = RejectionFilter(db_engine=MagicMock())
            result = filt.check_news(SYMBOL, "SELL")
        assert result == RejectionReason.POSITIVE_NEWS_RECENT

    def test_check_news_passes_long_on_no_news(self):
        with patch(
            "src.strategy.rejection_filter.NewsQuery",
            autospec=False,
        ) as MockNQ:
            MockNQ.return_value.has_negative_news.return_value = False
            filt = RejectionFilter(db_engine=MagicMock())
            result = filt.check_news(SYMBOL, "BUY")
        assert result is None

    def test_check_news_passes_sell_on_no_negative(self):
        with patch(
            "src.strategy.rejection_filter.NewsQuery",
            autospec=False,
        ) as MockNQ:
            MockNQ.return_value.has_positive_news.return_value = False
            filt = RejectionFilter(db_engine=MagicMock())
            result = filt.check_news(SYMBOL, "SELL")
        assert result is None

    def test_check_news_returns_none_without_db(self):
        filt = RejectionFilter(db_engine=None)
        assert filt.check_news(SYMBOL, "BUY") is None


# ---------------------------------------------------------------------------
# TestCheckVixSpike
# ---------------------------------------------------------------------------

class TestCheckVixSpike:
    def test_check_vix_spike_rejects_when_intraday_spike(self):
        engine = _make_engine_with_scalar(28.5)
        filt = RejectionFilter(db_engine=engine, config=RejectionConfig(vix_spike_check=True))
        result = filt.check_vix_spike()
        assert result == RejectionReason.VIX_SPIKE

    def test_check_vix_spike_passes_when_normal(self):
        engine = _make_engine_with_scalar(16.0)
        filt = RejectionFilter(db_engine=engine, config=RejectionConfig(vix_spike_check=True))
        result = filt.check_vix_spike()
        assert result is None

    def test_check_vix_spike_skipped_when_disabled(self):
        filt = RejectionFilter(db_engine=MagicMock(), config=RejectionConfig(vix_spike_check=False))
        result = filt.check_vix_spike()
        assert result is None

    def test_check_vix_spike_passes_when_no_db(self):
        filt = RejectionFilter(db_engine=None)
        assert filt.check_vix_spike() is None

    def test_check_vix_spike_passes_when_no_data(self):
        """No rows in vix_regime_intraday → skip check (None)."""
        engine = _make_engine_with_scalar(None)
        # Override fetchone to return None explicitly
        engine.connect.return_value.__enter__.return_value.execute.return_value.fetchone.return_value = None
        filt = RejectionFilter(db_engine=engine, config=RejectionConfig(vix_spike_check=True))
        result = filt.check_vix_spike()
        assert result is None


# ---------------------------------------------------------------------------
# TestCheckMarketClose
# ---------------------------------------------------------------------------

class TestCheckMarketClose:
    def test_check_market_close_rejects_after_buffer_time(self):
        """15:20 IST is past the default 15:15 IST cutoff."""
        late_ts = datetime(2026, 5, 8, 15, 20, 0, tzinfo=IST)
        filt = RejectionFilter(config=RejectionConfig(market_close_minutes_buffer=15))
        result = filt.check_market_close(late_ts)
        assert result == RejectionReason.NEAR_MARKET_CLOSE

    def test_check_market_close_passes_before_buffer_time(self):
        """10:00 IST is well before the cutoff."""
        early_ts = datetime(2026, 5, 8, 10, 0, 0, tzinfo=IST)
        filt = RejectionFilter(config=RejectionConfig(market_close_minutes_buffer=15))
        result = filt.check_market_close(early_ts)
        assert result is None

    def test_check_market_close_exact_cutoff_rejects(self):
        """15:15 IST is exactly at cutoff — should be rejected (>=)."""
        cutoff_ts = datetime(2026, 5, 8, 15, 15, 0, tzinfo=IST)
        filt = RejectionFilter(config=RejectionConfig(market_close_minutes_buffer=15))
        result = filt.check_market_close(cutoff_ts)
        assert result == RejectionReason.NEAR_MARKET_CLOSE

    def test_check_market_close_handles_naive_datetime(self):
        """Naive datetime (assumed IST) after cutoff should be rejected."""
        naive_ts = datetime(2026, 5, 8, 15, 20, 0)  # no tzinfo
        filt = RejectionFilter(config=RejectionConfig(market_close_minutes_buffer=15))
        result = filt.check_market_close(naive_ts)
        assert result == RejectionReason.NEAR_MARKET_CLOSE

    def test_check_market_close_handles_utc_datetime(self):
        """UTC datetime converted to IST correctly."""
        # 15:20 IST = 09:50 UTC
        utc_ts = datetime(2026, 5, 8, 9, 50, 0, tzinfo=timezone.utc)
        filt = RejectionFilter(config=RejectionConfig(market_close_minutes_buffer=15))
        result = filt.check_market_close(utc_ts)
        assert result == RejectionReason.NEAR_MARKET_CLOSE


# ---------------------------------------------------------------------------
# TestCheckSector
# ---------------------------------------------------------------------------

class TestCheckSector:
    def _engine_with_rank(self, rs_rank: int, total: int) -> MagicMock:
        row = (rs_rank, total)
        result = MagicMock()
        result.fetchone.return_value = row
        conn = MagicMock()
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=False)
        conn.execute.return_value = result
        engine = MagicMock()
        engine.connect.return_value = conn
        return engine

    def test_check_sector_rejects_long_in_bottom_quartile(self):
        """Rank 8 out of 9 (bottom quartile) → reject LONG."""
        engine = self._engine_with_rank(8, 9)
        filt = RejectionFilter(db_engine=engine)
        result = filt.check_sector("RELIANCE", "BUY")
        assert result == RejectionReason.SECTOR_BOTTOM_QUARTILE

    def test_check_sector_passes_long_in_top_quartile(self):
        """Rank 2 out of 9 (top) → allow LONG."""
        engine = self._engine_with_rank(2, 9)
        filt = RejectionFilter(db_engine=engine)
        result = filt.check_sector("RELIANCE", "BUY")
        assert result is None

    def test_check_sector_passes_short_in_bottom_quartile(self):
        """Sector filter only vetoes LONG — SHORT signals pass regardless."""
        engine = self._engine_with_rank(9, 9)
        filt = RejectionFilter(db_engine=engine)
        result = filt.check_sector("RELIANCE", "SELL")
        assert result is None

    def test_check_sector_skipped_when_disabled(self):
        filt = RejectionFilter(db_engine=MagicMock(), config=RejectionConfig(sector_filter_enabled=False))
        result = filt.check_sector(SYMBOL, "BUY")
        assert result is None

    def test_check_sector_passes_for_unknown_symbol(self):
        """Symbol not in sector map → skip check."""
        filt = RejectionFilter(db_engine=MagicMock())
        result = filt.check_sector("UNKNOWNSYM", "BUY")
        assert result is None


# ---------------------------------------------------------------------------
# TestCheckCorrelatedPositions
# ---------------------------------------------------------------------------

class TestCheckCorrelatedPositions:
    def test_check_correlated_positions_rejects_when_cap_exceeded(self):
        """3 LONG NIFTY_BANK positions already → reject 4th LONG on HDFCBANK."""
        config = RejectionConfig(correlated_position_max_count=3)
        filt = RejectionFilter(config=config)
        open_positions = [
            {"symbol": "HDFCBANK", "side": "LONG"},
            {"symbol": "ICICIBANK", "side": "LONG"},
            {"symbol": "AXISBANK", "side": "LONG"},
        ]
        result = filt.check_correlated_positions("KOTAKBANK", "BUY", open_positions)
        assert result == RejectionReason.CORRELATED_POSITION_CAP

    def test_check_correlated_positions_passes_below_cap(self):
        """2 LONG NIFTY_BANK positions → allow 3rd (cap=3)."""
        config = RejectionConfig(correlated_position_max_count=3)
        filt = RejectionFilter(config=config)
        open_positions = [
            {"symbol": "HDFCBANK", "side": "LONG"},
            {"symbol": "ICICIBANK", "side": "LONG"},
        ]
        result = filt.check_correlated_positions("AXISBANK", "BUY", open_positions)
        assert result is None

    def test_check_correlated_positions_ignores_opposite_side(self):
        """LONG positions do not count against SHORT cap."""
        config = RejectionConfig(correlated_position_max_count=2)
        filt = RejectionFilter(config=config)
        open_positions = [
            {"symbol": "HDFCBANK", "side": "LONG"},
            {"symbol": "ICICIBANK", "side": "LONG"},
            {"symbol": "AXISBANK", "side": "LONG"},
        ]
        result = filt.check_correlated_positions("KOTAKBANK", "SELL", open_positions)
        assert result is None

    def test_check_correlated_positions_passes_empty_positions(self):
        filt = RejectionFilter()
        result = filt.check_correlated_positions("RELIANCE", "BUY", [])
        assert result is None


# ---------------------------------------------------------------------------
# TestCheckLiquidity
# ---------------------------------------------------------------------------

class TestCheckLiquidity:
    def _engine_with_adv(self, adv_paisa: int | None) -> MagicMock:
        row = (adv_paisa,) if adv_paisa is not None else None
        result = MagicMock()
        result.fetchone.return_value = row
        conn = MagicMock()
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=False)
        conn.execute.return_value = result
        engine = MagicMock()
        engine.connect.return_value = conn
        return engine

    def test_check_liquidity_rejects_below_min_adv(self):
        """ADV below threshold → reject."""
        engine = self._engine_with_adv(10_00_00_000)  # ₹1 crore — below default ₹50 cr
        filt = RejectionFilter(db_engine=engine)
        result = filt.check_liquidity(SYMBOL, _TRADE_DATE)
        assert result == RejectionReason.LIQUIDITY_LOW

    def test_check_liquidity_passes_above_min_adv(self):
        """ADV above threshold → pass."""
        engine = self._engine_with_adv(200_00_00_000)  # ₹200 crore
        filt = RejectionFilter(db_engine=engine)
        result = filt.check_liquidity(SYMBOL, _TRADE_DATE)
        assert result is None

    def test_check_liquidity_passes_when_no_data(self):
        """No ADV row in DB → skip check (don't block)."""
        engine = self._engine_with_adv(None)
        filt = RejectionFilter(db_engine=engine)
        result = filt.check_liquidity(SYMBOL, _TRADE_DATE)
        assert result is None

    def test_check_liquidity_returns_none_without_db(self):
        filt = RejectionFilter(db_engine=None)
        assert filt.check_liquidity(SYMBOL, _TRADE_DATE) is None


# ---------------------------------------------------------------------------
# TestCheckInsiderAlignment
# ---------------------------------------------------------------------------

class TestCheckInsiderAlignment:
    def test_check_insider_alignment_rejects_long_on_promoter_selling(self):
        with patch(
            "src.strategy.rejection_filter.InsiderSignals",
            autospec=False,
        ) as MockIS:
            MockIS.return_value.promoter_selling_recent.return_value = True
            MockIS.return_value.promoter_buying_recent.return_value = False
            filt = RejectionFilter(db_engine=MagicMock())
            result = filt.check_insider_alignment(SYMBOL, "BUY")
        assert result == RejectionReason.PROMOTER_SELLING

    def test_check_insider_alignment_rejects_short_on_promoter_buying(self):
        with patch(
            "src.strategy.rejection_filter.InsiderSignals",
            autospec=False,
        ) as MockIS:
            MockIS.return_value.promoter_buying_recent.return_value = True
            filt = RejectionFilter(db_engine=MagicMock())
            result = filt.check_insider_alignment(SYMBOL, "SELL")
        assert result == RejectionReason.PROMOTER_BUYING

    def test_check_insider_alignment_passes_long_no_selling(self):
        with patch(
            "src.strategy.rejection_filter.InsiderSignals",
            autospec=False,
        ) as MockIS:
            MockIS.return_value.promoter_selling_recent.return_value = False
            filt = RejectionFilter(db_engine=MagicMock())
            result = filt.check_insider_alignment(SYMBOL, "BUY")
        assert result is None

    def test_check_insider_alignment_returns_none_without_db(self):
        filt = RejectionFilter(db_engine=None)
        assert filt.check_insider_alignment(SYMBOL, "BUY") is None


# ---------------------------------------------------------------------------
# TestEvaluate — integration-level tests for evaluate()
# ---------------------------------------------------------------------------

class TestEvaluate:
    def _filt_all_checks_pass(self) -> RejectionFilter:
        """Return a RejectionFilter where every check passes."""
        filt = RejectionFilter(db_engine=None, config=RejectionConfig())
        # Patch every check to return None (pass)
        for attr in [
            "check_market_close",
            "check_spread",
            "check_earnings_blackout",
            "check_corporate_blackout",
            "check_news",
            "check_insider_alignment",
            "check_sector",
            "check_vix_spike",
            "check_correlated_positions",
            "check_oi_dropping",
            "check_liquidity",
        ]:
            setattr(filt, attr, MagicMock(return_value=None))
        return filt

    def test_evaluate_passes_when_all_checks_clear(self):
        filt = self._filt_all_checks_pass()
        allowed, reason, details = filt.evaluate(
            signal=_base_signal(),
            context=_base_context(),
        )
        assert allowed is True
        assert reason == RejectionReason.NONE
        assert details == {}

    def test_evaluate_short_circuits_on_first_failure(self):
        """market_close fails → downstream checks are NOT called."""
        filt = RejectionFilter(db_engine=None)
        filt.check_market_close = MagicMock(return_value=RejectionReason.NEAR_MARKET_CLOSE)
        filt.check_spread = MagicMock(return_value=None)
        filt.check_earnings_blackout = MagicMock(return_value=None)
        filt.check_corporate_blackout = MagicMock(return_value=None)
        filt.check_news = MagicMock(return_value=None)
        filt.check_insider_alignment = MagicMock(return_value=None)
        filt.check_sector = MagicMock(return_value=None)
        filt.check_vix_spike = MagicMock(return_value=None)
        filt.check_correlated_positions = MagicMock(return_value=None)
        filt.check_oi_dropping = MagicMock(return_value=None)
        filt.check_liquidity = MagicMock(return_value=None)
        filt.log_rejection = MagicMock(return_value=1)

        allowed, reason, details = filt.evaluate(
            signal=_base_signal(),
            context=_base_context(ts=_LATE_TS),
        )
        assert allowed is False
        assert reason == RejectionReason.NEAR_MARKET_CLOSE
        # spread and subsequent checks should NOT have been called
        filt.check_spread.assert_not_called()
        filt.check_earnings_blackout.assert_not_called()

    def test_evaluate_spread_failure_after_market_close_passes(self):
        """market_close passes → spread check runs and can fail."""
        filt = RejectionFilter(db_engine=None, config=RejectionConfig(max_spread_bps=30.0))
        filt.check_market_close = MagicMock(return_value=None)  # pass
        filt.check_earnings_blackout = MagicMock(return_value=None)
        filt.check_corporate_blackout = MagicMock(return_value=None)
        filt.check_news = MagicMock(return_value=None)
        filt.check_insider_alignment = MagicMock(return_value=None)
        filt.check_sector = MagicMock(return_value=None)
        filt.check_vix_spike = MagicMock(return_value=None)
        filt.check_correlated_positions = MagicMock(return_value=None)
        filt.check_oi_dropping = MagicMock(return_value=None)
        filt.check_liquidity = MagicMock(return_value=None)
        filt.log_rejection = MagicMock(return_value=1)

        allowed, reason, details = filt.evaluate(
            signal=_base_signal(),
            context=_base_context(spread_bps=50.0),  # wide spread → fail
        )
        assert allowed is False
        assert reason == RejectionReason.SPREAD_TOO_WIDE

    def test_evaluate_calls_log_rejection_on_failure(self):
        filt = RejectionFilter(db_engine=None)
        filt.check_market_close = MagicMock(return_value=RejectionReason.NEAR_MARKET_CLOSE)
        filt.check_spread = MagicMock(return_value=None)
        filt.check_earnings_blackout = MagicMock(return_value=None)
        filt.check_corporate_blackout = MagicMock(return_value=None)
        filt.check_news = MagicMock(return_value=None)
        filt.check_insider_alignment = MagicMock(return_value=None)
        filt.check_sector = MagicMock(return_value=None)
        filt.check_vix_spike = MagicMock(return_value=None)
        filt.check_correlated_positions = MagicMock(return_value=None)
        filt.check_oi_dropping = MagicMock(return_value=None)
        filt.check_liquidity = MagicMock(return_value=None)
        filt.log_rejection = MagicMock(return_value=5)

        sig = _base_signal()
        ctx = _base_context()
        filt.evaluate(signal=sig, context=ctx)
        filt.log_rejection.assert_called_once()
        call_args = filt.log_rejection.call_args
        assert call_args[0][0] == sig
        assert call_args[0][1] == RejectionReason.NEAR_MARKET_CLOSE


# ---------------------------------------------------------------------------
# TestLogRejection
# ---------------------------------------------------------------------------

class TestLogRejection:
    def test_log_rejection_persists_to_db(self):
        """log_rejection inserts a row and returns the id."""
        row = MagicMock()
        row.__getitem__ = MagicMock(return_value=42)
        result = MagicMock()
        result.fetchone.return_value = row

        txn_conn = MagicMock()
        txn_conn.__enter__ = MagicMock(return_value=txn_conn)
        txn_conn.__exit__ = MagicMock(return_value=False)
        txn_conn.execute.return_value = result

        engine = MagicMock()
        engine.begin.return_value = txn_conn

        filt = RejectionFilter(db_engine=engine)
        rejection_id = filt.log_rejection(
            signal=_base_signal(),
            reason=RejectionReason.SPREAD_TOO_WIDE,
            details={"spread_bps": 55.0},
        )
        assert rejection_id == 42
        # Verify DB.begin() was called (transactional insert)
        engine.begin.assert_called_once()

    def test_log_rejection_returns_minus_one_without_db(self):
        filt = RejectionFilter(db_engine=None)
        result = filt.log_rejection(
            signal=_base_signal(),
            reason=RejectionReason.SPREAD_TOO_WIDE,
            details={"spread_bps": 55.0},
        )
        assert result == -1

    def test_log_rejection_returns_minus_one_on_db_error(self):
        engine = MagicMock()
        txn_conn = MagicMock()
        txn_conn.__enter__ = MagicMock(return_value=txn_conn)
        txn_conn.__exit__ = MagicMock(return_value=False)
        txn_conn.execute.side_effect = RuntimeError("DB down")
        engine.begin.return_value = txn_conn

        filt = RejectionFilter(db_engine=engine)
        result = filt.log_rejection(
            signal=_base_signal(),
            reason=RejectionReason.EARNINGS_BLACKOUT,
            details={},
        )
        assert result == -1


# ---------------------------------------------------------------------------
# TestGetRejectionStats
# ---------------------------------------------------------------------------

class TestGetRejectionStats:
    def _engine_with_stats_rows(self, rows: list[tuple]) -> MagicMock:
        result = MagicMock()
        result.fetchall.return_value = rows
        conn = MagicMock()
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=False)
        conn.execute.return_value = result
        engine = MagicMock()
        engine.connect.return_value = conn
        return engine

    def test_get_rejection_stats_aggregates_by_reason(self):
        """Returned dict has total_rejections and by_reason breakdown."""
        rows = [
            ("SPREAD_TOO_WIDE", 5),
            ("NEAR_MARKET_CLOSE", 3),
            ("EARNINGS_BLACKOUT", 1),
        ]
        engine = self._engine_with_stats_rows(rows)
        filt = RejectionFilter(db_engine=engine)
        stats = filt.get_rejection_stats(strategy_name=STRATEGY, days=7)
        assert stats["total_rejections"] == 9
        assert stats["by_reason"]["SPREAD_TOO_WIDE"] == 5
        assert stats["by_reason"]["NEAR_MARKET_CLOSE"] == 3
        assert stats["by_reason"]["EARNINGS_BLACKOUT"] == 1

    def test_get_rejection_stats_all_strategies(self):
        """Passing strategy_name=None queries all strategies."""
        rows = [("VIX_SPIKE", 2)]
        engine = self._engine_with_stats_rows(rows)
        filt = RejectionFilter(db_engine=engine)
        stats = filt.get_rejection_stats(strategy_name=None, days=7)
        assert stats["total_rejections"] == 2

    def test_get_rejection_stats_returns_zeros_without_db(self):
        filt = RejectionFilter(db_engine=None)
        stats = filt.get_rejection_stats()
        assert stats == {"total_rejections": 0, "by_reason": {}}

    def test_get_rejection_stats_returns_zeros_on_db_error(self):
        engine = MagicMock()
        conn = MagicMock()
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=False)
        conn.execute.side_effect = RuntimeError("DB down")
        engine.connect.return_value = conn

        filt = RejectionFilter(db_engine=engine)
        stats = filt.get_rejection_stats(days=7)
        assert stats == {"total_rejections": 0, "by_reason": {}}
