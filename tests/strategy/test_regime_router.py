"""Tests for Wave 5.3 — Regime Router.

All tests use mocked DB engines so no real Postgres connection is required.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest

from src.strategy.regime_router import (
    MarketRegime,
    RegimeContext,
    RegimeRouter,
    StrategyType,
    REGIME_ALLOWLIST,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TODAY = date(2026, 5, 9)


def _make_router(engine: Any = None) -> RegimeRouter:
    return RegimeRouter(db_engine=engine)


def _ctx(
    vix: str = "NORMAL",
    breadth: float = 0.5,
    macro: str = "NEUTRAL",
    flow: str = "MIXED",
) -> RegimeContext:
    return RegimeContext(
        vix_regime=vix,
        sector_breadth=breadth,
        macro_signal=macro,
        flow_signal=flow,
    )


# ---------------------------------------------------------------------------
# classify_regime tests
# ---------------------------------------------------------------------------


class TestClassifyRegime:

    def test_classify_regime_vix_spike_returns_high_vol(self):
        router = _make_router()
        ctx = _ctx(vix="SPIKE", breadth=0.8, macro="BULLISH", flow="TAILWIND")
        assert router.classify_regime(ctx) == MarketRegime.HIGH_VOL

    def test_classify_regime_low_vix_strong_breadth_bullish_returns_trend_bull(self):
        router = _make_router()
        ctx = _ctx(vix="LOW", breadth=0.7, macro="BULLISH", flow="TAILWIND")
        assert router.classify_regime(ctx) == MarketRegime.TREND_BULL

    def test_classify_regime_low_vix_weak_breadth_returns_trend_bear(self):
        router = _make_router()
        # breadth < 0.4 → TREND_BEAR regardless of macro/flow
        ctx = _ctx(vix="LOW", breadth=0.3, macro="BULLISH", flow="TAILWIND")
        assert router.classify_regime(ctx) == MarketRegime.TREND_BEAR

    def test_classify_regime_high_vix_headwind_returns_chop(self):
        router = _make_router()
        ctx = _ctx(vix="HIGH", breadth=0.5, macro="NEUTRAL", flow="HEADWIND")
        assert router.classify_regime(ctx) == MarketRegime.CHOP

    def test_classify_regime_default_returns_range(self):
        router = _make_router()
        # NORMAL VIX, mid breadth, neutral macro, mixed flow → RANGE
        ctx = _ctx(vix="NORMAL", breadth=0.5, macro="NEUTRAL", flow="MIXED")
        assert router.classify_regime(ctx) == MarketRegime.RANGE

    def test_classify_regime_normal_vix_breadth_below_threshold_returns_trend_bear(self):
        router = _make_router()
        ctx = _ctx(vix="NORMAL", breadth=0.2, macro="BEARISH", flow="HEADWIND")
        assert router.classify_regime(ctx) == MarketRegime.TREND_BEAR

    def test_classify_regime_low_vix_high_breadth_bearish_macro_returns_range(self):
        """LOW VIX + high breadth but BEARISH macro → not TREND_BULL, defaults RANGE."""
        router = _make_router()
        ctx = _ctx(vix="LOW", breadth=0.65, macro="BEARISH", flow="MIXED")
        # Rule 3 requires macro BULLISH; falls through to default
        assert router.classify_regime(ctx) == MarketRegime.RANGE

    def test_classify_regime_spike_beats_all_other_rules(self):
        """VIX SPIKE overrides even strong-breadth + bullish macro."""
        router = _make_router()
        ctx = _ctx(vix="SPIKE", breadth=0.9, macro="BULLISH", flow="TAILWIND")
        assert router.classify_regime(ctx) == MarketRegime.HIGH_VOL


# ---------------------------------------------------------------------------
# decide_allowed_strategies tests
# ---------------------------------------------------------------------------


class TestDecideAllowedStrategies:

    def test_decide_allowed_strategies_trend_bull_includes_trend_breakout(self):
        router = _make_router()
        allowed = router.decide_allowed_strategies(MarketRegime.TREND_BULL)
        assert StrategyType.TREND in allowed
        assert StrategyType.BREAKOUT in allowed

    def test_decide_allowed_strategies_trend_bull_excludes_mean_rev(self):
        router = _make_router()
        allowed = router.decide_allowed_strategies(MarketRegime.TREND_BULL)
        assert StrategyType.MEAN_REV not in allowed

    def test_decide_allowed_strategies_chop_only_options_sell(self):
        router = _make_router()
        allowed = router.decide_allowed_strategies(MarketRegime.CHOP)
        assert allowed == [StrategyType.OPTIONS_SELL]

    def test_decide_allowed_strategies_range_includes_mean_rev_options_sell(self):
        router = _make_router()
        allowed = router.decide_allowed_strategies(MarketRegime.RANGE)
        assert StrategyType.MEAN_REV in allowed
        assert StrategyType.OPTIONS_SELL in allowed

    def test_decide_allowed_strategies_high_vol_includes_options_buy_mean_rev(self):
        router = _make_router()
        allowed = router.decide_allowed_strategies(MarketRegime.HIGH_VOL)
        assert StrategyType.OPTIONS_BUY in allowed
        assert StrategyType.MEAN_REV in allowed

    def test_decide_allowed_strategies_trend_bear_no_options_sell(self):
        router = _make_router()
        allowed = router.decide_allowed_strategies(MarketRegime.TREND_BEAR)
        assert StrategyType.OPTIONS_SELL not in allowed


# ---------------------------------------------------------------------------
# evaluate_today + persistence tests
# ---------------------------------------------------------------------------


def _build_mock_engine(
    vix_regime: str = "NORMAL",
    sector_pos: int = 5,
    sector_total: int = 9,
    macro_trends: list[str] | None = None,
    flow_signal: str = "MIXED",
    existing_decision: dict | None = None,
) -> MagicMock:
    """Build a mock SQLAlchemy sync engine that returns preset query results."""
    if macro_trends is None:
        macro_trends = ["UP", "DOWN", "UP"]

    engine = MagicMock()
    conn = MagicMock()

    engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    engine.begin.return_value.__enter__ = MagicMock(return_value=conn)
    engine.begin.return_value.__exit__ = MagicMock(return_value=False)

    # Each execute() call returns different results based on call order
    vix_row = MagicMock()
    vix_row.__getitem__ = MagicMock(side_effect=lambda i: vix_regime)

    sector_row = MagicMock()
    sector_row.__getitem__ = MagicMock(
        side_effect=lambda i: sector_pos if i == 0 else sector_total
    )
    sector_row.__bool__ = MagicMock(return_value=True)

    macro_rows = [MagicMock() for _ in macro_trends]
    for row, trend in zip(macro_rows, macro_trends):
        row.__getitem__ = MagicMock(side_effect=lambda i, t=trend: t)

    flow_row = MagicMock()
    flow_row.__getitem__ = MagicMock(side_effect=lambda i: flow_signal if i == 0 else None)

    # Map call sequences
    call_results = [
        vix_row,          # vix_regime_daily
        sector_row,       # sector_rank_daily
        macro_rows,       # macro_regime_daily (list)
        flow_row,         # flow_regime_daily
    ]
    call_idx = [0]

    def _execute(*args, **kwargs):
        result = MagicMock()
        idx = call_idx[0]
        call_idx[0] += 1
        if idx == 0:
            result.fetchone.return_value = call_results[0]
        elif idx == 1:
            result.fetchone.return_value = call_results[1]
        elif idx == 2:
            result.fetchall.return_value = call_results[2]
        elif idx == 3:
            result.fetchone.return_value = call_results[3]
        else:
            # get_today_decision query
            if existing_decision:
                dr = MagicMock()
                dr.__getitem__ = MagicMock(side_effect=lambda i: [
                    existing_decision.get("trade_date"),
                    existing_decision.get("vix_regime"),
                    existing_decision.get("sector_breadth"),
                    existing_decision.get("macro_signal"),
                    existing_decision.get("flow_signal"),
                    existing_decision.get("market_regime"),
                    existing_decision.get("allowed_strategy_types", []),
                    existing_decision.get("blocked_strategy_types", {}),
                    existing_decision.get("evaluated_at"),
                ][i])
                result.fetchone.return_value = dr
            else:
                result.fetchone.return_value = None
        return result

    conn.execute.side_effect = _execute
    return engine


class TestEvaluateToday:

    def test_evaluate_today_persists_decision(self):
        """evaluate_today should call engine.begin() to persist the row."""
        engine = MagicMock()
        conn = MagicMock()
        engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
        engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        engine.begin.return_value.__enter__ = MagicMock(return_value=conn)
        engine.begin.return_value.__exit__ = MagicMock(return_value=False)

        # fetch_context returns None rows (triggers neutral fallback for each)
        exec_result = MagicMock()
        exec_result.fetchone.return_value = None
        exec_result.fetchall.return_value = []
        conn.execute.return_value = exec_result

        router = _make_router(engine)
        result = router.evaluate_today(_TODAY)

        # engine.begin() should have been called once (for the INSERT)
        engine.begin.assert_called_once()
        assert "market_regime" in result
        assert "allowed_strategy_types" in result
        assert isinstance(result["allowed_strategy_types"], list)

    def test_evaluate_today_returns_full_record(self):
        """evaluate_today should return all expected keys."""
        engine = MagicMock()
        conn = MagicMock()
        engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
        engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        engine.begin.return_value.__enter__ = MagicMock(return_value=conn)
        engine.begin.return_value.__exit__ = MagicMock(return_value=False)

        exec_result = MagicMock()
        exec_result.fetchone.return_value = None
        exec_result.fetchall.return_value = []
        conn.execute.return_value = exec_result

        router = _make_router(engine)
        result = router.evaluate_today(_TODAY)

        required_keys = {
            "trade_date", "vix_regime", "sector_breadth", "macro_signal",
            "flow_signal", "market_regime", "allowed_strategy_types",
            "blocked_strategy_types", "evaluated_at",
        }
        assert required_keys.issubset(set(result.keys()))

    def test_evaluate_today_no_engine_uses_fallback(self):
        """Without a DB engine, evaluate_today still returns a valid decision."""
        router = _make_router(engine=None)
        result = router.evaluate_today(_TODAY)
        # Neutral fallback → RANGE regime
        assert result["market_regime"] == MarketRegime.RANGE.value
        assert isinstance(result["allowed_strategy_types"], list)


# ---------------------------------------------------------------------------
# get_today_decision roundtrip
# ---------------------------------------------------------------------------


class TestGetTodayDecision:

    def test_get_today_decision_roundtrip(self):
        """get_today_decision returns values consistent with what evaluate_today computes."""
        # Use a router without DB to get the regime computed purely by classify logic.
        router_no_db = _make_router(engine=None)
        eval_result = router_no_db.evaluate_today(_TODAY)
        expected_regime = eval_result["market_regime"]
        expected_allowed = eval_result["allowed_strategy_types"]

        # Now build a mock engine whose SELECT returns those values.
        engine = MagicMock()
        conn = MagicMock()
        engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
        engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        row = MagicMock()
        row.__getitem__ = MagicMock(side_effect=lambda i: [
            _TODAY,
            "NORMAL",
            0.5,
            "NEUTRAL",
            "MIXED",
            expected_regime,
            expected_allowed,          # list — psycopg returns JSONB as Python list
            {},
            None,
        ][i])

        exec_result = MagicMock()
        exec_result.fetchone.return_value = row
        conn.execute.return_value = exec_result

        router2 = _make_router(engine)
        decision = router2.get_today_decision(_TODAY)

        assert decision is not None
        assert decision["market_regime"] == expected_regime
        assert decision["allowed_strategy_types"] == expected_allowed

    def test_get_today_decision_returns_none_when_no_row(self):
        engine = MagicMock()
        conn = MagicMock()
        engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
        engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        exec_result = MagicMock()
        exec_result.fetchone.return_value = None
        conn.execute.return_value = exec_result

        router = _make_router(engine)
        assert router.get_today_decision(_TODAY) is None

    def test_get_today_decision_returns_none_without_engine(self):
        router = _make_router(engine=None)
        assert router.get_today_decision(_TODAY) is None


# ---------------------------------------------------------------------------
# is_strategy_type_allowed
# ---------------------------------------------------------------------------


class TestIsStrategyTypeAllowed:

    def _make_router_with_decision(self, allowed_types: list[str]) -> RegimeRouter:
        """Return a router whose get_today_decision() returns *allowed_types*."""
        engine = MagicMock()
        conn = MagicMock()
        engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
        engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        row = MagicMock()
        blocked = {st: f"blocked in test" for st in ["trend","mean_rev","breakout","options_sell","options_buy"] if st not in allowed_types}
        row.__getitem__ = MagicMock(side_effect=lambda i: [
            _TODAY,
            "NORMAL",
            0.5,
            "NEUTRAL",
            "MIXED",
            "RANGE",
            allowed_types,    # list — psycopg returns JSONB as Python list
            blocked,
            None,
        ][i])

        exec_result = MagicMock()
        exec_result.fetchone.return_value = row
        conn.execute.return_value = exec_result

        return _make_router(engine)

    def test_is_strategy_type_allowed_true_when_in_list(self):
        router = self._make_router_with_decision(["trend", "breakout"])
        assert router.is_strategy_type_allowed("trend", _TODAY) is True

    def test_is_strategy_type_allowed_false_when_blocked(self):
        router = self._make_router_with_decision(["options_sell"])
        assert router.is_strategy_type_allowed("trend", _TODAY) is False

    def test_is_strategy_type_allowed_permissive_when_no_decision(self):
        """If no decision exists, default to True (permissive fallback)."""
        engine = MagicMock()
        conn = MagicMock()
        engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
        engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        exec_result = MagicMock()
        exec_result.fetchone.return_value = None
        conn.execute.return_value = exec_result

        router = _make_router(engine)
        assert router.is_strategy_type_allowed("trend", _TODAY) is True


# ---------------------------------------------------------------------------
# get_blocked_reason
# ---------------------------------------------------------------------------


class TestGetBlockedReason:

    def _make_router_with_blocked(self, blocked: dict[str, str]) -> RegimeRouter:
        engine = MagicMock()
        conn = MagicMock()
        engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
        engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        allowed_types = [st for st in ["trend","mean_rev","breakout","options_sell","options_buy"] if st not in blocked]

        row = MagicMock()
        row.__getitem__ = MagicMock(side_effect=lambda i: [
            _TODAY,
            "HIGH",
            0.4,
            "NEUTRAL",
            "HEADWIND",
            "CHOP",
            allowed_types,
            blocked,
            None,
        ][i])

        exec_result = MagicMock()
        exec_result.fetchone.return_value = row
        conn.execute.return_value = exec_result

        return _make_router(engine)

    def test_get_blocked_reason_returns_regime_label(self):
        blocked = {"trend": "Choppy market — trend signals unreliable"}
        router = self._make_router_with_blocked(blocked)
        reason = router.get_blocked_reason("trend", _TODAY)
        assert reason is not None
        assert "choppy" in reason.lower() or "trend" in reason.lower()

    def test_get_blocked_reason_returns_none_for_allowed(self):
        blocked = {"trend": "Choppy market — trend signals unreliable"}
        router = self._make_router_with_blocked(blocked)
        # options_sell is allowed in CHOP
        reason = router.get_blocked_reason("options_sell", _TODAY)
        assert reason is None

    def test_get_blocked_reason_returns_none_when_no_decision(self):
        engine = MagicMock()
        conn = MagicMock()
        engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
        engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        exec_result = MagicMock()
        exec_result.fetchone.return_value = None
        conn.execute.return_value = exec_result

        router = _make_router(engine)
        assert router.get_blocked_reason("trend", _TODAY) is None


# ---------------------------------------------------------------------------
# REGIME_ALLOWLIST integrity
# ---------------------------------------------------------------------------


class TestRegimeAllowlist:

    def test_all_regimes_have_allowlist_entries(self):
        for regime in MarketRegime:
            assert regime in REGIME_ALLOWLIST, f"No allowlist for {regime}"

    def test_chop_only_options_sell_in_allowlist(self):
        assert REGIME_ALLOWLIST[MarketRegime.CHOP] == [StrategyType.OPTIONS_SELL]

    def test_trend_bull_contains_expected_types(self):
        allowed = REGIME_ALLOWLIST[MarketRegime.TREND_BULL]
        assert StrategyType.TREND in allowed
        assert StrategyType.BREAKOUT in allowed
        assert StrategyType.OPTIONS_BUY in allowed

    def test_allowlist_values_are_strategy_type_instances(self):
        for regime, types in REGIME_ALLOWLIST.items():
            for t in types:
                assert isinstance(t, StrategyType), f"Non-StrategyType in {regime}: {t}"


# ---------------------------------------------------------------------------
# fetch_context fallback
# ---------------------------------------------------------------------------


class TestFetchContext:

    def test_fetch_context_no_engine_returns_neutral(self):
        router = _make_router(engine=None)
        ctx = router.fetch_context(_TODAY)
        assert ctx.vix_regime == "NORMAL"
        assert ctx.sector_breadth == 0.5
        assert ctx.macro_signal == "NEUTRAL"
        assert ctx.flow_signal == "MIXED"

    def test_fetch_context_handles_empty_db_rows_gracefully(self):
        engine = MagicMock()
        conn = MagicMock()
        engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
        engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        exec_result = MagicMock()
        exec_result.fetchone.return_value = None
        exec_result.fetchall.return_value = []
        conn.execute.return_value = exec_result

        router = _make_router(engine)
        ctx = router.fetch_context(_TODAY)
        # Should fall back to defaults
        assert ctx.vix_regime == "NORMAL"
        assert ctx.sector_breadth == 0.5
