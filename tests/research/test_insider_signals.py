"""Tests for src/research/insider_signals.py — insider/block flow aggregations."""
from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

import pytest

from src.research.insider_signals import InsiderSignals


def _engine_with_block_flow_rows(rows):
    """Engine where conn.execute(...).fetchall() returns row tuples (side, total_value, count)."""
    engine = MagicMock()
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchall.return_value = rows
    cursor.fetchone.return_value = rows[0] if rows else None
    conn.execute.return_value = cursor
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    engine.connect.return_value = conn
    return engine


class TestBlockFlowForSymbolAggregates:
    def test_returns_default_without_engine(self):
        s = InsiderSignals()
        result = s.block_flow_for_symbol("RELIANCE")
        assert result["net_value_crore"] == 0
        assert result["buy_value_crore"] == 0
        assert result["sell_value_crore"] == 0

    def test_aggregates_buy_minus_sell(self):
        # Two rows: BUY 100cr (3 deals), SELL 30cr (1 deal)
        engine = _engine_with_block_flow_rows([
            ("BUY", 100.0, 3),
            ("SELL", 30.0, 1),
        ])
        s = InsiderSignals(db_engine=engine)
        result = s.block_flow_for_symbol("RELIANCE", days=5)
        assert result["buy_value_crore"] == 100.0
        assert result["sell_value_crore"] == 30.0
        assert result["net_value_crore"] == 70.0
        assert result["buyer_count"] == 3
        assert result["seller_count"] == 1


class TestHasSignificantBlockBuy:
    def test_true_above_threshold(self):
        engine = _engine_with_block_flow_rows([("BUY", 60.0, 1)])
        s = InsiderSignals(db_engine=engine)
        assert s.has_significant_block_buy("RELIANCE", threshold_crore=50, days=3) is True

    def test_false_below_threshold(self):
        engine = _engine_with_block_flow_rows([("BUY", 20.0, 1)])
        s = InsiderSignals(db_engine=engine)
        assert s.has_significant_block_buy("RELIANCE", threshold_crore=50, days=3) is False


class TestPromoterBuyingRecent:
    def test_true_when_buy_in_window(self):
        engine = MagicMock()
        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchone.return_value = (1,)  # count > 0
        conn.execute.return_value = cursor
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=False)
        engine.connect.return_value = conn
        s = InsiderSignals(db_engine=engine)
        assert s.promoter_buying_recent("RELIANCE", days=30) is True

    def test_false_no_engine(self):
        s = InsiderSignals()
        assert s.promoter_buying_recent("RELIANCE") is False


class TestInsiderNetPosition:
    def test_dominant_buy(self):
        engine = _engine_with_block_flow_rows([("BUY", 100.0, 3), ("SELL", 30.0, 1)])
        s = InsiderSignals(db_engine=engine)
        result = s.insider_net_position("RELIANCE", days=30)
        assert result["dominant"] == "BUY"

    def test_dominant_sell(self):
        engine = _engine_with_block_flow_rows([("BUY", 20.0, 1), ("SELL", 80.0, 3)])
        s = InsiderSignals(db_engine=engine)
        result = s.insider_net_position("RELIANCE", days=30)
        assert result["dominant"] == "SELL"

    def test_dominant_none_when_no_engine(self):
        s = InsiderSignals()
        result = s.insider_net_position("RELIANCE")
        assert result["dominant"] == "NONE"
