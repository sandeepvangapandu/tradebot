"""Tests for src/strategy/conditions_insider.py — insider/block condition helpers."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.strategy import conditions_insider


class TestBlockBuyRecent:
    def test_false_without_engine(self):
        # InsiderSignals returns 0 flow when engine is None → False
        assert conditions_insider.block_buy_recent("RELIANCE") is False

    def test_true_when_significant(self):
        with patch("src.strategy.conditions_insider.InsiderSignals") as mock_cls:
            inst = MagicMock()
            inst.has_significant_block_buy.return_value = True
            mock_cls.return_value = inst
            assert conditions_insider.block_buy_recent("RELIANCE", db_engine=MagicMock()) is True

    def test_false_when_not_significant(self):
        with patch("src.strategy.conditions_insider.InsiderSignals") as mock_cls:
            inst = MagicMock()
            inst.has_significant_block_buy.return_value = False
            mock_cls.return_value = inst
            assert conditions_insider.block_buy_recent("RELIANCE", db_engine=MagicMock()) is False


class TestBlockSellRecent:
    def test_true_when_sell_dominates_threshold(self):
        with patch("src.strategy.conditions_insider.InsiderSignals") as mock_cls:
            inst = MagicMock()
            inst.block_flow_for_symbol.return_value = {
                "buy_value_crore": 0.0,
                "sell_value_crore": 100.0,
                "net_value_crore": -100.0,
            }
            mock_cls.return_value = inst
            assert conditions_insider.block_sell_recent(
                "RELIANCE", threshold_crore=50, db_engine=MagicMock()
            ) is True


class TestPromoterBuying:
    def test_true_when_recent(self):
        with patch("src.strategy.conditions_insider.InsiderSignals") as mock_cls:
            inst = MagicMock()
            inst.promoter_buying_recent.return_value = True
            mock_cls.return_value = inst
            assert conditions_insider.promoter_buying("RELIANCE", db_engine=MagicMock()) is True

    def test_false_without_engine(self):
        assert conditions_insider.promoter_buying("RELIANCE") is False


class TestPromoterSelling:
    def test_true_when_recent(self):
        with patch("src.strategy.conditions_insider.InsiderSignals") as mock_cls:
            inst = MagicMock()
            inst.promoter_selling_recent.return_value = True
            mock_cls.return_value = inst
            assert conditions_insider.promoter_selling("RELIANCE", db_engine=MagicMock()) is True


class TestInstitutionalFlowBullish:
    def test_true_when_net_above_threshold(self):
        with patch("src.strategy.conditions_insider.InsiderSignals") as mock_cls:
            inst = MagicMock()
            inst.block_flow_for_symbol.return_value = {"net_value_crore": 80.0}
            mock_cls.return_value = inst
            # Default threshold = 20.0; 80 >= 20 → True
            assert conditions_insider.institutional_flow_bullish(
                "RELIANCE", db_engine=MagicMock()
            ) is True

    def test_false_when_net_below_threshold(self):
        with patch("src.strategy.conditions_insider.InsiderSignals") as mock_cls:
            inst = MagicMock()
            inst.block_flow_for_symbol.return_value = {"net_value_crore": 5.0}
            mock_cls.return_value = inst
            assert conditions_insider.institutional_flow_bullish(
                "RELIANCE", db_engine=MagicMock()
            ) is False

    def test_false_without_engine(self):
        assert conditions_insider.institutional_flow_bullish("RELIANCE") is False
